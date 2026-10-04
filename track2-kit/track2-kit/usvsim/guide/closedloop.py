"""Closed-loop episodes for 2-2: the environment a policy drives in, and the recorder that writes
the records a dataset is made of.

One object, `Env`, takes an `Episode` -- the guide's path, the bodies on the water, where the ego
starts, the distance to keep -- and a plant, and steps them at the 0.5 s decision period: render the
camera, measure, ask the policy, apply the command for 0.5 s, score. The same class records a
dataset (drive it with a controller of yours, `record=True`) and scores a policy (the organisers
drive it with the trained learner, in their environment); the only thing that changes between the
two is who chooses the action. Because recording and scoring are one code path, the alignment rule
-- record[t]'s observation determines record[t]'s action, applied over the following interval --
is the same in both, and a one-step slip between them cannot happen by accident.

**What the policy sees**: the last four frames at stride 3 -- decisions `k-9, k-6, k-3, k`
spanning 4.5 s -- with `ego_motion` and `actuator_state` at the same four decisions and a
`history_mask` for the ones before the episode began. No window crosses an episode start,
because an episode *is* one `Env`. Frames are rendered once per decision and the window is an
index into them, which is also how the learner stores a dataset.

**What ends an episode early**: the guide out of the raster for `LOSS_STEPS` consecutive
decisions, the ego hull overlapping a scene body or the guide, or the ego leaving the water.
Every remaining step then scores `q_t = 0`, so precision and survival are one number.

**Measurement.** `ego_motion` carries 2-1's measurement model (`measure.SIGMA`): the environment
measures state, it does not know it. Actuator state is the plant's achieved `[T_stern, delta,
T_bow]` -- feedback, not the previous command.

**What is not here.** The camera here is clean: the exact top view -- the studio's picture, the
same geometry and colours as the reference environment's camera. What that camera adds to the
picture, and how the hidden episodes are composed, is withheld; the released real-driving
episodes show what its pictures look like. Anything you put on top of a frame before it goes into
your dataset is your design, and **`Env(camera=...)` is where it goes**: a function
`camera(frame, ctx) -> frame` called on every clean frame as it is rendered, with `ctx` (a
`CameraContext`) carrying what an image model needs and a stored frame does not have -- the time,
the decision index, the ego's pose, the frame's class map, its water mask, the world coordinates
of every pixel and the episode's seed. The studio has a selector for the functions in its work
folder's `camera.py`; `resolve_camera` accepts the same spellings from code and the command line.
"""
import os

import numpy as np

from . import path as guide, raster as R, scenes, station as W
from .measure import SIGMA
from .samples import DT, rel_pose, wrap

PER = int(round(0.5 / DT))                      # ticks per decision (0.5 s)
STRIDE, WINDOW = 3, 4                           # four frames, every third decision
if os.environ.get('T2_STRIDE'):                  # a diagnostic hook; never set in scoring
    STRIDE = int(os.environ['T2_STRIDE'])
LOSS_STEPS = 6                                  # guide out of frame for 3 s ends the episode
HORIZON = 120                                   # decisions; 60 s
EGO_HULL = (4.4, 1.9)

#: Widths in the guide's frame (m, m, rad) for diagnostics only; they do not enter the score.
SIGMA_SCORE = dict(long=2.0, lat=1.5, psi=0.35)
#: The score: the follower belongs in the guide's wake -- on the guide's own track, `d_target` behind the
#: guide measured along the track or in a straight line, up to `station.BAND` metres to either side without
#: loss. q_t = exp(-1/2 (e_s / 2 m)^2) x exp(-1/2 (max(0, n - 2 m) / 2 m)^2), `e_s` the along-track distance
#: from the station stretch, `n` the distance from the track (`station.py`). A follower that cuts inside a
#: loop the guide is circling scores close to zero, even at the right straight-line distance. `SCORE_ID`
#: names this scoring rule.
SIGMA_D = W.SIGMA                                # along-track width
SCORE_ID = f'wake/{W.SIGMA:.1f}m/band{W.BAND:.1f}m'


class Episode:
    """Everything that defines one episode, as values.

    path            a `path.GuidePath`: where the guide is at every tick
    bodies          the static scene: `(kind, shape)` pairs from `scenes.body()`
    boundary        the water's edge, a closed polygon `(k, 2)` with the water inside; None = open water
    init_state      the ego's `[x, y, heading, surge, sway, yaw_rate]` at tick 0; on station (`path.nominal`)
                    when omitted
    init_actuator   the achieved `[T_stern, delta, T_bow]` at tick 0; at rest when omitted
    d_target        the distance to keep, metres, in [D_MIN, D_MAX]
    d_schedule      `[(decision, new d_target), ...]`: changes of station during the episode
    horizon         decisions in the episode (0.5 s each)
    seed            the measurement-noise stream
    end_on_guide_loss, end_on_collision   the early terminations; both on when scoring
    terrain         the renderer's terrain extent; derived from the path, the boundary and the start
                    when None. Open water with no boundary: keep the ego inside the derived extent
    group, family, episode_id   labels copied into the result, for your bookkeeping
    """

    def __init__(self, path, bodies=(), boundary=None, init_state=None, init_actuator=None,
                 d_target=guide.D_TARGET, d_schedule=None, horizon=HORIZON, seed=0,
                 end_on_guide_loss=True, end_on_collision=True, terrain=None,
                 group=None, family=None, episode_id=0):
        self.path = path
        self.bodies = list(bodies)
        self.boundary = None if boundary is None else np.asarray(boundary, float)
        if init_state is None:
            init_state, a0 = guide.nominal(path, None, d_target)
            init_actuator = a0 if init_actuator is None else init_actuator
        self.init_state = np.asarray(init_state, float).reshape(6)
        self.init_actuator = (np.zeros(3) if init_actuator is None
                              else np.asarray(init_actuator, float).reshape(3))
        self.d_target = float(d_target)
        self.d_schedule = [(int(k), float(d)) for k, d in (d_schedule or [])]
        self.horizon = int(horizon)
        self.seed = int(seed)
        self.end_on_guide_loss = bool(end_on_guide_loss)
        self.end_on_collision = bool(end_on_collision)
        self.terrain = terrain
        self.group = group
        self.family = family if family is not None else getattr(path, 'family', None)
        self.episode_id = int(episode_id)

    def __repr__(self):
        return (f"Episode({self.family or 'path'}, seed={self.seed}, d_target={self.d_target}, "
                f"horizon={self.horizon}, bodies={len(self.bodies)})")


def _in_polygon(x, y, poly):
    px, py = poly[:, 0], poly[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    cross = ((py > y) != (qy > y)) & (x < (qx - px) * (y - py) / (qy - py + 1e-12) + px)
    return bool(np.count_nonzero(cross) % 2)


def _hits(x, y, psi, shape, pad):
    """Does the ego (centre x, y, heading psi) overlap `shape`, padded by `pad` metres?
    Ego approximated by a disc of its half-length -- conservative, and cheap."""
    rad = 0.5 * EGO_HULL[0] + pad
    if hasattr(shape, 'r'):
        return np.hypot(shape.x - x, shape.y - y) < shape.r + rad
    dx, dy = x - shape.x, y - shape.y
    c, s = np.cos(shape.heading), np.sin(shape.heading)
    lx, ly = dx * c + dy * s, -dx * s + dy * c
    return abs(lx) < 0.5 * shape.length + rad and abs(ly) < 0.5 * shape.width + rad

class Env:
    """`plant` is a plant object (`reset(state)`, `step(cmd, dt)`, `.s`, `.act` -- the shipped
    plant behind that interface is `ego.PublicPlant`), a callable returning one, or the string
    'public' for the shipped plant. `episode` is an `Episode`."""

    draw_guide = True           #: False draws no guide (the released real-driving frames have none)

    def __init__(self, plant, episode, plant_label='public', camera=None):
        ep = self.sc = episode
        self.plant_label = plant_label
        self.camera = resolve_camera(camera)
        self.camera_label = camera_label(camera)
        self.path = ep.path
        self.bodies = list(ep.bodies)
        self.boundary = ep.boundary
        if ep.terrain is not None:
            self.terrain = ep.terrain
        else:
            # Terrain must cover everywhere the ego can *be while still on the water*, not just the
            # guide's path: a policy that drives off is rendered right up to the bank, and a frame
            # that reaches past the terrain raises `outside_world` in the renderer. So the extent is
            # the shoreline's bounding box plus the raster's reach, and the water polygon does the rest.
            pts = [self.path.poses[:, :2], ep.init_state[None, :2]]
            if self.boundary is not None:
                pts.append(self.boundary)
            self.terrain = scenes.terrain_for(np.concatenate(pts), R.TRACK2.reach_m + 5.0)
        self.d = ep.d_target                      # current station, may change (d_schedule)
        self.track = W.Track(self.path.poses)     # the guide's own track, extended behind its start
        self.vessel = self._make_plant(plant)
        self.vessel.reset(ep.init_state.copy())
        self.vessel.act = np.array(ep.init_actuator, float)
        self.mrng = np.random.RandomState(ep.seed * 13 + 7)     # measurement noise
        self.k = 0                  # decision index
        self.tick = 0
        self.frames, self.motion, self.actuator, self.dts = [], [], [], []
        self.lost_for = 0
        self.done = None            # None while running, else the reason
        self.q = []                 # per-decision score contributions
        self.err = []               # per-decision (e_long, e_lat, e_psi) in the guide's frame -- diagnostics
        self.rd = []                # per-decision r_t - d_t (centre distance minus d_target) -- diagnostics
        self.es, self.en = [], []   # per-decision score errors: along the track from the station, across the track

    @staticmethod
    def _make_plant(plant):
        return make_plant(plant)

    # -- state -----------------------------------------------------------------------

    @property
    def pose(self):
        return tuple(self.vessel.s[:3])

    def guide_pose(self):
        return self.path.pose(self.tick)

    def distance(self):
        """Centre-to-centre distance from the ego to the guide, metres (a diagnostic; not the score)."""
        gx, gy, _ = self.guide_pose()
        return float(np.hypot(gx - self.vessel.s[0], gy - self.vessel.s[1]))

    def error(self):
        """(e_long, e_lat, e_psi): the ego relative to the slot (`path.GuidePath.slot`), in the guide's
        frame. **A diagnostic** (the `rms` of a run's result); the score is `station.py`'s.

        `e_psi` is measured against the **slot's course**, not the guide's heading. The two
        coincide on a straight; in a turn of radius R the slot moves on a circle of radius
        sqrt(R^2 + d^2) and its course is rotated by atan(d/R) from the guide's heading --
        25.6 deg at R = 25 m. An ego holding the slot points along the slot's course, so it
        has e_psi = 0 in a steady turn.
        """
        f, p, _ = rel_pose(self.guide_pose(), self.pose)
        dpsi = wrap(self.pose[2] - self.path.slot_course(self.tick, self.d))
        return np.array([f + self.d, p, dpsi])

    def guide_in_frame(self):
        f, p, _ = rel_pose(self.pose, self.guide_pose())
        half = 0.5 * R.span_m()
        return abs(f) <= half and abs(p) <= half

    def collided(self):
        x, y, psi = self.pose
        if self.boundary is not None and not _in_polygon(x, y, self.boundary):
            return 'bank'
        if _hits(x, y, psi, self.path.body(self.tick)[1], 0.0):
            return 'guide'
        for _, shape in self.bodies:
            if abs(shape.x - x) < 25 and abs(shape.y - y) < 25 and _hits(x, y, psi, shape, 0.0):
                return 'body'
        return None

    # -- observation -----------------------------------------------------------------

    def scene(self):
        """The renderer's scene at this tick: the bodies, the guide as the camera draws it, the ego."""
        bodies = self.bodies + (self.path.render_bodies(self.tick) if self.draw_guide else [])   # the guide: hull + mark
        return scenes.scene_at(bodies, self.boundary, self.terrain, self.pose, EGO_HULL)

    def render(self):
        """This decision's frame: the exact top view."""
        return R.render(self.scene(), R.TRACK2)

    def observe(self):
        """Render this decision's frame, put the camera (if any) on it, measure, and return the policy's window."""
        frame = self.render()
        if self.camera is not None:
            frame = apply_camera(self.camera, frame, CameraContext(self))
        self.frames.append(frame)
        s = self.vessel.s
        m = np.array([s[3] + self.mrng.normal(0, SIGMA['surge']),
                      s[4] + self.mrng.normal(0, SIGMA['sway']),
                      s[5] + self.mrng.normal(0, SIGMA['yawrate'])], np.float32)
        self.motion.append(m)
        self.actuator.append(np.array(self.vessel.act, np.float32))
        self.dts.append(np.float32(self.d))
        return window(self.frames, self.motion, self.actuator, self.k, self.dts)

    # -- dynamics --------------------------------------------------------------------

    def step(self, action):
        """Apply `action` over one decision period; score the state reached; check the end."""
        a = np.asarray(action, float)
        for _ in range(PER):
            self.vessel.step(a, DT)
            self.tick += 1
        self.k += 1
        for k0, d in self.sc.d_schedule:
            if k0 == self.k:
                self.d = d
        e = self.error()
        self.err.append(e)
        self.rd.append(self.distance() - self.d)
        e_s, n = self.track.errors(self.tick, self.d, self.vessel.s[:2])
        self.es.append(e_s); self.en.append(n)
        self.q.append(W.q(e_s, n))                                         # the score; rd is a diagnostic
        self.lost_for = 0 if self.guide_in_frame() else self.lost_for + 1
        hit = self.collided() if self.sc.end_on_collision else None
        if hit:
            self.done = 'collision:' + hit
        elif self.lost_for >= LOSS_STEPS and self.sc.end_on_guide_loss:
            self.done = 'guide_lost'
        elif self.k >= self.sc.horizon:
            self.done = 'horizon'
        return self.done

    def score(self):
        """The episode score: `100/H * sum q_t`, unreached steps worth zero."""
        return 100.0 * float(np.sum(self.q)) / self.sc.horizon


def window(frames, motion, actuator, k, dts=None):
    """The policy's input at decision `k`, oldest frame first, zero-padded before the episode.
    `d_target` is the distance to keep in force at each of the four decisions."""
    idx = [k - STRIDE * j for j in range(WINDOW - 1, -1, -1)]
    mask = np.array([i >= 0 for i in idx], np.float32)
    n = frames[0].shape[0]
    rgb = np.zeros((WINDOW, n, n, 3), np.uint8)
    mo = np.zeros((WINDOW, 3), np.float32)
    ac = np.zeros((WINDOW, 3), np.float32)
    dt = np.zeros(WINDOW, np.float32)
    for j, i in enumerate(idx):
        if i >= 0:
            rgb[j], mo[j], ac[j] = frames[i], motion[i], actuator[i]
            dt[j] = dts[i] if dts is not None else guide.D_TARGET
    return dict(rgb=rgb, ego_motion=mo, actuator_state=ac, history_mask=mask, d_target=dt)


# ------------------------------------------------------------------------------ the camera

class CameraContext:
    """What a camera function may read besides the frame, computed on demand.

    t              seconds since the episode began (0.5 per decision)
    k              the decision index; `tick` the plant tick
    pose           the ego's (x, y, heading) in world metres / radians -- the camera's pose
    seed           the episode's seed: key your randomness on (seed, k) and a dataset renders the same twice
    class_map      (128, 128) int: the class id at every pixel of the clean picture; `names[id]` is the class
    water          (128, 128) bool: the navigable water -- for effects that belong on the water only
    world_xy       (x, y) arrays (128, 128): the world position of every pixel centre, for effects that
                   stay put on the water while the camera moves
    cfg            the raster configuration (size, metres per pixel, frame)
    """

    def __init__(self, env):
        self._env = env
        self.k = env.k
        self.tick = env.tick
        self.t = env.tick * DT
        self.pose = tuple(float(v) for v in env.pose)
        self.seed = env.sc.seed
        self.cfg = R.TRACK2
        self._ids = None

    @property
    def names(self):
        return R.class_names(self.cfg)

    @property
    def class_map(self):
        if self._ids is None:
            self._ids = R.class_map(self._env.scene(), self.cfg)
        return self._ids

    @property
    def water(self):
        return self.class_map == self.names.index('water')

    @property
    def world_xy(self):
        from usvnav.look import world_coords
        return world_coords(self.cfg, self.pose, self.cfg.size_px, self.cfg.size_px)


def apply_camera(camera, frame, ctx):
    """`camera(frame, ctx)`, checked: the result must be the same shape and `uint8` -- the frame that
    goes into the record and the frame the learner trains on are one array."""
    out = camera(frame.copy(), ctx)
    out = np.asarray(out)
    if out.shape != frame.shape or out.dtype != np.uint8:
        raise ValueError(f"the camera returned {out.dtype} {out.shape}; a frame is uint8 {frame.shape}: "
                         "clip to [0, 255] and cast with .astype(np.uint8)")
    return out


def resolve_camera(spec):
    """`None` / 'none' -> no camera; a callable -> itself; 'module:function' -> imported; a path
    'dir/file.py:function' -> loaded from that file. The studio uses the last form for its work
    folder's `camera.py`."""
    if spec is None or callable(spec):
        return spec
    spec = str(spec).strip()
    if spec in ('', 'none'):
        return None
    if ':' not in spec:
        raise ValueError(f"camera {spec!r}: give 'module:function' or 'path/to/file.py:function' (or 'none')")
    where, _, fn = spec.rpartition(':')
    if where.endswith('.py'):
        import importlib.util
        mspec = importlib.util.spec_from_file_location('_camera_' + os.path.splitext(os.path.basename(where))[0], where)
        if mspec is None:
            raise ValueError(f"camera {spec!r}: cannot load {where}")
        mod = importlib.util.module_from_spec(mspec)
        mspec.loader.exec_module(mod)
    else:
        import importlib
        mod = importlib.import_module(where)
    target = getattr(mod, fn, None)
    if not callable(target):
        raise ValueError(f"camera {spec!r}: {where} has no function {fn}")
    return target


def camera_label(spec):
    """How a record names its camera: 'none', the function's name, or the spec as written."""
    if spec is None:
        return 'none'
    if callable(spec):
        return getattr(spec, '__name__', repr(spec))
    spec = str(spec).strip()
    return spec.rpartition(':')[2] if spec and spec != 'none' else 'none'


# ------------------------------------------------------------------------------ driving

def run_episode(env, policy, record=False):
    """Drive `env` with `policy` to termination. Returns a result dict, with the dataset records
    if asked. A `policy` is either privileged -- `policy.privileged` is true and it is called
    as `act(tick, state, actuator)` -- or observation-only, called as `act(obs)`."""
    priv = getattr(policy, 'privileged', False)
    if priv and hasattr(policy, 'bind'):
        policy.bind(env)
    actions = []
    while env.done is None:
        obs = env.observe()
        if priv:
            a = policy.act(env.tick, env.vessel.s.copy(), env.vessel.act.copy())
        else:
            a = policy.act(obs)
        actions.append(np.asarray(a, np.float32))
        env.step(a)
    err = np.array(env.err)
    rd = np.array(env.rd)
    out = dict(score=env.score(), steps=env.k, done=env.done, horizon=env.sc.horizon,
               rms=np.sqrt((err ** 2).mean(0)).tolist(), rms_d=float(np.sqrt((rd ** 2).mean())) if len(rd) else 0.0,
               group=env.sc.group,
               family=env.sc.family, seed=env.sc.seed, plant=env.plant_label, camera=env.camera_label)
    if record:
        out['records'] = dict(rgb=np.stack(env.frames), ego_motion=np.stack(env.motion),
                              actuator_state=np.stack(env.actuator), action=np.stack(actions),
                              step=np.arange(env.k, dtype=np.int32),
                              d_target=np.array(env.dts, np.float32),
                              error=err.astype(np.float32))
    return out


def make_plant(plant):
    """A plant object from whatever `Env` was handed: a plant object, a callable returning one, or
    'public' for the shipped plant."""
    if isinstance(plant, str):
        if plant == 'public':
            from .ego import PublicPlant
            return PublicPlant()
        raise ValueError(f"unknown plant {plant!r}: pass a plant object, a factory, or 'public'")
    if callable(plant) and not hasattr(plant, 'step'):
        return plant()
    return plant
