"""A Track 1 course file as a Track 2 episode (the studio's document format).

The studio embeds Track 1's map editor unchanged, and the editor writes `usvnav-course/1`. That
file already holds what a 2-2 episode is made of, under Track 1's names:

    world.boundary        the water's edge (a closed polygon)          -> Episode.boundary
    bodies                buoys, piers, moored vessels, docks          -> Episode.bodies
    start                 where the ego begins, and its heading        -> Episode.init_state
    traffic[0]            a vessel on a closed route at a speed        -> the guide's path
    waypoints             Track 1's goals                              -> not used here

The first traffic route is the guide: the loop is followed at its speed from its phase, corners
rounded over a couple of seconds so the guide's heading does not step. Its speed is capped at the
kit's `path.V_MAX`. `d_target`, the horizon and the seed are not in the file; the
studio asks for them beside the map.

`episode(course, d_target, horizon, seed)` builds the `Episode`; `check(course)` lists what the
file lacks or what was adjusted, in the editor's `{rule, severity, message}` form.
"""
import numpy as np

from .raster import USVNAV  # noqa: F401
from usvnav import render as _r
from usvnav.geometry import Circle, Rect

from . import closedloop as C, path as P, scenes as S
from .samples import DT, wrap

FORMAT = 'usvnav-course/1'
#: the editor's class names -> the raster's (identical where the renderer defines them)
KIND = {'buoy': 'buoy', 'pier': 'pier', 'dock': 'dock',
        'moored_vessel': _r.MOORED, 'moving_vessel': _r.MOVING}
CORNER_S = 2.0          # seconds over which a route corner's heading change is spread


class CourseError(ValueError):
    pass


def _need(d, key, what):
    if not isinstance(d, dict) or key not in d:
        raise CourseError(f"{what} lacks {key!r}")
    return d[key]


def bodies_of(course):
    out = []
    for i, b in enumerate(course.get('bodies', [])):
        cls = _need(b, 'class', f"bodies[{i}]")
        kind = KIND.get(cls)
        if kind is None:
            raise CourseError(f"bodies[{i}].class {cls!r} is not one of {sorted(KIND)}")
        x, y = float(_need(b, 'x', f"bodies[{i}]")), float(_need(b, 'y', f"bodies[{i}]"))
        if b.get('primitive', 'rect') == 'circle':
            out.append((kind, Circle(x, y, 0.5 * float(_need(b, 'diameter_m', f"bodies[{i}]")))))
        else:
            out.append((kind, Rect(x, y, float(_need(b, 'length_m', f"bodies[{i}]")),
                                   float(_need(b, 'width_m', f"bodies[{i}]")), float(b.get('heading_rad', 0.0)))))
    return out


def route_path(route, secs):
    """A closed route at a constant speed -> `GuidePath` covering `secs` seconds."""
    pts = np.asarray(_need(route, 'route', 'traffic'), float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise CourseError("the guide's route needs at least two [x, y] points")
    speed = min(float(route.get('speed_mps', 1.0)), P.V_MAX)
    if speed <= 0:
        raise CourseError("the guide's speed must be positive")
    loop = np.vstack([pts, pts[:1]])                      # closed
    seg = np.diff(loop, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    total = float(seg_len.sum())
    if total < 1.0:
        raise CourseError("the guide's route is shorter than a metre")
    n = int(round(secs / DT)) + 1
    s = (float(route.get('phase_m', 0.0)) + np.arange(n) * speed * DT) % total
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    idx = np.clip(np.searchsorted(cum, s, side='right') - 1, 0, len(seg) - 1)
    frac = (s - cum[idx]) / np.maximum(seg_len[idx], 1e-9)
    xy = loop[idx] + seg[idx] * frac[:, None]
    heading = np.arctan2(seg[idx, 1], seg[idx, 0])
    # round the corners: the heading's steps spread over CORNER_S seconds (a boat behind cannot follow a step)
    w = max(int(round(CORNER_S / DT)), 1)
    if w > 1 and n > w:
        unwrapped = np.unwrap(heading)
        kernel = np.ones(w) / w
        padded = np.concatenate([np.full(w // 2, unwrapped[0]), unwrapped, np.full(w - 1 - w // 2, unwrapped[-1])])
        heading = wrap(np.convolve(padded, kernel, mode='valid'))
    poses = np.column_stack([xy, heading])
    v = np.full(n, speed)
    r = np.zeros(n)
    r[:-1] = wrap(np.diff(poses[:, 2])) / DT
    r[-1] = r[-2] if n > 1 else 0.0
    return P.GuidePath(poses, v, r, family='route')


def episode(course, d_target=P.D_TARGET, horizon=C.HORIZON, seed=0, init_state=None):
    """The `Episode` a course file describes. The ego starts at the course's `start`, at rest, unless
    `init_state` overrides it."""
    if course.get('format') != FORMAT:
        raise CourseError(f"format is {course.get('format')!r}, expected {FORMAT!r}")
    world = _need(course, 'world', 'the course')
    boundary = np.asarray(_need(world, 'boundary', 'world'), float)
    if boundary.ndim != 2 or boundary.shape[1] != 2 or len(boundary) < 3:
        raise CourseError("world.boundary must be at least 3 [x, y] pairs")
    traffic = course.get('traffic') or []
    if not traffic:
        raise CourseError("the course has no traffic loop: add one -- it is the guide's path")
    secs = (int(horizon) + 2) * C.PER * DT
    path = route_path(traffic[0], secs)
    if init_state is None:
        st = _need(course, 'start', 'the course')
        init_state = np.array([float(st['x']), float(st['y']), float(st.get('heading_rad', 0.0)), 0.0, 0.0, 0.0])
    return C.Episode(path, bodies_of(course), boundary, init_state, np.zeros(3), d_target=float(d_target),
                     horizon=int(horizon), seed=int(seed), family='route')


def check(course, d_target=P.D_TARGET):
    """What the file lacks or what the loader adjusts, as `{rule, severity, message}` rows; an
    `error` row means `episode()` would refuse it."""
    rows = []
    try:
        ep = episode(course, d_target)
    except CourseError as exc:
        return [dict(rule='episode', severity='error', message=str(exc))]
    traffic = course.get('traffic') or []
    speed = float(traffic[0].get('speed_mps', 1.0))
    if speed > P.V_MAX:
        rows.append(dict(rule='guide', severity='warning',
                         message=f"the guide's route speed {speed:.2f} m/s is capped at {P.V_MAX} m/s"))
    if len(traffic) > 1:
        rows.append(dict(rule='guide', severity='warning',
                         message=f"{len(traffic)} traffic loops: the first is the guide, the others are not used"))
    if course.get('waypoints'):
        rows.append(dict(rule='waypoints', severity='warning',
                         message="waypoints are Track 1's goals and are not used by 2-2"))
    gx, gy, _ = ep.path.pose(0)
    d0 = float(np.hypot(gx - ep.init_state[0], gy - ep.init_state[1]))
    reach = 32.0
    if d0 > reach:
        rows.append(dict(rule='start', severity='warning',
                         message=f"the guide starts {d0:.0f} m from the ego; beyond {reach:.0f} m it is out of the camera's frame, "
                                 f"and {C.LOSS_STEPS} decisions without it end the episode"))
    elif d0 < 7.0:
        rows.append(dict(rule='start', severity='error',
                         message=f"the guide starts {d0:.1f} m from the ego's centre: inside the hulls"))
    if not C._in_polygon(ep.init_state[0], ep.init_state[1], ep.boundary):
        rows.append(dict(rule='start', severity='error', message="the ego starts on the bank"))
    off = [k for k in range(0, ep.path.n, 20) if not C._in_polygon(ep.path.poses[k, 0], ep.path.poses[k, 1], ep.boundary)]
    if off:
        rows.append(dict(rule='guide', severity='warning',
                         message=f"the guide's route leaves the water for {len(off) * 20 * DT:.0f} s of the episode"))
    return rows
