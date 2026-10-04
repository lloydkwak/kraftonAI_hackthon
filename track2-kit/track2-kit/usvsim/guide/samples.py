"""Shared helpers for 2-2: the timing constants, pose geometry, and per-index seeding.

`DT` is the plant tick (0.05 s), `DECISION_S` the decision period (one action per 0.5 s) and
`FRAME_STRIDE` the window stride (four frames, every third decision, spanning 4.5 s). `wrap`,
`rel_pose` and `rear_slot` are the angle and pose helpers the harness uses; `scene_seed` gives scene
`k` a random stream of its own.

`trajectory` and `decision_indices` produce open-loop ego paths (random commands, no guide). In the
kit `trajectory` needs a plant module that is not shipped; drive the public plant through
`closedloop.Env` instead.
"""
import numpy as np

from . import raster, scenes

DT = 0.05
DECISION_S = 0.5        # one action per 0.5 s
FRAME_STRIDE = 3        # the four-frame window is taken every third decision, 4.5 s

#: Seconds of trajectory discarded before the first record.
#:
#: `trajectory` starts the vessel at rest, so the opening seconds are an acceleration transient
#: rather than a draw from the motion distribution. With few records per episode, keeping them
#: would skew the whole set toward low speed.
WARMUP_S = 4.0


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def rel_pose(pose_from, pose_to):
    """`pose_to` expressed in `pose_from`'s body frame, as (forward, port, heading).

    Used for the guide-frame diagnostics (`closedloop.Env.error`) and for the check that the
    guide is inside the camera frame (`closedloop.Env.guide_in_frame`). Purely geometric.
    """
    x0, y0, p0 = pose_from
    x1, y1, p1 = pose_to
    dx, dy = x1 - x0, y1 - y0
    return np.array([dx * np.cos(p0) + dy * np.sin(p0),
                     -dx * np.sin(p0) + dy * np.cos(p0),
                     wrap(p1 - p0)])


def rear_slot(pose_guide, d_target):
    """The point `d_target` behind the guide along the guide's own heading, with the guide's heading.

    Where `path.nominal` starts the ego, and the reference point of the guide-frame diagnostics.
    The score uses the guide's track instead (`station.py`).
    """
    x, y, psi = pose_guide
    return np.array([x - d_target * np.cos(psi), y - d_target * np.sin(psi), psi])


def trajectory(rng, secs, cfg=None, seg_s=DECISION_S, start=(0.0, 0.0, 0.0)):
    """Poses at `DT`, driven by random commands held for `seg_s` (the decision period).

    The command distribution is wider than the 2-1 logs': a 2-2 dataset has to cover a *state*
    space, and a dataset whose motions all look alike teaches the policy one motion.
    """
    from . import track2                       # not in the kit: calling `trajectory` raises ImportError
    from .vessel import Vessel
    v = Vessel(track2.thruster_map, *(cfg or track2.pub_config()))
    v.reset(np.array([start[0], start[1], start[2], 0.0, 0.0, 0.0]))
    n = int(round(secs / DT))
    per = max(int(round(seg_s / DT)), 1)
    poses = np.empty((n, 3))
    states = np.empty((n, 6))
    cmds = np.empty((n, 3))
    cmd = None
    for k in range(n):
        if k % per == 0:
            cmd = [rng.uniform(-0.2, 1.0), rng.uniform(-1.4, 1.4), rng.uniform(-1.0, 1.0)]
        s = v.step(cmd, DT)
        poses[k], states[k], cmds[k] = s[:3], s, cmd
    return poses, states, cmds


def decision_indices(n_steps, decision_s=DECISION_S, warmup_s=WARMUP_S):
    """Tick indices at which a record is taken, after the warm-up."""
    step = max(int(round(decision_s / DT)), 1)
    off = int(round(warmup_s / DT))
    return list(range(off, n_steps, step))


def scene_seed(seed, k):
    """A stream per scene index, so scene `k` is the same scene however many are asked for.

    Drawing scenes from one running generator would make every dataset a prefix of a different
    sequence: a 150-scene set and a 300-scene set would share no scene, and the two could not be
    compared. It also makes rendering parallel over `k` for free.
    """
    return np.random.RandomState((int(seed) * 1000003 + int(k) * 7919) % (2 ** 31 - 1))


def render_at(bodies, boundary, terrain, pose, cfg=None):
    cfg = cfg or raster.TRACK2
    return raster.render(scenes.scene_at(bodies, boundary, terrain, pose), cfg)
