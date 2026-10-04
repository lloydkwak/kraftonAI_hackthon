"""The guide vessel's path: what the camera draws ahead of the ego and what the score is measured
against.

A `GuidePath` is a sequence of poses at `DT`: where the guide is at every tick of an episode. How
that sequence is made is the author's business -- a scripted manoeuvre, a recorded track, a course
drawn by hand, `GuidePath.from_rates` over a speed and yaw-rate profile of your own. The harness
only reads poses. The organisers' hidden episodes come from a generator that is not in the kit;
the problem statement says what kind of motion it produces.

**The guide is kinematic, not a second plant.** It is scene content and defines the task;
nothing in 2-2 is scored on how the guide moves, only on how the ego follows it (in its wake, `station.py`).

**`V_MAX` and `A_MAX` are this kit's constants, not a description of the hidden episodes.**
`V_MAX` caps the speed of a guide the studio builds from a course file (`course.py`), below the
2.0 m/s the shipped plant reaches at full stern thrust; `A_MAX` is a gentle acceleration you may use
in profiles of your own, and nothing in the kit enforces it. Of the guide's motion in the organisers'
hidden episodes, the problem statement promises only that its speed, acceleration and curvature stay
within what the USV can follow.
"""
import copy

import numpy as np

from .raster import USVNAV  # noqa: F401  (puts Track 1's renderer on the path first)
from usvnav.geometry import Circle, Rect
from usvnav.render import MOVING

from .samples import DT, rear_slot, wrap

#: The default distance to keep, metres. `d_target` varies per episode in
#: [D_MIN, D_MAX] and may change during an episode. Below 8 m the hull gap is ~2 m and a stopping
#: guide is a collision; above 18 m the guide leaves the top of the 64 m raster in a turn.
D_TARGET = 12.0
D_MIN, D_MAX = 8.0, 18.0

#: The guide's hull, 7.0 x 2.6 m: 14 x 5 px at 0.5 m/px. For physics it is a vessel like any
#: other underway vessel; the camera draws it in its own light colour with a black centre mark,
#: the way a particular boat is recognised.
GUIDE_LENGTH, GUIDE_WIDTH = 7.0, 2.6
GUIDE_HULL = 'guide_hull'
GUIDE_MARK = 'guide_mark'
GUIDE_MARK_M = 1.5

V_MAX = 1.5             # m/s, the kit's speed cap for a guide built from a course file (course.py)
A_MAX = 0.15            # m/s^2, a gentle acceleration for profiles of your own; nothing enforces it


def integrate(v, r, start=(0.0, 0.0, 0.0)):
    """Unicycle integration at `DT`: speed `v` and yaw rate `r` per tick -> poses `(n, 3)`."""
    n = len(v)
    poses = np.empty((n, 3))
    x, y, psi = start
    for k in range(n):
        poses[k] = (x, y, psi)
        x += v[k] * np.cos(psi) * DT
        y += v[k] * np.sin(psi) * DT
        psi = wrap(psi + r[k] * DT)
    return poses


def rates(poses):
    """Speed and yaw rate per tick from a pose sequence, by finite differences: what `GuidePath`
    uses when it is given poses alone."""
    p = np.asarray(poses, float)
    n = len(p)
    v, r = np.zeros(n), np.zeros(n)
    if n > 1:
        d = np.diff(p[:, :2], axis=0)
        v[:-1] = np.hypot(d[:, 0], d[:, 1]) / DT
        r[:-1] = wrap(np.diff(p[:, 2])) / DT
        v[-1], r[-1] = v[-2], r[-2]
    return v, r


class GuidePath:
    """The guide's motion for one episode: poses `(n, 3)` = (x, y, heading) at every tick of `DT`,
    with its speed `v` and yaw rate `r` per tick (derived from the poses unless given)."""

    def __init__(self, poses, v=None, r=None, family=None):
        self.poses = np.asarray(poses, float).reshape(-1, 3).copy()
        self.n = len(self.poses)
        if self.n == 0:
            raise ValueError("a guide path needs at least one pose")
        if v is None or r is None:
            dv, dr = rates(self.poses)
            v = dv if v is None else v
            r = dr if r is None else r
        self.v, self.r = np.asarray(v, float), np.asarray(r, float)
        if len(self.v) != self.n or len(self.r) != self.n:
            raise ValueError("v and r must have one value per pose")
        self.secs = self.n * DT
        self.family = family

    @classmethod
    def from_rates(cls, v, r, start=(0.0, 0.0, 0.0), family=None):
        """A path from a speed and yaw-rate profile per tick, integrated from `start`."""
        v, r = np.asarray(v, float), np.asarray(r, float)
        return cls(integrate(v, r, start), v, r, family)

    def shifted(self, k0):
        """The same course, entered `k0` ticks in: every array re-indexed so tick 0 of the episode
        is tick `k0` of the course."""
        k0 = int(max(0, min(k0, self.n - 1)))
        out = copy.copy(self)
        out.v, out.r, out.poses = self.v[k0:], self.r[k0:], self.poses[k0:]
        out.n = len(out.v)
        out.secs = out.n * DT
        return out

    def pose(self, k):
        return self.poses[min(k, self.n - 1)]

    def slot(self, k, d_target=D_TARGET):
        """The point `d_target` behind the guide along its heading (where `nominal()` starts the ego)."""
        return rear_slot(self.pose(k), d_target)

    def slot_velocity(self, k, d_target=D_TARGET):
        """World velocity of the slot: the guide's, plus the sweep of the slot as the guide turns."""
        k = min(k, self.n - 1)
        x, y, psi = self.poses[k]
        v, r = self.v[k], self.r[k]
        return np.array([v * np.cos(psi) + d_target * r * np.sin(psi),
                         v * np.sin(psi) - d_target * r * np.cos(psi)])

    def slot_course(self, k, d_target=D_TARGET):
        """Direction the slot is moving, radians. Equal to the guide's heading on a straight and
        rotated by `atan(d_target * r / v)` in a turn -- the slot traces a circle of radius
        `sqrt(R^2 + d^2)`, outside the guide's, so its tangent is not the guide's heading."""
        k = min(k, self.n - 1)
        vel = self.slot_velocity(k, d_target)
        if np.hypot(*vel) < 0.15:
            return self.poses[k][2]
        return float(np.arctan2(vel[1], vel[0]))

    def body(self, k):
        """The guide's hull, for physics (collision) and as the vessel-class body it is."""
        x, y, psi = self.pose(k)
        return (MOVING, Rect(x, y, GUIDE_LENGTH, GUIDE_WIDTH, psi))

    def render_bodies(self, k):
        """What the camera draws for the guide: the hull in `GUIDE_HULL`'s light colour and
        a `GUIDE_MARK` disc at its centre. Both are classes outside the renderer's painting order,
        so they paint after every other class, hull first then mark."""
        x, y, psi = self.pose(k)
        return [(GUIDE_HULL, Rect(x, y, GUIDE_LENGTH, GUIDE_WIDTH, psi)),
                (GUIDE_MARK, Circle(x, y, GUIDE_MARK_M / 2))]


def nominal(path, rng=None, d_target=D_TARGET):
    """On station at tick 0: at the slot, aligned, moving with the guide, actuators at rest.
    Returns `(state, actuator)` for an `Episode`."""
    x, y, psi = path.slot(0, d_target)
    return np.array([x, y, psi, path.v[0], 0.0, 0.0]), np.zeros(3)
