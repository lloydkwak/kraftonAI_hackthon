"""A deterministic pilot for tests that need a trajectory rather than an agent.

Several tests want the vessel to travel a course so they can look at something along the
way -- the raster at twenty poses, a replay reproducing tick for tick. What they do *not*
want is a competent agent.

This pilot reads only the waypoint chain every agent is given and steers along it. It
has no perception and no obstacle avoidance, so it is tutorial-grade by construction.
Being open-loop against obstacles is also what makes it a *better* fixture than an
avoider: the trajectory is a property of the course, not of a policy someone may improve
later.
"""

from __future__ import annotations

import math

import numpy as np

from usvnav.geometry import wrap
from usvnav.plant import V_CRUISE, V_MAX, W_MAX


class ChainPilot:
    """Pure pursuit along the disclosed chain. Deterministic; no state across episodes."""

    def __init__(self, config_path=None, *, gain: float = 1.4, cruise: float = V_CRUISE):
        self.gain = gain
        self.cruise = cruise

    def reset(self, meta):
        self.meta = meta

    def act(self, obs):
        dist, bearing = (float(v) for v in obs["wp_polar"])
        w = float(np.clip(self.gain * wrap(bearing), -W_MAX, W_MAX))
        # Slow for a sharp turn and on the final approach, so the trace has a range of
        # speeds in it rather than one.
        v = self.cruise * (1.0 - 0.6 * abs(w) / W_MAX)
        v = min(v, max(0.4, 0.5 * dist))
        return np.array([float(np.clip(v, 0.0, V_MAX)), w])


def chain_poses(course, step_m: float = 8.0):
    """Poses along the straight waypoint chain, tangent-aligned.

    Not a driven path -- a geometric sweep of the whole route, which is what a coverage
    check wants. An agent's trajectory stops where the agent does, so a test that asks
    "is `outside_world` ever reachable along the route" would only have checked as far as
    the agent got.
    """
    chain = [np.asarray(course.start[:2], dtype=float)]
    chain += [np.asarray(w, dtype=float) for w in course.waypoints]
    out = []
    for a, b in zip(chain, chain[1:]):
        seg = b - a
        L = float(np.linalg.norm(seg))
        if L < 1e-9:
            continue
        head = math.atan2(seg[1], seg[0])
        for s in np.arange(0.0, L, step_m):
            p = a + seg * (s / L)
            out.append((float(p[0]), float(p[1]), head))
    return out
