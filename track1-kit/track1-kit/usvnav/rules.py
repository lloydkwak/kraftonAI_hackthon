"""The course rules a participant's tooling needs, without the generator.

`coursefile.validate` and the editor check hand-authored courses against rules the
generator honours by construction -- the reach's extent, the waypoint chain's shape, the
terrain margin. They live here so that the validator works without the generator, which
is not in the kit.
"""

from __future__ import annotations

import numpy as np

REACH_X, REACH_Y = 400.0, 150.0          # the world box, m (rule `reach`)
LEG_MIN, LEG_MAX = 40.0, 160.0           # leg length, m (rule `waypoints`); a U-turn leg back downstream needs the room
WP_MIN, WP_MAX = 2, 10                   # waypoint count (rule `waypoints`); an out-and-back route has the return's waypoints too
RADIUS_INTERMEDIATE, RADIUS_GOAL = 5.0, 3.0   # arrival radii, m (rule `radius`)


def _terrain_margin() -> float:
    """The terrain margin, derived from the raster rather than restated as a constant.

    The raster's square extent reaches 70.71 m into its corners -- further than the 50 m
    sensing range -- so a smaller margin leaves `outside_world` reachable there, and
    Track 1's palette has no colour for that class on purpose (it raises). Deriving the
    margin from `reach_m` means changing the raster cannot silently reintroduce the gap.
    """
    from .render import TRACK1
    return TRACK1.reach_m


TERRAIN_MARGIN = _terrain_margin()       # 70.71 m: the square raster's corner reach


def chain_conforms(course):
    """Waypoint-chain conformance: `WP_MIN`-`WP_MAX` waypoints (2-10), legs of
    `LEG_MIN`-`LEG_MAX` m (40-160) from the start pose onward.

    Checked rather than assumed: the generator rejects and redraws a course that fails
    it, and the validator applies the same test to a hand-authored course.
    """
    k = course.n_waypoints
    if not WP_MIN <= k <= WP_MAX:
        return False, f"{k} waypoints; {WP_MIN}-{WP_MAX} are allowed"
    chain = [np.asarray(course.start[:2], dtype=float)] + [np.asarray(w, dtype=float)
                                                           for w in course.waypoints]
    for i, (a, b) in enumerate(zip(chain, chain[1:])):
        d = float(np.linalg.norm(b - a))
        if not LEG_MIN - 1e-6 <= d <= LEG_MAX + 1e-6:
            return False, f"leg {i} is {d:.0f} m; {LEG_MIN:.0f}-{LEG_MAX:.0f} m is allowed"
    return True, "ok"
