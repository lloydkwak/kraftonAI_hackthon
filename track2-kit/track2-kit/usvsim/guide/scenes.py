"""Scenes for 2-2: what is on the water around the guide, in Track 1's class vocabulary.

A 2-2 record is one top-down raster plus the state and action at that decision. The ego
sits at the centre of every frame pointing up (`bow_up`), so it never moves in the image:
*the motion is visible only in how the surroundings shift*. What is in the scene is therefore not
decoration, it is the signal, and composing it is the author's design.

A scene is a list of bodies from `body()` and, optionally, a `boundary`: the water's edge as a
closed polygon, water inside, bank outside. How many bodies, of what kinds, where -- and how the
organisers compose the hidden episodes' scenes -- is not in the kit; the released
real-driving episodes show scenes through the reference camera.
"""
import numpy as np

from .raster import USVNAV  # noqa: F401  (ensures the renderer is importable first)
from usvnav import render as _r
from usvnav.geometry import Circle, Rect

#: The kinds of body a scene may hold, with plausible size ranges in metres: `(kind, min, max)`
#: for round bodies (a radius), `(kind, min_length, max_length, min_width, max_width)` for hulls
#: and structures.
BODY_KINDS = (
    ('buoy',            0.30, 0.60, None, None),
    (_r.MOORED,         6.0, 18.0, 2.0, 5.0),
    (_r.MOVING,         5.0, 14.0, 1.8, 4.0),
    ('pier',            8.0, 30.0, 3.0, 8.0),
    ('dock',            6.0, 20.0, 4.0, 12.0),
)
KINDS = tuple(k for k, *_ in BODY_KINDS)


def body(kind, x, y, psi=0.0, size=None):
    """One scene body at world `(x, y)` with heading `psi`. `size` is a radius (buoy) or
    `(length, width)` in metres; the middle of the kind's range when omitted."""
    for k, lo, hi, wlo, whi in BODY_KINDS:
        if k == kind:
            break
    else:
        raise ValueError(f"unknown body kind {kind!r}; one of {KINDS}")
    if wlo is None:
        return kind, Circle(float(x), float(y), float(size if size is not None else 0.5 * (lo + hi)))
    if size is None:
        size = (0.5 * (lo + hi), 0.5 * (wlo + whi))
    return kind, Rect(float(x), float(y), float(size[0]), float(size[1]), float(psi))


def terrain_for(poses, reach_m, pad=2.0):
    """Terrain extent guaranteed to cover everything the observer can see from `poses`.

    The square raster reaches its *half-diagonal*, not its half-width, so pass the raster's
    `reach_m` (`raster.TRACK2.reach_m`), not its range: a frame that reaches past the terrain
    contains `outside_world`, and the renderer raises. Deriving the extent from the poses that
    will actually be rendered keeps it right whatever the trajectory length.
    """
    p = np.asarray(poses, float)[:, :2]
    lo = p.min(axis=0) - reach_m - pad
    hi = p.max(axis=0) + reach_m + pad
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


def scene_at(bodies, boundary, terrain, pose, hull=(4.4, 1.9)):
    """The renderer's `Scene` for one observer pose `(x, y, psi)`."""
    x, y, psi = pose
    return _r.Scene(observer=(x, y, psi), bodies=bodies, boundary=boundary,
                    terrain_extent=terrain,
                    ego_hull=Rect(x, y, hull[0], hull[1], psi))
