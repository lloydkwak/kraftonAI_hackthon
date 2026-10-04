"""The 2-2 camera: Track 2's configuration of Track 1's top-view renderer.

The renderer itself is Track 1's `usvnav` package, used unchanged (vendored beside this package in
the kit, else found through `USVNAV_PATH`). Track 2 supplies the configuration `TRACK2` and the
guide's two classes, nothing else.

**What a frame is.** A 128 x 128 RGB top view at 0.5 m/px -- 64 m across, +/- 32 m around the
ego -- in the `bow_up` frame: the ego at the centre, its bow toward the top of the image. Every
class of Track 1's vocabulary is drawn as the studio map draws it (`usvnav.look`): the map's base
colours, a wet shoreline, shadows, decks and outlines, painted at 0.25 m/px and box-downsampled to
0.5 m/px, so edges are anti-aliased. The kit's clean camera, the studio's frame panel and the
reference environment all draw this picture; the reference camera then adds effects of its own on
top, which are not in the kit (the released real-driving frames show the result).

**The guide** is drawn in two classes Track 2 adds to Track 1's vocabulary: a light grey-white
hull (`GUIDE_HULL`; no other vessel is light) with a black centre mark (`GUIDE_MARK`). They are
added to Track 1's look tables at import; nothing in Track 1 is modified.

`renderer_id()` stamps which build of the renderer drew a frame; `dataset.write` records it in
the manifest.
"""
import functools
import hashlib
import os
import subprocess
import sys

_KIT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#: Track 1's renderer: the copy vendored beside this package (the kit), else a Track 1 checkout.
def _track1_checkout():
    """Where Track 1's `usvnav` package is when the kit carries no copy: `USVNAV_PATH` if set, else a
    checkout in the home directory."""
    env = os.environ.get('USVNAV_PATH')
    if env:
        return env
    for d in ('~/usvnav', '~/mallard'):
        if os.path.isdir(os.path.join(os.path.expanduser(d), 'usvnav')):
            return os.path.expanduser(d)
    return os.path.expanduser('~/usvnav')


USVNAV = (_KIT_ROOT if os.path.isfile(os.path.join(_KIT_ROOT, 'usvnav', 'render.py')) else _track1_checkout())
if USVNAV not in sys.path:
    sys.path.insert(0, USVNAV)

import numpy as np

from usvnav import render as _r                                    # noqa: E402
from usvnav.colour import (palette_margins, srgb_to_lab,                # noqa: E402
                            srgb_to_linear, linear_to_srgb)

if 'look' not in getattr(_r.RasterConfig, '__dataclass_fields__', {}):          # pragma: no cover
    raise ImportError(f"Track 1's renderer at {USVNAV} is too old: "
                      "Track 2 needs RasterConfig.look; point USVNAV_PATH at a current Track 1 checkout")
from usvnav import look as _look                                   # noqa: E402

#: Every Track 1 module the raster path loaded, snapshotted at the moment that path is established
#: -- except the package's `__init__`, which carries Track 1's exports and version, not drawing code.
#: `renderer_id()` hashes these: the modules a pixel depends on and nothing else. `look.py` is one.
RENDER_SOURCES = tuple(sorted(
    m.__file__ for n, m in sys.modules.items()
    if n.split('.')[0] == 'usvnav' and getattr(m, '__file__', None)
    and not os.path.basename(m.__file__).startswith('__init__')))
assert any(os.path.basename(f) == 'look.py' for f in RENDER_SOURCES), RENDER_SOURCES

SIZE_PX = 128           # pixels per side
M_PER_PX = 0.5          # -> 64 m span, +/- 32 m
FRAME = 'bow_up'        # ego at the centre, bow toward the top of the image

#: Track 2 draws from Track 1's class vocabulary in Track 1's base colours (`usvnav.look.BASE`, which
#: `render.PALETTE_TRACK1` is): the water, bank, pier, dock, vessel, buoy and ego of the studio map.
#: `unobserved` is Track 1's black.
#:
#: The guide's own hull colour (light grey-white; no other vessel is light) and its black centre mark:
#: the light hull is what makes the guide stand out, the mark identifies it.
GUIDE_HULL = (225, 228, 222)
GUIDE_MARK = (15, 15, 15)
GUIDE_HULL_EDGE = (70, 72, 68)
PALETTE = dict(_r.PALETTE_TRACK1, guide_hull=GUIDE_HULL, guide_mark=GUIDE_MARK)

#: Track 2's classes in Track 1's look tables (`usvnav.look.paint` reads them by name at call time):
#: the hull is a body -- shadow, deck panel, outline like every other hull -- and the mark is a plain
#: disc painted on it. Added at import; Track 1's own entries are untouched. Any Track 1 change to
#: `look.py` changes `renderer_id()`.
_look.MAP_LOOK.setdefault('guide_hull', dict(rgb=GUIDE_HULL, edge=GUIDE_HULL_EDGE))
_look.MAP_LOOK.setdefault('guide_mark', dict(rgb=GUIDE_MARK))
if 'guide_hull' not in _look.BODIES:
    _look.BODIES = tuple(_look.BODIES) + ('guide_hull',)

#: Bumped whenever Track 2's own configuration changes a pixel; hashed into `renderer_id()` beside
#: the raster modules, so a change of picture is a change of stamp.
PIXEL_VERSION = 5

TRACK2 = _r.RasterConfig(
    size_px=SIZE_PX,
    m_per_px=M_PER_PX,
    range_m=SIZE_PX * M_PER_PX / 2.0,       # 32 m; the extent is the whole square
    frame=FRAME,
    circular_mask=False,
    draw_ego=True,
    occlusion=False,                        # no occlusion is drawn
    look=True,                              # the studio's picture
    subsamples=2,                           # painted at 0.25 m/px, box-downsampled to 0.5 m/px
    palette=PALETTE,
)

Scene = _r.Scene
render = _r.render
class_map = _r.class_map
class_names = _r.class_names


@functools.lru_cache(maxsize=1)
def renderer_id():
    """Which build of the renderer this is: `render:<hash>`, optionally followed by `@<commit>`.

    The hash covers the raster path's own sources (`RENDER_SOURCES`) and `PIXEL_VERSION`, not Track
    1's repository, so it changes when the drawing code or Track 2's pixel configuration changes, and
    not otherwise. The Track 1 git commit, when one can be read, is appended for provenance only:
    two commits with the same raster sources give the same identity. Compare stamps with
    `renderer_key()`.
    """
    h = hashlib.blake2b(digest_size=6)
    for f in RENDER_SOURCES:
        with open(f, 'rb') as fh:
            h.update(os.path.basename(f).encode() + b'\0' + fh.read() + b'\0')
    h.update(f'track2-pixel-version:{PIXEL_VERSION}'.encode())
    try:
        rev = subprocess.run(['git', '-C', USVNAV, 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        rev = ''
    return f"render:{h.hexdigest()}" + (f"@{rev}" if rev else "")


def renderer_key(stamp=None):
    """The identity half of a stamp, with the provenance commit dropped. Compare on this."""
    return (stamp or renderer_id()).split('@')[0]


def dim(palette, level):
    """A palette under illumination `level` (1.0 is nominal), every colour scaled in linear light.

    **Linear light, not sRGB code values.** Halving the light halves the photons, and sRGB is a
    ~2.2-gamma encoding of those, so half the light is code value ~0.73 rather than 0.50; scaling
    the stored bytes would darken far too fast. Uses Track 1's `srgb_to_linear` / `linear_to_srgb`,
    so both tracks compute the same numbers.
    """
    return {k: tuple(int(round(float(x)))
                     for x in linear_to_srgb(srgb_to_linear(v) * level))
            for k, v in palette.items()}


def margin_at(level, palette=None):
    """The palette's general-position colour margin at illumination `level`, in CIELAB units. A
    reference number only: the frame carries shoreline, shadow and outline tones that are not class
    colours, so no colour-lookup segmentation is promised."""
    return palette_margins(dim(palette or PALETTE, level), subsamples=8, max_mix=3)['general_position']


def darkest_usable(threshold, palette=None, lo=0.05, hi=1.0, tol=1e-3):
    """The dimmest illumination whose margin (`margin_at`) still clears `threshold`.

    Bisection is safe because dimming is monotone in the margin: every colour scales toward
    black together, so every pairwise distance shrinks together.
    """
    if margin_at(hi, palette) < threshold:
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if margin_at(mid, palette) >= threshold:
            hi = mid
        else:
            lo = mid
    return hi


def span_m():
    return SIZE_PX * M_PER_PX


def px_per_m():
    return 1.0 / M_PER_PX
