"""The palette checks that ship with the renderer.

Two palettes, two properties. `PALETTE_TRACK1` is what a Track 1 agent receives: the
studio map's base colours, on which the picture is painted (outlines, shadows and
anti-aliased edges add tones of their own), so what it needs is that its eight colours are
distinct and that the map is drawn in the same values. `PALETTE_GENERAL_POSITION` is the
blend-safe palette computed for plain anti-aliased class-colour renders; Track 2 renders
with it, and its margins are asserted here so editing a colour cannot silently break it.
"""

from __future__ import annotations

import itertools

import numpy as np

from usvnav import figure, render
from usvnav.colour import palette_margins, reachable_blends, srgb_to_lab
from usvnav.render import PALETTE_GENERAL_POSITION, PALETTE_TRACK1, TRACK1, VESSEL
from usvnav.world import MOORED, MOVING

AA = 8                                                # what an anti-aliased track renders with
MARGINS = palette_margins(PALETTE_GENERAL_POSITION, subsamples=AA, max_mix=3)


# --- Track 1's palette: distinct, exact, and the map's own -----------------------------

def test_track1_palette_has_eight_distinct_entries():
    """Eight classes, moored and moving vessels sharing `vessel`."""
    assert len(PALETTE_TRACK1) == 8, sorted(PALETTE_TRACK1)
    assert render.COLOUR_OF[MOORED] == VESSEL and render.COLOUR_OF[MOVING] == VESSEL
    assert len(set(PALETTE_TRACK1.values())) == 8, "two classes share a colour"


def test_track1_renders_the_studio_picture_anti_aliased():
    """The studio's picture, painted at `subsamples` times the
    resolution and box-downsampled."""
    assert TRACK1.look and TRACK1.subsamples >= 2
    assert dict(TRACK1.palette) == dict(PALETTE_TRACK1)


def test_the_raster_palette_is_the_map_palette():
    """What the studio shows is what the agent gets: `figure.MAP_LOOK` draws its textures
    on `PALETTE_TRACK1`'s values, class for class."""
    for cls, rgb in PALETTE_TRACK1.items():
        assert tuple(figure.MAP_LOOK[cls]["rgb"]) == tuple(rgb), cls


def test_track1_classes_stay_apart_for_a_human_and_in_grayscale():
    """Not a blend margin but a reader's margin: every pair of
    classes that can appear in a Track 1 raster is at least a few JND apart in CIELAB and
    keeps an L* gap, so the raster reads at a glance and survives a grayscale conversion."""
    reachable = {k: v for k, v in PALETTE_TRACK1.items() if k != "unobserved"}
    lab = {k: srgb_to_lab(np.array(v, dtype=float)) for k, v in reachable.items()}
    worst = min(float(np.linalg.norm(lab[a] - lab[b])) for a, b in itertools.combinations(lab, 2))
    assert worst >= 5.0, worst
    lstar = sorted(v[0] for v in lab.values())
    assert min(b - a for a, b in zip(lstar, lstar[1:])) >= 2.0, lstar


def test_the_class_map_under_the_picture_is_exact():
    """The truth the picture is painted from: without `look`, every pixel is the exact
    colour of the class at its centre."""
    from dataclasses import replace
    from fixtures import course as fixture_course
    from usvnav.plant import Vessel

    course = fixture_course(0)
    img = render.top_view(course, Vessel(*course.start), 0.0, replace(TRACK1, look=False, subsamples=1))
    pal = np.array(list(PALETTE_TRACK1.values()), dtype=np.uint8)
    exact = (img.reshape(-1, 1, 3) == pal.reshape(1, -1, 3)).all(axis=2).any(axis=1)
    assert exact.all(), f"{(~exact).sum()} pixels equal no palette colour"


# --- the blend-safe palette, for plain anti-aliased renders (Track 2) ----------------------

def test_general_position_palette_has_eight_distinct_entries():
    assert len(PALETTE_GENERAL_POSITION) == 8
    assert len(set(PALETTE_GENERAL_POSITION.values())) == 8


def test_general_position_margin():
    """The palette's margins: 19.0 on the idealised CIELAB segment, 19.6 on the blend space the
    renderer actually emits at 8x8 subsamples."""
    assert MARGINS["lab_segment"] >= 18.9, MARGINS["lab_segment"]
    assert MARGINS["general_position"] >= 19.0, MARGINS["general_position"]


def test_grayscale_and_colour_vision():
    assert MARGINS["lstar_gap"] >= 8.6, MARGINS["lstar_gap"]
    assert MARGINS["deuteranopia"] >= 22.5, MARGINS["deuteranopia"]


def test_no_blend_is_ever_exactly_another_class():
    """The palette's literal promise: no two- or three-class blend equals a third class."""
    names = list(PALETTE_GENERAL_POSITION)
    rgb = np.array([PALETTE_GENERAL_POSITION[n] for n in names], dtype=float)
    for m in (2, 3):
        for target in range(len(names)):
            others = [i for i in range(len(names)) if i != target]
            for combo in itertools.combinations(others, m):
                blends = np.rint(reachable_blends(rgb[list(combo)], AA))
                assert not (blends == rgb[target]).all(axis=1).any(), (
                    f"a {m}-class blend of {[names[i] for i in combo]} is exactly {names[target]}")


def test_three_class_corner_margin():
    """Three-class corners come within 0.9 CIELAB of a class over all eight entries."""
    assert MARGINS["mix_3"] >= 0.8, MARGINS["mix_3"]


def test_margins_over_the_classes_that_can_actually_appear():
    """Without `unobserved` (black, in every worst pair) the margins are 22.5 / 5.8 / 8.7."""
    reachable = {k: v for k, v in PALETTE_GENERAL_POSITION.items() if k != "unobserved"}
    m = palette_margins(reachable, subsamples=AA, max_mix=3)
    assert m["general_position"] >= 22.0, m["general_position"]
    assert m["mix_3"] >= 5.0, m["mix_3"]
    assert m["lstar_gap"] >= 8.6, m["lstar_gap"]
