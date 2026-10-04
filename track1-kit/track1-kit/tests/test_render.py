"""Renderer contract tests: the raster geometry (index mapping, `bow_up` frame, square
extent), the minimum-marker rule, occlusion, and determinism.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from usvnav import render
from usvnav.geometry import Circle, Rect
from usvnav.plant import HULL_LENGTH, HULL_WIDTH, Vessel
from usvnav.render import PaletteError, RasterConfig, Scene, TRACK1, class_map, render as draw
from usvnav.world import BANK, BUOY, EGO, MOORED, MOVING, OUTSIDE_WORLD, PIER, UNOBSERVED, WATER

NAMES = render.class_names()
ID = {n: i for i, n in enumerate(NAMES)}

# A square of open water big enough that the shoreline never enters a 50 m raster.
OPEN_WATER = np.array([[-400.0, -400.0], [400.0, -400.0], [400.0, 400.0], [-400.0, 400.0]])


def _scene(bodies=(), psi=0.0, **kw):
    """Open water, so the shoreline never enters a 70.71 m corner reach."""
    return Scene(observer=(0.0, 0.0, psi), bodies=list(bodies), boundary=OPEN_WATER, **kw)


def test_index_mapping_signs():
    """`row = r0 - x/s`, `col = c0 - y/s`. The two minus signs are the thing an
    implementer gets wrong, and getting them wrong mirrors the whole scene.

    A body on the **port bow** -- body `x = +10`, `y = +10`, and `+y` is port -- must
    land **upper-left**, as docs/OBSERVATION.md §1 says.
    """
    ids = class_map(_scene([(BUOY, Circle(10.0, 10.0, 0.3))]),
                    replace(TRACK1, occlusion=False))
    r, c = np.nonzero(ids == ID[BUOY])
    assert len(r), "the buoy did not render at all"
    row, col = r.mean(), c.mean()
    assert row < 100 and col < 100, f"port-bow buoy landed at row {row:.0f}, col {col:.0f}"
    # 10 m at 0.5 m/px is 20 px from the centre, on both axes.
    assert abs(row - 80) < 1.5 and abs(col - 80) < 1.5, (row, col)


def test_ego_is_drawn_at_the_centre_at_its_true_size():
    """The raster draws the ego hull, 3.0 x 2.0 m. Under `bow_up` it is axis-aligned
    and centred on the corner shared by the four central pixels, so it is exactly
    6 x 4 = 24 pixels with no partial coverage anywhere -- which is also a check that
    `r0 = c0 = size/2` is the pixel corner and not a pixel centre."""
    hull = Rect(0.0, 0.0, HULL_LENGTH, HULL_WIDTH, 0.0)
    ids = class_map(_scene(ego_hull=hull), replace(TRACK1, occlusion=False))
    ego = ids == ID[EGO]
    assert ego.sum() == 24, ego.sum()
    r, c = np.nonzero(ego)
    assert (r.min(), r.max(), c.min(), c.max()) == (97, 102, 98, 101), (r.min(), r.max(), c.min(), c.max())


def test_extent_is_the_whole_square_and_nothing_is_unobserved():
    """The extent is the square, with no circular mask.

    Two things follow and both are asserted. Every pixel carries a real world class, so
    `unobserved` -- whose only sources would be a mask and occlusion -- cannot appear in a
    Track 1 raster at all. And the square reaches 70.71 m into its corners against 50 m
    along its axes, which is the figure a course's terrain margin has to cover.
    """
    ids = class_map(_scene(), TRACK1)
    assert (ids != ID[UNOBSERVED]).all(), "something is still unobserved"
    assert abs(TRACK1.reach_m - 50.0 * math.sqrt(2.0)) < 1e-9, TRACK1.reach_m
    # the corner pixel's own centre, in metres from the observer
    xb, yb = render._Frame(_scene(), TRACK1).pixel_to_frame(np.array([0.5]), np.array([0.5]))
    assert abs(math.hypot(xb[0], yb[0]) - 70.36) < 0.05, (xb[0], yb[0])


PLAIN = replace(TRACK1, look=False, subsamples=1)     # the class map under the picture


def test_a_buoy_paints_at_every_sub_pixel_offset():
    """A 0.6 m buoy is drawn at the 2.0 px minimum marker: a disc of radius 1 px always
    contains a pixel centre (the square grid's covering radius is 0.71 px), so the class
    map carries the buoy at every sub-pixel offset, and the picture painted from it has
    the buoy's tones there too."""
    buoy = np.array(PLAIN.palette[BUOY], dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            scene = _scene([(BUOY, Circle(20.0 + 0.5 * i / 8.0, 0.5 * j / 8.0, 0.3))])
            assert (draw(scene, PLAIN) == buoy).all(axis=2).any(), f"buoy missing at offset {i},{j}"
            pic = draw(scene, TRACK1)
            yellow = (pic[..., 0] > 120) & (pic[..., 1] > 90) & (pic[..., 2] < 90)
            assert yellow.any(), f"the picture lost the buoy's colour at offset {i},{j}"


def test_the_plain_class_map_is_exact_and_the_picture_is_the_studio_look():
    """Two renders of one scene. Without `look` every pixel is the exact colour of its
    centre class (the class map underneath). With `look` -- Track 1's setting -- the
    picture is `usvnav.look.paint` of that class map at `subsamples`
    times the resolution, box-downsampled: deterministic, and anchored to the world."""
    from usvnav import look
    scene = _scene([(MOORED, Rect(14.0, 3.0, 30.0, 8.0, 0.37)), (BUOY, Circle(-12.3, 7.7, 0.3))])
    plain = replace(PLAIN, occlusion=False)
    img = draw(scene, plain)
    lut = np.array([plain.palette[n] for n in render.class_names(plain)], dtype=np.uint8)
    assert (img == lut[class_map(scene, plain)]).all(), "a plain pixel is not its centre class colour"
    assert TRACK1.look and TRACK1.subsamples >= 2 and TRACK1.noise is None
    pic = draw(scene, TRACK1)
    assert (pic == draw(scene, TRACK1)).all(), "the picture is not deterministic"
    assert (pic != img).any(axis=2).sum() > 200, "the picture carries no shadow/outline/edge tones"
    # no noise unless asked: open water far from anything is the base colour exactly
    assert (pic[5:40, 5:40] == np.array(TRACK1.palette[WATER], dtype=np.uint8)).all(), "the scorer's picture is not flat"
    # the example noise is anchored to the world: move the observer sideways and the
    # texture shifts with the scene rather than staying stuck to the screen
    no_ego = replace(TRACK1, draw_ego=False, noise=look.grain)
    shift_m = 2.0                                          # 4 raster px, 8 fine px: exact
    still = Scene(observer=(0.0, 0.0, 0.0), bodies=[], boundary=OPEN_WATER)
    shifted = Scene(observer=(0.0, shift_m, 0.0), bodies=[], boundary=OPEN_WATER)
    a, b = draw(still, no_ego).astype(int), draw(shifted, no_ego).astype(int)
    k = int(round(shift_m / TRACK1.m_per_px))
    assert (b[:, k:] == a[:, :-k]).all(), "grain did not travel with the world"
    assert (a[:, k:] != a[:, :-k]).any(), "the water has no grain to travel"


def test_supersampling_makes_sub_pixel_motion_visible_when_enabled():
    """The renderer keeps its plain supersampling path for a track that renders with
    blends (Track 2, with `PALETTE_GENERAL_POSITION`): two poses 0.15 m apart -- one tick
    of cruise -- change many more pixels at `subsamples=8` than with hard edges. Track 1
    renders the studio picture (`look`) instead; this is a property of the plain option.
    """
    a = _scene([(MOORED, Rect(14.0, 3.0, 30.0, 8.0, 0.37))])
    b = Scene(observer=(0.15, 0.0, 0.0), bodies=list(a.bodies), boundary=OPEN_WATER)
    for subs, expect in ((1, "hard"), (8, "anti-aliased")):
        cfg = replace(TRACK1, occlusion=False, look=False, subsamples=subs)
        d = (draw(a, cfg).astype(int) != draw(b, cfg).astype(int)).any(axis=2).sum()
        if subs == 1:
            hard = d
        else:
            aa = d
    assert aa > 3 * max(hard, 1), f"anti-aliasing added little: {hard} -> {aa} px changed"


def test_render_is_deterministic():
    """The same world state yields the same visible content through the same code path."""
    scene = _scene([(MOORED, Rect(14.0, 3.0, 30.0, 8.0, 0.37)),
                    (BUOY, Circle(-8.0, 12.0, 0.3))])
    assert np.array_equal(draw(scene), draw(scene))


def test_minimum_marker_keeps_a_buoy_present_at_every_sub_pixel_offset():
    """The minimum-marker rule: a body must not be present in one rendering and absent
    from another.

    A 0.6 m buoy is 1.2 px across. Swept across a pixel it sometimes falls between pixel
    centres and, at the true size, disappears from the class map -- so the counter-case
    is asserted too: the rule is load-bearing, not decorative.
    """
    cfg = replace(TRACK1, occlusion=False)
    true_size = replace(cfg, min_marker_px=1.2)
    misses = 0
    for k in range(16):
        off = 20.0 + 0.5 * k / 16.0                     # sweep one pixel of 0.5 m
        scene = _scene([(BUOY, Circle(off, 0.17 * k, 0.3))])
        assert (class_map(scene, cfg) == ID[BUOY]).sum() >= 1, f"buoy vanished at {off}"
        if (class_map(scene, true_size) == ID[BUOY]).sum() == 0:
            misses += 1
    assert misses > 0, ("a 1.2 px buoy never vanished, so this test no longer "
                        "demonstrates why the minimum-marker rule exists")


def test_occlusion_hides_what_is_behind_and_keeps_the_near_body_whole():
    """With `occlusion` on (`TRACK1` has it off), the nearest body along a bearing is drawn
    as a whole footprint and everything past its far surface is `unobserved` -- the
    bird's-eye convention."""
    near = (MOORED, Rect(20.0, 0.0, 30.0, 10.0, math.pi / 2))     # broadside, 20 m ahead
    far = (BUOY, Circle(35.0, 0.0, 0.3))                          # directly behind it
    on = class_map(_scene([near, far]), replace(TRACK1, occlusion=True))
    off = class_map(_scene([near, far]), TRACK1)
    assert (off == ID[BUOY]).sum() >= 1, "the far buoy is not visible even unoccluded"
    assert (on == ID[BUOY]).sum() == 0, "the far buoy survived being occluded"
    # the near vessel keeps its full 30 x 10 m footprint = 1200 px at 0.5 m/px
    assert abs(int((on == ID["vessel"]).sum()) - 1200) < 60, (on == ID["vessel"]).sum()


def test_the_shoreline_does_not_occlude():
    """Deliberate: `meta` discloses the boundary polygon exactly, so shadowing the land
    behind it would remove information the agent already has. Land behind land is land
    in any case."""
    poly = np.array([[-400.0, -400.0], [400.0, -400.0], [400.0, 10.0], [-400.0, 10.0]])
    ids = class_map(Scene(observer=(0.0, 0.0, 0.0), boundary=poly),
                    replace(TRACK1, occlusion=True))
    assert (ids == ID[BANK]).sum() > 5000, (ids == ID[BANK]).sum()


def test_outside_world_never_appears_along_a_real_trajectory():
    """The terrain margin, turned into a check that fires. Track 1's palette has no
    `outside_world` colour, so the renderer raises rather than substituting one; if a
    course's terrain margin were ever too small, this test is what says so.

    The margin is the raster's 70.71 m half-diagonal, not the 50 m sensing range: a
    200 x 200 square reaches further into its corners than along its axes, and there is
    no circular mask to hide them.
    """
    from fixtures import course as fixture_course
    from pilot import chain_poses

    course = fixture_course(0)
    assert course.terrain_extent is not None
    # The whole route, sampled geometrically: driving an agent would check only as far as
    # the agent got.
    poses = chain_poses(course, step_m=8.0)
    assert len(poses) > 25, len(poses)
    for k, (x, y, psi) in enumerate(poses):
        ids = class_map(render.scene_from_course(course, Vessel(x, y, psi), k * 1.0))
        assert OUTSIDE_WORLD not in NAMES or (ids != ID.get(OUTSIDE_WORLD, -1)).all()


def test_outside_world_raises_rather_than_being_painted_as_something_else():
    """The other half of the same guarantee: when it *is* reachable, it is loud."""
    try:
        class_map(Scene(observer=(0.0, 0.0, 0.0), boundary=OPEN_WATER,
                        terrain_extent=(-20.0, -20.0, 20.0, 20.0)), TRACK1)
    except PaletteError:
        pass
    else:
        raise AssertionError("outside_world was painted with no colour configured")


def test_entry_point_tolerates_worlds_track_1_never_emits():
    """The entry point must be usable by Track 2, which composes arbitrary scenes from
    the same vocabulary. None of these can come out of a Track 1 course."""
    cfgs = [TRACK1,
            replace(TRACK1, frame="world_aligned", circular_mask=False, occlusion=False),
            replace(TRACK1, size_px=64, m_per_px=0.25, range_m=8.0),
            RasterConfig(size_px=48, m_per_px=1.0, range_m=24.0, occlusion=False,
                         palette=dict(render.PALETTE_TRACK1, outside_world=(255, 255, 255)))]
    scenes = [Scene(observer=(0.0, 0.0, 0.4)),                            # nothing at all
              Scene(observer=(0.0, 0.0, 0.0), bodies=[(PIER, Rect(0.0, 0.0, 400.0, 400.0, 0.2))]),
              Scene(observer=(5.0, -3.0, -2.0), bodies=[(MOVING, Rect(5.0, -3.0, 6.0, 2.5, 1.0))]),
              Scene(observer=(0.0, 0.0, 0.0), boundary=OPEN_WATER[::-1],   # clockwise winding
                    terrain_extent=(-1e4, -1e4, 1e4, 1e4)),
              Scene(observer=(0.0, 0.0, 0.0), bodies=[(BUOY, Circle(0.0, 0.0, 1e-6))])]
    for cfg in cfgs:
        for scene in scenes:
            img = draw(scene, cfg)
            assert img.shape == (cfg.size_px, cfg.size_px, 3) and img.dtype == np.uint8


def test_bad_frame_name_is_rejected():
    """`bow_up`, `world_aligned` and `map` are the only frame names; the ambiguous
    phrase "ego-centric fixed heading" is not one of them."""
    try:
        draw(_scene(), replace(TRACK1, frame="ego_centric_fixed_heading"))
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown frame mode was accepted")


def test_frame_transform_is_native_not_a_rotated_bitmap():
    """A rotated bitmap blurs interiors as well as edges, so the test
    is that interior pixels remain exact class colours at an arbitrary heading -- which
    resampling cannot achieve."""
    scene = _scene([(MOORED, Rect(18.0, 6.0, 30.0, 8.0, 0.0))], psi=0.6180339887)
    cfg = replace(TRACK1, occlusion=False, look=False)       # the class map under the picture
    img = draw(scene, cfg)
    ids = class_map(scene, cfg)
    pal = np.array(list(cfg.palette.values()), dtype=np.uint8)
    exact = (img.reshape(-1, 1, 3) == pal.reshape(1, -1, 3)).all(axis=2).any(axis=1)
    interior = ~render._edge_pixels(ids)
    assert exact.reshape(200, 200)[interior].all()
    assert interior.sum() > 30000, interior.sum()


def test_culling_keeps_a_body_that_only_just_reaches_the_raster():
    """The renderer drops bodies whose bounding circle misses the painted disc, which is
    what makes one code path affordable for both the pixel-centre and the subsample
    pass. The cull is by bounding circle, so it must keep a large body whose *centre* is
    well outside the sensing range but whose footprint is not.

    A 40 m moored vessel -- the largest moored-vessel length -- centred at 65 m has a
    bounding radius of 20.6 m, so it reaches to 44.4 m and must still be drawn.
    """
    cfg = replace(TRACK1, occlusion=False)
    far = Rect(65.0, 0.0, 40.0, 10.0, 0.0)
    assert (class_map(_scene([(MOORED, far)]), cfg) == ID["vessel"]).sum() > 0
    # Along an axis the raster stops at 50 m, so a body reaching only to 59.4 m in +x
    # paints nothing -- but the cull works on the 70.71 m corner reach, so it is the
    # exact test and not the cull that decides this one.
    beyond = Rect(80.0, 0.0, 40.0, 10.0, 0.0)
    assert (class_map(_scene([(MOORED, beyond)]), cfg) == ID["vessel"]).sum() == 0
    far_corner = Rect(300.0, 300.0, 40.0, 10.0, 0.0)                  # culled outright
    assert (class_map(_scene([(MOORED, far_corner)]), cfg) == ID["vessel"]).sum() == 0


def test_a_body_out_of_range_casts_no_shadow_into_the_raster():
    """The other half of the same cull: a shadow begins past its own body's far surface,
    so a body outside the painted disc cannot shadow anything inside it. If that were
    wrong the cull would delete visible unobserved area."""
    cfg = replace(TRACK1, occlusion=True)
    near = (MOORED, Rect(20.0, 0.0, 30.0, 10.0, math.pi / 2))
    a = class_map(_scene([near]), cfg)
    b = class_map(_scene([near, (MOORED, Rect(90.0, 20.0, 40.0, 10.0, 0.5))]), cfg)
    assert np.array_equal(a, b)


def test_moored_and_moving_share_one_colour():
    """The reason for an eight-entry palette: a dedicated colour for the moving
    vessel *is* a velocity label, and it would hand 1-2 for free the thing 1-1 is
    supposed to have exclusively."""
    cfg = replace(TRACK1, occlusion=False)
    a = draw(_scene([(MOORED, Rect(20.0, 0.0, 20.0, 6.0, 0.0))]), cfg)
    b = draw(_scene([(MOVING, Rect(20.0, 0.0, 20.0, 6.0, 0.0))]), cfg)
    assert np.array_equal(a, b)


def test_noise_is_an_option_with_one_example():
    """`resolve_noise`: the scorer's `none`, the kit's `grain`, a callable, `module:function`."""
    from usvnav import look
    assert look.resolve_noise(None) is None and look.resolve_noise("none") is None
    assert look.resolve_noise("grain") is look.grain
    f = lambda c, x, y: np.zeros(x.shape, dtype=np.int16)
    assert look.resolve_noise(f) is f
    assert look.resolve_noise("usvnav.look:grain") is look.grain
    for bad in ("sparkle", "usvnav.look:nope"):
        try:
            look.resolve_noise(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} was accepted")
    # the example has the documented shape and is a pure function of world position
    x, y = np.meshgrid(np.arange(10.0), np.arange(10.0))
    g = look.grain(WATER, x, y)
    assert g.shape == x.shape and g.dtype == np.int16 and (g == look.grain(WATER, x, y)).all()
    assert np.abs(g).max() <= look.GRAIN_EXAMPLE[WATER][1]
    assert (look.grain(EGO, x, y) == 0).all(), "the example leaves the hull alone"
