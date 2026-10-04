"""Condition 1-1's object list, and its region: the raster's square, not a 50 m circle.

The test that matters most is the region test: a body in the raster's **corner**, past
50 m along neither axis but inside the square. Under a 50 m radius that body would be in
1-2 and not in 1-1, which would invert 1-1's role as the perception-free upper bound over
21% of the frame.
"""

from __future__ import annotations

import math

import numpy as np

from usvnav import render
from usvnav.geometry import Circle, Rect
from usvnav.objects import (CAPACITY, CIRCLE, CLASS_ID, RECT, CapacityExceeded,
                             object_list, peak_occupancy)
from usvnav.plant import Vessel
from usvnav.render import TRACK1, class_map
from usvnav.world import BUOY, MOORED, MOVING, PIER, Course, Lane, StaticBody, TrafficRoute

BOX = np.array([[-200.0, -200.0], [200.0, -200.0], [200.0, 200.0], [-200.0, 200.0]])


def _course(bodies, traffic=()):
    return Course(BOX, list(bodies), np.array([[100.0, 0.0]]), np.array([3.0]),
                  (0.0, 0.0, 0.0), list(traffic),
                  terrain_extent=(-300.0, -300.0, 300.0, 300.0))


def _rows(ol):
    n = int(ol["valid"].sum())
    return [{k: (ol[k][i].tolist() if ol[k].ndim > 1 else ol[k][i].item())
             for k in ol} for i in range(n)]


def test_positions_and_velocities_are_in_body_axes():
    """Body axes: +x through the bow, +y to port. The two minus signs are the thing to get
    wrong, so they are asserted rather than assumed."""
    course = _course([StaticBody(BUOY, Circle(10.0, 10.0, 0.3))])
    # Heading +90 deg: the bow points along +y_world, so a body at world (10, 10) is
    # 10 m ahead and 10 m to *starboard*, i.e. y_body negative.
    ol = object_list(course, Vessel(0.0, 0.0, math.pi / 2.0), 0.0)
    x, y = _rows(ol)[0]["pos"]
    assert abs(x - 10.0) < 1e-4 and abs(y + 10.0) < 1e-4, (x, y)


def test_static_bodies_have_velocity_zero_and_traffic_does_not():
    """The list's stated invariant, and the reason its velocity is over ground."""
    route = TrafficRoute(MOVING, 12.0, 4.0,
                         np.array([[0.0, 20.0], [40.0, 20.0], [40.0, 60.0], [0.0, 60.0]]),
                         speed=2.0)
    course = _course([StaticBody(PIER, Rect(10.0, 0.0, 12.0, 5.0, 0.0))], [route])
    rows = _rows(object_list(course, Vessel(0.0, 0.0, 0.0), 0.0))
    by_cls = {r["cls"]: r for r in rows}
    assert by_cls[CLASS_ID[PIER]]["vel"] == [0.0, 0.0]
    assert np.hypot(*by_cls[CLASS_ID[MOVING]]["vel"]) > 1.9


def test_shape_parameters_not_a_size_scalar():
    course = _course([StaticBody(BUOY, Circle(5.0, 0.0, 0.3)),
                      StaticBody(MOORED, Rect(20.0, 0.0, 30.0, 8.0, 0.4))])
    rows = {r["cls"]: r for r in _rows(object_list(course, Vessel(0.0, 0.0, 0.0), 0.0))}
    buoy, moored = rows[CLASS_ID[BUOY]], rows[CLASS_ID[MOORED]]
    # All real-valued fields are `float32`, so compare with a tolerance rather than for
    # equality -- 0.6 is not exact in binary32.
    assert buoy["primitive"] == CIRCLE
    assert max(abs(v - 0.6) for v in buoy["extent"]) < 1e-6, buoy["extent"]
    assert moored["primitive"] == RECT
    assert abs(moored["extent"][0] - 30.0) < 1e-4 and abs(moored["extent"][1] - 8.0) < 1e-4
    assert abs(moored["heading"] - 0.4) < 1e-5


def test_occluded_objects_are_listed():
    """1-1 is the upper bound, so nothing may silently remove information. A buoy
    directly behind a 40 m moored vessel is invisible to 1-3 and present here."""
    course = _course([StaticBody(MOORED, Rect(20.0, 0.0, 10.0, 40.0, math.pi / 2)),
                      StaticBody(BUOY, Circle(30.0, 0.0, 0.3))])
    classes = {r["cls"] for r in _rows(object_list(course, Vessel(0.0, 0.0, 0.0), 0.0))}
    assert CLASS_ID[BUOY] in classes and CLASS_ID[MOORED] in classes


def test_rows_are_nearest_first_and_the_rest_is_masked_off():
    bodies = [StaticBody(BUOY, Circle(float(d), 0.0, 0.3)) for d in (30, 5, 18)]
    ol = object_list(_course(bodies), Vessel(0.0, 0.0, 0.0), 0.0)
    xs = [r["pos"][0] for r in _rows(ol)]
    assert xs == sorted(xs), xs
    assert len(ol["valid"]) == CAPACITY
    assert not ol["valid"][len(xs):].any()
    assert (ol["pos"][len(xs):] == 0).all()          # zero-padded


def test_overflow_raises_rather_than_truncating():
    """The overflow fallback, which should never be needed, does not fire quietly."""
    bodies = [StaticBody(BUOY, Circle(float(i % 20) * 2.0, float(i // 20) * 2.0, 0.3))
              for i in range(200)]
    try:
        object_list(_course(bodies), Vessel(0.0, 0.0, 0.0), 0.0)
    except CapacityExceeded:
        pass
    else:
        raise AssertionError("the list truncated instead of refusing")


def test_the_region_is_the_raster_square_not_a_50_m_circle():
    """A buoy at body (46, 46) is 65 m away -- outside the 50 m sensing range, inside the
    raster's square, and drawn by the renderer. If 1-1 stopped at the radius it would be missing
    from the object list while visible in the image, over the 21% of the frame the square
    adds. The counter-case at (60, 60) is outside both and must be in neither.
    """
    inside = StaticBody(BUOY, Circle(46.0, 46.0, 0.3))
    outside = StaticBody(BUOY, Circle(60.0, 60.0, 0.3))
    course = _course([inside, outside])
    vessel = Vessel(0.0, 0.0, 0.0)

    assert math.hypot(46.0, 46.0) > TRACK1.range_m, "the corner case has to be past 50 m"

    rows = _rows(object_list(course, vessel, 0.0))
    assert len(rows) == 1, [r["pos"] for r in rows]
    assert abs(rows[0]["pos"][0] - 46.0) < 1e-4

    # And 1-2 agrees, which is the half that makes it an *upper bound* rather than just
    # a different region: the listed buoy is drawn and the unlisted one is not.
    buoy_id = render.class_names(TRACK1).index(BUOY)
    with_both = class_map(render.scene_from_course(course, vessel, 0.0))
    only_far = class_map(render.scene_from_course(
        _course([outside]), vessel, 0.0))
    assert (with_both == buoy_id).sum() >= 1, "the corner buoy is not in the raster either"
    assert (only_far == buoy_id).sum() == 0, "the 60 m buoy should be outside the square"


def test_a_lane_vessel_is_listed_with_its_velocity_and_only_while_it_exists():
    # Along +x at 2.0 m/s past the observer (heading 0 at the origin): enters at x = -40 at t = 10.
    lane = Lane(closed=False, points=np.array([[-40.0, 8.0], [40.0, 8.0]]),
                speeds=np.array([2.0, 2.0]), vessels=[(10.0, 3.0)], headway_s=500.0, offset_s=10.0)
    course = Course(BOX, [], np.array([[100.0, 0.0]]), np.array([3.0]), (0.0, 0.0, 0.0), [], [lane],
                    terrain_extent=(-300.0, -300.0, 300.0, 300.0))
    ego = Vessel(0.0, 0.0, 0.0)
    before = object_list(course, ego, 5.0)                   # the lane's first vessel has not entered
    assert int(before["valid"].sum()) == 0
    at = object_list(course, ego, 30.0)                      # 20 s in: x = -40 + 40 = 0, y = 8
    assert int(at["valid"].sum()) == 1
    assert at["cls"][0] == CLASS_ID[MOVING] and at["primitive"][0] == RECT
    assert abs(at["pos"][0][0] - 0.0) < 1e-4 and abs(at["pos"][0][1] - 8.0) < 1e-4
    assert abs(at["vel"][0][0] - 2.0) < 1e-5 and abs(at["vel"][0][1]) < 1e-5
    assert abs(at["extent"][0][0] - 10.0) < 1e-5 and abs(at["extent"][0][1] - 3.0) < 1e-5
    gone = object_list(course, ego, 10.0 + 40.0 + 1.0)       # 80 m at 2 m/s = 40 s; then it has left
    assert int(gone["valid"].sum()) == 0


def test_peak_occupancy_matches_a_dense_time_sweep_with_two_schedules():
    # An open lane (population repeats every 30 s, warm-up 300 s) and a closed loop (lap 60 s)
    # share one observation square; a sweep that collapsed both into a few samples of one
    # period would under-count the moment both are densest.
    #
    # The lane's speed is 1.4 m/s rather than the more obvious 1.0: at 1.0 the lane's own
    # population cycle happens to land a same-phase sample in a fixed 8-sample sweep by
    # coincidence (it reports the true peak anyway), so the aliasing this test exists to
    # catch would go unexercised. 1.4 m/s changes the spacing between candidate vessels
    # enough that a fixed 8-sample sweep under-counts by one (5 instead of 6), while every
    # step count from 9 to 64 finds the true peak -- so whatever this course's lane/loop
    # ratio sends the resolution formula to, it lands on one that works.
    lane = Lane(closed=False, points=np.array([[-150.0, 0.0], [150.0, 0.0]]),
                speeds=np.array([1.4, 1.4]), vessels=[(6.0, 2.5)], headway_s=30.0, offset_s=0.0)
    loop = Lane(closed=True, points=np.array([[-20.0, 20.0], [10.0, 20.0], [10.0, 50.0], [-20.0, 50.0]]),
                speeds=np.full(4, 2.0), vessels=[(6.0, 2.5)] * 3)          # lap 120 m / 2 m/s = 60 s
    course = Course(BOX, [], np.array([[100.0, 0.0]]), np.array([3.0]), (0.0, 0.0, 0.0), [], [lane, loop],
                    terrain_extent=(-300.0, -300.0, 300.0, 300.0))
    xs = ys = np.linspace(-200.0, 200.0, 3)                                 # samples=4 -> a 3 x 3 grid
    dense = max(int(object_list(course, Vessel(float(x), float(y), 0.0), float(t), capacity=10 ** 6)["valid"].sum())
                for x in xs for y in ys for t in np.arange(0.0, 330.0, 0.5))
    assert dense >= 4, dense                                               # the scenario is dense enough to matter
    assert peak_occupancy(course, samples=4) == dense
