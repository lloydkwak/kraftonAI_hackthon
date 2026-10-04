"""Lanes: open end-to-end lanes with scheduled vessels and closed circulating lanes.

A vessel's pose on a lane is a closed-form function of time -- nothing depends on the ego or
the seed -- which is what keeps a replay reproducible with lanes present.
"""

from __future__ import annotations

import math

import numpy as np

from usvnav.world import A_MAX, Lane, _segment_time


def _straight(length=300.0, speed=1.5, headway=45.0, offset=0.0, vessels=((12.0, 3.5),)):
    return Lane(closed=False, points=np.array([[0.0, 0.0], [length, 0.0]]),
                speeds=np.array([speed, speed]), vessels=list(vessels),
                headway_s=headway, offset_s=offset)


def test_segment_time_matches_the_integral_of_one_over_speed():
    # constant speed: d / v
    assert abs(_segment_time(30.0, 1.5, 1.5) - 20.0) < 1e-9
    # linear ramp 0.5 -> 2.5 over 100 m: numeric integral of ds / v(s)
    s = np.linspace(0.0, 100.0, 200001)
    v = 0.5 + (2.5 - 0.5) * s / 100.0
    numeric = float(np.trapezoid(1.0 / v, s))
    assert abs(_segment_time(100.0, 0.5, 2.5) - numeric) < 1e-6


def test_arc_and_time_are_inverses_along_a_ramped_lane():
    lane = Lane(closed=False, points=np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 80.0]]),
                speeds=np.array([0.5, 2.5, 1.0]), vessels=[(10.0, 3.0)], headway_s=60.0)
    assert abs(lane.length - 180.0) < 1e-9
    for s in (0.0, 12.5, 99.9, 100.0, 150.0, 180.0):
        assert abs(lane.arc_at_time(lane.time_at_arc(s)) - s) < 1e-6
    assert abs(lane.speed_at_arc(50.0) - 1.5) < 1e-9        # halfway up the 0.5 -> 2.5 ramp
    assert abs(lane.speed_at_arc(140.0) - 1.75) < 1e-9      # halfway down the 2.5 -> 1.0 ramp


def test_an_open_lane_spawns_on_schedule_and_despawns_at_the_end():
    lane = _straight(length=300.0, speed=1.5, headway=45.0, offset=10.0)
    assert abs(lane.traversal_s - 200.0) < 1e-9
    assert lane.alive(9.9) == []                            # nothing before the first entry
    (k, s), = lane.alive(10.0)
    assert k == 0 and abs(s) < 1e-9                         # vessel 0 at the start point
    (k, s), = lane.alive(40.0)
    assert k == 0 and abs(s - 45.0) < 1e-9                  # 30 s at 1.5 m/s
    alive = lane.alive(100.0)                               # entries at 10, 55, 100
    assert sorted(k for k, _ in alive) == [0, 1, 2]
    assert all(0.0 <= s <= 300.0 for _, s in alive)
    # vessel 0 leaves at 10 + 200 = 210 s
    assert 0 in [k for k, _ in lane.alive(209.9)]
    assert 0 not in [k for k, _ in lane.alive(210.1)]
    # never more than ceil(traversal / headway) + 1 vessels
    bound = math.ceil(200.0 / 45.0) + 1
    for t in np.arange(0.0, 1000.0, 0.7):
        assert len(lane.alive(float(t))) <= bound


def test_a_closed_lane_keeps_its_vessels_equally_spaced_in_time():
    square = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    lane = Lane(closed=True, points=square, speeds=np.full(4, 1.0),
                vessels=[(10.0, 3.0)] * 4, offset_s=0.0)
    assert abs(lane.length - 400.0) < 1e-9 and abs(lane.traversal_s - 400.0) < 1e-9
    assert abs(lane.headway() - 100.0) < 1e-9
    for t in (0.0, 37.0, 250.0, 1234.5):
        arcs = sorted(s for _, s in lane.alive(t))
        assert len(arcs) == 4
        gaps = np.diff(arcs + [arcs[0] + 400.0])
        assert np.allclose(gaps, 100.0, atol=1e-6)          # 100 m apart at 1 m/s = 100 s apart


def test_poses_follow_the_polyline_with_the_tangent_heading():
    lane = Lane(closed=False, points=np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0]]),
                speeds=np.array([1.0, 1.0, 1.0]), vessels=[(8.0, 3.0)], headway_s=500.0)
    x, y, h = lane.pose_at_arc(25.0)
    assert (abs(x - 25.0) < 1e-9 and abs(y) < 1e-9 and abs(h) < 1e-9)
    x, y, h = lane.pose_at_arc(75.0)
    assert (abs(x - 50.0) < 1e-9 and abs(y - 25.0) < 1e-9 and abs(h - math.pi / 2) < 1e-9)
    (rect, vel), = lane.moving_at(75.0)                     # entry at 0, 75 s at 1 m/s
    assert abs(rect.x - 50.0) < 1e-9 and abs(rect.y - 25.0) < 1e-9
    assert rect.length == 8.0 and rect.width == 3.0
    assert abs(vel[0]) < 1e-9 and abs(vel[1] - 1.0) < 1e-9


def test_vessel_sizes_cycle_by_entry_index_on_an_open_lane():
    lane = _straight(headway=10.0, vessels=((12.0, 3.5), (8.0, 2.8)))
    assert lane.vessel(0) == (12.0, 3.5) and lane.vessel(1) == (8.0, 2.8) and lane.vessel(2) == (12.0, 3.5)


def test_max_accel_is_the_largest_v_dv_ds():
    steep = Lane(closed=False, points=np.array([[0.0, 0.0], [10.0, 0.0]]),
                 speeds=np.array([0.5, 2.5]), vessels=[(6.0, 2.5)], headway_s=30.0)
    gentle = Lane(closed=False, points=np.array([[0.0, 0.0], [50.0, 0.0]]),
                  speeds=np.array([0.5, 2.5]), vessels=[(6.0, 2.5)], headway_s=30.0)
    assert abs(steep.max_accel() - 2.5 * 2.0 / 10.0) < 1e-9     # 0.5 m/s², over the bound
    assert steep.max_accel() > A_MAX
    assert abs(gentle.max_accel() - 2.5 * 2.0 / 50.0) < 1e-9    # 0.1 m/s², within it
    assert gentle.max_accel() <= A_MAX


def test_construction_rejects_what_cannot_be_a_lane():
    def bad(**kw):
        try:
            Lane(**kw)
        except ValueError:
            return True
        return False
    base = dict(closed=False, points=np.array([[0.0, 0.0], [10.0, 0.0]]),
                speeds=np.array([1.0, 1.0]), vessels=[(6.0, 2.5)], headway_s=30.0)
    assert bad(**{**base, "points": np.array([[0.0, 0.0]]), "speeds": np.array([1.0])})   # one point
    assert bad(**{**base, "speeds": np.array([1.0])})                                       # speeds mismatch
    assert bad(**{**base, "headway_s": None})                                               # open needs headway
    assert bad(**{**base, "headway_s": 0.0})
    assert bad(**{**base, "vessels": []})
    assert bad(**{**base, "closed": True})                                                  # closed has no headway
    assert bad(**{**base, "points": np.array([[0.0, 0.0], [0.0, 0.0]])})                    # zero-length segment


from pilot import ChainPilot
from usvnav.collide import CourseIndex
from usvnav.geometry import Rect
from usvnav.plant import HULL_LENGTH, HULL_WIDTH
from usvnav.sim import run_episode
from usvnav.world import MOVING, Course, TrafficRoute

BOX = np.array([[-200.0, -200.0], [200.0, -200.0], [200.0, 200.0], [-200.0, 200.0]])


def _box_course(lanes=(), traffic=()):
    return Course(BOX, [], np.array([[100.0, 0.0]]), np.array([3.0]), (0.0, 0.0, 0.0),
                  list(traffic), list(lanes), terrain_extent=(-300.0, -300.0, 300.0, 300.0))


def _crossing_lane():
    # A slow, long, wide vessel crossing the ego's line at x = 60, centred on y = 0 at t = 41 s:
    # it enters at (60, -150) and reaches y = 0 after 150 m / 0.5 m/s = 300 s, so offset = 41 - 300;
    # it is on the lane from t = -259 s to t = 341 s and spans y in [-10, 10] from t = 21 s to 61 s.
    return Lane(closed=False, points=np.array([[60.0, -150.0], [60.0, 150.0]]),
                speeds=np.array([0.5, 0.5]), vessels=[(20.0, 6.0)], headway_s=1000.0,
                offset_s=41.0 - 300.0)


def test_course_traffic_rects_and_moving_union_loops_and_lanes():
    loop = TrafficRoute(MOVING, 10.0, 3.0,
                        np.array([[0.0, 100.0], [50.0, 100.0], [50.0, 150.0], [0.0, 150.0]]), 1.0)
    course = _box_course(lanes=[_crossing_lane()], traffic=[loop])
    rects = course.traffic_rects(41.0)
    assert len(rects) == 2 and all(isinstance(r, Rect) for r in rects)
    assert abs(rects[1].x - 60.0) < 1e-6 and abs(rects[1].y) < 1e-6      # the lane vessel, at the crossing
    moving = course.moving(41.0)
    assert len(moving) == 2
    (rect, vel) = moving[1]
    assert abs(vel[0]) < 1e-9 and abs(vel[1] - 0.5) < 1e-9                # heading +y at 0.5 m/s
    assert course.traffic_rects(-500.0) == [loop.rect_at(-500.0)]           # before the lane's first entry


def test_the_collision_index_reports_contact_with_a_lane_vessel():
    course = _box_course(lanes=[_crossing_lane()])
    index = CourseIndex(course)
    hull = Rect(60.0, 0.0, HULL_LENGTH, HULL_WIDTH, 0.0)                    # sitting on the vessel
    clearance, hit, inside = index.tick(hull, 41.0, 10.0)
    assert hit == MOVING and clearance == 0.0 and inside
    clearance, hit, inside = index.tick(hull, -500.0, 10.0)                # vessel not yet on the lane
    assert hit is None and clearance > 5.0


def test_an_unaware_pilot_collides_with_a_lane_vessel_in_the_simulator():
    course = _box_course(lanes=[_crossing_lane()])
    res = run_episode(course, ChainPilot(), condition="1-1", seed=0)
    assert res.outcome == "dynamic_collision", (res.outcome, res.ticks)
    assert 300 <= res.ticks <= 650                                          # around the 41 s crossing
    clear = run_episode(_box_course(), ChainPilot(), condition="1-1", seed=0)
    assert clear.outcome == "goal"


from usvnav import record as _rec


def test_a_run_record_lists_the_alive_vessels_with_their_sizes_each_tick():
    assert _rec.RUN_FORMAT == "usvnav-run/2" and "usvnav-run/1" in _rec.READ_RUN_FORMATS
    course = _box_course(lanes=[_crossing_lane()])
    rows0 = _rec.traffic_at(course, -500.0)
    assert rows0 == []                                                    # nobody on the lane yet
    rows = _rec.traffic_at(course, 41.0)
    assert len(rows) == 1 and len(rows[0]) == 5
    x, y, h, L, W = rows[0]
    assert abs(x - 60.0) < 1e-3 and abs(y) < 1e-3 and L == 20.0 and W == 6.0
    recorder = _rec.Recorder()
    res = run_episode(course, ChainPilot(), condition="1-1", seed=0, on_observation=recorder.on_observation)
    rec = _rec.episode_record(course, res, recorder, 0.1)
    assert "traffic_dims" not in rec
    assert len(rec["traffic"]) == len(rec["frames"])
    assert all(len(row) == 5 for tick in rec["traffic"] for row in tick)
    counts = {len(tick) for tick in rec["traffic"]}
    assert counts == {1}                                                   # the vessel is on the lane throughout this run
    failed = _rec.failed_record(course, "crash", "boom")
    assert failed["traffic"] == [_rec.traffic_at(course, 0.0)] and "traffic_dims" not in failed


from usvnav.figure import LANE_RGB, course_image, figure_course


def test_the_map_figure_draws_a_lane_in_its_own_colour():
    import pathlib, tempfile
    with_lane = course_image(_box_course(lanes=[_crossing_lane()]))
    without = course_image(_box_course())
    lane_rgb = np.array(LANE_RGB, dtype=np.uint8)
    assert np.any(np.all(with_lane == lane_rgb, axis=-1))
    assert not np.any(np.all(without == lane_rgb, axis=-1))
    out = pathlib.Path(tempfile.gettempdir()) / "usvnav-lane-figure.png"
    figure_course(_box_course(lanes=[_crossing_lane()]), out)
    assert out.exists() and out.stat().st_size > 1000


def test_the_ego_constants_follow_the_hull_and_the_cruise_speed():
    from usvnav.plant import HULL_LENGTH, HULL_WIDTH, V_CRUISE
    from usvnav.world import EGO_SLOT, V_WAIT
    assert abs(EGO_SLOT - (HULL_LENGTH + 2.0 * HULL_WIDTH)) < 1e-12 and abs(EGO_SLOT - 7.0) < 1e-12
    assert abs(V_WAIT - V_CRUISE / 2.0) < 1e-12 and abs(V_WAIT - 0.75) < 1e-12


def test_lane_size_helpers_take_the_largest_vessel():
    from usvnav.plant import HULL_WIDTH
    from usvnav.world import LANE_CLEAR
    lane = _straight(vessels=((12.0, 3.5), (8.0, 4.0), (20.0, 2.5)))
    assert lane.max_size() == (20.0, 4.0)
    assert abs(lane.corridor_half_width() - (2.0 + LANE_CLEAR)) < 1e-12
    assert abs(lane.coexist_width() - (4.0 + HULL_WIDTH + 2.0 * LANE_CLEAR)) < 1e-12


def test_rect_at_arc_places_the_largest_vessel_on_the_tangent():
    lane = Lane(closed=False, points=np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0]]),
                speeds=np.array([1.0, 1.0, 1.0]), vessels=[(8.0, 3.0), (12.0, 2.6)], headway_s=500.0)
    r = lane.rect_at_arc(25.0)
    assert (r.x, r.y) == (25.0, 0.0) and (r.length, r.width) == (12.0, 3.0) and abs(r.heading) < 1e-12
    r = lane.rect_at_arc(75.0, (5.0, 2.0))
    assert abs(r.x - 50.0) < 1e-9 and abs(r.y - 25.0) < 1e-9 and (r.length, r.width) == (5.0, 2.0)
    assert abs(r.heading - math.pi / 2) < 1e-9


def test_segments_cover_the_chain_and_close_a_closed_lane():
    square = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    closed = Lane(closed=True, points=square, speeds=np.full(4, 1.0), vessels=[(10.0, 3.0)])
    segs = closed.segments()
    assert len(segs) == 4
    a, b, s0, s1 = segs[-1]
    assert np.allclose(a, [0.0, 100.0]) and np.allclose(b, [0.0, 0.0]) and (s0, s1) == (300.0, 400.0)
    open_ = _straight(length=300.0)
    (a, b, s0, s1), = open_.segments()
    assert np.allclose(a, [0.0, 0.0]) and np.allclose(b, [300.0, 0.0]) and (s0, s1) == (0.0, 300.0)


def test_moving_at_is_memoised_by_time_exactly_and_boundedly():
    from usvnav.world import MEMO_MAX
    lane = _straight(length=300.0, speed=1.5, headway=45.0, offset=10.0, vessels=((12.0, 3.5), (8.0, 2.8)))
    fresh = _straight(length=300.0, speed=1.5, headway=45.0, offset=10.0, vessels=((12.0, 3.5), (8.0, 2.8)))
    times = [10.0 + 0.1 * k for k in range(300)]
    for t in times + times:                                   # every time twice: the second pass hits the memo
        got = lane.moving_at(t)
        want = fresh.alive(t)
        assert len(got) == len(want)
        for (rect, vel), (k, s) in zip(got, want):
            x, y, h = fresh.pose_at_arc(s)
            assert abs(rect.x - x) < 1e-12 and abs(rect.y - y) < 1e-12 and rect.length == fresh.vessel(k)[0]
            assert abs(vel[0] - fresh.speed_at_arc(s) * math.cos(h)) < 1e-12
        assert lane.rects_at(t) == [r for r, _ in got]
    assert len(lane._memo) <= MEMO_MAX                        # bounded: 300 distinct times, 256 kept
    assert lane.moving_at(11.0) is not lane.moving_at(11.0)   # a fresh list each time; callers extend it
    assert lane._memo == {} or fresh._memo == {}              # the untouched lane has never cached anything


def test_draw_lanes_paints_the_lane_colour_and_nothing_else():
    from usvnav import figure
    course = _box_course(lanes=[_crossing_lane()])
    img, cfg, centre, off = figure.map_view(course)
    before = img.copy()
    figure.draw_lanes(img, course, cfg, centre, off)
    changed = np.any(img != before, axis=2)
    assert changed.sum() > 50                                 # a 300 m lane plus three arrowheads
    assert np.all(img[changed] == np.array(figure.LANE_RGB, dtype=np.uint8))
    blank = img.copy()
    figure.draw_lanes(blank, _box_course(), cfg, centre, off)  # no lanes: no change
    assert np.array_equal(blank, img)
