"""The lane design rules (L1-L11, listed in docs/OBSERVATION.md §9) and the geometry they are built on.

Every rule is tested with one hand-built violation on `lanefix`'s straight reach, whose numbers
are exact, and the clean reach with a lane through its gate reports nothing.
"""

from __future__ import annotations

import math

import numpy as np

import lanefix as fx
from usvnav import coursefile as cf
from usvnav import lanerules as lr
from usvnav.freespace import FreeSpace
from usvnav.world import Course, Lane


def test_cap_edges_are_the_reach_ends_and_padding_moves_only_them():
    b = fx.boundary()
    caps = lr.cap_edges(b)
    assert len(caps) == 2
    xs = sorted(float(a[0]) for a, _ in caps)
    assert xs == [0.0, 400.0]
    padded = lr.cap_padded(b, 13.0)
    assert padded[:, 0].min() == -13.0 and padded[:, 0].max() == 413.0
    assert np.array_equal(padded[:, 1], b[:, 1])                      # banks untouched
    assert lr.distance_to_caps((5.0, 70.0), caps) == 5.0
    assert lr.distance_to_caps((5.0, 70.0), []) == math.inf
    triangle = np.array([[0.0, 0.0], [100.0, 10.0], [50.0, 100.0]])
    assert lr.cap_edges(triangle) == [] and np.array_equal(lr.cap_padded(triangle, 5.0), triangle)


def test_the_free_width_profile_reads_the_gate_and_the_open_reach():
    lane = fx.along()
    course = fx.course(fx.gate(), [lane])
    fs = FreeSpace(course)
    s, w = lr.free_width_profile(lane, fs, lr.cap_edges(course.boundary))
    assert s[0] == 0.0 and abs(s[-1] - 400.0) < 1e-9 and abs(s[1] - s[0] - 1.0) < 1e-9
    assert np.isnan(w[0]) and np.isnan(w[-1])                         # on a cap: the world's edge
    assert not np.isnan(w[10]) and not np.isnan(w[390])
    at = lambda x: float(w[int(np.argmin(np.abs(s - x)))])
    assert abs(at(150.0) - 80.0) <= 1.0                               # the open channel
    assert abs(at(200.0) - 5.0) <= 0.6                                # inside the aperture
    assert abs(at(194.0) - 80.0) <= 1.0 and abs(at(206.0) - 80.0) <= 1.0


def test_free_width_sides_reads_each_side_and_sums_to_the_profile():
    lane = fx.along(90.0)
    course = fx.course(fx.gate(), [lane])
    fs = FreeSpace(course)
    caps = lr.cap_edges(course.boundary)
    s, left, right = lr.free_width_sides(lane, fs, caps)
    inside = (s >= 195.0) & (s <= 205.0)
    assert np.all(np.abs(left[inside] - 2.5) <= 0.6), left[inside]
    assert np.all(np.abs(right[inside] - 2.5) <= 0.6), right[inside]
    _, w = lr.free_width_profile(lane, fs, caps)
    both_finite = ~np.isnan(left) & ~np.isnan(right)
    assert np.allclose((left + right)[both_finite], w[both_finite])
    assert np.allclose(left + right, w, equal_nan=True)


def test_narrow_stretches_find_the_gate_once_with_its_depth_and_speed():
    lane = fx.along(speed=1.0)
    course = fx.course(fx.gate(), [lane])
    fs = FreeSpace(course)
    caps = lr.cap_edges(course.boundary)
    got = lr.narrow_stretches(lane, fs, lane.coexist_width(), caps)
    assert len(got) == 1
    st = got[0]
    assert abs(st.s0 - 195.0) <= 1.0 and abs(st.s1 - 206.0) <= 1.5
    assert 10.0 <= st.depth <= 12.0
    assert abs(st.w_min - 5.0) <= 0.6 and st.v_min == 1.0
    # A precomputed profile is used as-is, agreeing with the profile computed internally.
    assert lr.narrow_stretches(lane, fs, lane.coexist_width(), caps,
                                profile=lr.free_width_profile(lane, fs, caps)) == got
    assert lr.narrow_stretches(lane, fs, 4.0, lr.cap_edges(course.boundary)) == []   # 5 m is not narrower than 4
    open_reach = fx.course([], [lane])
    # The caps argument matters here too: the lane's own endpoints sit, by design, exactly on
    # the reach's cap edges at x = 0 and x = 400, and marching from there runs straight along
    # the cap line itself -- a true (not approximate) zero-width reading, not a bottleneck.
    # `caps` is what tells `narrow_stretches` to leave those samples out.
    assert lr.narrow_stretches(lane, FreeSpace(open_reach), lane.coexist_width(),
                                lr.cap_edges(open_reach.boundary)) == []


def test_a_clean_reach_with_a_lane_through_its_gate_reports_nothing():
    course = fx.course(fx.gate(), [fx.along()])
    assert cf.validate(course) == [], cf.validate(course)
    plain = fx.course([], [fx.along(50.0), fx.along(90.0)])
    assert cf.validate(plain) == [], cf.validate(plain)


def test_L2_speeds_out_of_range_and_acceleration_over_A_MAX():
    slow = fx.course([], [fx.along(speed=0.4)])
    assert "L2" in fx.rules(cf.validate(slow))
    steep = fx.course([], [fx.Lane(closed=False, points=np.array([[0.0, 90.0], [10.0, 90.0], [400.0, 90.0]]),
                                   speeds=np.array([0.5, 2.5, 2.5]), vessels=[(12.0, 3.0)], headway_s=60.0)])
    problems = cf.validate(steep)
    assert "L2" in fx.rules(problems)
    assert any("0.50 m/s" in p["message"] and "0.2" in p["message"] for p in problems if p["rule"] == "L2")


def test_L3_the_space_inequality_needs_an_ego_slot_between_vessels():
    # 15 s at 1 m/s leaves 15 - 12 = 3 m between a 12 m vessel and the next; the ego needs 7.
    course = fx.course([], [fx.along(headway=15.0, offset=0.0)])
    problems = cf.validate(course)
    assert "L3" in fx.rules(problems)
    assert "L4" not in fx.rules(problems)                            # an open reach has no passage
    assert "L3" not in fx.rules(cf.validate(fx.course([], [fx.along(headway=19.0)])))   # 19 - 12 = 7: just enough


def test_L8_a_vessel_within_sensing_range_of_the_start_at_t0():
    # offset -70 s puts vessel 0 at arc 70 -> (70, 90), 20.5 m from the start pose (74.7, 70).
    course = fx.course([], [fx.along(offset=-70.0)])
    problems = cf.validate(course)
    assert "L8" in fx.rules(problems)
    assert any("20." in p["message"] and "50" in p["message"] for p in problems if p["rule"] == "L8")
    assert "L8" not in fx.rules(cf.validate(fx.course([], [fx.along(offset=0.0)])))   # vessel 0 at the cap, 77 m away


def test_L9_vessel_sizes_outside_the_size_range_are_errors_on_a_lane():
    long_ = fx.course([], [fx.along(vessels=((25.0, 3.0),))])
    problems = cf.validate(long_)
    assert "L9" in fx.rules(problems)
    assert all(p["severity"] == "error" for p in problems if p["rule"] == "L9")
    thin = fx.course([], [fx.along(vessels=((12.0, 1.0),))])     # L9 allows 1.2-6 m wide
    assert "L9" in fx.rules(cf.validate(thin))


def test_lane_problems_is_not_consulted_for_a_course_without_lanes():
    course = fx.course(fx.gate())
    assert cf.validate(course) == []
    assert lr.lane_problems(course) == []


def test_L1_a_corridor_through_a_pier_or_over_the_bank():
    # The gate's lower pier ends at y = 87.5; a lane at y = 88 sweeps a 4 m corridor (3 m hull
    # plus LANE_CLEAR each side) from 86 to 90, through the pier.
    course = fx.course(fx.gate(), [fx.along(88.0)])
    problems = cf.validate(course)
    assert "L1" in fx.rules(problems)
    assert any("bodies[0]" in p["message"] for p in problems if p["rule"] == "L1")
    # A lane at y = 109 sweeps up to 111, over the bank at 110.
    over = fx.course([], [fx.along(109.0)])
    problems = cf.validate(over)
    assert "L1" in fx.rules(problems)
    assert any("shoreline" in p["message"] for p in problems if p["rule"] == "L1")


def test_L1_tolerates_a_vessel_half_outside_its_spawn_cap():
    # A lane that runs cap to cap has its first vessel centred on x = 0 -- half of it beyond
    # the world's edge, by design. That is not a shoreline breach.
    course = fx.course([], [fx.along(90.0, vessels=((20.0, 6.0),))])
    assert "L1" not in fx.rules(cf.validate(course))


def test_L1_sweeps_by_arc_length_so_a_body_between_vertices_is_found():
    # A single 40 m lane segment passes a buoy row; the vertices are far from the buoy.
    from usvnav.geometry import Circle
    from usvnav.world import BUOY, StaticBody
    buoy = StaticBody(BUOY, Circle(150.0, 90.0, 0.3))
    course = fx.course([buoy], [fx.along(90.0)])
    assert "L1" in fx.rules(cf.validate(course))


def test_L10_an_open_lane_that_does_not_reach_the_caps_is_a_warning():
    # Starts at x = 130 so its first vessel sits 59 m from the start pose (L8 stays quiet); ends at 350.
    course = fx.course([], [fx.along(90.0, x0=130.0, x1=350.0)])
    problems = cf.validate(course)
    assert "L10" in fx.rules(problems)
    assert sum(p["rule"] == "L10" for p in problems) == 2                    # one per end
    assert all(p["severity"] == "warning" for p in problems if p["rule"] == "L10")
    assert fx.rules(problems) == {"L10"}                             # nothing else is wrong with it
    closed = fx.Lane(closed=True, points=np.array([[150.0, 45.0], [230.0, 45.0], [230.0, 55.0], [150.0, 55.0]]),
                     speeds=np.full(4, 1.0), vessels=[(10.0, 3.0), (10.0, 3.0)])
    assert "L10" not in fx.rules(cf.validate(fx.course([], [closed])))   # a closed lane has no ends


def test_L4_the_time_inequality_at_the_gate():
    # Through a 10 m deep passage at 1 m/s a 12 m vessel occupies it for ~22 s (the finder
    # over-reads the depth by up to one metre); the ego needs (D + 7) / 0.75 ≈ 23-24 s.
    # So headway >= ~47 s: 60 passes, 40 fails, and the open reach has no passage at all.
    clean = fx.course(fx.gate(), [fx.along(headway=60.0)])
    assert "L4" not in fx.rules(cf.validate(clean))
    tight = fx.course(fx.gate(), [fx.along(headway=40.0)])
    problems = cf.validate(tight)
    assert "L4" in fx.rules(problems)
    msg = next(p["message"] for p in problems if p["rule"] == "L4")
    assert "195" in msg or "196" in msg                              # names where the passage is
    assert "40 s" in msg
    assert "L3" not in fx.rules(problems)                            # 40 · 1 − 12 = 28 ≥ 7: space is fine
    assert "L4" not in fx.rules(cf.validate(fx.course([], [fx.along(headway=40.0)])))


def test_L4_uses_the_slowest_speed_inside_the_passage():
    # Slowing to 0.5 m/s across the gate doubles the occupancy: 22 m / 0.5 = 44 s, plus the
    # ego's 23 s -> headway must be ~67 s; 60 s is no longer enough.
    slow_gate = fx.Lane(closed=False,
                        points=np.array([[0.0, 90.0], [150.0, 90.0], [190.0, 90.0], [210.0, 90.0], [250.0, 90.0], [400.0, 90.0]]),
                        speeds=np.array([1.0, 1.0, 0.5, 0.5, 1.0, 1.0]), vessels=[(12.0, 3.0)], headway_s=60.0)
    course = fx.course(fx.gate(), [slow_gate])
    problems = cf.validate(course)
    assert "L2" not in fx.rules(problems)                            # 1.0·0.5/40 = 0.0125 m/s² is gentle
    assert "L4" in fx.rules(problems)


def test_L9_a_vessel_wider_than_the_passage_minus_a_metre():
    # The aperture is 5 m; the widest vessel may be 4 m (5 − 1). A 4.5 m hull also breaks L1
    # (its 5.5 m corridor exceeds the aperture); both are reported.
    course = fx.course(fx.gate(), [fx.along(vessels=((12.0, 4.5),))])
    problems = cf.validate(course)
    assert {"L1", "L9"} <= fx.rules(problems)
    assert any("passage" in p["message"] for p in problems if p["rule"] == "L9")
    fits = fx.course(fx.gate(), [fx.along(vessels=((12.0, 3.5),))])   # 3.5 + 1 = 4.5 m corridor fits 5 m
    assert "L9" not in fx.rules(cf.validate(fits))


def test_L5_two_lanes_a_metre_apart_share_a_corridor():
    course = fx.course([], [fx.along(90.0), fx.along(91.0)])
    problems = cf.validate(course)
    assert "L5" in fx.rules(problems)
    assert any("lanes[0]" in p["message"] and "lanes[1]" in p["message"] for p in problems if p["rule"] == "L5")
    # 3 m hulls: corridors are 4 m wide, so centrelines 4 m apart just touch and 4.5 m apart are clear.
    assert "L5" in fx.rules(cf.validate(fx.course([], [fx.along(90.0), fx.along(93.9)])))
    assert "L5" not in fx.rules(cf.validate(fx.course([], [fx.along(90.0), fx.along(94.5)])))


def test_L5_a_legacy_loop_crossing_a_lane_is_caught_too():
    from usvnav.world import MOVING, TrafficRoute
    # An octagon (45° turns, so the loop's own fillet rule is quiet) around (300, 90), 20 m across.
    ang = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    octagon = np.column_stack([300.0 + 20.0 * np.cos(ang), 90.0 + 20.0 * np.sin(ang)])
    loop = TrafficRoute(MOVING, 10.0, 3.0, octagon, 1.0)
    course = fx.course([], [fx.along(90.0)], [loop])
    problems = cf.validate(course)
    assert "L5" in fx.rules(problems)
    assert any("traffic[0]" in p["message"] for p in problems if p["rule"] == "L5")
    # Two loops without any lane are the `traffic` rule's business, not L5's: no lanes, no lane rules.
    far = TrafficRoute(MOVING, 10.0, 3.0, octagon + np.array([0.0, -40.0]), 1.0)
    assert "L5" not in fx.rules(cf.validate(fx.course([], [], [loop, far])))


def test_L7_a_corridor_through_the_waypoints_or_the_start_pose():
    course = fx.course([], [fx.along(70.0)])                          # down the route itself
    problems = cf.validate(course)
    assert "L7" in fx.rules(problems)
    msgs = [p["message"] for p in problems if p["rule"] == "L7"]
    assert any("start pose" in m for m in msgs) and any("goal" in m for m in msgs)
    # L7 guards the start pose and the goal only (intermediate waypoints are deliberately dangerous
    # places). The goal at (325.3, 70), radius 3 m: the disc is 3 + 1.8 = 4.8 m; plus a 2 m corridor
    # half-width the lane's centreline must stay 6.8 m off (y >= 76.8): 76.5 is caught, 77.5 is not --
    # even though 77.5 passes 7.5 m from waypoint 0's centre.
    assert "L7" in fx.rules(cf.validate(fx.course([], [fx.along(76.5)])))
    assert "L7" not in fx.rules(cf.validate(fx.course([], [fx.along(77.5)])))


def test_free_width_profile_measures_a_passage_close_to_a_cap():
    # A passage close to a cap is still measured: a sample reads `nan` only where its march is
    # stopped within a grid cell (0.5 m) of the cap itself. Here the gate's piers sit 6 m from
    # the x = 400 cap (GATE_X + 194 = 394, GATE_DEPTH = 10 so they run 389..399) -- closer than
    # a 12 m vessel's half-length plus LANE_CLEAR (6.5 m) but far outside a grid cell of the cap,
    # so the true ~11 m deep passage should still be found and fail L4 at a headway of 40 s,
    # exactly as the gate at its usual x = 200 does.
    tight = fx.course(fx.gate(194.0), [fx.along(headway=40.0)])
    problems = cf.validate(tight)
    assert "L4" in fx.rules(problems)
    clean = fx.course(fx.gate(194.0), [fx.along(headway=60.0)])
    assert "L4" not in fx.rules(cf.validate(clean))


def test_L9_width_clause_is_not_missed_next_to_a_cap():
    # With the gate at x = 397 (3 m from the cap) the aperture is still measured, so L9's width
    # clause runs: the ~5 m gap allows 5 - 1 = 4 m, less than the 4.5 m hull.
    course = fx.course(fx.gate(197.0), [fx.along(vessels=((12.0, 4.5),))])
    problems = cf.validate(course)
    assert "L9" in fx.rules(problems)
    assert any("passage" in p["message"] for p in problems if p["rule"] == "L9")


def test_polyline_distance_matches_brute_force_far_apart_and_crossing():
    # The vectorised bounding-box shortcut must agree with the exact brute-force double
    # loop over `_seg_seg_dist` to the last bit, both when the chains never come close and when
    # they cross (the zero-distance short-circuit).
    def brute(chain_a, chain_b):
        best = math.inf
        for a0, a1 in zip(chain_a, chain_a[1:]):
            for b0, b1 in zip(chain_b, chain_b[1:]):
                best = min(best, lr._seg_seg_dist(a0, a1, b0, b1))
                if best == 0.0:
                    return 0.0
        return best

    rng = np.random.default_rng(3)
    a = np.cumsum(rng.normal(size=(60, 2)), axis=0)
    b_far = np.cumsum(rng.normal(size=(60, 2)), axis=0) + 500.0
    assert abs(lr.polyline_distance(a, b_far) - brute(a, b_far)) < 1e-9

    b_cross = a[::-1] + rng.normal(scale=0.01, size=(60, 2))   # retraces `a` in reverse: crosses it
    got, want = lr.polyline_distance(a, b_cross), brute(a, b_cross)
    assert want == 0.0                                        # confirm this pair actually crosses
    assert abs(got - want) < 1e-9

    # `stop_below` (L5's use) returns an exact distance below the threshold, not necessarily
    # the global minimum, but never a false negative: whenever the true minimum is below the
    # threshold, the returned value is too.
    assert lr.polyline_distance(a, b_far, stop_below=1e9) < 1e9
    if brute(a, b_far) < 200.0:
        assert lr.polyline_distance(a, b_far, stop_below=200.0) < 200.0


def test_L1_a_closed_lane_corridor_over_the_bank_is_not_cap_padded():
    # A closed lane never spawns at a cap, so L1's shoreline test runs on the boundary
    # unpadded for it -- a corner 5 m outside the world box is caught.
    closed = lr.Lane(closed=True,
                     points=np.array([[-5.0, 45.0], [60.0, 45.0], [60.0, 55.0], [-5.0, 55.0]]),
                     speeds=np.full(4, 1.0), vessels=[(10.0, 3.0), (10.0, 3.0)])
    problems = cf.validate(fx.course([], [closed]))
    assert "L1" in fx.rules(problems)
    assert any("shoreline" in p["message"] for p in problems if p["rule"] == "L1")


def test_L11_a_hairpin_lane_is_an_error_and_a_filleted_bend_is_not():
    hairpin = fx.Lane(closed=False, points=np.array([[0.0, 90.0], [200.0, 90.0], [400.0, 50.0]]),
                      speeds=np.array([1.0, 1.0, 1.0]), vessels=[(12.0, 3.0)], headway_s=60.0)
    assert abs(lr.max_turn_deg(hairpin) - math.degrees(math.atan2(40.0, 200.0))) < 1e-9   # 11.3°: fine
    assert "L11" not in fx.rules(cf.validate(fx.course([], [hairpin])))
    sharp = fx.Lane(closed=False, points=np.array([[0.0, 90.0], [200.0, 90.0], [200.0, 50.0], [400.0, 50.0]]),
                    speeds=np.array([1.0, 1.0, 1.0, 1.0]), vessels=[(12.0, 3.0)], headway_s=60.0)
    assert abs(lr.max_turn_deg(sharp) - 90.0) < 1e-9
    problems = cf.validate(fx.course([], [sharp]))
    assert "L11" in fx.rules(problems)
    assert any("90" in p["message"] and "60" in p["message"] for p in problems if p["rule"] == "L11")
    # A closed lane's corner between its last and first segment counts too.
    square = fx.Lane(closed=True, points=np.array([[150.0, 45.0], [230.0, 45.0], [230.0, 55.0], [150.0, 55.0]]),
                     speeds=np.full(4, 1.0), vessels=[(10.0, 3.0), (10.0, 3.0)])
    assert abs(lr.max_turn_deg(square) - 90.0) < 1e-9
    assert "L11" in fx.rules(cf.validate(fx.course([], [square])))
    assert lr.max_turn_deg(fx.along()) == 0.0


def test_spawn_walls_are_the_caps_and_every_rectangular_notch():
    plain = lr.spawn_walls(fx.boundary())
    assert [w.kind for w in plain] == ["cap", "cap"]
    assert {tuple(np.sign(w.outward).astype(int)) for w in plain} == {(-1, 0), (1, 0)}
    assert lr.notch_walls(fx.boundary()) == []
    b = fx.berth_boundary(side=+1, x=300.0, width=5.0, depth=8.0)
    walls = lr.spawn_walls(b)
    assert [w.kind for w in walls] == ["cap", "cap", "berth"]
    berth = walls[-1]
    assert np.allclose(sorted([berth.a[0], berth.b[0]]), [297.5, 302.5]) and berth.a[1] == berth.b[1] == 118.0
    assert np.allclose(berth.outward, [0.0, 1.0])                        # into the land behind the bank
    assert len(berth.sides) == 2 and all(abs(a[0] - c[0]) < 1e-9 for a, c in berth.sides)   # vertical side walls
    low = lr.spawn_walls(fx.berth_boundary(side=-1, x=120.0))
    assert low[-1].kind == "berth" and np.allclose(low[-1].outward, [0.0, -1.0]) and low[-1].a[1] == 22.0
    # A pier-like bump sticking *into* the water has the opposite turn signature and is not a wall.
    bump = np.array([[0.0, 30.0], [400.0, 30.0], [400.0, 110.0], [302.5, 110.0], [302.5, 102.0], [297.5, 102.0],
                     [297.5, 110.0], [0.0, 110.0]])
    assert lr.notch_walls(bump) == []
    # A gentle bank bend (small turns) is not a wall either.
    bend = np.array([[0.0, 30.0], [400.0, 30.0], [400.0, 110.0], [300.0, 110.0], [290.0, 108.0], [280.0, 110.0], [0.0, 110.0]])
    assert lr.notch_walls(bend) == []


def test_spawn_padded_moves_caps_along_x_and_berth_walls_into_the_land():
    b = fx.berth_boundary(side=+1, x=300.0)
    padded = lr.spawn_padded(b, 7.0)
    assert np.allclose(lr.cap_padded(fx.boundary(), 7.0)[:, 0].min(), padded[:, 0].min())   # caps as before
    assert padded[:, 1].max() == 118.0 + 7.0                              # the back wall moved 7 m deeper
    assert np.sum(np.abs(padded[:, 1] - 125.0) < 1e-9) == 2                # exactly its two vertices
    assert np.allclose(lr.spawn_padded(fx.boundary(), 3.0), lr.cap_padded(fx.boundary(), 3.0))
    assert lr.distance_to_walls(np.array([300.0, 118.0]), lr.spawn_walls(b)) < 1e-9
    assert abs(lr.distance_to_walls(np.array([300.0, 112.0]), lr.spawn_walls(b)) - 6.0) < 1e-9
    edges = lr.wall_edges(lr.spawn_walls(b))
    assert len(edges) == 2 + 3                                            # two caps, a berth's back wall and two sides


def test_L1_and_L10_accept_a_lane_that_ends_on_a_berth_wall():
    b = fx.berth_boundary(side=+1, x=300.0, width=5.0, depth=8.0)
    lane = fx.berth_lane(x=300.0, side=+1, depth=8.0)                    # 3 m vessel: 4 m corridor in a 5 m slip
    course = Course(b, [], np.array([[160.0, 70.0], [240.0, 70.0], [325.3, 70.0]]), np.array([5.0, 5.0, 3.0]),
                    (74.7, 70.0, 0.0), [], [lane], terrain_extent=cf.default_terrain_extent(b))
    problems = cf.validate(course)
    assert "L10" not in fx.rules(problems), problems                       # the end lies on a spawn wall
    assert "L1" not in fx.rules(problems), problems                        # the spawn rect is half behind the padded wall
    # The same lane on the un-notched reach ends in open water: L10 warns and L1 finds the shore.
    plain = Course(fx.boundary(), [], course.waypoints, course.arrival_radii, course.start, [], [lane],
                   terrain_extent=cf.default_terrain_extent(fx.boundary()))
    rules = fx.rules(cf.validate(plain))
    assert "L10" in rules and "L1" in rules
    # A lane ending two metres short of the back wall is still open water to L10.
    short = Lane(closed=False, points=lane.points[:-1].tolist() + [[300.0, 116.0]], speeds=np.full(5, 1.0),
                 vessels=[(12.0, 3.0)], headway_s=60.0)
    assert "L10" in fx.rules(cf.validate(Course(b, [], course.waypoints, course.arrival_radii, course.start, [], [short],
                                                terrain_extent=cf.default_terrain_extent(b))))


def test_the_profile_does_not_read_a_berth_slip_as_a_passage():
    b = fx.berth_boundary(side=+1, x=300.0, width=5.0, depth=8.0)
    lane = fx.berth_lane(x=300.0, side=+1, depth=8.0, headway=20.0)     # 20 s: L3 passes (needs 19 s); L4 would fail if the 5 m slip read as an 8 m deep passage
    course = Course(b, [], np.array([[160.0, 70.0], [240.0, 70.0], [325.3, 70.0]]), np.array([5.0, 5.0, 3.0]),
                    (74.7, 70.0, 0.0), [], [lane], terrain_extent=cf.default_terrain_extent(b))
    fs = FreeSpace(course)
    walls = lr.wall_edges(lr.spawn_walls(b))
    s, w = lr.free_width_profile(lane, fs, walls)
    inside = s > lane.length - 8.0                                       # the last 8 m are inside the slip
    assert np.all(np.isnan(w[inside]))                                   # unmeasurable: the slip's own walls
    assert lr.narrow_stretches(lane, fs, lane.coexist_width(), walls) == []
    assert "L4" not in fx.rules(cf.validate(course))


def test_corridor_conflicts_finds_the_gate_hit_and_is_empty_on_the_clean_line():
    course = fx.course(fx.gate(), [fx.along(88.0)])                       # the L1 fixture: the corridor hits the lower pier
    hits = lr.corridor_conflicts(course.lanes[0], course)
    assert hits, "the 88 m line's corridor should clip the lower pier near the gate"
    assert any(w == 0 for _, w in hits), hits                             # bodies[0] is the lower pier
    assert any(180.0 <= arc <= 220.0 for arc, w in hits if w == 0), hits   # the gate sits at x = 200
    clean = fx.course(fx.gate(), [fx.along(90.0)])                        # dead centre of the 5 m aperture
    assert lr.corridor_conflicts(clean.lanes[0], clean) == []
