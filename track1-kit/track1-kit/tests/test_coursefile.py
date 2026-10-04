"""The course file format and its rules, on the courses a participant holds.

The format is the contract between the editor, the scorer and the participant. The tests
here use the kit's own course files as fixtures (`fixtures.py`); the tests that need the
generator are organiser-side and not in the kit, because the generator does not ship.
"""

from __future__ import annotations

import json

import numpy as np

from fixtures import course as fixture_course, course_with_traffic, courses as fixture_courses
from usvnav import coursefile as cf
from usvnav.agent import TutorialAgent
from usvnav.geometry import Rect
from usvnav.sim import run_episode
from usvnav.world import MOORED, MOVING, StaticBody


def _tmp(name):
    import pathlib
    import tempfile
    return pathlib.Path(tempfile.gettempdir()) / name


def test_round_trip_preserves_the_geometry():
    for course in fixture_courses():
        back = cf.from_dict(cf.to_dict(course))
        assert np.allclose(course.boundary, back.boundary, atol=1e-3)
        assert np.allclose(course.waypoints, back.waypoints, atol=1e-3)
        assert np.allclose(course.arrival_radii, back.arrival_radii, atol=1e-3)
        assert np.allclose(course.start, back.start, atol=1e-4)
        assert len(course.bodies) == len(back.bodies)
        for a, b in zip(course.bodies, back.bodies):
            assert a.cls == b.cls and type(a.shape) is type(b.shape)
            assert abs(a.shape.x - b.shape.x) < 1e-3
        assert len(course.traffic) == len(back.traffic)
        for a, b in zip(course.traffic, back.traffic):
            for t in (0.0, 7.3, 91.1):
                assert np.allclose(a.pose_at(t), b.pose_at(t), atol=1e-3)


def test_a_saved_course_loads_and_runs_identically_twice():
    """A scored course is the file (`coursefile`): two loads of the same file agree
    exactly, in every condition. The generator-object-versus-file comparison, which is
    where the 0.1 mm rounding shows, needs the generator and is not in the kit."""
    course = course_with_traffic()
    path = cf.save(course, _tmp("rt-fixture.json"))
    a, b = cf.load(path), cf.load(path)
    for cond in ("1-1", "1-3", "1-4"):
        x = run_episode(a, TutorialAgent(), condition=cond, seed=3, tick_limit=400)
        y = run_episode(b, TutorialAgent(), condition=cond, seed=3, tick_limit=400)
        assert (x.outcome, x.ticks, x.clearance, x.distance) == \
               (y.outcome, y.ticks, y.clearance, y.distance), cond


def test_the_bundled_courses_carry_no_hidden_field_and_validate_clean():
    """Nothing in a course file a participant holds says how it was made, and every
    bundled course passes every rule."""
    for course in fixture_courses():
        text = json.dumps(cf.to_dict(course))
        for leak in ("situation", "seed", "difficulty", "gap_ratio", "density", "composite",
                     "easy", "medium", "hard", "scenario"):
            assert leak not in text, f"{leak!r} is in a bundled course file"
        assert set(cf.to_dict(course)) == {"format", "world", "start", "waypoints", "bodies",
                                           "traffic", "lanes"}
        assert not cf.validate(course), cf.validate(course)


def _clean_course():
    return cf.from_dict(cf.to_dict(fixture_course(0)))


def _rules(problems):
    return {p["rule"] for p in problems}


def test_validate_catches_a_body_on_the_start_pose():
    course = _clean_course()
    x, y, _ = course.start
    course.bodies.append(StaticBody(MOORED, Rect(x, y, 20.0, 6.0, 0.0)))
    assert "start" in _rules(cf.validate(course))


def test_validate_catches_a_body_in_an_arrival_disc():
    """`keep-clear`. A body inside the disc makes arrival unsatisfiable at every heading, so
    this is unfinishable rather than merely hard -- which is why it is an error."""
    course = _clean_course()
    goal = course.waypoints[-1]
    course.bodies.append(StaticBody(MOORED, Rect(float(goal[0]) + 2.0, float(goal[1]),
                                                 20.0, 6.0, 0.0)))
    assert "keep-clear" in _rules(cf.validate(course))


def test_validate_catches_a_waypoint_chain_outside_the_rules():
    course = _clean_course()
    course.waypoints = course.waypoints[:1]
    course.arrival_radii = course.arrival_radii[:1]
    assert "waypoints" in _rules(cf.validate(course))

    course = _clean_course()
    far = np.asarray(course.start[:2], dtype=float) + np.array([5.0, 0.0])
    course.waypoints = np.vstack([far[None, :], course.waypoints])
    course.arrival_radii = np.concatenate([[5.0], course.arrival_radii])
    assert "waypoints" in _rules(cf.validate(course))


def test_validate_catches_a_body_outside_its_size_range():
    course = _clean_course()
    course.bodies.append(StaticBody(MOORED, Rect(200.0, 30.0, 90.0, 6.0, 0.0)))
    problems = cf.validate(course)
    assert "size" in _rules(problems)
    assert all(p["severity"] == "warning" for p in problems if p["rule"] == "size"), (
        "a size outside the range still runs, so it is a warning; an author may mean it")
    assert "size" not in _rules(cf.validate(course, strict_sizes=False))


def test_validate_catches_a_shoreline_that_crosses_itself():
    course = _clean_course()
    b = np.asarray(course.boundary, dtype=float).copy()
    b[[1, len(b) // 2]] = b[[len(b) // 2, 1]]          # swap two vertices
    course.boundary = b
    assert "shoreline" in _rules(cf.validate(course))


def test_validate_catches_a_terrain_margin_that_lets_outside_world_in():
    """`terrain`: the figure it needs is the raster's 70.71 m half-diagonal rather than
    the 50 m sensing range -- the square extent reaches further into its corners than
    along its axes."""
    course = _clean_course()
    b = np.asarray(course.boundary, dtype=float)
    course.terrain_extent = (float(b[:, 0].min()) - 50.0, float(b[:, 1].min()) - 50.0,
                             float(b[:, 0].max()) + 50.0, float(b[:, 1].max()) + 50.0)
    assert "terrain" in _rules(cf.validate(course))


def test_structural_errors_are_raised_not_reported():
    """`from_dict` raises when a file cannot become a course; `validate` reports when a
    course breaks a rule. The editor depends on the split: a half-finished course should
    still open and draw so its author can see what is wrong with it."""
    good = cf.to_dict(fixture_course(0))

    for mangle, why in (
        (lambda d: d.update(format="something-else"), "wrong format"),
        (lambda d: d.update(waypoints=[]), "no waypoints"),
        (lambda d: d["bodies"].append({"class": MOVING, "primitive": "rect", "x": 1.0,
                                       "y": 1.0, "length_m": 8.0, "width_m": 3.0}),
         "a moving vessel among the static bodies"),
        (lambda d: d["bodies"].append({"class": "buoy", "primitive": "rect", "x": 1.0,
                                       "y": 1.0, "length_m": 1.0, "width_m": 1.0}),
         "a buoy declared as a rectangle"),
        (lambda d: d["world"].update(boundary=[[0.0, 0.0], [1.0, 1.0]]), "a two-point polygon"),
        (lambda d: d["start"].pop("heading_rad"), "a start pose with no heading"),
    ):
        d = json.loads(json.dumps(good))
        mangle(d)
        try:
            cf.from_dict(d)
        except cf.CourseFileError:
            pass
        else:
            raise AssertionError(f"{why} was accepted")


def test_terrain_extent_is_optional_and_derived():
    """Hand-authoring should not require restating something the loader derives."""
    d = cf.to_dict(fixture_course(0))
    d["world"].pop("terrain_extent")
    back = cf.from_dict(d)
    assert back.terrain_extent is not None
    assert not [p for p in cf.validate(back) if p["rule"] == "terrain"]


def test_the_editor_and_the_validator_share_their_constants():
    """The editor re-implements the cheap checks in JavaScript, so its constants are a
    copy and a copy can drift. This asserts the ones that matter are still the same
    numbers, by reading them out of the HTML.

    It cannot assert the *logic* agrees -- that is what driving the editor from a
    headless browser would show, and there is no browser in the test environment.
    """
    import pathlib
    import re

    from usvnav.lanerules import LANE_TURN_MAX_DEG
    from usvnav.rules import LEG_MAX, LEG_MIN, REACH_X, REACH_Y, WP_MAX, WP_MIN
    from usvnav.plant import HULL_LENGTH, HULL_WIDTH
    from usvnav.sim import SENSING_RANGE
    from usvnav.world import SIZE_RANGES, TRAFFIC_SPEED_MAX, TRAFFIC_SPEED_MIN

    html = (pathlib.Path(__file__).resolve().parent.parent
            / "tools" / "editor.html").read_text()

    def js(name):
        m = re.search(rf"\b{name}\s*=\s*([0-9.]+)", html)
        assert m, f"{name} is not in editor.html any more"
        return float(m.group(1))

    assert js("HULL_L") == HULL_LENGTH and js("HULL_W") == HULL_WIDTH
    assert js("REACH_X") == REACH_X and js("REACH_Y") == REACH_Y
    assert js("LEG_MIN") == LEG_MIN and js("LEG_MAX") == LEG_MAX
    assert js("WP_MIN") == WP_MIN and js("WP_MAX") == WP_MAX
    assert js("LANE_TURN_MAX") == LANE_TURN_MAX_DEG                   # L11, the lane tool's own check
    assert js("SENSING") == SENSING_RANGE
    assert js("SPEED_MIN") == TRAFFIC_SPEED_MIN and js("SPEED_MAX") == TRAFFIC_SPEED_MAX
    assert js("BUOY_D") == SIZE_RANGES["buoy"][0]
    assert f'"{cf.FORMAT}"' in html
    for tag in (t for t in cf.READ_FORMATS if t.startswith("usvnav-")):   # the Open dialog reads what a participant can hold
        assert f'"{tag}"' in html, f"editor.html does not accept {tag}"
    for cls, (lr, wr) in SIZE_RANGES.items():
        if cls == "buoy":
            continue
        assert f"{cls}:[[{lr[0]:g},{lr[1]:g}],[{wr[0]:g},{wr[1]:g}]]" in \
            html.replace(" ", ""), f"{cls}'s size range differs in editor.html"
    # Colours: the editor must track the **map** palette (`figure.MAP_LOOK`), not the
    # observation palette. Pinning it to the map palette is what keeps the editor and
    # `usvnav view` agreeing about what a barge looks like, which is the drift that
    # matters to an author comparing the two windows.
    from usvnav.figure import MAP_LOOK
    from usvnav.world import BANK, BUOY, WATER

    for name in (WATER, BANK, BUOY):
        r, g, b = MAP_LOOK[name]["rgb"]
        assert f"#{r:02X}{g:02X}{b:02X}" in html, \
            f"the {name} colour differs between editor.html and figure.MAP_LOOK"
    for gone in ("#12263E", "#BE0091", "#22B5A5"):
        assert gone not in html, \
            f"{gone} is from the observation palette; the editor is not an observation"


def test_the_editor_carries_lanes_through_open_and_save():
    """`saveObject()` builds the saved file field by field, so it must carry `lanes`, and the
    Open handler must default `lanes` as it does `bodies` and `traffic` -- otherwise a course
    with lanes, opened in the editor (or the studio, which saves through the same `toJSON`)
    and then saved, loses every lane silently, and the studio's dirty check -- which compares
    two `JSON.stringify(toJSON())` snapshots -- does not notice because both sides miss the
    same field."""
    import pathlib

    html = (pathlib.Path(__file__).resolve().parent.parent
            / "tools" / "editor.html").read_text()

    i = html.index("function saveObject")
    assert "lanes" in html[i:i + 1500], "saveObject() does not carry lanes into the saved file"
    assert "course.lanes = course.lanes || []" in html, \
        "the Open handler does not default lanes on an old-format file"
    # a lane can be drawn, not only carried: the Add button and the placement branch
    assert 'data-add="lane"' in html and 'what === "lane"' in html and 'kind==="lane"' in html


def test_validate_catches_a_traffic_route_tunnelling_through_a_body_between_vertices():
    """`traffic`. A route with a long straight leg and a body sitting between its vertices must
    be caught: sampling at the vertices only would step right over the body, so the check
    sweeps the leg by arc length with overlapping rectangles. Driven through
    `_route_problems` directly so only the route/body geometry is under test."""
    from usvnav.geometry import Circle, body_distance
    from usvnav.world import Course, TrafficRoute
    base = _clean_course()
    route = TrafficRoute(MOVING, 6.0, 3.0, np.array([[120, 60], [280, 60], [280, 95], [120, 95]], float), 1.5, 0.0)
    body = StaticBody(MOORED, Circle(200.0, 60.0, 2.0))                    # mid bottom leg, far from any vertex
    course = Course(base.boundary, [body], base.waypoints, base.arrival_radii, base.start, [route])
    # sampling only at the four vertices never touches the body
    vertex_hit = any(body_distance(route.rect_at(route._cum[k] / max(route.speed, 1e-9)), body.shape) <= 0.0
                     for k in range(len(route.points)))
    assert not vertex_hit
    problems = list(cf._route_problems(route, course, 0, __import__("usvnav.collide", fromlist=["CourseIndex"]).CourseIndex(course)))
    assert problems and "passes through" in problems[0]

from usvnav.world import Course, Lane


def _lane_course():
    base = fixture_course(0)
    open_lane = Lane(closed=False,
                     points=np.array([[10.0, 5.0], [80.0, 5.0], [80.0, 40.0]]),
                     speeds=np.array([1.2, 2.4, 1.0]), vessels=[(12.0, 3.5), (8.0, 2.8)],
                     headway_s=45.0, offset_s=-30.0)
    closed_lane = Lane(closed=True,
                       points=np.array([[100.0, 60.0], [140.0, 60.0], [140.0, 90.0], [100.0, 90.0]]),
                       speeds=np.array([1.0, 1.5, 1.0, 1.5]), vessels=[(10.0, 3.0), (10.0, 3.0)])
    return Course(base.boundary, list(base.bodies), base.waypoints, base.arrival_radii, base.start,
                  list(base.traffic), [open_lane, closed_lane], terrain_extent=base.terrain_extent)


def test_the_format_is_version_2_and_version_1_files_still_load():
    assert cf.FORMAT == "usvnav-course/2"
    assert cf.READ_FORMATS[0] == cf.FORMAT and "usvnav-course/1" in cf.READ_FORMATS
    frozen = [tag for tag in cf.READ_FORMATS if not tag.startswith("usvnav-")]
    assert len(frozen) == 1 and frozen[0].endswith("-course/1")     # the older tag stays readable
    plain = next((c for c in fixture_courses() if not c.lanes), None)      # tools/demo-course.json
    assert plain is not None, "a lane-free fixture course is needed here"
    d = cf.to_dict(plain)
    assert d["format"] == "usvnav-course/2" and d["lanes"] == []
    d1 = dict(d); d1["format"] = "usvnav-course/1"; d1.pop("lanes")
    assert cf.from_dict(d1).lanes == []
    d2 = dict(d); d2["format"] = frozen[0]; d2.pop("lanes")
    assert cf.from_dict(d2).lanes == []


def test_lanes_round_trip_through_the_file():
    course = _lane_course()
    d = cf.to_dict(course)
    assert len(d["lanes"]) == 2
    o, c = d["lanes"]
    assert o["closed"] is False and o["headway_s"] == 45.0 and o["offset_s"] == -30.0
    assert o["speeds_mps"] == [1.2, 2.4, 1.0] and len(o["route"]) == 3
    assert o["vessels"] == [{"length_m": 12.0, "width_m": 3.5}, {"length_m": 8.0, "width_m": 2.8}]
    assert c["closed"] is True and "headway_s" not in c
    back = cf.from_dict(json.loads(json.dumps(d)))
    assert len(back.lanes) == 2
    for a, b in zip(course.lanes, back.lanes):
        assert a.closed == b.closed
        assert np.allclose(a.points, b.points, atol=1e-3) and np.allclose(a.speeds, b.speeds, atol=1e-4)
        assert a.vessels == b.vessels and a.headway_s == b.headway_s and abs(a.offset_s - b.offset_s) < 1e-4
        for t in (0.0, 17.3, 250.0):
            assert np.allclose([r.x for r in a.rects_at(t)], [r.x for r in b.rects_at(t)], atol=1e-3)
    path = _tmp("lane-course.json")
    cf.save(course, path)
    assert len(cf.load(path).lanes) == 2


def test_a_lane_that_cannot_be_one_is_a_file_error_not_a_crash():
    good = cf.to_dict(_lane_course())["lanes"][0]
    def rejects(mutate):
        d = cf.to_dict(fixture_course(0)); lane = json.loads(json.dumps(good)); mutate(lane); d["lanes"] = [lane]
        try:
            cf.from_dict(d)
        except cf.CourseFileError:
            return True
        return False
    assert rejects(lambda l: l.pop("route"))
    assert rejects(lambda l: l.__setitem__("route", [[0.0, 0.0]]))
    assert rejects(lambda l: l.__setitem__("speeds_mps", [1.0]))
    assert rejects(lambda l: l.pop("headway_s"))                       # open needs a headway
    assert rejects(lambda l: l.__setitem__("vessels", []))
    assert rejects(lambda l: l.__setitem__("class", "moored_vessel"))
    assert rejects(lambda l: (l.__setitem__("closed", True)))          # closed with a headway present
    assert rejects(lambda l: l.__setitem__("vessels", [{"length_m": 12.0}]))   # width missing
    assert rejects(lambda l: l.__setitem__("closed", "false"))         # truthy string, not a bool
    assert rejects(lambda l: l.__setitem__("route", [[0.0, 90.0], [400.0]]))          # ragged
    assert rejects(lambda l: l.__setitem__("route", [[0.0, 90.0], ["x", 90.0]]))      # not a number
    assert rejects(lambda l: l.__setitem__("route", [[0.0, 90.0], [float("nan"), 90.0]]))


def test_a_ragged_traffic_route_or_boundary_is_a_file_error_not_a_crash():
    """A ragged `traffic` route or `world.boundary` is a `CourseFileError` naming the field,
    not numpy's shape error in numpy's wording."""
    # No fixture carries a legacy loop any more (moving vessels are on lanes), so one is written in by hand:
    # the point is the file loader's error, not the loop.
    d = cf.to_dict(fixture_course(0))
    d["traffic"] = [{"class": "moving_vessel", "length_m": 10.0, "width_m": 3.0, "speed_mps": 1.0, "phase_m": 0.0,
                     "route": [[100.0, 60.0], [140.0, 60.0], [140.0, 80.0], [100.0, 80.0]]}]
    broken = json.loads(json.dumps(d)); broken["traffic"][0]["route"] = [[0.0, 0.0], [1.0], [2.0, 2.0]]
    try:
        cf.from_dict(broken)
    except cf.CourseFileError as e:
        assert "traffic[0].route" in str(e)
    else:
        raise AssertionError("a ragged traffic route loaded")
    broken = json.loads(json.dumps(d)); broken["world"]["boundary"] = [[0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0]]
    try:
        cf.from_dict(broken)
    except cf.CourseFileError as e:
        assert "boundary" in str(e)
    else:
        raise AssertionError("a ragged boundary loaded")
