"""The broad phase must change no number. This is the test that says so.

`collide.CourseIndex` does the three per-tick tests (clearance, collision, out-of-bounds)
in one pass, and skips exact tests it can prove cannot win. A broad phase that got a bound
wrong would move a clearance statistic, and the clearance item divides by the best
clearance among completing submissions -- so the error would land straight in the score.
"""

from __future__ import annotations

import numpy as np

from usvnav.collide import CourseIndex
from fixtures import easy_course, courses as fixture_courses
from usvnav.geometry import Rect, hull_inside_polygon
from usvnav.plant import DT, HULL_LENGTH, HULL_WIDTH, Vessel
from usvnav.sim import clearance_per_tick, run_episode
from pilot import ChainPilot


def _poses(course, n=400, seed=0):
    """Poses over the whole reach, not just the driven line: the broad phase has to be
    right where the hull is far from everything as well as where it is touching."""
    rng = np.random.default_rng(seed)
    b = course.boundary
    xs = rng.uniform(b[:, 0].min() - 5, b[:, 0].max() + 5, n)
    ys = rng.uniform(b[:, 1].min() - 5, b[:, 1].max() + 5, n)
    ps = rng.uniform(-np.pi, np.pi, n)
    return np.column_stack([xs, ys, ps])


def test_broad_phase_matches_the_reference_implementation():
    cap = 10.0
    for seed, course in enumerate(fixture_courses()[:3], start=7):
        index = CourseIndex(course)
        for x, y, psi in _poses(course, 300, seed):
            hull = Rect(x, y, HULL_LENGTH, HULL_WIDTH, psi)
            for t in (0.0, 13.7):
                clear, hit, inside = index.tick(hull, t, cap)
                ref_clear = clearance_per_tick(hull, course, t, cap)
                assert abs(clear - ref_clear) < 1e-9, (seed, x, y, psi, clear, ref_clear)
                ref_inside = hull_inside_polygon(hull, course.boundary)
                # A hull in contact with a body short-circuits before the bounds test,
                # which is the documented precedence (sim.run_episode), so only compare
                # when there is no contact.
                if hit is None:
                    assert inside == ref_inside, (seed, x, y, psi, inside, ref_inside)


def test_contact_is_reported_with_the_class_that_owns_it():
    """The outcomes separate `static_collision` from `dynamic_collision`, so the index has to
    return *which* body is in contact, not just that one is."""
    course = fixture_courses()[0]
    index = CourseIndex(course)
    body = course.bodies[0]
    hull = Rect(body.shape.x, body.shape.y, HULL_LENGTH, HULL_WIDTH, 0.3)
    clear, hit, _ = index.tick(hull, 0.0, 10.0)
    assert hit == body.cls, (hit, body.cls)
    assert clear == 0.0


def test_same_seed_reruns_identically():
    """A rerun with the same seed reproduces the trajectory, which an audit replay relies on.

    Under 1-4, so the disturbance stream is in the comparison: an episode's independent
    random streams all derive from its one seed, and a drift that did not replay would show
    up here and nowhere else in the shipped suite.
    """
    import lanefix
    course = easy_course(4, lanes=[lanefix.along(speed=1.5, headway=40.0)])   # moving vessels are in the comparison
    a = run_episode(course, ChainPilot(), condition="1-4", seed=11, record_trace=True)
    b = run_episode(course, ChainPilot(), condition="1-4", seed=11, record_trace=True)
    assert len(a.trace) > 100, f"a {len(a.trace)}-tick trace tests almost nothing"
    assert a.outcome == b.outcome and a.ticks == b.ticks
    assert np.array_equal(np.array(a.trace, dtype=float), np.array(b.trace, dtype=float))
