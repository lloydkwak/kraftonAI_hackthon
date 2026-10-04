"""Episode runner: observation, outcome classification, and per-episode raw metrics.

Outcome tiers. Navigational: goal, static_collision, dynamic_collision, out_of_bounds,
timeout. Submission fault: crash, invalid_action, time_overrun. Only `goal` scores in
full; a `timeout` earns partial credit for the waypoints it reached (`score.py`); every
other outcome scores 0. Only the navigational tier is published as the failure-cause
diagnostic.

Precedence inside one tick, fixed so a re-run reproduces it:
    submission fault -> collision -> out_of_bounds -> goal -> timeout
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field

import numpy as np

from . import plant
from .collide import CourseIndex
from .geometry import (Circle, Rect, body_distance, hull_inside_polygon,
                       polygon_distance, wrap)
from .plant import DT, InvalidAction, NoDisturbance, OUDisturbance, Vessel, sanitize
from .world import MOVING, Course

SENSING_RANGE = 50.0
TICK_LIMIT = 6000           # the episode time limit in ticks, one published constant for every course

#: The wall-clock limit of one episode, as the scoring runner (`usvnav.runner`) applies it:
#: from the `reset(meta)` call to the end of the episode, the agent's time and the scorer's
#: together. An episode past it is stopped as `time_overrun`. There is no per-tick limit --
#: the 100 ms control period is simulated time, not a deadline -- so a slow tick only spends
#: the episode's budget: an episode that runs to `TICK_LIMIT` has 20 ms per tick on average.
EPISODE_TIME_LIMIT_S = 120.0

NAVIGATIONAL = ("goal", "static_collision", "dynamic_collision", "out_of_bounds", "timeout")
SUBMISSION_FAULT = ("crash", "invalid_action", "time_overrun")


class TimeOverrun(Exception):
    """The episode's wall-clock limit ran out (`EPISODE_TIME_LIMIT_S`), raised by the runner's proxy;
    `run_episode` records it as `time_overrun`. In-process agents never raise it -- the limit is
    applied at the process boundary."""


@dataclass
class Result:
    outcome: str
    tier: str
    ticks: int
    distance: float
    elapsed: float
    clearance: float                 # episode statistic, see `clearance_stat`
    waypoints_reached: int
    #: Control effort ("fuel"), a scored item: the sum over ticks of ((dv_cmd / V_MAX)^2 + (dw_cmd / W_MAX)^2) of the
    #: *applied* action -- how hard the agent changed its setpoints. Lower is smoother. Squared, so a few large jumps
    #: count more than steady dithering.
    fuel: float = 0.0
    trace: list = field(default_factory=list, repr=False)
    #: For a `submission_fault`, what the agent raised -- the traceback text for `crash`,
    #: the sanitizer's message for `invalid_action`. Diagnostic only: the scorer reads
    #: `outcome`, the studio shows this to the participant whose code it is. Empty otherwise.
    detail: str = ""
    #: The hull's pose (x, y, psi) when the episode ended -- after the step that produced a
    #: collision or reached the goal, which the trace does not contain (its last row is the
    #: tick *before*). The studio draws the run to the point it stopped.
    pose: tuple = ()
    #: Every tick's minimum surface-to-surface distance, before the clearance statistic is
    #: taken. Scoring reads `clearance`; this is kept for diagnostics.
    per_tick_clearance: list = field(default_factory=list, repr=False)

    @property
    def completed(self) -> bool:
        return self.outcome == "goal"


def observation(vessel: Vessel, course: Course, t_tick: int, wp_index: int, prev_applied):
    """The six common fields, identical in all four conditions."""
    if wp_index < course.n_waypoints:
        target = course.waypoints[wp_index]
    else:
        target = course.waypoints[-1]
    dx, dy = float(target[0] - vessel.x), float(target[1] - vessel.y)
    dist = math.hypot(dx, dy)
    bearing = wrap(math.atan2(dy, dx) - vessel.psi)
    return {
        "t": np.int32(t_tick),
        "pose": np.array([vessel.x, vessel.y, vessel.psi], dtype=np.float32),
        "vel": np.array([vessel.u, vessel.v, vessel.r], dtype=np.float32),
        "wp_index": np.int32(min(wp_index, course.n_waypoints - 1)),
        "wp_polar": np.array([dist, bearing], dtype=np.float32),
        "prev_action": np.asarray(prev_applied, dtype=np.float32),
    }


def clearance_per_tick(hull: Rect, course: Course, t: float, cap: float) -> float:
    """Minimum surface-to-surface distance to any body or the boundary, capped.

    Every surface counts, the bank included. Capping happens per tick, before the
    episode statistic, so the statistic is a function of the tight moments alone.

    **Reference implementation.** The episode loop uses `collide.CourseIndex`, which
    computes this and the collision and out-of-bounds tests in one pass with a broad
    phase. This function is kept as the thing that definition is checked against
    (`tests/test_collide.py`), not as dead code: a broad phase that silently changed a
    clearance would move every score.
    """
    d = polygon_distance(hull, course.boundary)
    for shape in course.static_shapes():
        d = min(d, body_distance(hull, shape))
        if d <= 0.0:
            return 0.0
    for rect in course.traffic_rects(t):
        d = min(d, body_distance(hull, rect))
        if d <= 0.0:
            return 0.0
    return min(d, cap)


def clearance_stat(per_tick, percentile: float) -> float:
    """Mean of the lowest `percentile`% of per-tick minima.

    The scorer uses 5% (`contest.CLEARANCE_PERCENTILE`); this default matches it.
    """
    if not len(per_tick):
        return 0.0
    a = np.sort(np.asarray(per_tick, dtype=float))
    k = max(1, int(math.ceil(len(a) * percentile / 100.0)))
    return float(a[:k].mean())


def run_episode(course: Course, agent, *, condition: str = "1-1", seed: int = 0,
                tick_limit: int = TICK_LIMIT, d_cap: float = 10.0,
                clearance_percentile: float = 5.0, record_trace: bool = False,
                on_observation=None, raster=None) -> Result:
    """Closed-loop execution of one episode.

    `on_observation(tick, meta, obs)`, if given, sees every observation exactly as the
    agent does, before `act`. It exists so the schema document's example dumps
    (`usvnav.docs`) are captured from the scoring loop itself rather than
    from a second observation builder that could drift from it.

    `raster`, if given, is the `RasterConfig` for 1-2 / 1-4 -- the studio and the CLI pass
    one carrying the participant's chosen noise. Scoring passes nothing: `render.TRACK1`.
    """
    drift = OUDisturbance(seed ^ 0x9E3779B9) if condition == "1-4" else NoDisturbance()
    x0, y0, psi0 = course.start
    vessel = Vessel(x0, y0, psi0)

    wp_index = 0
    dist_travelled = 0.0
    fuel = 0.0
    prev_applied = np.zeros(2, dtype=float)
    per_tick_clearance = []
    trace = []

    meta = {
        "boundary": course.boundary,
        "waypoints": course.waypoints,
        "arrival_radii": course.arrival_radii,
        "hull": (plant.HULL_LENGTH, plant.HULL_WIDTH),
        "observation_mode": condition,
    }
    try:
        agent.reset(meta)
    except TimeOverrun as exc:
        return Result("time_overrun", "submission_fault", 0, 0.0, 0.0, 0.0, 0, trace=trace,
                      detail=str(exc), pose=tuple(course.start))
    except Exception:
        return Result("crash", "submission_fault", 0, 0.0, 0.0, 0.0, 0, trace=trace,
                      detail=traceback.format_exc(), pose=tuple(course.start))

    index = CourseIndex(course)
    for tick in range(tick_limit + 1):
        t = tick * DT
        hull = vessel.hull()
        clearance, hit, inside = index.tick(hull, t, d_cap)
        per_tick_clearance.append(clearance)

        # --- outcome checks, in the fixed precedence order ---
        if hit is not None:
            outcome = "dynamic_collision" if hit == MOVING else "static_collision"
            return _finish(fuel, outcome, "navigational", tick, dist_travelled,
                           per_tick_clearance, clearance_percentile, wp_index, trace,
                           pose=(vessel.x, vessel.y, vessel.psi))
        if not inside:
            return _finish(fuel, "out_of_bounds", "navigational", tick, dist_travelled,
                           per_tick_clearance, clearance_percentile, wp_index, trace,
                           pose=(vessel.x, vessel.y, vessel.psi))

        # Sequential arrival, measured to the hull centre.
        if wp_index < course.n_waypoints:
            target = course.waypoints[wp_index]
            if math.hypot(vessel.x - target[0], vessel.y - target[1]) <= course.arrival_radii[wp_index]:
                wp_index += 1
        if wp_index >= course.n_waypoints:
            return _finish(fuel, "goal", "navigational", tick, dist_travelled,
                           per_tick_clearance, clearance_percentile, wp_index, trace,
                           pose=(vessel.x, vessel.y, vessel.psi))
        if tick == tick_limit:
            return _finish(fuel, "timeout", "navigational", tick, dist_travelled,
                           per_tick_clearance, clearance_percentile, wp_index, trace,
                           pose=(vessel.x, vessel.y, vessel.psi))

        obs = observation(vessel, course, tick, wp_index, prev_applied)
        obs["perception"] = _perception(condition, vessel, course, t, raster)
        if on_observation is not None:
            on_observation(tick, meta, obs)
        try:
            action = agent.act(obs)
            applied = sanitize(action)
        except InvalidAction as exc:
            return _finish(fuel, "invalid_action", "submission_fault", tick, dist_travelled,
                           per_tick_clearance, clearance_percentile, wp_index, trace,
                           detail=str(exc), pose=(vessel.x, vessel.y, vessel.psi))
        except TimeOverrun as exc:
            return _finish(fuel, "time_overrun", "submission_fault", tick, dist_travelled,
                           per_tick_clearance, clearance_percentile, wp_index, trace,
                           detail=str(exc), pose=(vessel.x, vessel.y, vessel.psi))
        except Exception:
            return _finish(fuel, "crash", "submission_fault", tick, dist_travelled,
                           per_tick_clearance, clearance_percentile, wp_index, trace,
                           detail=traceback.format_exc(), pose=(vessel.x, vessel.y, vessel.psi))

        if record_trace:
            trace.append((tick, vessel.x, vessel.y, vessel.psi, float(applied[0]), float(applied[1])))

        fuel += ((float(applied[0]) - float(prev_applied[0])) / plant.V_MAX) ** 2 \
            + ((float(applied[1]) - float(prev_applied[1])) / plant.W_MAX) ** 2
        px, py = vessel.x, vessel.y
        vessel.step(applied, drift.step())
        dist_travelled += math.hypot(vessel.x - px, vessel.y - py)
        prev_applied = applied

    raise AssertionError("unreachable")  # pragma: no cover


def _collision(hull: Rect, course: Course, t: float):
    for body in course.bodies:
        if body_distance(hull, body.shape) <= 0.0:
            return body.cls
    for rect in course.traffic_rects(t):
        if body_distance(hull, rect) <= 0.0:
            return MOVING
    return None


def _finish(fuel, outcome, tier, tick, dist, clear, pct, wp_index, trace, detail: str = "", pose=()):
    return Result(outcome, tier, tick, dist, tick * DT, clearance_stat(clear, pct), wp_index, fuel=float(fuel),
                  trace=trace, per_tick_clearance=clear, detail=detail, pose=pose)


def _perception(condition, vessel, course, t, raster=None):
    """The per-condition half of the observation. The common half is `observation`."""
    if condition == "1-1":
        from .objects import object_list
        return object_list(course, vessel, t)
    if condition == "1-3":
        from .lidar import scan
        return scan(vessel, course, t)
    if condition in ("1-2", "1-4"):
        from .render import TRACK1, top_view
        return top_view(course, vessel, t, raster if raster is not None else TRACK1)
    return None
