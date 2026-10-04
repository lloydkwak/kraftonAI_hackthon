"""Episode scoring: the relative items among completers, and partial credit for a safe timeout -- (k/N)^2 times the lowest completer's score, nothing where nobody completes."""

from __future__ import annotations

from usvnav.contest import WEIGHTS
from usvnav.score import ITEMS, episode_ratios, partial_credit


def _total(r):
    return sum(WEIGHTS[i] * r[i] for i in ITEMS)


def _e(outcome, fuel=10.0, clearance=1.0, time=3000.0, waypoints=6, of=6):
    return {"outcome": outcome, "fuel": fuel, "clearance": clearance, "time": time, "waypoints": waypoints, "of": of}


FIELD = {"a": _e("goal", 10.0, 1.0, 3000.0), "b": _e("goal", 20.0, 0.5, 4000.0),
         "slow": _e("timeout", 1.0, 10.0, 6000.0, waypoints=5), "parked": _e("timeout", 0.0, 10.0, 6000.0, waypoints=2),
         "crash": _e("static_collision", 5.0, 0.0, 900.0, waypoints=5)}


def test_completers_are_scored_against_completers_only():
    r = episode_ratios(FIELD, WEIGHTS)
    assert _total(r["a"]) == 1.0
    assert abs(_total(r["b"]) - (0.3 * 0.5 + 0.4 * 0.5 + 0.3 * 0.75)) < 1e-12
    # the timeouts' fuel (1.0, 0.0) and clearance (10 m) would beat a's; they never become a best
    assert r["a"]["fuel"] == 1.0 and r["a"]["clearance"] == 1.0


def test_a_timeout_scores_the_square_of_its_progress_times_the_worst_completion():
    r = episode_ratios(FIELD, WEIGHTS)
    worst = _total(r["b"])
    assert abs(_total(r["slow"]) - (5 / 6) ** 2 * worst) < 1e-12
    assert abs(_total(r["parked"]) - (2 / 6) ** 2 * worst) < 1e-12
    assert _total(r["slow"]) < worst                   # below every completion
    assert all(r["slow"][i] == r["slow"]["fuel"] for i in ITEMS)


def test_a_collision_scores_nothing_however_far_it_got():
    assert _total(episode_ratios(FIELD, WEIGHTS)["crash"]) == 0.0


def test_nobody_gets_partial_credit_where_nobody_completes():
    r = episode_ratios({"slow": FIELD["slow"], "crash": FIELD["crash"]}, WEIGHTS)
    assert _total(r["slow"]) == 0.0 and _total(r["crash"]) == 0.0


def test_without_weights_the_rule_is_binary_as_before():
    r = episode_ratios(FIELD)
    assert _total(r["slow"]) == 0.0 and _total(r["a"]) == 1.0


def test_partial_credit_guards_its_inputs():
    assert partial_credit({"waypoints": 3, "of": 0}, 0.8) == 0.0
    assert partial_credit({"waypoints": 9, "of": 6}, 0.8) == 0.8          # k is capped at N
