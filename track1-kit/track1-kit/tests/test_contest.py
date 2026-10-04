"""The end-to-end path: a set of episodes, a field of submissions, their scores.

`usvnav.contest` runs the whole system rather than one episode, so what these tests
protect is the scoring *procedure* of RULES.md §5 -- per-episode normalisation against
completing submissions only, the item weights, the weighted mean over conditions (a
timeout's partial credit is tested in `test_score.py`) -- and the shape of a set written
from courses in hand.
"""

from __future__ import annotations

import json

import numpy as np

from usvnav import coursefile as cf
from usvnav.agent import TutorialAgent
from fixtures import scratch, write_fixture_set
from usvnav.contest import CONDITION_WEIGHTS, CONDITIONS, WEIGHTS, leaderboard, load_set, score_set
from usvnav.score import ITEMS
from pilot import ChainPilot


def _tmp(name):
    return scratch("usvnav-set-") / name


def test_a_set_round_trips_and_its_course_files_are_clean():
    out = _tmp("public")
    manifest = write_fixture_set(out, n=2, conditions=CONDITIONS)
    assert manifest["kind"] == "public" and [e["id"] for e in manifest["episodes"]] == ["fx-01", "fx-02"]
    for ep in manifest["episodes"]:
        assert set(ep) == {"id", "course", "seed", "conditions"}
        assert ep["conditions"] == list(CONDITIONS)
        course = cf.load(out / ep["course"])
        assert not cf.validate(course), ep["id"]
    loaded, root = load_set(out)
    assert loaded == manifest and root == out


def test_scoring_a_field_follows_the_relative_procedure_and_the_weights():
    """Two submissions on two fixture courses under two conditions.

    The chain pilot follows the disclosed chain and the tutorial agent steers at the
    waypoint, so where either completes the field has a completer and usually a
    non-completer, which is where the scoring rules bite: the completer's items are 1.0 (it is the best completing submission), the
    non-completer's are 0 on all three, and if both complete, each item's ratio is in
    (0, 1] with at least one of them at 1.0.
    """
    out = _tmp("field")
    manifest = write_fixture_set(out, n=2, conditions=("1-1", "1-3"))
    agents = {"pilot": ChainPilot, "tutorial": TutorialAgent}
    scores = score_set(manifest, out, agents)

    assert scores["weights"] == WEIGHTS and abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
    assert set(scores["raw"]) == {"1-1", "1-3"}
    for cond, eps in scores["per_episode"].items():
        for ep_id, ratios in eps.items():
            raw = scores["raw"][cond][ep_id]
            completers = [t for t in raw if raw[t]["outcome"] == "goal"]
            for team, r in ratios.items():
                if raw[team]["outcome"] != "goal":
                    assert all(r[i] == 0.0 for i in ITEMS), (team, r)
                else:
                    assert all(0.0 < r[i] <= 1.0 for i in ITEMS), (team, r)
            if completers:
                for item in ITEMS:
                    assert max(ratios[t][item] for t in completers) == 1.0, (item, ratios)
    # The final score is the weighted mean of the condition
    # totals, 1-1/1-2 at 20% and 1-3/1-4 at 30%, renormalised over the conditions scored --
    # here 1-1 and 1-3, so 0.4 and 0.6.
    assert CONDITION_WEIGHTS == {"1-1": 0.2, "1-2": 0.2, "1-3": 0.3, "1-4": 0.3}
    assert abs(sum(CONDITION_WEIGHTS.values()) - 1.0) < 1e-9 and scores["condition_weights"] == CONDITION_WEIGHTS
    for team, total in scores["final"].items():
        t11 = scores["condition"]["1-1"][team]["total"]
        t13 = scores["condition"]["1-3"][team]["total"]
        assert abs(total - (0.2 * t11 + 0.3 * t13) / 0.5) < 1e-9, (team, total, t11, t13)
    # A condition total is the weighted sum of the item means.
    for c, table in scores["condition"].items():
        for team, items in table.items():
            want = sum(WEIGHTS[i] * items[i] for i in ITEMS)
            assert abs(items["total"] - want) < 1e-9
    assert scores["final"]["pilot"] >= scores["final"]["tutorial"]
    board = leaderboard(scores)
    assert "pilot" in board and "tutorial" in board and "1-3" in board


def test_a_submission_alone_scores_one_on_what_it_completes():
    """No organiser reference figures. A submission scored alone is its own best
    completing submission, so every completed episode is 1.0 on all items and every
    other episode is 0 -- the number it reads is its completion rate."""
    out = _tmp("alone")
    manifest = write_fixture_set(out, n=2, conditions=("1-1",))
    scores = score_set(manifest, out, {"me": ChainPilot})
    raw = scores["raw"]["1-1"]
    done = sum(raw[e]["me"]["outcome"] == "goal" for e in raw)
    assert abs(scores["final"]["me"] - done / len(raw)) < 1e-9
    for ep_id, ratios in scores["per_episode"]["1-1"].items():
        want = 1.0 if raw[ep_id]["me"]["outcome"] == "goal" else 0.0
        assert all(ratios["me"][i] == want for i in ITEMS)


def test_the_manifest_round_trips_as_json_and_names_its_format():
    out = _tmp("fmt")
    manifest = write_fixture_set(out, n=1, conditions=("1-1",))
    text = (out / "manifest.json").read_text()
    assert json.loads(text) == manifest
    assert manifest["format"] == "usvnav-set/1"
