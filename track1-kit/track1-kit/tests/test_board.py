"""The persistent leaderboard.

`usvnav score` is tested in `test_contest.py`; what a board adds is the store and the
re-scoring, so these tests drive submissions in one at a time and check that the field is
re-scored from raw items each time, that the ranking follows the tie-break chain, that a
participant's page carries every submission's per-episode scores, the failure causes and nothing
a hidden set must keep, and that a rejected submission leaves the board untouched.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil

from usvnav import board
from usvnav import submission as sub
from fixtures import scratch, write_fixture_set

EXAMPLE = pathlib.Path(__file__).resolve().parent.parent / "examples" / "submission"


def _tmp():
    return scratch("usvnav-board-")


def _team(root, name, **config):
    d = root / name
    shutil.copytree(EXAMPLE, d)                 # a local board names a submission after its directory
    if config:
        c = json.loads((d / "config.json").read_text()); c.update(config)
        (d / "config.json").write_text(json.dumps(c))
    return d


def test_submissions_accumulate_and_the_field_is_rescored_each_time():
    root = _tmp()
    write_fixture_set(root / "set", n=2, conditions=("1-1",))
    board.init(root / "board", root / "set", intake_dir=root / "intake", runs_dir=root / "runs")
    assert (root / "board" / "leaderboard.md").exists()
    assert "no submission yet" in (root / "board" / "leaderboard.md").read_text()

    r1 = board.submit(root / "board", _team(root, "alpha"), validate_first=False)
    assert r1["accepted"] and r1["rank"] == 1 and r1["of"] == 1
    # The submission ran from a read-only intake copy through the child-process runner, and
    # every episode left a record.
    stored = board.load(root / "board")["teams"]["alpha"]["submissions"][0]
    frozen = pathlib.Path(stored["dir"])
    assert frozen == (root / "intake" / "alpha" / "s01").resolve() and stored["source"] == str((root / "alpha").resolve())
    assert not os.access(frozen / "agent.py", os.W_OK) and not os.access(frozen, os.W_OK)
    runs = pathlib.Path(stored["runs"])
    assert runs == (root / "runs" / "alpha" / "s01").resolve()
    assert (runs / "1-1" / "pair.json").is_file() and (runs / "run.json").is_file()
    assert len(list((runs / "1-1").glob("alpha-1-1-fx-*.json"))) == 2 * 2      # record + meta per episode
    assert all({"ms_mean", "wall_s"} <= set(e) and "overruns" not in e for e in stored["raw"]["1-1"].values())
    assert stored["restarts"] == 0
    alone = r1["scores"]["final"]["alpha"]
    # Alone, a submission is its own best on everything it completed: its score is its
    # completion rate, exactly as `usvnav score` reports it.
    raw = board.counting(board.load(root / "board"), "alpha")["raw"]["1-1"]
    done = sum(e["outcome"] == "goal" for e in raw.values())
    assert abs(alone - done / len(raw)) < 1e-9

    r2 = board.submit(root / "board", _team(root, "beta", cruise_fraction=0.5),
                      validate_first=False)
    assert r2["accepted"] and r2["of"] == 2
    # The arrival of beta re-scores alpha: the stored items are the same, the ratios move.
    b = board.load(root / "board")
    assert set(b["teams"]) == {"alpha", "beta"}
    scores = board.rescore(b)
    assert set(scores["final"]) == {"alpha", "beta"}
    for cond, eps in scores["per_episode"].items():
        for ep_id, ratios in eps.items():
            completers = [t for t in ratios if board.counting(b, t)["raw"][cond][ep_id]["outcome"] == "goal"]
            if completers:
                for item in ("fuel", "clearance", "time"):
                    assert max(ratios[t][item] for t in completers) == 1.0
    rows = board.ranking(b, scores)
    assert [r["rank"] for r in rows] == [1, 2]
    assert rows[0]["final"] >= rows[1]["final"]
    text = (root / "board" / "leaderboard.md").read_text()
    assert "| alpha |" in text and "| beta |" in text
    # The head: the title alone, the time of posting, the pointer at the statement, the
    # table, a rule, one footnote line.
    assert text.startswith("# Track 1 리더보드\n\n게시 20") and "+09:00." in text.splitlines()[2]
    assert "평가 기준: 완주와 상대점수" in text.splitlines()[4]
    tail = text.split("\n---\n")[-1].strip()
    assert tail.startswith("세트 `set`") and "참가자 2명" in tail and "×100" in tail and "usvnav-runtime/1" in tail
    for team in ("alpha", "beta"):
        page = (root / "board" / "teams" / f"{team}.md").read_text()
        assert "Per-episode scores" in page and "Failure causes" in page and "late ticks" not in page
        assert "Submission faults" in page and not re.search(r"\b\d-[A-Z]\d+\b", page)   # no internal ids


def test_every_submission_is_stored_and_the_best_one_counts():
    """A participant's best accepted submission counts. Two submissions with different
    gains are both kept; the counting one is whichever scores higher against the field, and
    the page lists both with the counting one marked."""
    root = _tmp()
    write_fixture_set(root / "set", n=2, conditions=("1-1",))
    board.init(root / "board", root / "set", intake_dir=root / "intake", runs_dir=root / "runs")
    board.submit(root / "board", _team(root, "beta", cruise_fraction=0.5), validate_first=False)
    r1 = board.submit(root / "board", _team(root, "alpha"), validate_first=False)
    r2 = board.submit(root / "board", _team(root, "alpha2", cruise_fraction=0.4), validate_first=False)
    # alpha2 is a copy of the example under another name; resubmit it *as* alpha (a directory named so).
    d = root / "again" / "alpha"
    shutil.copytree(root / "alpha2", d)
    r3 = board.submit(root / "board", d, validate_first=False)
    b = board.load(root / "board")
    rec = b["teams"]["alpha"]
    assert [s_["id"] for s_ in rec["submissions"]] == ["s01", "s02"]
    picks = {t: r["counting"] for t, r in b["teams"].items()}
    scores = {s_["id"]: board.submission_score(b, "alpha", s_["id"], picks) for s_ in rec["submissions"]}
    best = max(scores, key=lambda k: (scores[k], -rec["submissions"][int(k[1:]) - 1]["submitted_epoch"]))
    assert rec["counting"] == best == r3["counting"], (scores, rec["counting"])
    assert board.select_counting(b) == picks                    # a fixed point
    page = (root / "board" / "teams" / "alpha.md").read_text()
    assert "## Submissions" in page and "| s01 |" in page and "| s02 |" in page
    assert page.count("**yes**") == 1
    # Every submission's per-episode table, newest first, each headed with its id, the counting
    # one marked; each row scored as that submission would score against the field now.
    heads = [l for l in page.splitlines() if l.startswith("### ")]
    assert heads == [f"### s02{' (counting)' if best == 's02' else ''}",
                     f"### s01{' (counting)' if best == 's01' else ''}"], heads
    for sid in ("s01", "s02"):
        section = page.split(f"### {sid}")[1].split("\n## ")[0].split("\n### ")[0]
        rows = [l for l in section.splitlines() if l.startswith("| 1-1 | fx-")]
        assert len(rows) == 2, section
        own = next(s_ for s_ in rec["submissions"] if s_["id"] == sid)["raw"]["1-1"]
        alt = board.submission_scores(b, "alpha", sid, picks)["per_episode"]["1-1"]
        for row, (ep_id, e) in zip(rows, own.items()):
            cells = [c.strip() for c in row.strip("|").split("|")]
            total = sum(b["weights"][i] * alt[ep_id]["alpha"][i] for i in ("fuel", "clearance", "time"))
            assert cells[1] == ep_id and cells[2] == f"{100 * total:.1f}" and cells[6] == e["outcome"], row
    text = (root / "board" / "leaderboard.md").read_text()
    assert f"{best} of 2" in text


def test_the_ranking_breaks_ties_by_waypoints_then_by_earlier_submission():
    """The tie-break chain, exercised where ties happen: participants that completed
    nothing all score 0."""
    b = {"format": board.BOARD_FORMAT, "set": "x", "set_kind": "public", "conditions": None,
         "weights": {"fuel": 0.3, "clearance": 0.4, "time": 0.3}, "intake": "x", "runs": "x", "teams": {}}

    def rec(stamp, epoch, waypoints):
        return {"counting": "s01", "submissions": [
            {"id": "s01", "submitted_at": stamp, "submitted_epoch": epoch, "dir": "", "validation": None,
             "raw": {"1-1": {"ep-01": {"outcome": "static_collision", "tier": "navigational",
                                       "path": 10.0, "fuel": 10.0, "clearance": 1.0, "time": 100.0,
                                       "ticks": 100, "waypoints": waypoints, "of": 3,
                                       "wall_s": 0.1}}}}]}
    b["teams"]["late_far"] = rec("t3", 3.0, 2)
    b["teams"]["early_near"] = rec("t1", 1.0, 1)
    b["teams"]["mid_near"] = rec("t2", 2.0, 1)
    rows = board.ranking(b, board.rescore(b))
    assert [r["team"] for r in rows] == ["late_far", "early_near", "mid_near"]
    assert all(r["final"] == 0.0 for r in rows)


def test_a_rejected_submission_is_not_run_and_not_stored():
    root = _tmp()
    write_fixture_set(root / "set", n=1, conditions=("1-1",))
    board.init(root / "board", root / "set", intake_dir=root / "intake", runs_dir=root / "runs")
    d = _team(root, "alpha")
    m = json.loads((d / sub.MANIFEST).read_text()); m["runtime"] = "somebody-elses/9"
    (d / sub.MANIFEST).write_text(json.dumps(m))
    r = board.submit(root / "board", d, validate_first=True,
                     validate_kwargs=dict(conditions=("1-1",), ticks=20))
    assert not r["accepted"] and r["scores"] is None
    assert board.load(root / "board")["teams"] == {}


def test_a_hidden_style_board_shows_opaque_episode_ids_and_no_manifest_extras():
    """A participant's page shows per-episode scores by id, so a hidden set's ids are
    opaque (`ep-NN`). Whatever else the hidden manifest carries -- extra fields, the seed --
    must reach neither the participant's page nor the board's store."""
    root = _tmp()
    from usvnav.contest import write_set
    from fixtures import courses
    eps = [(f"ep-{i + 1:02d}", c, 13300 + i, ("1-1",), {"extra": {"secret": 3.3, "notes": ["x"]}})
           for i, c in enumerate(courses()[:2])]
    write_set(root / "set", eps, kind="hidden")
    board.init(root / "board", root / "set", intake_dir=root / "intake", runs_dir=root / "runs")
    board.submit(root / "board", _team(root, "alpha"), validate_first=False)
    page = (root / "board" / "teams" / "alpha.md").read_text()
    assert "ep-01" in page and "ep-02" in page
    for leak in ("13300", "13301", "secret", "notes", "extra"):
        assert leak not in page, leak
    assert "secret" not in (root / "board" / "board.json").read_text()


def test_a_board_refuses_a_set_that_changed_under_it():
    """A board ranks one set. The digest taken at init is checked at every submit, so an
    edited course file -- or a swapped seed -- stops the board rather than scoring two
    participants on different episodes."""
    root = _tmp()
    write_fixture_set(root / "set", n=1, conditions=("1-1",))
    board.init(root / "board", root / "set", intake_dir=root / "intake", runs_dir=root / "runs")
    b = board.load(root / "board")
    assert len(b["set_digest"]) == 64 and b["set_frozen"] is False
    m = json.loads((root / "set" / "manifest.json").read_text())
    m["episodes"][0]["seed"] += 1
    (root / "set" / "manifest.json").write_text(json.dumps(m))
    try:
        board.submit(root / "board", _team(root, "alpha"), validate_first=False)
    except board.BoardError as exc:
        assert "has changed since the board was created" in str(exc)
    else:
        raise AssertionError("a changed set must be refused")
    assert board.load(root / "board")["teams"] == {}


def test_intake_leaves_compiled_files_and_caches_behind():
    root = _tmp()
    src = _team(root, "alpha")
    (src / "__pycache__").mkdir(exist_ok=True)
    (src / "__pycache__" / "agent.cpython-313.pyc").write_bytes(b"\x00")
    (src / "agent.pyc").write_bytes(b"\x00")
    (src / "helper.pyo").write_bytes(b"\x00")
    dst = board.intake(src, root / "intake" / "alpha" / "s01")
    names = {p.relative_to(dst).as_posix() for p in dst.rglob("*")}
    assert "agent.py" in names
    assert not any(n.endswith((".pyc", ".pyo")) or "__pycache__" in n for n in names), names
