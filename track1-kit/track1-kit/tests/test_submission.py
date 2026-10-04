"""The packaging rules and the local validator.

Every requirement in docs/PACKAGING.md is mechanically checkable, so each test here breaks
one requirement in a copy of the worked example and checks that the validator names it. The
example itself has to pass, which is what makes it a worked example rather than a template.
"""

from __future__ import annotations

import json
import pathlib
import shutil

from usvnav import submission as sub
from fixtures import scratch

EXAMPLE = pathlib.Path(__file__).resolve().parent.parent / "examples" / "submission"
FAST = dict(conditions=("1-1", "1-3"), ticks=60)      # the raster conditions cost 30 ms/tick


def _copy(name="herons"):
    d = scratch("usvnav-sub-") / name          # a local board names a submission after its directory
    shutil.copytree(EXAMPLE, d)
    return d


def _errors(report):
    return {p["check"] for p in report["problems"] if p["severity"] == "error"}


def test_the_worked_example_is_accepted_under_every_condition():
    report = sub.validate(EXAMPLE, ticks=40)
    assert report["ok"], sub.render(report)
    assert set(report["runs"]) == set(sub.CONDITIONS)
    checks = {p["check"] for p in report["problems"] if p["severity"] == "ok"}
    for want in ("manifest", "entry", "config", "runtime", "size", "import",
                 "construct", "act", "determinism", "tick_ms", "network"):
        assert want in checks, want
    # Nothing beyond the manifest, the code and its config is required: no SUBMISSION.md.
    assert "doc" not in {c[0] for c in sub.CHECKS} and not (EXAMPLE / "SUBMISSION.md").exists()
    # Nor a name: a submission is identified by the submission ID it is uploaded with.
    assert "team" not in {c[0] for c in sub.CHECKS} and "team" not in sub.REQUIRED
    assert "team" not in json.loads((EXAMPLE / sub.MANIFEST).read_text())


def test_an_old_manifest_with_a_team_is_accepted_and_the_name_ignored():
    d = _copy("egrets")
    m = json.loads((d / sub.MANIFEST).read_text())
    m["team"] = "has space"                     # any old value: neither checked nor used
    (d / sub.MANIFEST).write_text(json.dumps(m))
    problems, manifest = sub.check_static(d)
    assert not [p for p in problems if p["severity"] == "error"], problems
    assert manifest["team"] == "egrets"


def test_every_reported_check_is_a_documented_one():
    """`docs/PACKAGING.md` is generated from `CHECKS`; a check the validator can report but
    the document does not list would be a rule nobody was told about."""
    documented = {c[0] for c in sub.CHECKS}
    d = _copy()
    (d / "config.json").unlink()
    static, _ = sub.check_static(d)
    full = sub.validate(EXAMPLE, **FAST)
    seen = {p["check"] for p in static} | {p["check"] for p in full["problems"]}
    assert seen <= documented, seen - documented


def test_static_checks_name_each_broken_requirement():
    d = _copy()
    m = json.loads((d / sub.MANIFEST).read_text())
    m["config"] = "missing.json"
    m["runtime"] = "somebody-elses/9"
    (d / sub.MANIFEST).write_text(json.dumps(m))
    problems, manifest = sub.check_static(d)
    errs = {p["check"] for p in problems if p["severity"] == "error"}
    assert errs == {"config", "runtime"}, errs
    # And a rejected static pass never starts the child process.
    report = sub.validate(d, **FAST)
    assert not report["ok"] and report["runs"] == {}


def test_a_missing_manifest_and_a_wrong_format_are_errors():
    d = _copy()
    (d / sub.MANIFEST).unlink()
    problems, m = sub.check_static(d)
    assert m is None and {p["check"] for p in problems} == {"manifest"}
    d = _copy()
    (d / sub.MANIFEST).write_text(json.dumps({"format": "usvnav-submission/0",
                                              "entry": "agent:Agent", "config": "config.json",
                                              "runtime": sub.RUNTIME_ID}))
    problems, m = sub.check_static(d)
    assert m is None and "format" in problems[0]["message"]


def test_the_entry_point_must_live_inside_the_directory():
    d = _copy()
    m = json.loads((d / sub.MANIFEST).read_text())
    m["entry"] = "usvnav.agent:TutorialAgent"               # the organisers' module
    (d / sub.MANIFEST).write_text(json.dumps(m))
    problems, _ = sub.check_static(d)
    assert {p["check"] for p in problems if p["severity"] == "error"} == {"entry"}


def test_the_size_cap_is_applied():
    d = _copy()
    (d / "weights.bin").write_bytes(b"\0" * (sub.SIZE_CAP_BYTES + 1))
    problems, _ = sub.check_static(d)
    assert {p["check"] for p in problems if p["severity"] == "error"} == {"size"}


def test_a_non_finite_action_fails_as_invalid_action():
    d = _copy()
    src = (d / "agent.py").read_text()
    src = src.replace("return np.array([v, w], dtype=np.float32)",
                      "return np.array([float('nan'), w], dtype=np.float32)")
    (d / "agent.py").write_text(src)
    report = sub.validate(d, **FAST)
    assert not report["ok"]
    assert "act" in _errors(report)
    assert all(r["outcome"] == "invalid_action" for r in report["runs"].values()), report["runs"]


def test_state_carried_across_episodes_is_caught():
    """An agent whose behaviour depends on how many episodes it has seen gives a
    different action sequence the second time the same episode is driven."""
    d = _copy()
    src = (d / "agent.py").read_text()
    src = src.replace("        self.mode = meta[\"observation_mode\"]",
                      "        self.mode = meta[\"observation_mode\"]\n"
                      "        self.episodes = getattr(self, 'episodes', 0) + 1")
    src = src.replace("        w = max(-W_MAX, min(W_MAX, self.gain * bearing))",
                      "        w = max(-W_MAX, min(W_MAX, self.gain * bearing * self.episodes))")
    (d / "agent.py").write_text(src)
    report = sub.validate(d, **FAST)
    assert "determinism" in _errors(report), sub.render(report)


def test_a_network_attempt_is_reported_and_the_run_survives_it():
    d = _copy()
    src = (d / "agent.py").read_text()
    src = src.replace("        self.mode = meta[\"observation_mode\"]",
                      "        self.mode = meta[\"observation_mode\"]\n"
                      "        import socket\n"
                      "        try:\n"
                      "            socket.create_connection(('example.invalid', 80), timeout=1)\n"
                      "        except OSError:\n"
                      "            pass")
    (d / "agent.py").write_text(src)
    report = sub.validate(d, **FAST)
    assert "network" in _errors(report), sub.render(report)


def test_a_crash_in_the_agent_is_a_report_line_not_a_validator_death():
    d = _copy()
    src = (d / "agent.py").read_text()
    src = src.replace("        self.mode = meta[\"observation_mode\"]",
                      "        import os\n        os._exit(3)")
    (d / "agent.py").write_text(src)
    report = sub.validate(d, **FAST)
    assert not report["ok"] and "crash" in _errors(report), sub.render(report)


def test_load_agent_and_unload_let_two_submissions_share_one_interpreter():
    a, b = _copy("alpha"), _copy("beta")
    cfg_b = json.loads((b / "config.json").read_text()); cfg_b["steer_gain"] = 0.7
    (b / "config.json").write_text(json.dumps(cfg_b))
    ma, fa = sub.load_agent(a)
    agent_a = fa()
    sub.unload(a)
    mb, fb = sub.load_agent(b)
    agent_b = fb()
    sub.unload(b)
    assert (ma["team"], mb["team"]) == ("alpha", "beta")
    assert agent_a.gain != agent_b.gain, "the second load reused the first module"
