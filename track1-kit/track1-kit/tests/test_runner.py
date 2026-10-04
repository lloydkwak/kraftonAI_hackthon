"""The scoring runner: the agent in a child process, one pair per (participant, condition)
with `reset()` between episodes, each episode's wall-clock limit at the boundary, and a
record per episode.

Each test copies the worked example and edits one line of its `agent.py` into the behaviour
under test -- a sleep at one tick, a raise, an `os._exit`, a look for the hidden course --
because the runner's promises are all about what happens at the boundary when the far side
misbehaves, and the example is the one agent every copy of the kit has.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import time

import numpy as np

from fixtures import scratch, write_easy_set
from usvnav import coursefile, record as _rec, runner, submission as sub
from usvnav.contest import CLEARANCE_PERCENTILE, D_CAP, load_set
from usvnav.sim import _perception, observation, run_episode
from usvnav.plant import Vessel

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "submission"
RESET_LINE = '        self.mode = meta["observation_mode"]'
ACT_LINE = '        dist, bearing = (float(v) for v in obs["wp_polar"])      # bearing > 0 is to port'
INIT_LINE = '        self.mode = None'


def _submission(root, name="herons", *, reset=None, act=None, init=None, top=None):
    """A copy of the example under `name`, with code inserted at the head of `reset`,
    `act` or `__init__`, and/or at the top of the module."""
    d = root / name
    shutil.copytree(EXAMPLE, d)                 # named after its directory
    src = (d / "agent.py").read_text()
    if top:
        src = src.replace("import json\n", "import json\n" + top.rstrip("\n") + "\n", 1)
    for anchor, code in ((RESET_LINE, reset), (ACT_LINE, act), (INIT_LINE, init)):
        if code:
            assert anchor in src, anchor
            src = src.replace(anchor, anchor + "\n" + "\n".join("        " + l for l in code.splitlines()))
    (d / "agent.py").write_text(src)
    return d


def _set(root, n=1, conditions=("1-1",)):
    """Easy hand-built courses (`fixtures.easy_course`): fx-01 has four waypoints, fx-02 three, and
    the example agent completes both -- the promises tested here are about the boundary, not the course."""
    write_easy_set(root / "set", n=n, conditions=conditions)
    return root / "set"


def _record(out, team, cond, ep):
    return json.loads((out / f"{team}-{cond}-{ep}.json").read_text())


# --------------------------------------------------------------------------- the codec

def test_the_codec_round_trips_every_condition_s_observation_and_the_metadata():
    course = coursefile.load(ROOT / "tools" / "demo-course.json")
    vessel = Vessel(*course.start)
    meta = {"boundary": course.boundary, "waypoints": course.waypoints,
            "arrival_radii": course.arrival_radii, "hull": (3.0, 2.0), "observation_mode": "1-2",
            "nested": {"none": None, "flag": True, "n": 7, "x": 1.5, "s": "text", "l": [1, [2, 3]]}}
    back = runner.decode(runner.encode(meta))
    for k in ("boundary", "waypoints", "arrival_radii"):
        assert isinstance(back[k], np.ndarray) and back[k].dtype == meta[k].dtype
        assert np.array_equal(back[k], meta[k])
    assert back["hull"] == [3.0, 2.0] and back["observation_mode"] == "1-2"
    assert back["nested"] == {"none": None, "flag": True, "n": 7, "x": 1.5, "s": "text", "l": [1, [2, 3]]}
    for cond in ("1-1", "1-2", "1-3", "1-4"):
        obs = observation(vessel, course, 17, 1, np.array([0.4, -0.1]))
        obs["perception"] = _perception(cond, vessel, course, 1.7)
        back = runner.decode(runner.encode(obs))
        assert set(back) == set(obs)
        for k in ("pose", "vel", "wp_polar", "prev_action"):
            assert back[k].dtype == obs[k].dtype and np.array_equal(back[k], obs[k]), k
        for k in ("t", "wp_index"):                 # numpy scalars stay numpy scalars
            assert type(back[k]) is type(obs[k]) and back[k] == obs[k], k
        p, q = obs["perception"], back["perception"]
        if isinstance(p, dict):
            assert set(p) == set(q)
            for k in p:
                assert q[k].dtype == p[k].dtype and q[k].shape == p[k].shape and np.array_equal(q[k], p[k]), k
        else:
            assert q.dtype == p.dtype and q.shape == p.shape and np.array_equal(q, p)
        assert back["perception"] is not obs["perception"]
    data = runner.encode({"a": np.zeros(4)})
    for bad in (data[:-1], data + b"x"):
        try:
            runner.decode(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("a truncated or padded message must not decode")


# --------------------------------------------------------------------------- parity

def test_an_isolated_run_gives_the_in_process_numbers_tick_for_tick():
    """The runner wraps the agent, not the simulator. The isolated episode is the in-process
    episode -- same outcome, same items, the same applied action on every tick."""
    root = scratch("usvnav-runner-")
    set_dir = _set(root)
    d = _submission(root)
    out = root / "out"
    pair = runner.run_pair(d, set_dir, "1-1", out)
    assert pair["format"] == runner.PAIR_FORMAT and pair["team"] == "herons"
    assert pair["restarts"] == 0 and pair["agent"]["ok"] and pair["agent"]["entry"] == "agent:Agent"
    manifest, sroot = load_set(set_dir)
    ep = manifest["episodes"][0]
    course = coursefile.load(sroot / ep["course"])
    m, factory = sub.load_agent(d)
    try:
        res = run_episode(course, factory(), condition="1-1", seed=ep["seed"], d_cap=D_CAP,
                          clearance_percentile=CLEARANCE_PERCENTILE, record_trace=True)
    finally:
        sub.unload(d)
    e = pair["episodes"][ep["id"]]
    assert (e["outcome"], e["ticks"], e["waypoints"]) == (res.outcome, res.ticks, res.waypoints_reached)
    assert e["path"] == res.distance and e["fuel"] == res.fuel and e["clearance"] == res.clearance and e["time"] == float(res.ticks)
    assert "overruns" not in e and e["ms_mean"] > 0 and {"ms_p99", "wall_s", "of", "tier"} <= set(e)
    rec = _record(out, "herons", "1-1", ep["id"])
    assert rec["format"] == _rec.RUN_FORMAT and rec["frame_fields"] == list(_rec.FRAME_FIELDS)
    applied = [(f[0], f[8], f[9]) for f in rec["frames"] if f[8] is not None]
    assert applied == [(t, v, w) for t, _, _, _, v, w in res.trace], "the applied actions differ"
    assert "overrun_ticks" not in rec and rec["tick_ms"]["n"] == res.ticks
    assert rec["episode_limit_s"] == runner.EPISODE_TIME_LIMIT_S == 120.0
    assert rec["result"]["outcome"] == res.outcome and rec["course_snapshot"] == json.loads((sroot / ep["course"]).read_text())
    assert (out / "agent.log").is_file() and (out / "pair.json").is_file()
    meta = json.loads((out / f"herons-1-1-{ep['id']}.meta.json").read_text())
    assert meta["status"] == "done" and meta["result"]["outcome"] == res.outcome and meta["id"] == rec["id"]
    # Replay on the scorer's own record: the actions fed back reproduce the run.
    rep = _rec.replay(rec)
    assert _rec.replay_matches(rec, rep) == [] and rep.outcome == res.outcome and rep.ticks == res.ticks
    broken = json.loads(json.dumps(rec))
    broken["frames"][100][8] = 2.0
    assert _rec.replay_matches(broken, _rec.replay(broken)), "a tampered action must not replay"


# --------------------------------------------------------------------------- isolation

def test_the_hidden_course_is_unreachable_from_the_agent_process():
    """Tested rather than asserted: in the runner's own process the `Course`
    is one `gc.get_objects()` away; in the child there is none to find. The agent encodes
    how many it found in its first action."""
    root = scratch("usvnav-runner-")
    set_dir = _set(root)
    d = _submission(root, top="import gc\n",
                    reset="self.found = sum(type(o).__name__ == 'Course' for o in gc.get_objects())",
                    act="if int(obs['t']) == 0:\n    return np.array([0.1 * min(self.found, 5), 0.0])")
    manifest, sroot = load_set(set_dir)
    course = coursefile.load(sroot / manifest["episodes"][0]["course"])
    m, factory = sub.load_agent(d)
    try:
        res = run_episode(course, factory(), condition="1-1", seed=0, record_trace=True, tick_limit=3)
    finally:
        sub.unload(d)
    assert res.trace[0][4] >= 0.1, "in-process, the course is reachable (which is the problem)"
    pair = runner.run_pair(d, set_dir, "1-1", root / "out", tick_limit=3)
    rec = _record(root / "out", "herons", "1-1", "fx-01")
    assert rec["frames"][0][8] == 0.0, "the agent process found a Course object"
    assert pair["episodes"]["fx-01"]["outcome"] == "timeout"


def test_organiser_only_modules_are_refused_and_the_network_is_closed():
    root = scratch("usvnav-runner-")
    set_dir = _set(root)
    d = _submission(root, top="import socket\n",
                    reset="self.flags = 0.0\n"
                          "try:\n    import usvnav.generate\nexcept ImportError:\n    self.flags += 0.1\n"
                          "try:\n    socket.create_connection(('example.invalid', 80), timeout=1)\n"
                          "except OSError:\n    self.flags += 0.2\n"
                          "try:\n    import usvnav.lanegen\nexcept ImportError:\n    self.flags += 0.4",
                    act="if int(obs['t']) == 0:\n    return np.array([self.flags, 0.0])")
    runner.run_pair(d, set_dir, "1-1", root / "out", tick_limit=2)
    rec = _record(root / "out", "herons", "1-1", "fx-01")
    assert abs(rec["frames"][0][8] - 0.7) < 1e-6, rec["frames"][0]


# --------------------------------------------------------------------------- the episode's time limit

def test_a_slow_tick_is_applied_as_returned_and_costs_only_time():
    """There is no per-tick limit: a reply 150 ms late is the action applied on its tick, and the run
    is the one an agent without the sleep drives, action for action."""
    root = scratch("usvnav-runner-")
    set_dir = _set(root)
    runner.run_pair(_submission(root, "quick"), set_dir, "1-1", root / "out-quick")
    d = _submission(root, top="import time\n", act="if int(obs['t']) == 40:\n    time.sleep(0.15)")
    pair = runner.run_pair(d, set_dir, "1-1", root / "out")
    e = pair["episodes"]["fx-01"]
    rec = _record(root / "out", "herons", "1-1", "fx-01")
    quick = _record(root / "out-quick", "quick", "1-1", "fx-01")
    assert e["outcome"] == "goal" and rec["tick_ms"]["max"] >= 150 and pair["restarts"] == 0
    assert [f[8:10] for f in rec["frames"]] == [f[8:10] for f in quick["frames"]]


def test_quick_ticks_add_up_to_the_episode_limit_and_the_process_is_replaced():
    """The limit is the episode's, from reset(): 20 ms a tick is no single slow tick, but 1.5 s
    runs out before the course is done. That episode is `time_overrun`; the process is started
    again and the next episode runs in full."""
    root = scratch("usvnav-runner-")
    set_dir = _set(root, n=2)
    d = _submission(root, top="import time\n", reset="self.slow = len(meta['waypoints']) == 4",
                    act="if self.slow:\n    time.sleep(0.02)")
    pair = runner.run_pair(d, set_dir, "1-1", root / "out", episode_limit_s=1.5)
    e1, e2 = pair["episodes"]["fx-01"], pair["episodes"]["fx-02"]
    assert e1["outcome"] == "time_overrun" and e1["tier"] == "submission_fault"
    assert 20 <= e1["ticks"] <= 75 and e1["wall_s"] >= 1.5, e1
    rec = _record(root / "out", "herons", "1-1", "fx-01")
    assert "the episode's wall-clock limit of 1.5 s ran out" in rec["result"]["detail"]
    assert pair["restarts"] == 1 and e2["outcome"] == "goal"
    assert _record(root / "out", "herons", "1-1", "fx-02")["restarts_before"] == 1


def test_a_reply_that_never_comes_ends_the_episode_at_its_limit_and_the_process_is_replaced():
    root = scratch("usvnav-runner-")
    set_dir = _set(root, n=2)
    d = _submission(root, top="import time\n", reset="self.hang = len(meta['waypoints']) == 4",
                    act="if self.hang and int(obs['t']) == 10:\n    time.sleep(30)")
    t0 = time.perf_counter()
    pair = runner.run_pair(d, set_dir, "1-1", root / "out", episode_limit_s=3.0)
    assert time.perf_counter() - t0 < 15, "a hung agent must not hold the runner for its sleep"
    e1, e2 = pair["episodes"]["fx-01"], pair["episodes"]["fx-02"]
    assert e1["outcome"] == "time_overrun" and e1["ticks"] == 10
    rec = _record(root / "out", "herons", "1-1", "fx-01")
    assert "tick 10: no action" in rec["result"]["detail"]
    assert pair["restarts"] == 1 and e2["outcome"] == "goal"
    rec2 = _record(root / "out", "herons", "1-1", "fx-02")
    assert rec2["restarts_before"] == 1


def test_reset_is_inside_the_episode_limit():
    """No separate limit on reset(meta): it spends the episode's time like any tick."""
    root = scratch("usvnav-runner-")
    set_dir = _set(root, n=2)
    d = _submission(root, top="import time\n",
                    reset="if len(meta['waypoints']) == 4:\n    time.sleep(30)")
    t0 = time.perf_counter()
    pair = runner.run_pair(d, set_dir, "1-1", root / "out", episode_limit_s=2.0)
    assert time.perf_counter() - t0 < 15
    e1, e2 = pair["episodes"]["fx-01"], pair["episodes"]["fx-02"]
    assert e1["outcome"] == "time_overrun" and e1["ticks"] == 0
    assert "reset() did not return" in _record(root / "out", "herons", "1-1", "fx-01")["result"]["detail"]
    assert pair["restarts"] == 1 and e2["outcome"] == "goal"


def test_a_reset_that_raises_is_a_crash_on_that_episode():
    root = scratch("usvnav-runner-")
    set_dir = _set(root, n=2)
    d = _submission(root, reset="if len(meta['waypoints']) == 4:\n    raise KeyError('no such mode')")
    pair = runner.run_pair(d, set_dir, "1-1", root / "out")
    e1, e2 = pair["episodes"]["fx-01"], pair["episodes"]["fx-02"]
    assert e1["outcome"] == "crash" and e1["ticks"] == 0 and e1["fuel"] == 0.0
    assert "no such mode" in _record(root / "out", "herons", "1-1", "fx-01")["result"]["detail"]
    assert pair["restarts"] == 0 and e2["outcome"] == "goal"


# --------------------------------------------------------------------------- faults

def test_a_crash_carries_the_child_s_traceback_and_a_death_restarts_the_process():
    root = scratch("usvnav-runner-")
    set_dir = _set(root, n=2)
    d = _submission(root, "raiser", act="if int(obs['t']) == 30:\n    raise ZeroDivisionError('boom')")
    pair = runner.run_pair(d, set_dir, "1-1", root / "out-a")
    e = pair["episodes"]["fx-01"]
    rec = _record(root / "out-a", "raiser", "1-1", "fx-01")
    assert e["outcome"] == "crash" and e["ticks"] == 30 and "ZeroDivisionError: boom" in rec["result"]["detail"]
    assert "agent.py" in rec["result"]["detail"], "the traceback is the child's, naming the participant's file"
    assert pair["restarts"] == 0 and pair["episodes"]["fx-02"]["outcome"] == "crash"

    d = _submission(root, "exiter", top="import os\n", reset="self.die = len(meta['waypoints']) == 4",
                    act="if self.die and int(obs['t']) == 30:\n    os._exit(3)")
    pair = runner.run_pair(d, set_dir, "1-1", root / "out-b")
    e1, e2 = pair["episodes"]["fx-01"], pair["episodes"]["fx-02"]
    rec = _record(root / "out-b", "exiter", "1-1", "fx-01")
    assert e1["outcome"] == "crash" and e1["ticks"] == 30 and "exited with 3" in rec["result"]["detail"]
    assert pair["restarts"] == 1 and e2["outcome"] in ("goal", "static_collision")


def test_a_malformed_or_non_finite_action_is_invalid_action():
    root = scratch("usvnav-runner-")
    set_dir = _set(root)
    d = _submission(root, "words", act="if int(obs['t']) == 5:\n    return 'full ahead'")
    pair = runner.run_pair(d, set_dir, "1-1", root / "out-a")
    e = pair["episodes"]["fx-01"]
    rec = _record(root / "out-a", "words", "1-1", "fx-01")
    assert e["outcome"] == "invalid_action" and e["ticks"] == 5 and "full ahead" in rec["result"]["detail"]
    d = _submission(root, "nans", act="if int(obs['t']) == 5:\n    return np.array([float('nan'), 0.0])")
    pair = runner.run_pair(d, set_dir, "1-1", root / "out-b")
    assert pair["episodes"]["fx-01"]["outcome"] == "invalid_action"


def test_a_construction_failure_is_recorded_on_every_episode_as_the_agent_s_fault():
    root = scratch("usvnav-runner-")
    set_dir = _set(root, n=2)
    d = _submission(root, init="raise RuntimeError('no weights here')")
    pair = runner.run_pair(d, set_dir, "1-1", root / "out")
    assert pair["agent"] is None
    for ep in ("fx-01", "fx-02"):
        e = pair["episodes"][ep]
        rec = _record(root / "out", "herons", "1-1", ep)
        assert e["outcome"] == "crash" and e["ticks"] == 0 and e["waypoints"] == 0
        assert "no weights here" in rec["result"]["detail"] and len(rec["frames"]) == 1


# --------------------------------------------------------------------------- one process per condition

def test_one_process_serves_every_episode_of_a_condition_with_reset_between():
    root = scratch("usvnav-runner-")
    set_dir = _set(root, n=2)
    d = _submission(root, top="import os\n",
                    init="open(os.path.join(os.path.dirname(config_path), 'calls.txt'), 'a').write('init\\n')",
                    reset="open(os.path.join(os.path.dirname(self.cfg_path), 'calls.txt'), 'a').write('reset\\n')")
    src = (d / "agent.py").read_text().replace(INIT_LINE, INIT_LINE + "\n        self.cfg_path = config_path", 1)
    (d / "agent.py").write_text(src)
    pair = runner.run_pair(d, set_dir, "1-1", root / "out", tick_limit=5)
    calls = (d / "calls.txt").read_text().split()
    assert calls == ["init", "reset", "reset"], calls
    assert pair["restarts"] == 0 and len(pair["episodes"]) == 2


# --------------------------------------------------------------------------- the submission

def test_score_submission_runs_the_pairs_in_parallel_and_the_records_play_back():
    root = scratch("usvnav-runner-")
    set_dir = _set(root, conditions=("1-1", "1-3"))
    d = _submission(root)
    lines = []
    t0 = time.perf_counter()
    out = runner.score_submission(d, set_dir, root / "out", jobs=2, log=lines.append)
    assert out["team"] == "herons" and set(out["raw"]) == {"1-1", "1-3"}
    for cond in ("1-1", "1-3"):
        assert set(out["raw"][cond]) == {"fx-01"} and set(out["raw"][cond]["fx-01"]) == {"herons"}
        assert out["raw"][cond]["fx-01"]["herons"]["outcome"] == "goal"
        assert (root / "out" / cond / "pair.json").is_file() and (root / "out" / cond / "agent.log").is_file()
    assert any("1-1 fx-01" in l for l in lines) and any("1-3 fx-01" in l for l in lines)
    summary = json.loads((root / "out" / "run.json").read_text())
    assert summary["conditions"] == ["1-1", "1-3"] and summary["jobs"] == 2 and "episodes" not in summary["pairs"]["1-1"]

    # The record is the studio's format: copied into a work folder it is listed and observed.
    from usvnav import studio
    wd = studio.Workdir(root / "studio")
    wd.init()
    for p in (root / "out" / "1-1").glob("herons-1-1-fx-01*.json"):
        shutil.copy(p, wd.runs / p.name)
    (run,) = wd.list_runs()
    assert run["id"] == "herons-1-1-fx-01" and run["status"] == "done"
    rec = wd.read_record(run["id"])
    ctype, body = studio.observe(rec, 50, "1-2")
    assert ctype == "image/png" and body[:4] == b"\x89PNG"
    ctype, body = studio.observe(rec, 50)
    assert ctype == "application/json" and json.loads(body)["condition"] == "1-1"


# --------------------------------------------------------------------------- rerun

def test_a_rerun_reproduces_a_deterministic_submission_and_attributes_a_divergence():
    root = scratch("usvnav-runner-")
    set_dir = _set(root)
    d = _submission(root)
    runner.run_pair(d, set_dir, "1-1", root / "out")
    rec = _record(root / "out", "herons", "1-1", "fx-01")
    rep = runner.rerun(rec, d, root / "rerun-a")
    assert rep["attribution"] == "reproduces" and rep["first_divergence"] is None
    assert rep["outcome"] == [rec["result"]["outcome"]] * 2 and rep["actions"][0] == rep["actions"][1]
    assert (root / "rerun-a" / "rerun.json").is_file() and pathlib.Path(rep["rerun_record"]).is_file()

    # A record the episode's time limit stopped: the rerun under the default limit goes on from
    # there, and the divergence is attributed to the stop, not to the agent.
    slow = _submission(root, "sleepy", top="import time\n", act="if int(obs['t']) < 60:\n    time.sleep(0.05)")
    runner.run_pair(slow, set_dir, "1-1", root / "out-slow", episode_limit_s=1.0)
    rec = _record(root / "out-slow", "sleepy", "1-1", "fx-01")
    stop = rec["result"]["ticks"]
    assert rec["result"]["outcome"] == "time_overrun" and 0 < stop < 60
    rep = runner.rerun(rec, slow, root / "rerun-b")
    assert rep["attribution"] == "time_limit" and rep["first_divergence"] == stop, rep["verdict"]
    assert rep["stopped"] == ["the record"] and rep["outcome"][1] == "goal"

    # Nothing stopped and still different: the agent is the cause.
    rnd = _submission(root, "dicey", top="import random\n", act="w_noise = random.random() * 1e-3",
                      )
    src = (rnd / "agent.py").read_text().replace("        w = max(-W_MAX, min(W_MAX, self.gain * bearing))",
                                                 "        w = max(-W_MAX, min(W_MAX, self.gain * bearing + w_noise))")
    (rnd / "agent.py").write_text(src)
    runner.run_pair(rnd, set_dir, "1-1", root / "out-rnd", tick_limit=30)
    rec = _record(root / "out-rnd", "dicey", "1-1", "fx-01")
    rep = runner.rerun(rec, rnd, root / "rerun-c", tick_limit=30)
    assert rep["attribution"] == "agent" and rep["first_divergence"] == 0, rep["verdict"]


# --------------------------------------------------------------------------- the pipe's deadlines and the sandbox prefix

def test_a_send_that_is_never_taken_times_out_instead_of_blocking():
    r, w = os.pipe()
    ch = runner.Channel(r, w)
    big = {"op": "act", "obs": {"raster": np.zeros(300_000, np.uint8)}}     # more than a pipe holds
    t0 = time.perf_counter()
    try:
        ch.send(big, timeout=0.3)
        raise AssertionError("send returned with nobody reading")
    except runner.SendTimeout:
        pass
    assert time.perf_counter() - t0 < 2.0
    ch.close()


def test_an_agent_that_stops_reading_is_a_timeout_not_a_hang():
    root = scratch("usvnav-runner-")
    # The host loop's recv becomes a sleep as soon as the agent is constructed (the host is `python -m
    # usvnav.runner`, so its module is `__main__`): the child sends its hello and never reads again. The runner's next big message must come back as a timeout.
    d = _submission(root, top="import sys, time\n",
                    init="sys.modules['__main__'].Channel.recv = lambda self, timeout=None: time.sleep(600)")
    proc = runner.AgentProcess(d, log_path=root / "agent.log", scratch_dir=root / "scratch")
    try:
        big = {"op": "act", "obs": {"raster": np.zeros(300_000, np.uint8)}, "t": 0}
        t0 = time.perf_counter()
        reply = proc.call(big, timeout=0.5)
        assert reply is None and proc.broken and time.perf_counter() - t0 < 3.0
    finally:
        proc.kill()


def test_child_prefix_fills_the_placeholders_and_refuses_junk():
    env = {runner.CHILD_PREFIX_ENV: json.dumps(["bwrap", "--ro-bind", "{submission}", "{submission}",
                                                 "--bind", "{scratch}", "{scratch}"])}
    got = runner.child_prefix("/tmp/sub", "/tmp/scr", environ=env)
    assert got == ["bwrap", "--ro-bind", "/tmp/sub", "/tmp/sub", "--bind", "/tmp/scr", "/tmp/scr"]
    assert runner.child_prefix("/s", "/t", environ={}) == []
    assert runner.child_prefix("/s", "/t", environ={runner.CHILD_PREFIX_ENV: "  "}) == []
    for junk in ("not json", '{"a": 1}', '["x", 1]'):
        try:
            runner.child_prefix("/s", "/t", environ={runner.CHILD_PREFIX_ENV: junk})
            raise AssertionError(junk)
        except ValueError:
            pass


def test_the_prefix_goes_in_front_of_the_agent_process():
    import os as _os, sys as _sys
    root = scratch("usvnav-runner-")
    # a recording wrapper stands in for bwrap: it notes what it was given and runs the rest
    wrapper = root / "wrap.py"
    wrapper.write_text("import json, os, sys\n"
                       "marks, sub, scr, *cmd = sys.argv[1:]\n"
                       "open(marks, 'a').write(json.dumps({'sub': sub, 'scr': scr, 'cmd': cmd}) + '\\n')\n"
                       "os.execv(cmd[0], cmd)\n")
    marks = root / "argv.jsonl"
    set_dir = _set(root, n=1)
    d = _submission(root)
    _os.environ[runner.CHILD_PREFIX_ENV] = json.dumps([_sys.executable, str(wrapper), str(marks), "{submission}", "{scratch}"])
    try:
        pair = runner.run_pair(d, set_dir, "1-1", root / "out")
    finally:
        del _os.environ[runner.CHILD_PREFIX_ENV]
    assert pair["episodes"]["fx-01"]["outcome"] in ("goal", "static_collision")
    lines = [json.loads(l) for l in marks.read_text().splitlines()]
    assert lines, "the wrapper never ran"
    for m in lines:
        assert m["sub"] == str(d.resolve()) and pathlib.Path(m["scr"]).is_dir()
        assert m["cmd"][0] == _sys.executable and m["cmd"][-1] == str(d.resolve()) and "usvnav.runner" in m["cmd"]
