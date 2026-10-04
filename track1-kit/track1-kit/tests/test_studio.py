"""The studio's server and worker.

There is no browser in the test environment, so the page (`tools/studio.html`) is checked
by hand, and what is tested here is every
call the page makes: the work folder, saving a course with the full rule check, a run as a
child process that records every tick, the observation rebuilt at a tick, a crash reported
as a result rather than as a dead studio, and cancelling. The tests start a real server on
a free port and speak HTTP to it, because that is what the page does.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import struct
import tempfile
import threading
import time
import urllib.error
import urllib.request

from fixtures import easy_course
from usvnav import coursefile, studio

KIT = pathlib.Path(__file__).resolve().parent.parent


class _Studio:
    def __init__(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="usvnav-studio-"))
        self.wd = studio.Workdir(self.tmp)
        self.notes = self.wd.init()
        # `easy`: a hand-built straight reach the tutorial agent completes (fixtures.easy_course); the
        # practice courses follow the difficulty design and the tutorial agent fails them, as RULES.md says.
        coursefile.save(easy_course(4), self.wd.courses / "easy.json")
        self.srv = studio.StudioServer(self.wd, 0)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = self.srv.url.rstrip("/")

    def close(self):
        self.srv.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def call(self, method, path, body=None, raw=False):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
                return r.status, (b if raw else json.loads(b))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def run(self, course, agent, condition, seed, timeout=300):
        status, r = self.call("POST", "/api/run", {"course": course, "agent": agent,
                                                   "condition": condition, "seed": seed})
        assert status == 201, r
        return self.wait(r["id"], timeout)

    def wait(self, run_id, timeout=300):
        t0 = time.time()
        while time.time() - t0 < timeout:
            m = self.call("GET", f"/api/run/{run_id}")[1]
            if m["status"] != "running":
                return m
            time.sleep(0.1)
        raise AssertionError(f"run {run_id} did not finish in {timeout} s")


def test_the_work_folder_is_seeded_and_listed():
    s = _Studio()
    try:
        assert any("agents/example" in n for n in s.notes) and any("practice-01" in n for n in s.notes)
        st = s.call("GET", "/api/state")[1]
        assert {c["name"] for c in st["courses"]} >= {"practice-01", "practice-02", "practice-03",
                                                       "demo-course"}
        (agent,) = st["agents"]
        assert agent["name"] == "example" and agent["ok"] and agent["team"] == "example"
        assert tuple(st["conditions"]) == studio.CONDITIONS and st["hull"] == [3.0, 2.0]
        assert st["runs"] == []
        # a second init on a populated folder adds nothing
        assert studio.Workdir(s.tmp).init() == []
        # the page and the editor are served; anything else is a 404 with a message
        assert s.call("GET", "/", raw=True)[0] == 200
        assert s.call("GET", "/editor.html", raw=True)[0] == 200
        assert s.call("GET", "/api/nothing")[0] == 404
    finally:
        s.close()


def test_a_course_saves_with_the_full_rule_check_and_a_bad_one_is_refused():
    s = _Studio()
    try:
        data = s.call("GET", "/api/course/practice-01")[1]
        status, r = s.call("PUT", "/api/course/mine", data)
        assert status == 200 and r["saved"] == "mine" and r["problems"] == []
        saved = json.loads((s.wd.courses / "mine.json").read_text())
        assert saved == data, "what the editor sends is what is written, field for field"
        # the full check, not the editor's subset: a body on the start pose is `start`
        data["bodies"].append({"class": "moored_vessel", "primitive": "rect",
                               "x": data["start"]["x"], "y": data["start"]["y"],
                               "length_m": 20, "width_m": 6, "heading_rad": 0})
        r = s.call("PUT", "/api/course/mine", data)[1]
        assert any(p["rule"] == "start" and p["severity"] == "error" for p in r["problems"]), r
        r = s.call("POST", "/api/validate", data)[1]
        assert any(p["rule"] == "start" for p in r["problems"])
        # not a course file at all
        status, r = s.call("PUT", "/api/course/bad", {"format": "nope"})
        assert status == 400 and "not a course file" in r["error"]
        assert not (s.wd.courses / "bad.json").exists()
        status, r = s.call("PUT", "/api/course/../escape", data)
        assert status in (400, 404)
        assert not (s.tmp / "escape.json").exists()
        assert s.call("DELETE", "/api/course/mine")[1] == {"deleted": "mine"}
        assert s.call("GET", "/api/course/mine")[0] == 400
    finally:
        s.close()


def test_a_run_is_a_child_process_that_records_every_tick_and_the_end():
    s = _Studio()
    try:
        m = s.run("easy", "example", "1-1", 3)
        assert m["status"] == "done", m
        res = m["result"]
        assert res["outcome"] == "goal" and res["completed"], res
        rec = s.call("GET", f"/api/run/{m['id']}/record")[1]
        assert rec["format"] == studio.RUN_FORMAT and rec["frame_fields"] == list(studio.FRAME_FIELDS)
        frames = rec["frames"]
        # one frame per observation plus the terminal pose, which the trace does not hold
        assert len(frames) == res["ticks"] + 1
        assert [f[0] for f in frames] == list(range(res["ticks"] + 1))
        assert frames[-1][8] is None and frames[-1][9] is None, "no action at the end"
        assert frames[0][8] is not None and all(f[10] is not None for f in frames)
        assert res["clearance_min"] == min(f[10] for f in frames)
        assert frames[res["clearance_min_tick"]][10] == res["clearance_min"]
        # the hull ends inside the goal's arrival disc
        gx, gy = rec["course_snapshot"]["waypoints"][-1]["x"], rec["course_snapshot"]["waypoints"][-1]["y"]
        assert ((frames[-1][1] - gx) ** 2 + (frames[-1][2] - gy) ** 2) ** 0.5 <= 3.0 + 1e-6
        assert len(rec["traffic"]) == len(frames)
        assert rec["course_snapshot"] == json.loads((s.wd.courses / "easy.json").read_text())
        assert rec["agent_team"] == "example" and rec["agent_entry"] == "agent:Agent"
        # the same episode twice is the same run
        m2 = s.run("easy", "example", "1-1", 3)
        rec2 = s.call("GET", f"/api/run/{m2['id']}/record")[1]
        assert rec2["frames"] == frames
        # the list carries summaries only, newest first
        runs = s.call("GET", "/api/state")[1]["runs"]
        assert [r["id"] for r in runs] == [m2["id"], m["id"]]
        assert "frames" not in runs[0] and runs[0]["result"]["outcome"] == "goal"
        # a run's record survives a new server on the same folder
        s.srv.close()
        s.srv = studio.StudioServer(s.wd, 0)
        threading.Thread(target=s.srv.serve_forever, daemon=True).start()
        s.base = s.srv.url.rstrip("/")
        assert s.call("GET", f"/api/run/{m['id']}/record")[1]["frames"] == frames
    finally:
        s.close()


def test_the_observation_at_a_tick_is_rebuilt_from_the_recorded_state():
    s = _Studio()
    try:
        m = s.run("practice-01", "example", "1-1", 3)
        rid = m["id"]
        rec = s.call("GET", f"/api/run/{rid}/record")[1]
        f = rec["frames"][40]
        o = s.call("GET", f"/api/run/{rid}/observe/40")[1]
        assert o["condition"] == "1-1" and o["tick"] == 40
        assert [round(v, 4) for v in o["common"]["pose"]] == [round(v, 4) for v in f[1:4]]
        prev = rec["frames"][39]
        assert [round(v, 4) for v in o["common"]["prev_action"]] == [round(prev[8], 4), round(prev[9], 4)]
        assert o["objects"] and all({"cls", "pos", "extent", "vel"} <= set(row) for row in o["objects"])
        # the same state, rendered as the raster: a 200 x 200 PNG upscaled 3x for the eye
        status, png = s.call("GET", f"/api/run/{rid}/observe/40?view=1-2", raw=True)
        assert status == 200 and png[:8] == b"\x89PNG\r\n\x1a\n"
        w, h = struct.unpack(">II", png[16:24])
        assert (w, h) == (600, 600)
        o3 = s.call("GET", f"/api/run/{rid}/observe/40?view=1-3")[1]
        assert len(o3["ranges"]) == 180 and o3["max_range"] == 50.0
        assert s.call("GET", f"/api/run/{rid}/observe/{len(rec['frames'])}")[0] == 400
        assert s.call("GET", f"/api/run/{rid}/observe/40?view=9-9")[0] == 400
        # the terminal frame has no velocity; it still renders
        assert s.call("GET", f"/api/run/{rid}/observe/{len(rec['frames']) - 1}")[0] == 200
    finally:
        s.close()


def test_a_crash_in_the_agent_is_a_result_with_its_traceback_not_a_dead_studio():
    s = _Studio()
    try:
        status, r = s.call("POST", "/api/agent/team-a", {})
        assert status == 200 and (s.wd.agents / "team-a" / "submission.json").is_file()
        assert "team" not in json.loads((s.wd.agents / "team-a" / "submission.json").read_text())
        assert s.call("POST", "/api/agent/team-a", {})[0] == 400, "exists already"
        src = s.wd.agents / "team-a" / "agent.py"
        src.write_text(src.read_text().replace(
            "def act(self, obs):",
            "def act(self, obs):\n        print('tick', int(obs['t']))\n"
            "        if obs['t'] > 5: raise RuntimeError('kaboom')", 1))
        m = s.run("practice-01", "team-a", "1-3", 1)
        res = m["result"]
        assert res["outcome"] == "crash" and res["tier"] == "submission_fault" and res["ticks"] == 6
        assert "RuntimeError: kaboom" in res["detail"] and "agent.py" in res["detail"]
        assert "tick 6" in m["log"], "what the agent printed is kept"
        rec = s.call("GET", f"/api/run/{m['id']}/record")[1]
        assert len(rec["frames"]) == 7 and rec["frames"][-1][8] is None
        # the code is reloaded by the next run: fix it, run again
        src.write_text(src.read_text().replace("if obs['t'] > 5: raise RuntimeError('kaboom')", "pass"))
        assert s.run("easy", "team-a", "1-1", 1)["result"]["outcome"] == "goal"
        # a module that does not import is a crash with the ImportError, not a failed run
        src.write_text("import nonexistent_module_xyz\n" + src.read_text())
        res = s.run("practice-01", "team-a", "1-1", 1)["result"]
        assert res["outcome"] == "crash" and "nonexistent_module_xyz" in res["detail"]
        # the server is still there
        assert s.call("GET", "/api/state")[0] == 200
    finally:
        s.close()


def test_a_run_can_be_cancelled_and_deleted():
    s = _Studio()
    try:
        # the raster conditions cost 30 ms a tick, so this run would take minutes
        status, r = s.call("POST", "/api/run", {"course": "practice-02", "agent": "example",
                                                "condition": "1-2", "seed": 5})
        rid = r["id"]
        for _ in range(200):
            m = s.call("GET", f"/api/run/{rid}")[1]
            if m.get("tick", 0) > 0 or m["status"] != "running":
                break
            time.sleep(0.1)
        assert m["status"] == "running" and m["tick"] > 0, m
        assert s.call("POST", f"/api/run/{rid}/cancel")[1] == {"cancelled": True}
        m = s.wait(rid)
        assert m["status"] == "cancelled"
        assert s.call("GET", f"/api/run/{rid}/record")[0] == 400
        assert s.call("DELETE", f"/api/run/{rid}")[1] == {"deleted": rid}
        assert not list(s.wd.runs.glob(f"{rid}*"))
        assert s.call("GET", f"/api/run/{rid}")[0] == 400
        # refused before it starts
        assert s.call("POST", "/api/run", {"course": "practice-02", "agent": "example",
                                           "condition": "1-9", "seed": 0})[0] == 400
        assert s.call("POST", "/api/run", {"course": "nope", "agent": "example",
                                           "condition": "1-1", "seed": 0})[0] == 400
        assert s.call("POST", "/api/run", {"course": "practice-02", "agent": "nope",
                                           "condition": "1-1", "seed": 0})[0] == 400
    finally:
        s.close()


def test_the_editor_keeps_the_hooks_the_studio_relies_on():
    """The page drives the editor through `window.usvnavEditor`; if a hook is renamed the
    studio breaks silently in a browser nobody runs in CI, so the contract is pinned here."""
    html = (KIT / "tools" / "editor.html").read_text()
    for hook in ("get overlay()", "set overlay(fn)", "frame,", "draw,", "blank: blankCourse",
                 "toJSON: () => saveObject()", "set course(c)", 'has("embedded")'):
        assert hook in html, hook
    page = (KIT / "tools" / "studio.html").read_text()
    for call in ("/api/state", "/api/course/", "/api/agent/", "/api/run",
                 "/observe/", "/cancel", "editor.html?embedded=1"):
        assert call in page, call
    # what the page is told about a run is what the worker writes
    for field in studio.FRAME_FIELDS:
        assert field in ("tick", "x", "y", "psi", "u", "v", "r", "wp_index", "v_cmd", "w_cmd", "clearance")


def test_noise_is_a_studio_wide_choice_recorded_with_the_run():
    """The work folder gets a `noise.py`; the state lists the kit's names and its functions;
    a run carries its noise, the record keeps it, and *what the agent saw* renders with it."""
    s = _Studio()
    try:
        assert (s.tmp / "noise.py").is_file()
        st = s.call("GET", "/api/state")[1]
        assert st["noises"][:2] == ["none", "grain"] and "noise:my_noise" in st["noises"], st["noises"]
        status, r = s.call("POST", "/api/run", {"course": "practice-01", "agent": "example",
                                                  "condition": "1-2", "seed": 1, "noise": "sparkle"})
        assert status == 400, r
        status, r = s.call("POST", "/api/run", {"course": "practice-01", "agent": "example",
                                                  "condition": "1-2", "seed": 1, "noise": "grain"})
        assert status == 201, r
        m = s.wait(r["id"])
        assert m["noise"] == "grain", m
        rec = s.call("GET", f"/api/run/{m['id']}/record")[1]
        assert rec["noise"] == "grain"
        status, png = s.call("GET", f"/api/run/{m['id']}/observe/5", raw=True)
        assert status == 200 and png[:4] == b"\x89PNG"
    finally:
        s.close()
