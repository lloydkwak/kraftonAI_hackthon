"""The scoring runner: the agent in a child process, one process per condition, each
episode's wall-clock limit applied at the boundary, and a record per episode.

**Why a process.** Called in the runner's own interpreter, a submission would be one
`gc.get_objects()` away from the hidden course, the traffic's future and the disturbance.
Here the agent lives in a child interpreter that receives the episode metadata and the
observations and returns actions, over a pipe pair on the same machine; the `Course`
object never exists on its side, so the hidden state is unreachable rather than merely
forbidden. In every run the child also gets a stripped environment, a scratch `HOME`,
socket calls that fail, and a refusal to import the organiser-only modules -- guards against
accidents, not containment. On the scoring machine the child additionally runs in a private
filesystem with no network (`CHILD_PREFIX_ENV`).

**One process per condition**: `run_pair` starts one agent process, constructs
`Agent(config_path)` once, and drives that condition's episodes in sequence with `reset()`
between them. A process that dies, or that an episode's time limit stopped, is started again
for the next episode and the restart is recorded. Records are named as the studio names
its runs (`<team>-<condition>-<episode>.json` with a `.meta.json` beside it), so copying a
scored episode into a studio's `runs/` lists it there and plays it back.

**The boundary is where the episode is timed.** `RemoteAgent.reset` starts the episode's
clock (`sim.EPISODE_TIME_LIMIT_S`, from the `reset(meta)` call to the end of the episode -- the
agent's time and the simulator's together), and every message after it is sent and its reply
awaited within what is left of it. The runner is synchronous and applies every reply, however
slow, so *which* actions were applied never depends on the pipe's scheduling; there is no
per-tick limit. When the episode's time runs out -- while the runner waits for a reply, or
before it sends the next observation -- the episode ends as `time_overrun` and the process is
replaced for the next one. The wall clock of every `act`, send to reply with the pipe
included, is kept as information (`tick_ms`).

**The simulator runs in the runner's process** through `sim.run_episode`, exactly as
`usvnav run`, `usvnav score` and the studio run it: the proxy is an agent to the loop, so the
numbers do not have a second code path. `score_submission` runs the conditions of one
submission as processes of their own, `jobs` at a time; the board calls it.

**What is written** (under the board's `runs` directory): one `usvnav-run/2` record per
episode (`usvnav.record`, the studio's format, so a scored episode plays back in a studio),
with `tick_ms` and `episode_limit_s` added; the agent process's stdout and stderr in
`agent.log`; and `pair.json` with the raw items of every episode, which is what the board
stores.

**Nothing crosses the pipe as a pickle.** Observations are dicts of numpy arrays, so the
codec is a JSON tree with the arrays as raw buffers behind it; the parent decodes only
JSON and `np.frombuffer`, and a hostile child can at worst send a malformed action.
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import select
import struct
import subprocess
import sys
import time
import traceback

import numpy as np

from . import coursefile
from . import record as _rec
from . import submission as _sub
from .contest import CLEARANCE_PERCENTILE, CONDITIONS, D_CAP, load_set
from .plant import InvalidAction
from .sim import EPISODE_TIME_LIMIT_S, TimeOverrun, run_episode

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAIR_FORMAT = "usvnav-pair/1"

#: `Agent(config_path)` must construct within `CONSTRUCT_TIMEOUT_S` (a model may be loading onto
#: the GPU); otherwise every remaining episode of that condition is `time_overrun`. Construction is
#: outside the episodes' limit (`sim.EPISODE_TIME_LIMIT_S`); `reset(meta)` is inside it.
CONSTRUCT_TIMEOUT_S = 120.0

#: Condition processes run at once by `score_submission` (`usvnav leaderboard submit --jobs`).
#: The scoring machine sets its own.
DEFAULT_JOBS = 2

#: The scoring machine's sandbox for the agent's process: a JSON list
#: of argv tokens that goes in front of the child's interpreter -- bubblewrap with a private mount
#: namespace in which only the interpreter, this package, the submission (read-only) and the
#: child's scratch directory exist, and no network. The scoring machine sets it; the tokens `{submission}`
#: and `{scratch}` are filled in here. Unset (a participant's machine, the studio): no wrapper.
CHILD_PREFIX_ENV = "USVNAV_CHILD_PREFIX"


def child_prefix(submission_dir, scratch_dir, environ=None) -> list[str]:
    """The tokens to put before the child's interpreter, placeholders filled. Unset or empty:
    none. Malformed: `ValueError` -- an organiser's broken sandbox setting stops the run
    rather than quietly running the agent unwrapped."""
    raw = (os.environ if environ is None else environ).get(CHILD_PREFIX_ENV, "")
    if not raw.strip():
        return []
    try:
        tokens = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{CHILD_PREFIX_ENV} is not valid JSON ({exc})") from None
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        raise ValueError(f"{CHILD_PREFIX_ENV} must be a JSON list of strings (argv tokens)")
    fill = {"{submission}": str(pathlib.Path(submission_dir).resolve()),
            "{scratch}": str(pathlib.Path(scratch_dir).resolve())}
    return [fill.get(t, t) for t in tokens]


#: Modules the child refuses to import: a submission cannot rely on organiser-only code
#: that the participant kit does not contain. Courtesy, not containment.
_REFUSED = ("usvnav.internal", "usvnav.generate", "usvnav.scenario", "usvnav.channel",
            "usvnav.lanegen")


def data_home() -> pathlib.Path | None:
    """`$USVNAV_HOME`, where a board keeps its `intake/` and `runs/` (the scoring machine sets
    it); `None` when unset, and a board keeps them inside its own directory."""
    env = os.environ.get("USVNAV_HOME")
    return pathlib.Path(env) if env else None


# --------------------------------------------------------------------------- codec

_ND = "$nd"


def encode(obj) -> bytes:
    """A JSON tree with numpy arrays (and numpy scalars) lifted out as raw buffers.
    Tuples become lists; dict keys must be strings."""
    bufs: list[bytes] = []

    def conv(x):
        if isinstance(x, (np.ndarray, np.generic)):
            a = np.asarray(x)                   # a numpy scalar stays 0-d: `ascontiguousarray`
            if a.ndim and not a.flags.c_contiguous:   # would make it shape (1,), and `int(obs["t"])`
                a = np.ascontiguousarray(a)     # in the agent would then raise
            bufs.append(a.tobytes())
            return {_ND: len(bufs) - 1, "dtype": a.dtype.str, "shape": list(a.shape), "n": a.nbytes}
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if x is None or isinstance(x, (str, bool, int, float)):
            return x
        raise TypeError(f"cannot send a {type(x).__name__} across the boundary")

    head = json.dumps(conv(obj), separators=(",", ":")).encode()
    return b"".join([struct.pack(">II", len(head), len(bufs)), head, *bufs])


def decode(data: bytes):
    hl, nb = struct.unpack_from(">II", data, 0)
    head = json.loads(data[8:8 + hl].decode())
    descs: list[dict] = []

    def collect(x):
        if isinstance(x, dict):
            if _ND in x:
                descs.append(x)
            else:
                for v in x.values():
                    collect(v)
        elif isinstance(x, list):
            for v in x:
                collect(v)

    collect(head)
    if len(descs) != nb or sorted(d[_ND] for d in descs) != list(range(nb)):
        raise ValueError("malformed message: array table")
    offsets = {}
    off = 8 + hl
    for d in sorted(descs, key=lambda d: d[_ND]):
        offsets[d[_ND]] = off
        off += int(d["n"])
    if off != len(data):
        raise ValueError("malformed message: length")

    def build(x):
        if isinstance(x, dict):
            if _ND in x:
                dt = np.dtype(x["dtype"])
                a = np.frombuffer(data, dtype=dt, count=int(x["n"]) // max(dt.itemsize, 1),
                                  offset=offsets[x[_ND]]).reshape(x["shape"]).copy()
                return a[()] if a.ndim == 0 else a
            return {k: build(v) for k, v in x.items()}
        if isinstance(x, list):
            return [build(v) for v in x]
        return x

    return build(head)


class SendTimeout(Exception):
    """The other side stopped taking messages: `Channel.send` could not hand one over in time."""


class Channel:
    """Length-prefixed messages over a read fd and a write fd. The write end is non-blocking, so a
    send has a deadline like a receive: a peer that stops reading fills the pipe (64 KiB here; one
    1-2/1-4 observation is about 120 KB), and a blocking write would then wait forever."""

    def __init__(self, rfd: int, wfd: int):
        self.rfd, self.wfd = rfd, wfd
        self._buf = bytearray()
        os.set_blocking(self.wfd, False)

    def send(self, obj, timeout: float | None = None) -> None:
        """Write the message whole; `SendTimeout` if it is not taken within `timeout` seconds."""
        payload = encode(obj)
        data = struct.pack(">I", len(payload)) + payload
        view = memoryview(data)
        deadline = None if timeout is None else time.perf_counter() + timeout
        while view:
            left = None
            if deadline is not None:
                left = deadline - time.perf_counter()
                if left <= 0:
                    raise SendTimeout(f"{len(view)} of {len(data)} bytes not taken within {timeout:.3f} s")
            _, w, _ = select.select([], [self.wfd], [], left)
            if not w:
                raise SendTimeout(f"{len(view)} of {len(data)} bytes not taken within {timeout:.3f} s")
            try:
                n = os.write(self.wfd, view)
            except BlockingIOError:
                continue
            view = view[n:]

    def recv(self, timeout: float | None = None):
        """The next message; `None` if `timeout` seconds pass first (the partial message
        stays buffered for the next call); `EOFError` when the other side is gone."""
        deadline = None if timeout is None else time.perf_counter() + timeout
        while True:
            if len(self._buf) >= 4:
                n = struct.unpack_from(">I", self._buf, 0)[0]
                if len(self._buf) >= 4 + n:
                    payload = bytes(self._buf[4:4 + n])
                    del self._buf[:4 + n]
                    return decode(payload)
            if deadline is not None:
                left = deadline - time.perf_counter()
                if left <= 0:
                    return None
                r, _, _ = select.select([self.rfd], [], [], left)
                if not r:
                    return None
            chunk = os.read(self.rfd, 1 << 20)
            if not chunk:
                raise EOFError
            self._buf += chunk

    def close(self) -> None:
        for fd in (self.rfd, self.wfd):
            try:
                os.close(fd)
            except OSError:
                pass


# --------------------------------------------------------------------------- the child

def _refuse_internal_imports() -> None:
    import importlib.abc

    class Refuse(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name in _REFUSED or any(name.startswith(p + ".") for p in _REFUSED):
                raise ImportError(f"{name} is organiser-only and not available to a submission")
            return None

    sys.meta_path.insert(0, Refuse())


def _host_main(rfd: int, wfd: int, root: str) -> int:
    """The agent's process: construct once, then answer `reset` and `act` until told to quit.
    Everything the agent raises is reported, never propagated; only a process death ends it."""
    ch = Channel(rfd, wfd)
    attempts: list = []
    _sub._disable_network(attempts)
    _refuse_internal_imports()
    try:
        t0 = time.perf_counter()
        manifest, factory = _sub.load_agent(root)
        agent = factory()
        ch.send({"ok": True, "team": manifest["team"], "entry": manifest["entry"],
                 "construct_ms": 1000.0 * (time.perf_counter() - t0), "pid": os.getpid()})
    except BaseException:  # noqa: BLE001 - reported to the runner, which records `crash`
        ch.send({"ok": False, "exception": traceback.format_exc()})
        return 1
    while True:
        try:
            msg = ch.recv()
        except EOFError:
            return 0
        op = msg.get("op") if isinstance(msg, dict) else None
        if op == "quit":
            return 0
        if op == "reset":
            try:
                agent.reset(msg["meta"])
                ch.send({"ok": True})
            except BaseException:  # noqa: BLE001
                ch.send({"ok": False, "exception": traceback.format_exc()})
        elif op == "act":
            try:
                action = agent.act(msg["obs"])
            except BaseException:  # noqa: BLE001
                ch.send({"ok": False, "exception": traceback.format_exc()})
                continue
            try:
                arr = np.asarray(action, dtype=float)
            except Exception:  # noqa: BLE001 - not even array-like: a malformed action
                ch.send({"ok": True, "malformed": repr(action)[:200]})
                continue
            ch.send({"ok": True, "action": arr})
        else:
            ch.send({"ok": False, "exception": f"unknown op {op!r}"})


# --------------------------------------------------------------------------- the parent

class AgentProcessDied(Exception):
    """The child exited or closed the pipe; `run_episode` records the episode as `crash`."""


class AgentRaised(Exception):
    """The agent raised in the child; the message is the child's traceback."""


class AgentProcess:
    """One agent process: started with `Agent(config_path)` already constructed, or raising."""

    def __init__(self, submission_dir, *, log_path, scratch_dir,
                 construct_timeout_s: float = CONSTRUCT_TIMEOUT_S):
        self.dir = pathlib.Path(submission_dir).resolve()
        self.broken = False
        self.log_path = pathlib.Path(log_path)
        r_in, w_in = os.pipe()          # runner -> agent
        r_out, w_out = os.pipe()        # agent -> runner
        scratch = pathlib.Path(scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               "HOME": str(scratch), "TMPDIR": str(scratch),
               "PYTHONPATH": os.pathsep.join(
                   [str(ROOT)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]),
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
        for k, v in os.environ.items():
            if k.startswith(("CUDA_", "NVIDIA_", "LC_", "LD_LIBRARY_PATH", "LANG")):
                env.setdefault(k, v)
        flags = (["-S"] if sys.flags.no_site else []) + (["-P"] if sys.flags.safe_path else [])
        cmd = [*child_prefix(self.dir, scratch),
               sys.executable, *flags, "-m", "usvnav.runner", "--host", str(r_in), str(w_out), str(self.dir)]
        self._log = open(self.log_path, "a")
        self.proc = subprocess.Popen(cmd, cwd=self.dir, env=env, stdin=subprocess.DEVNULL,
                                     stdout=self._log, stderr=subprocess.STDOUT,
                                     pass_fds=(r_in, w_out), close_fds=True)
        os.close(r_in)
        os.close(w_out)
        self.ch = Channel(r_out, w_in)
        try:
            hello = self.ch.recv(timeout=construct_timeout_s)
        except EOFError:
            self._reap()
            why = self._died("during construction")
            self.kill()
            raise AgentProcessDied(why)
        if hello is None:
            self.kill()
            raise TimeOverrun(f"Agent(config_path) did not construct within {construct_timeout_s:.0f} s")
        if not hello.get("ok"):
            self.kill()
            raise AgentRaised(hello.get("exception", ""))
        self.hello = hello

    # -- messages

    def call(self, msg: dict, timeout: float | None):
        """Send, then wait for the reply, both within one `timeout`; `None` when it passes (the
        process is then out of step and marked broken) -- whether the reply never came or the
        message was never taken; `AgentProcessDied` if it is gone."""
        t0 = time.perf_counter()
        try:
            self.ch.send(msg, timeout=timeout)
            left = None if timeout is None else max(0.0, timeout - (time.perf_counter() - t0))
            reply = self.ch.recv(timeout=left)
        except SendTimeout:
            reply = None
        except (BrokenPipeError, EOFError, OSError):
            self.broken = True
            self._reap()
            raise AgentProcessDied(self._died(f"on {msg.get('op')}")) from None
        if reply is None:
            self.broken = True
        return reply

    # -- lifecycle

    def alive(self) -> bool:
        return self.proc.poll() is None and not self.broken

    def close(self, grace_s: float = 2.0) -> None:
        if self.proc.poll() is None and not self.broken:
            try:
                self.ch.send({"op": "quit"}, timeout=grace_s)
            except (OSError, SendTimeout):
                pass
            try:
                self.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                pass
        self.kill()

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self.ch.close()
        self._log.close()

    def _reap(self):
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)

    def _died(self, when: str) -> str:
        self._log.flush()
        try:
            tail = self.log_path.read_text(errors="replace")[-2000:]
        except OSError:
            tail = ""
        code = self.proc.returncode
        how = f"exited with {code}" if code is not None and code >= 0 else f"was killed by signal {-code}" if code is not None else "is gone"
        return f"the agent process {how} {when}\n--- agent.log (tail) ---\n{tail}"


class RemoteAgent:
    """What `sim.run_episode` drives: an agent whose `reset` and `act` cross the boundary.

    `reset` starts the episode's wall clock; every call after it waits at most for what is left
    of `episode_limit_s`. Keeps the per-tick wall clock (`tick_ms`) for the record; raises
    `TimeOverrun` when the episode's time runs out (the process is then marked for replacement),
    `InvalidAction` for a malformed reply, `AgentRaised` or `AgentProcessDied` for a crash."""

    def __init__(self, proc: AgentProcess, *, episode_limit_s: float = EPISODE_TIME_LIMIT_S):
        self.proc = proc
        self.limit_s = float(episode_limit_s)
        self.deadline = time.perf_counter() + self.limit_s
        self.tick_ms: list[float] = []

    def _overrun(self, what: str) -> TimeOverrun:
        self.proc.broken = True         # stopped mid-episode: a fresh process for the next one
        return TimeOverrun(f"{what}: the episode's wall-clock limit of {self.limit_s:g} s ran out "
                           f"(counted from reset(meta), the agent's time and the scorer's together)")

    def reset(self, meta):
        self.tick_ms = []
        self.deadline = time.perf_counter() + self.limit_s
        reply = self.proc.call({"op": "reset", "meta": meta}, timeout=self.limit_s)
        if reply is None:
            raise self._overrun("reset() did not return")
        if not reply.get("ok"):
            raise AgentRaised(reply.get("exception", ""))

    def act(self, obs):
        tick = int(obs["t"])
        left = self.deadline - time.perf_counter()
        if left <= 0:
            raise self._overrun(f"tick {tick}")
        t0 = time.perf_counter()
        reply = self.proc.call({"op": "act", "obs": obs}, timeout=left)
        self.tick_ms.append(1000.0 * (time.perf_counter() - t0))
        if reply is None:
            raise self._overrun(f"tick {tick}: no action")
        if not reply.get("ok"):
            raise AgentRaised(reply.get("exception", ""))
        if "malformed" in reply:
            raise InvalidAction(f"act() returned {reply['malformed']} (two finite numbers are required)")
        action = reply.get("action")
        if not isinstance(action, np.ndarray):
            raise InvalidAction(f"act() reply carried no action ({type(action).__name__})")
        return action


# --------------------------------------------------------------------------- a pair

def _ms_summary(ms: list[float]) -> dict:
    if not ms:
        return {"n": 0, "mean": 0.0, "max": 0.0, "p99": 0.0}
    a = np.sort(np.asarray(ms, dtype=float))
    return {"n": int(a.size), "mean": round(float(a.mean()), 3), "max": round(float(a[-1]), 3),
            "p99": round(float(a[min(a.size - 1, int(0.99 * a.size))]), 3)}


def _raw_entry(res, course, wall_s: float, remote: RemoteAgent | None) -> dict:
    """`run_field`'s per-episode items, plus what the boundary measured."""
    ms = _ms_summary(remote.tick_ms if remote else [])
    return {"outcome": res.outcome, "tier": res.tier, "path": float(res.distance), "fuel": float(res.fuel),
            "clearance": float(res.clearance), "time": float(res.ticks), "ticks": res.ticks,
            "waypoints": res.waypoints_reached, "of": course.n_waypoints,
            "wall_s": round(wall_s, 2), "ms_mean": ms["mean"], "ms_p99": ms["p99"]}


def _enable_acceleration(log=None) -> None:
    """The organiser-side simulator kernels (`usvnav.accel`, held back from the bundle), under
    `USVNAV_ACCEL` (unset: auto -- enabled when they import, match the pinned source and pass their
    self-check; `0`: never; `1`: required). Imported by name so that the shipped runner does not depend
    on them: in a participant's copy the module is absent and the Python simulator runs, as always."""
    if (os.environ.get("USVNAV_ACCEL") or "auto").strip().lower() in ("0", "off", "no", "false", "python"):
        if log:
            log("simulator: python (USVNAV_ACCEL off)")
        return
    try:
        accel = importlib.import_module("usvnav.accel")
    except ImportError:
        return                                      # the bundle: nothing to enable
    accel.enable_for_scoring(log=log)


def simulator_status() -> dict:
    """Which simulator this process runs, for the records: `{"backend": "python"|"native", "kernels"}`."""
    from . import render as _render
    if _render._native_render is None:
        return {"backend": "python", "kernels": None}
    accel = sys.modules.get("usvnav.accel")
    return {"backend": "native", "kernels": getattr(accel, "__version__", None)}


def run_pair(submission_dir, set_path, condition: str, out_dir, *, episodes=None, name=None,
             episode_limit_s: float = EPISODE_TIME_LIMIT_S,
             construct_timeout_s: float = CONSTRUCT_TIMEOUT_S, tick_limit=None, log=None) -> dict:
    """One (submission, condition): one agent process, the condition's episodes in sequence with
    `reset()` between them. Writes `<out_dir>/<episode>.json` records, `agent.log`
    and `pair.json`; returns the pair summary (`pair.json`'s content).

    `episodes` restricts the set's episode ids. `tick_limit` is for tests. Every record and the
    pair carry `simulator` -- which implementation drew the observations (`simulator_status`)."""
    manifest, root = load_set(set_path)
    simulator = simulator_status()
    sub_dir = pathlib.Path(submission_dir).resolve()
    sub_manifest = _sub.load_manifest(sub_dir)
    team = name or sub_manifest["team"]          # the board's name for it (an intake copy's directory is its id)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "scratch"
    log_path = out / "agent.log"
    params = dict(episode_limit_s=episode_limit_s, construct_timeout_s=construct_timeout_s)
    started = _rec.now()

    proc: AgentProcess | None = None
    failure: tuple[str, str] | None = None        # (outcome, detail) when construction failed
    restarts = 0
    hello = None

    def start():
        nonlocal proc, failure, hello
        try:
            proc = AgentProcess(sub_dir, log_path=log_path, scratch_dir=scratch,
                                construct_timeout_s=construct_timeout_s)
            hello = hello or proc.hello
            failure = None
        except AgentRaised as exc:
            failure = ("crash", f"Agent(config_path) raised:\n{exc}")
        except AgentProcessDied as exc:
            failure = ("crash", str(exc))
        except TimeOverrun as exc:
            failure = ("time_overrun", str(exc))

    start()
    entries: dict[str, dict] = {}
    for ep in manifest["episodes"]:
        if condition not in ep["conditions"] or (episodes and ep["id"] not in episodes):
            continue
        snapshot = json.loads((root / ep["course"]).read_text())
        course = coursefile.from_dict(snapshot)
        record = {"id": f"{team}-{condition}-{ep['id']}", "course": ep["id"], "agent": team,
                  "condition": condition, "seed": int(ep["seed"]), "started": _rec.now(),
                  "course_snapshot": snapshot, "n_waypoints": course.n_waypoints,
                  "agent_team": team, "agent_entry": sub_manifest["entry"],
                  "set": str(root / "manifest.json"), "episode": ep["id"], "episode_limit_s": episode_limit_s,
                  "simulator": simulator, **_rec.header()}
        if proc is not None and not proc.alive():
            proc.close()
            proc = None
            restarts += 1
            start()
        if proc is None and failure is None:
            restarts += 1
            start()
        remote = None
        t0 = time.perf_counter()
        if failure is not None:
            outcome, detail = failure
            res = None
            record.update(_rec.failed_record(course, outcome, detail))
            entry = {"outcome": outcome, "tier": "submission_fault", "path": 0.0, "fuel": 0.0, "clearance": 0.0,
                     "time": 0.0, "ticks": 0, "waypoints": 0, "of": course.n_waypoints,
                     "wall_s": 0.0, "ms_mean": 0.0, "ms_p99": 0.0}
            if outcome == "crash":
                failure = None          # try a fresh process for the next episode
        else:
            remote = RemoteAgent(proc, episode_limit_s=episode_limit_s)
            recorder = _rec.Recorder()
            kw = {} if tick_limit is None else {"tick_limit": int(tick_limit)}
            res = run_episode(course, remote, condition=condition, seed=int(ep["seed"]),
                              d_cap=D_CAP, clearance_percentile=CLEARANCE_PERCENTILE,
                              record_trace=True, on_observation=recorder.on_observation, **kw)
            wall = time.perf_counter() - t0
            record.update(_rec.episode_record(course, res, recorder, wall))
            record["tick_ms"] = _ms_summary(remote.tick_ms)
            entry = _raw_entry(res, course, wall, remote)
        record["restarts_before"] = restarts
        _write_json(out / f"{record['id']}.json", record)
        _write_json(out / f"{record['id']}.meta.json",
                    {k: record[k] for k in ("id", "course", "agent", "condition", "seed", "started")}
                    | {"status": "done", "result": record["result"], "finished": record["finished"],
                       "episode": ep["id"]})
        entries[ep["id"]] = entry
        if log:
            log(f"{condition} {ep['id']:<28} {team:<18} {entry['outcome']:<18} "
                f"{entry['ticks']:>5} ticks  {entry['waypoints']}/{entry['of']}  {entry['wall_s']:.1f} s")
    if proc is not None:
        proc.close()
    pair = {"format": PAIR_FORMAT, "team": team, "condition": condition, "submission": str(sub_dir),
            "set": str(root / "manifest.json"), "started": started, "finished": _rec.now(),
            "params": params, "agent": hello, "restarts": restarts, "episodes": entries,
            "agent_log": str(log_path), "simulator": simulator}
    _write_json(out / "pair.json", pair)
    return pair


def _write_json(path: pathlib.Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- a submission

def score_submission(submission_dir, set_path, out_dir, *, conditions=None, jobs: int = DEFAULT_JOBS,
                     log=None, name=None, **params) -> dict:
    """Every (condition) pair of one submission, each in a runner process of its own
    (`python -m usvnav.runner --pair`), `jobs` at a time. Returns `{"team", "raw", "pairs"}`
    with `raw` in `contest.run_field`'s shape -- `{condition: {episode: {team: entry}}}` --
    so the board stores it as it stored `run_field`'s items."""
    manifest, root = load_set(set_path)
    sub_dir = pathlib.Path(submission_dir).resolve()
    team = name or _sub.load_manifest(sub_dir)["team"]
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    present = {c for ep in manifest["episodes"] for c in ep["conditions"]}
    conds = [c for c in CONDITIONS if c in present and (not conditions or c in conditions)]
    if not conds:
        raise ValueError(f"the set has no episodes under {conditions}")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    flags = (["-S"] if sys.flags.no_site else []) + (["-P"] if sys.flags.safe_path else [])
    pending = list(conds)
    live: dict[str, tuple[subprocess.Popen, object]] = {}
    pairs: dict[str, dict] = {}

    def launch(cond):
        spec = {"submission": str(sub_dir), "set": str(root / "manifest.json"), "condition": cond,
                "out": str(out / cond), "name": team, "params": params}
        spec_path = out / f"{cond}.spec.json"
        _write_json(spec_path, spec)
        plog = open(out / f"{cond}.log", "w")
        p = subprocess.Popen([sys.executable, *flags, "-m", "usvnav.runner", "--pair", str(spec_path)],
                             cwd=str(ROOT), env=env, stdout=plog, stderr=subprocess.STDOUT)
        live[cond] = (p, plog)

    def collect(cond):
        p, plog = live.pop(cond)
        plog.close()
        pair_path = out / cond / "pair.json"
        if p.returncode != 0 or not pair_path.is_file():
            tail = (out / f"{cond}.log").read_text(errors="replace")[-2000:]
            raise RuntimeError(f"the {cond} pair process exited with {p.returncode} and wrote "
                               f"no pair.json:\n{tail}")
        pairs[cond] = json.loads(pair_path.read_text())
        if log:
            for line in (out / f"{cond}.log").read_text(errors="replace").splitlines():
                log(line)

    while pending or live:
        while pending and len(live) < max(1, int(jobs)):
            launch(pending.pop(0))
        done = [c for c, (p, _) in live.items() if p.poll() is not None]
        if not done:
            time.sleep(0.05)
            continue
        for c in done:
            collect(c)

    raw = {c: {ep_id: {team: entry} for ep_id, entry in pairs[c]["episodes"].items()} for c in conds}
    summary = {"format": "usvnav-submission-run/1", "team": team, "submission": str(sub_dir),
               "set": str(root / "manifest.json"), "conditions": conds, "jobs": int(jobs),
               "params": params, "pairs": {c: {k: v for k, v in pairs[c].items() if k != "episodes"}
                                           for c in conds}}
    _write_json(out / "run.json", summary)
    return {"team": team, "raw": raw, "pairs": pairs}


# --------------------------------------------------------------------------- entry points

def _pair_main(spec_path: str) -> int:
    spec = json.loads(pathlib.Path(spec_path).read_text())
    _enable_acceleration(log=print)
    run_pair(spec["submission"], spec["set"], spec["condition"], spec["out"],
             name=spec.get("name"), log=print, **spec.get("params", {}))
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 4 and argv[0] == "--host":
        return _host_main(int(argv[1]), int(argv[2]), argv[3])
    if len(argv) == 2 and argv[0] == "--pair":
        return _pair_main(argv[1])
    raise SystemExit("internal: python -m usvnav.runner --host RFD WFD SUBMISSION_DIR | "
                     "--pair SPEC.json (use `usvnav leaderboard submit`)")


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------- rerun

def rerun(record: dict, submission_dir, out_dir, **params) -> dict:
    """Run the *submission* again on the record's episode -- same course, seed and condition,
    through the isolating runner -- and compare the applied actions tick for tick. Every reply is
    applied, so a deterministic agent reproduces its actions whatever the machine's load; a
    divergence is attributed to the episode's time limit when one side was stopped by it exactly
    where the actions part (the actions before agree), and otherwise to the agent itself (state
    across episodes or unseeded randomness). `params` reach `run_pair` (`episode_limit_s` and the rest)."""
    out = pathlib.Path(out_dir)
    set_dir = out / "set"
    (set_dir / "courses").mkdir(parents=True, exist_ok=True)
    ep_id = record.get("episode") or record["course"]
    (set_dir / "courses" / f"{ep_id}.json").write_text(json.dumps(record["course_snapshot"]) + "\n")
    manifest = {"format": "usvnav-set/1", "kind": "rerun",
                "episodes": [{"id": ep_id, "course": f"courses/{ep_id}.json",
                              "seed": int(record["seed"]), "conditions": [record["condition"]]}]}
    (set_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    pair = run_pair(submission_dir, set_dir, record["condition"], out / "run", **params)
    (new_path,) = (out / "run").glob(f"*-{record['condition']}-{ep_id}.json")
    new = json.loads(new_path.read_text())

    def applied(rec):
        return [(f[0], f[8], f[9]) for f in rec["frames"] if f[8] is not None]

    a, b = applied(record), applied(new)
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    stopped = [side for side, rec, n in (("the record", record, len(a)), ("the rerun", new, len(b)))
               if first is not None and first == n and rec["result"]["outcome"] == "time_overrun"]
    if first is None:
        attribution, verdict = "reproduces", (
            f"the submission reproduces the record: {len(a)} actions identical, outcome "
            f"{new['result']['outcome']} at tick {new['result']['ticks']} both times")
    elif stopped:
        attribution, verdict = "time_limit", (
            f"diverges at tick {first}: {' and '.join(stopped)} was stopped there by the episode's "
            f"wall-clock limit (time_overrun); the {first} actions before it are identical, so the "
            f"agent reproduces its run up to the stop")
    else:
        attribution, verdict = "agent", (
            f"diverges at tick {first}: the agent does not reproduce its own run (state across "
            f"episodes, unseeded randomness, or an environment that differs)")
    report = {"format": "usvnav-rerun/1", "record": record["id"], "submission": str(pathlib.Path(submission_dir).resolve()),
              "outcome": [record["result"]["outcome"], new["result"]["outcome"]],
              "ticks": [record["result"]["ticks"], new["result"]["ticks"]],
              "actions": [len(a), len(b)], "first_divergence": first,
              "stopped": stopped, "attribution": attribution, "verdict": verdict, "rerun_record": str(new_path),
              "restarts": pair["restarts"]}
    _write_json(out / "rerun.json", report)
    return report
