"""The studio: the participant's one test surface for Track 2, as a page (modelled on Track 1's
`usvnav studio`, whose map editor it embeds unchanged).

    usvsim studio [DIR] [--port N] [--no-open]

**What it is for.** You design episodes -- a scene, a guide's path, a start, a distance -- drive
your own controller through them in the plant of your choice, and need to *look* at what came out:
the map with the guide, the ego and the station stretch on the guide's track, the 128 px frame the
fixed learner will see, the per-decision score q_t, the score. The loop is: draw the scene, pick a
controller and a plant, press Run, scrub the run, save it as an episode. The
same page opens any dataset directory -- the kit's real-driving library among them -- and pages
through its frames, and replays a 2-1 simulator against the released logs with the measured
trajectory drawn over it.

**What it is.** One HTML page (`tools/studio.html`) served by one stdlib HTTP server on the
loopback interface. The page embeds Track 1's course editor (`tools/editor.html`) and reads its
document as a 2-2 episode (`usvsim.guide.course`): bodies and the water's edge as they are, the
first traffic loop as the guide's path, the start as the ego's start. A run is a **child process**
running `closedloop.run_episode` -- the same loop the organisers score with -- so an edit to the
controller's code is picked up by the next run without a restart, and a crash or an infinite loop
in your controller cannot take the studio down with it.

**What it is not.** Not an evaluation the organisers chose for you: the scenes, the guide's
motion, the controller and the plant are yours; the camera is the clean one. It writes nothing
outside the work folder.

The work folder:

    scenes/*.json            course files (`usvnav-course/1`), the editor's documents
    controllers/<name>/      a directory with `policy.py` defining `make_policy()`
    camera.py                your camera models: functions `name(frame, ctx) -> frame` (one example seeded)
    plants/<name>/           2-1 submission directories (or a bare .py), selectable as the plant
    runs/<id>.json           one record per run: the scene as run, the result, every decision, the trail
    runs/<id>.npz            the run's camera frames
    runs/<id>.meta.json      the record's summary, for the list
    runs/<id>.log            what the controller process printed
    datasets/<name>/         `t2-2-dataset/1` directories the studio appends runs to

An empty `scenes/` is seeded with a demo scene, an empty `controllers/` with a placeholder
controller that holds still (not a driving example), and a missing `camera.py` with the
template and one deliberately simple example: edit them.

**The camera selector.** The clean picture is what the kit renders and what scoring renders before the
reference camera adds its own appearance (not in the kit). Your model of that appearance is a
function in `camera.py`; the selector in the Run panel applies the chosen one to every frame of a run --
what the map shows, what the frame panel shows, what Save writes -- through `closedloop.Env(camera=...)`.
`none` is the clean picture. The record and the saved episode's manifest entry name the camera used.

The API the page uses, all JSON unless said otherwise:

    GET    /api/state                          scenes, controllers, plants, cameras, runs, datasets, constants
    GET    /api/scene/<name>                   the course file
    PUT    /api/scene/<name>                   save; body is the course; returns the episode check
    DELETE /api/scene/<name>
    POST   /api/check                          the episode check without saving
    POST   /api/controller/<name>              new controller directory, copied from the placeholder
    POST   /api/run                            {scene, controller, plant, camera, d_target, horizon, seed} -> {id}
    GET    /api/run/<id>                       status: running (with the decision), done, failed, cancelled
    GET    /api/run/<id>/record                the full record
    GET    /api/run/<id>/frame/<k>.png         the camera frame at decision k (?scale=3)
    GET    /api/run/<id>/window/<k>.png        the four-frame window the learner sees at decision k
    POST   /api/run/<id>/save                  {dataset} -> append the run to datasets/<dataset>
    POST   /api/run/<id>/cancel
    DELETE /api/run/<id>
    GET    /api/dataset/<name>                 manifest and episode list (`released_frames` is the kit's)
    GET    /api/dataset/<name>/<i>/<k>.png     frame k of episode i (?scale=3)
    GET    /api/dataset/<name>/<i>/<k>         the record's other fields at k
    POST   /api/replay                         {plant, replays: [names]} -> trajectories and errors (2-1)
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGE = ROOT / 'tools' / 'studio.html'
RUN_FORMAT = 'usvsim-run/1'
PROGRESS_EVERY = 2
NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')
BUDGET = 18000
PLACEHOLDER_POLICY = '''"""A controller for the studio: `make_policy()` returns the object that drives the ego.

This one holds still. It is a placeholder for yours, not an example of driving.

Two interfaces, your choice:

  privileged = True    act(tick, state, actuator) -> [T_stern, delta, T_bow]
                       `state` is the true [x, y, heading, surge, sway, yaw_rate]; `bind(env)` is called
                       first, and through `env` you may read the guide: env.guide_pose() -> (x, y, heading),
                       env.distance(), env.d (the current d_target), env.path (its whole course).
  privileged = False   act(obs) -> [T_stern, delta, T_bow]
                       `obs` is what the learner sees: rgb (4, 128, 128, 3) uint8, ego_motion (4, 3),
                       actuator_state (4, 3), history_mask (4,), d_target (4,) -- the last four decisions.

The command: stern thruster magnitude in [-1, 1], its azimuth in [-pi/2, pi/2] (positive turns the
thrust to port), bow thruster in [-1, 1]. One decision every 0.5 s.
"""
import numpy as np


class Policy:
    privileged = True

    def bind(self, env):
        self.env = env

    def act(self, tick, state, actuator):
        return np.zeros(3)


def make_policy():
    return Policy()
'''


CAMERA_TEMPLATE = '''"""Your camera models -- where what the reference camera does to a picture gets built.

The kit's camera is clean: the exact top view in the studio's colours. The reference camera that
renders the hidden evaluation frames adds its own appearance to that picture, and what it adds is
not in the kit: `data/released_frames/` shows it, on 12 real-driving episodes. Anything you learn
from looking at those frames, you implement here.

A camera is a function

    def my_camera(frame, ctx):
        ...
        return frame                      # uint8, the same (128, 128, 3) shape

called on every frame as it is rendered, before the frame goes into the run's record and into the
dataset you save. Pick it in the studio's *camera* selector (the Run panel) and every run you start
renders with it; from code, `closedloop.Env(plant, episode, camera=my_camera)` or
`camera="camera.py:my_camera"` does the same. `none` is the clean picture.

`ctx` carries what a stored frame does not have:

    ctx.t            seconds since the episode began (0.5 per decision); ctx.k the decision index
    ctx.pose         the ego's (x, y, heading) in the world -- the camera's own pose
    ctx.seed         the episode's seed. Key your randomness on (ctx.seed, ctx.k) and the same
                     episode renders the same twice, which is what makes a dataset reproducible.
    ctx.class_map    (128, 128) int, the class of every pixel of the clean picture; ctx.names[id]
    ctx.water        (128, 128) bool, the water -- an effect that belongs on the water goes here
    ctx.world_xy     (x, y) arrays (128, 128): where every pixel is in the world, so a pattern
                     can stay put on the water while the camera moves over it

Below is one deliberately simple example -- sensor grain and a slowly wandering brightness -- so
the shape of the thing is clear. It is not a model of the reference camera. Look at the released
frames and write yours; several functions may live here, the selector lists them all.
"""
import numpy as np


def speckle(frame, ctx):
    """Additive sensor grain, a little stronger on dark pixels, and a brightness that wanders slowly
    over the episode. Deterministic in (seed, decision)."""
    rng = np.random.default_rng([ctx.seed, ctx.k])
    f = frame.astype(np.float32)
    gain = 1.0 + 0.08 * np.sin(0.15 * ctx.t + 0.7 * ctx.seed)          # slow, per-episode phase
    sigma = 4.0 + 6.0 * (1.0 - f.mean(axis=2, keepdims=True) / 255.0)  # darker -> grainier
    out = f * gain + rng.normal(0.0, 1.0, f.shape) * sigma
    return np.clip(out, 0, 255).astype(np.uint8)
'''


class StudioError(Exception):
    """A request the studio refuses, with a message for the page."""


def _name(s):
    if not isinstance(s, str) or not NAME_RE.match(s) or s in ('.', '..'):
        raise StudioError(f"{s!r} is not a usable name: letters, digits, '.', '_' and '-' only")
    return s


def _now():
    return _dt.datetime.now().isoformat(timespec='seconds')


def _write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj) + '\n')
    os.replace(tmp, path)


def editor_page():
    """Track 1's editor: the kit's copy in `tools/`, else the one in the Track 1 checkout the renderer
    was found in."""
    from .guide.raster import USVNAV
    for p in (ROOT / 'tools' / 'editor.html', pathlib.Path(USVNAV) / 'tools' / 'editor.html'):
        if p.is_file():
            return p
    raise StudioError("Track 1's editor.html is missing (tools/editor.html)")


def demo_scene():
    """A straight reach with a few bodies and a guide loop, so the first thing you see runs."""
    y0, hw = 75.0, 30.0
    xs = [0, 100, 200, 300, 400]
    boundary = [[x, y0 + hw] for x in xs] + [[x, y0 - hw] for x in reversed(xs)]
    return {
        'format': 'usvnav-course/1',
        'world': {'boundary': boundary},
        'start': {'x': 60.0, 'y': y0, 'heading_rad': 0.0},
        'waypoints': [],
        'bodies': [
            {'class': 'buoy', 'primitive': 'circle', 'x': 130.0, 'y': y0 + 18, 'diameter_m': 0.8},
            {'class': 'buoy', 'primitive': 'circle', 'x': 230.0, 'y': y0 - 20, 'diameter_m': 0.8},
            {'class': 'moored_vessel', 'primitive': 'rect', 'x': 180.0, 'y': y0 - 22, 'length_m': 12.0,
             'width_m': 3.5, 'heading_rad': 0.1},
            {'class': 'pier', 'primitive': 'rect', 'x': 300.0, 'y': y0 + 26, 'length_m': 20.0,
             'width_m': 5.0, 'heading_rad': 0.0},
        ],
        'traffic': [{'class': 'moving_vessel', 'length_m': 7.0, 'width_m': 2.6, 'speed_mps': 1.2,
                     'phase_m': 0.0, 'route': [[72.0, y0], [340.0, y0], [340.0, y0 + 12], [72.0, y0 + 12]]}],
    }


# --------------------------------------------------------------------------- the work folder

class Workdir:
    def __init__(self, path):
        self.path = pathlib.Path(path).resolve()
        self.scenes = self.path / 'scenes'
        self.controllers = self.path / 'controllers'
        self.plants = self.path / 'plants'
        self.runs = self.path / 'runs'
        self.datasets = self.path / 'datasets'
        self.camera_file = self.path / 'camera.py'

    def init(self):
        notes = []
        for d in (self.scenes, self.controllers, self.plants, self.runs, self.datasets):
            d.mkdir(parents=True, exist_ok=True)
        if not any(self.scenes.glob('*.json')):
            (self.scenes / 'demo.json').write_text(json.dumps(demo_scene(), indent=1) + '\n')
            notes.append('scenes/demo.json (a reach, a few bodies, a guide loop)')
        if not any(p.is_dir() for p in self.controllers.iterdir()):
            (self.controllers / 'example').mkdir()
            (self.controllers / 'example' / 'policy.py').write_text(PLACEHOLDER_POLICY)
            notes.append('controllers/example/policy.py (holds still; edit it)')
        if not self.camera_file.is_file():
            self.camera_file.write_text(CAMERA_TEMPLATE)
            notes.append('camera.py (your camera models; one simple example)')
        return notes

    # cameras
    def list_cameras(self):
        """'none', then every module-level function of `camera.py` that takes two or more arguments --
        read from the source, not imported: the file is yours and may not import cleanly here.
        A file that does not parse is reported as such rather than hidden."""
        out = [{'name': 'none', 'ok': True, 'note': 'the clean picture'}]
        if not self.camera_file.is_file():
            return out
        import ast
        try:
            tree = ast.parse(self.camera_file.read_text(), filename=str(self.camera_file))
        except SyntaxError as exc:
            out.append({'name': f"camera.py line {exc.lineno}", 'ok': False, 'note': f"does not parse: {exc.msg}"})
            return out
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                nargs = len(node.args.args) + len(node.args.posonlyargs)
                if nargs >= 2 or node.args.vararg is not None:
                    doc = ast.get_docstring(node) or ''
                    out.append({'name': node.name, 'ok': True, 'note': doc.split('\n')[0][:120]})
        return out

    def camera_spec(self, name):
        """The spec `closedloop.resolve_camera` loads for a selector name: None for 'none', else
        `<work folder>/camera.py:<function>`."""
        if name in (None, '', 'none'):
            return None
        names = {c['name'] for c in self.list_cameras() if c['ok']}
        if name not in names:
            raise StudioError(f"no camera named {name!r} in {self.camera_file.name}: one of {sorted(names)}")
        return f"{self.camera_file}:{name}"

    # scenes
    def scene_file(self, name):
        return self.scenes / f"{_name(name)}.json"

    def list_scenes(self):
        out = []
        for p in sorted(self.scenes.glob('*.json')):
            st = p.stat()
            out.append({'name': p.stem, 'modified': _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds'),
                        'bytes': st.st_size})
        return out

    def read_scene(self, name):
        p = self.scene_file(name)
        if not p.is_file():
            raise StudioError(f"no scene named {name!r}")
        return json.loads(p.read_text())

    def save_scene(self, name, data, d_target):
        from .guide import course
        if not isinstance(data, dict):
            raise StudioError("the scene must be a JSON object")
        problems = course.check(data, d_target)
        self.scene_file(name).write_text(json.dumps(data, indent=1) + '\n')
        return problems

    def delete_scene(self, name):
        p = self.scene_file(name)
        if not p.is_file():
            raise StudioError(f"no scene named {name!r}")
        p.unlink()

    # controllers
    def controller_dir(self, name):
        return self.controllers / _name(name)

    def list_controllers(self):
        out = []
        for d in sorted(p for p in self.controllers.iterdir() if p.is_dir()):
            ok = (d / 'policy.py').is_file()
            out.append({'name': d.name, 'path': str(d), 'ok': ok,
                        'problem': None if ok else 'no policy.py'})
        return out

    def new_controller(self, name, source=None):
        dest = self.controller_dir(name)
        if dest.exists():
            raise StudioError(f"controllers/{name} exists already")
        if source:
            src = self.controller_dir(source)
            if not (src / 'policy.py').is_file():
                raise StudioError(f"controllers/{source} has no policy.py")
            shutil.copytree(src, dest, ignore=shutil.ignore_patterns('__pycache__'))
        else:
            dest.mkdir()
            (dest / 'policy.py').write_text(PLACEHOLDER_POLICY)
        return dest

    # plants
    def list_plants(self):
        out = [{'name': 'public', 'path': None, 'ok': True, 'note': 'the shipped plant (usvsim.plant)'}]
        from . import submission
        for p in sorted(self.plants.iterdir()):
            if p.name.startswith('.'):
                continue
            row = {'name': p.name, 'path': str(p)}
            try:
                entry, man = submission.resolve(str(p))
                row.update(ok=True, note=(f"entry {man['entry']}" if man else 'bare module'))
            except Exception as exc:
                row.update(ok=False, note=str(exc))
            out.append(row)
        return out

    def plant_spec(self, name):
        if name == 'public':
            return 'public'
        p = self.plants / _name(name)
        if not p.exists():
            raise StudioError(f"no plant named {name!r} under plants/")
        return str(p)

    # runs
    def run_paths(self, run_id):
        _name(run_id)
        base = self.runs / run_id
        return {'record': base.with_suffix('.json'), 'frames': base.with_suffix('.npz'),
                'meta': self.runs / f"{run_id}.meta.json", 'log': base.with_suffix('.log'),
                'spec': self.runs / f"{run_id}.spec.json", 'progress': self.runs / f"{run_id}.progress"}

    def list_runs(self):
        out = []
        for p in self.runs.glob('*.meta.json'):
            try:
                out.append(json.loads(p.read_text()))
            except (OSError, ValueError):
                continue
        out.sort(key=lambda m: (m.get('started', ''), m.get('started_epoch', 0.0)), reverse=True)
        return out

    def read_meta(self, run_id):
        p = self.run_paths(run_id)['meta']
        if not p.is_file():
            raise StudioError(f"no run {run_id!r}")
        return json.loads(p.read_text())

    def read_record(self, run_id):
        p = self.run_paths(run_id)['record']
        if not p.is_file():
            meta = self.read_meta(run_id)
            raise StudioError(f"run {run_id} has no record ({meta.get('status')})")
        return json.loads(p.read_text())

    def read_frames(self, run_id):
        p = self.run_paths(run_id)['frames']
        if not p.is_file():
            raise StudioError(f"run {run_id} has no frames")
        return np.load(p)['rgb']

    # datasets
    def dataset_dir(self, name):
        if name == 'released_frames':
            from . import data
            return pathlib.Path(data.DATA) / 'released_frames'
        return self.datasets / _name(name)

    def list_datasets(self):
        from .guide import dataset as F
        out = []
        dirs = [p for p in sorted(self.datasets.iterdir()) if p.is_dir()]
        rel = self.dataset_dir('released_frames')
        if (rel / 'manifest.json').is_file():
            dirs.insert(0, rel)
        for d in dirs:
            row = {'name': 'released_frames' if d == rel else d.name, 'path': str(d), 'kit': d == rel}
            try:
                man = F.read_manifest(str(d))
                row.update(ok=True, episodes=len(man.get('episodes', [])), records=int(man.get('n_records', 0)))
            except Exception as exc:
                row.update(ok=False, problem=str(exc))
            out.append(row)
        return out


# --------------------------------------------------------------------------- runs

class Runner:
    """Starts run workers, watches the live ones, and writes each run's summary at the end."""

    def __init__(self, wd):
        self.wd = wd
        self.live = {}
        self.lock = threading.Lock()

    def start(self, scene, controller, plant, d_target, horizon, seed, camera='none'):
        from .guide import course, path as P
        try:
            seed, horizon, d_target = int(seed), int(horizon), float(d_target)
        except (TypeError, ValueError):
            raise StudioError("seed and horizon must be integers, d_target a number") from None
        if not P.D_MIN <= d_target <= P.D_MAX:
            raise StudioError(f"d_target must be in [{P.D_MIN}, {P.D_MAX}] m")
        if not 1 <= horizon <= 600:
            raise StudioError("horizon must be between 1 and 600 decisions")
        data = self.wd.read_scene(scene)
        problems = course.check(data, d_target)
        errors = [p['message'] for p in problems if p['severity'] == 'error']
        if errors:
            raise StudioError("the scene cannot run: " + '; '.join(errors))
        cdir = self.wd.controller_dir(controller)
        if not (cdir / 'policy.py').is_file():
            raise StudioError(f"controllers/{controller} has no policy.py")
        plant_spec = self.wd.plant_spec(plant)
        camera = camera or 'none'
        camera_spec = self.wd.camera_spec(camera)
        run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(2)
        paths = self.wd.run_paths(run_id)
        spec = {'format': RUN_FORMAT, 'id': run_id, 'scene': scene, 'controller': controller, 'plant': plant,
                'camera': camera, 'd_target': d_target, 'horizon': horizon, 'seed': seed, 'started': _now(),
                'scene_snapshot': data, 'controller_dir': str(cdir), 'plant_spec': plant_spec, 'camera_spec': camera_spec}
        _write_json(paths['spec'], spec)
        meta = {k: spec[k] for k in ('id', 'scene', 'controller', 'plant', 'camera', 'd_target', 'horizon', 'seed', 'started')}
        meta.update(status='running', decision=0, started_epoch=time.time())
        _write_json(paths['meta'], meta)

        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT)] + [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p])
        env['PYTHONUNBUFFERED'] = '1'
        cmd = [sys.executable, '-m', 'usvsim.studio', '--worker', str(paths['spec']), str(paths['record'])]
        log = open(paths['log'], 'w')
        proc = subprocess.Popen(cmd, cwd=cdir, stdout=log, stderr=subprocess.STDOUT, env=env)
        with self.lock:
            self.live[run_id] = {'proc': proc, 'log': log, 'cancelled': False, 'meta': meta}
        threading.Thread(target=self._watch, args=(run_id,), daemon=True).start()
        return run_id

    def _watch(self, run_id):
        live = self.live[run_id]
        proc, paths = live['proc'], self.wd.run_paths(run_id)
        proc.wait()
        live['log'].close()
        meta = dict(live['meta'])
        record = None
        if paths['record'].is_file():
            try:
                record = json.loads(paths['record'].read_text())
            except ValueError:
                record = None
        if record is not None and record.get('status') == 'done':
            meta.update(status='done', result=record['result'], finished=record['finished'])
        else:
            meta.update(status='cancelled' if live['cancelled'] else 'failed', finished=_now(),
                        exit_code=proc.returncode,
                        error=(self._log_tail(run_id, 4000) or
                               f"the run process exited with {proc.returncode} and wrote no record"))
        meta.pop('decision', None)
        _write_json(paths['meta'], meta)
        for k in ('spec', 'progress'):
            paths[k].unlink(missing_ok=True)
        with self.lock:
            self.live.pop(run_id, None)

    def _log_tail(self, run_id, n):
        p = self.wd.run_paths(run_id)['log']
        try:
            return p.read_text(errors='replace')[-n:]
        except OSError:
            return ''

    def status(self, run_id):
        meta = self.wd.read_meta(run_id)
        if meta.get('status') == 'running':
            p = self.wd.run_paths(run_id)['progress']
            try:
                meta['decision'] = int(p.read_text().strip() or 0)
            except (OSError, ValueError):
                pass
            with self.lock:
                if run_id not in self.live:
                    meta.update(status='failed', error="the studio was restarted while this run was in progress")
                    _write_json(self.wd.run_paths(run_id)['meta'], meta)
        meta['log'] = self._log_tail(run_id, 2000)
        return meta

    def cancel(self, run_id):
        with self.lock:
            live = self.live.get(run_id)
            if live is None:
                return False
            live['cancelled'] = True
            live['proc'].terminate()
        return True

    def delete(self, run_id):
        self.cancel(run_id)
        paths = self.wd.run_paths(run_id)
        if not paths['meta'].is_file():
            raise StudioError(f"no run {run_id!r}")
        for _ in range(50):
            with self.lock:
                if run_id not in self.live:
                    break
            time.sleep(0.05)
        for p in paths.values():
            p.unlink(missing_ok=True)

    def shutdown(self):
        with self.lock:
            ids = list(self.live)
        for run_id in ids:
            self.cancel(run_id)


# --------------------------------------------------------------------------- the worker

class _TracedPlant:
    """The chosen plant with every tick's state kept, so the map can draw the hull between decisions.
    The harness is not touched: alignment, measurement and the score are its own."""

    def __init__(self, plant):
        self._p = plant
        self.trail = []

    def reset(self, state=None):
        out = self._p.reset(state)
        self.trail = [np.asarray(self._p.s, float).copy()]
        return out

    def step(self, cmd, dt=0.05):
        out = self._p.step(cmd, dt)
        self.trail.append(np.asarray(self._p.s, float).copy())
        return out

    @property
    def s(self):
        return self._p.s

    @property
    def act(self):
        return self._p.act

    @act.setter
    def act(self, v):
        self._p.act = v


def _make_plant(spec):
    from .guide import ego
    if spec == 'public':
        return ego.PublicPlant()
    from . import submission
    mod = submission.load(spec)
    return ego.SimPlant(mod.make_sim(None, 0))


def _load_policy(controller_dir):
    import importlib.util
    path = os.path.join(controller_dir, 'policy.py')
    spec = importlib.util.spec_from_file_location('_studio_policy', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_studio_policy'] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'make_policy'):
        raise RuntimeError(f"{path} defines no make_policy()")
    return mod.make_policy()


def _worker_main(spec_path, record_path):
    """One run, in its own interpreter: build the episode, load the controller, drive, write the record."""
    from .guide import closedloop as C, course
    from .guide.samples import DT
    spec = json.loads(pathlib.Path(spec_path).read_text())
    record_path = pathlib.Path(record_path)
    frames_path = record_path.with_suffix('.npz')
    progress_path = record_path.parent / f"{spec['id']}.progress"
    record = {k: v for k, v in spec.items() if k not in ('controller_dir', 'plant_spec', 'camera_spec')}
    record.setdefault('camera', 'none')
    t0 = time.perf_counter()
    try:
        ep = course.episode(spec['scene_snapshot'], spec['d_target'], spec['horizon'], spec['seed'])
        plant = _TracedPlant(_make_plant(spec['plant_spec']))
        policy = _load_policy(spec['controller_dir'])
        env = C.Env(plant, ep, spec['plant'], camera=spec.get('camera_spec'))   # a camera.py that fails to load is a failed run
    except Exception:
        record.update(status='failed', finished=_now(), error=traceback.format_exc())
        _write_json(record_path, record)
        return 0

    class Progress:                                   # progress on disk every few decisions, for the page
        def __init__(self, inner):
            self.inner = inner
            self.privileged = getattr(inner, 'privileged', False)

        def bind(self, e):
            if hasattr(self.inner, 'bind'):
                self.inner.bind(e)

        def act(self, *a):
            if env.k % PROGRESS_EVERY == 0:
                try:
                    progress_path.write_text(str(env.k))
                except OSError:
                    pass
            return self.inner.act(*a)

    try:
        r = C.run_episode(env, Progress(policy), record=True)
    except Exception:
        record.update(status='failed', finished=_now(), error=traceback.format_exc())
        _write_json(record_path, record)
        return 0
    rec = r['records']
    K = r['steps']
    dec = []

    def stretch(tick, d):                  # the station stretch as at most 8 points on the guide's track
        s_c, s_a = env.track.station(tick, d)
        ss = np.linspace(s_c, s_a, 8 if s_a - s_c > 0.05 else 1)
        return np.round(np.stack([np.interp(ss, env.track.s, env.track.pts[:, 0]),
                                  np.interp(ss, env.track.s, env.track.pts[:, 1])], 1), 3).tolist()
    for k in range(K):
        tick = k * C.PER
        d_s = float(env.dts[k + 1]) if k + 1 < len(env.dts) else float(env.d)   # the station decision k was scored at
        gx, gy, gpsi = env.path.pose(tick)
        s = plant.trail[min(tick, len(plant.trail) - 1)]
        dec.append({'k': k, 'tick': tick, 'x': float(s[0]), 'y': float(s[1]), 'psi': float(s[2]), 'u': float(s[3]),
                    'gx': float(gx), 'gy': float(gy), 'gpsi': float(gpsi),
                    'r': float(env.rd[k] + env.dts[k]), 'd': float(env.dts[k]), 'q': float(env.q[k]),
                    'es': float(env.es[k]), 'en': float(env.en[k]), 'station': stretch(tick + C.PER, d_s),
                    'action': [float(v) for v in rec['action'][k]],
                    'actuator': [float(v) for v in rec['actuator_state'][k]],
                    'ego_motion': [float(v) for v in rec['ego_motion'][k]]})
    trail = np.asarray(plant.trail, float)[:, :4]
    guide_line = env.path.poses[::C.PER][:, :3]
    record.update(status='done', finished=_now(),
                  result={'score': float(r['score']), 'steps': int(K), 'done': str(r['done']),
                          'rms_d': float(r['rms_d']), 'elapsed_s': K * C.PER * DT,
                          'wall_s': time.perf_counter() - t0, 'plant_label': spec['plant'], 'camera': env.camera_label},
                  decisions=dec, trail=np.round(trail, 3).tolist(), guide_line=np.round(guide_line, 3).tolist(),
                  hull=list(C.EGO_HULL), guide_dims=[7.0, 2.6], per=C.PER, dt=DT)
    np.savez_compressed(frames_path, rgb=rec['rgb'])
    _write_json(record_path, record)
    return 0


def _replay_main(spec_path, out_path):
    """2-1 in the studio: replay a simulator over chosen released logs, in its own interpreter."""
    from . import data, metric, plant as public_plant, submission
    spec = json.loads(pathlib.Path(spec_path).read_text())
    lset = data.load('released')
    names = spec.get('replays') or list(lset.names[:5])
    rows = [lset.names.index(n) for n in names if n in lset.names]
    if not rows:
        _write_json(pathlib.Path(out_path), {'error': 'no such replays'})
        return 0
    mod = public_plant if spec['plant_spec'] == 'public' else submission.load(spec['plant_spec'])
    small = data.LogSet([lset.names[i] for i in rows], [lset.groups[i] for i in rows], lset.commands[rows], lset.states[rows])
    t0 = time.perf_counter()
    sim = data.replay(mod, small)
    base = data.replay(public_plant, small)
    e_sim = metric.error(sim, small.states)
    e_base = metric.error(base, small.states)
    out = {'replays': [], 'wall_s': time.perf_counter() - t0}
    for j, i in enumerate(rows):
        meas, tr = small.states[j], sim[j]
        d = tr - meas
        chan = {'position': np.hypot(d[:, 0], d[:, 1]) / metric.NORM['position'],
                'heading': np.abs(np.arctan2(np.sin(d[:, 2]), np.cos(d[:, 2]))) / metric.NORM['heading'],
                'surge': np.abs(d[:, 3]) / metric.NORM['surge'],
                'yawrate': np.abs(d[:, 5]) / metric.NORM['yawrate']}
        out['replays'].append({
            'name': lset.names[i], 'group': lset.groups[i], 'E': float(e_sim[j]), 'E_base': float(e_base[j]),
            'measured': np.round(meas[:, :3], 3).tolist(), 'simulated': np.round(tr[:, :3], 3).tolist(),
            'channels': {k: np.round(np.minimum(v, metric.CLIP), 3).tolist() for k, v in chan.items()},
            'log_hz': 1.0 / (metric.DT * metric.LOG_EVERY)})
    _write_json(pathlib.Path(out_path), out)
    return 0


# --------------------------------------------------------------------------- frames, datasets, saving

def frame_png(rgb, scale=3):
    from .guide.png import encode_png
    return encode_png(rgb, scale=max(1, min(int(scale), 6)))


def window_png(frames, k, scale=2):
    """The learner's input at decision k: the four frames of the window, oldest first, side by side;
    frames before the episode began are blank."""
    from .guide import closedloop as C
    from .guide.png import encode_png
    idx = [k - C.STRIDE * j for j in range(C.WINDOW - 1, -1, -1)]
    n = frames.shape[1]
    strip = np.zeros((n, n * C.WINDOW + 2 * (C.WINDOW - 1), 3), np.uint8)
    for j, i in enumerate(idx):
        x0 = j * (n + 2)
        strip[:, x0:x0 + n] = frames[i] if i >= 0 else 24
    return encode_png(strip, scale=scale)


class _EpisodeCache:
    def __init__(self):
        self.key, self.ep = None, None

    def get(self, ddir, i):
        from .guide import dataset as F
        key = (str(ddir), int(i))
        if self.key != key:
            man = F.read_manifest(str(ddir))
            entries = man.get('episodes', [])
            if not 0 <= i < len(entries):
                raise StudioError(f"episode index must be in [0, {len(entries) - 1}]")
            self.ep, self.key = F.read_episode(str(ddir), entries[i]), key
        return self.ep


def append_run(wd, run_id, dataset):
    """Append a finished run to `datasets/<dataset>/` as one more episode, creating the dataset if needed."""
    from .guide import dataset as F
    rec = wd.read_record(run_id)
    if rec.get('status') != 'done':
        raise StudioError("only a finished run can be saved")
    frames = wd.read_frames(run_id)
    dec = rec['decisions']
    records = dict(rgb=frames, ego_motion=np.array([d['ego_motion'] for d in dec], np.float32),
                   actuator_state=np.array([d['actuator'] for d in dec], np.float32),
                   action=np.array([d['action'] for d in dec], np.float32),
                   d_target=np.array([d['d'] for d in dec], np.float32))
    ep = F.episode_from_records(records)
    ddir = wd.dataset_dir(dataset)
    if dataset == 'released_frames':
        raise StudioError("the kit's real-driving library is read-only here; save into a dataset of yours")
    camera = rec.get('camera', 'none')
    if (ddir / 'manifest.json').is_file():
        man = F.read_manifest(str(ddir))
        entries = man.get('episodes', [])
        name = f"ep_{len(entries):05d}.npz"
        np.savez_compressed(str(ddir / 'episodes' / name), **ep)
        entries.append(dict(file=f"episodes/{name}", n=int(len(ep['action'])), camera=camera))
        man.update(n_episodes=len(entries), n_records=int(sum(e['n'] for e in entries)), episodes=entries)
    else:
        man = F.write(str(ddir), [ep], meta=dict(source='studio'))
        man['episodes'][0]['camera'] = camera                 # informational; the validator ignores extra keys
    json.dump(man, open(ddir / 'manifest.json', 'w'), indent=1)
    return {'dataset': dataset, 'episodes': man['n_episodes'], 'records': man['n_records'], 'budget': BUDGET}


# --------------------------------------------------------------------------- HTTP

class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        if self.server.verbose:
            sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))

    def _send(self, status, ctype, body):
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(status, 'application/json', (json.dumps(obj) + '\n').encode())

    def _body(self):
        n = int(self.headers.get('Content-Length') or 0)
        raw = self.rfile.read(n) if n else b''
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            raise StudioError("the request body is not JSON") from None

    def _route(self, method):
        url = urllib.parse.urlsplit(self.path)
        parts = [urllib.parse.unquote(p) for p in url.path.split('/') if p]
        query = dict(urllib.parse.parse_qsl(url.query))
        try:
            handled = self._dispatch(method, parts, query)
        except StudioError as exc:
            self._json({'error': str(exc)}, 400)
            return
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            self._json({'error': traceback.format_exc()}, 500)
            return
        if not handled:
            self._json({'error': f"no such route: {method} {url.path}"}, 404)

    def do_GET(self):
        self._route('GET')

    def do_POST(self):
        self._route('POST')

    def do_PUT(self):
        self._route('PUT')

    def do_DELETE(self):
        self._route('DELETE')

    def _dispatch(self, method, parts, query):
        wd, runner = self.server.wd, self.server.runner
        if method == 'GET' and parts in ([], ['index.html']):
            self._send(200, 'text/html; charset=utf-8', PAGE.read_bytes())
            return True
        if method == 'GET' and parts == ['editor.html']:
            self._send(200, 'text/html; charset=utf-8', editor_page().read_bytes())
            return True
        if method == 'GET' and parts == ['favicon.ico']:
            self._send(204, 'image/x-icon', b'')
            return True
        if not parts or parts[0] != 'api':
            return False
        api = parts[1:]

        if api == ['state'] and method == 'GET':
            from .guide import closedloop as C, path as P, raster as R, station as W
            from .guide.samples import DT
            self._json({'workdir': str(wd.path), 'scenes': wd.list_scenes(), 'controllers': wd.list_controllers(),
                        'plants': wd.list_plants(), 'cameras': wd.list_cameras(), 'camera_file': str(wd.camera_file),
                        'runs': wd.list_runs(), 'datasets': wd.list_datasets(),
                        'dt': DT, 'per': C.PER, 'horizon': C.HORIZON, 'd_default': P.D_TARGET,
                        'd_range': [P.D_MIN, P.D_MAX], 'hull': list(C.EGO_HULL), 'guide_dims': [7.0, 2.6],
                        'size_px': R.SIZE_PX, 'm_per_px': R.M_PER_PX, 'window': C.WINDOW, 'stride': C.STRIDE,
                        'v_max': P.V_MAX, 'budget': BUDGET, 'sigma_d': C.SIGMA_D, 'band': W.BAND})
            return True

        if len(api) == 2 and api[0] == 'scene':
            name = api[1]
            if method == 'GET':
                self._json(wd.read_scene(name))
            elif method == 'PUT':
                d = float(query.get('d_target', 12.0))
                self._json({'saved': name, 'problems': wd.save_scene(name, self._body(), d)})
            elif method == 'DELETE':
                wd.delete_scene(name)
                self._json({'deleted': name})
            else:
                return False
            return True

        if api == ['check'] and method == 'POST':
            from .guide import course
            body = self._body() or {}
            self._json({'problems': course.check(body.get('scene') or {}, float(body.get('d_target', 12.0)))})
            return True

        if len(api) == 2 and api[0] == 'controller' and method == 'POST':
            body = self._body() or {}
            dest = wd.new_controller(api[1], body.get('from'))
            self._json({'created': api[1], 'path': str(dest)}, 201)
            return True

        if api == ['run'] and method == 'POST':
            body = self._body() or {}
            for key in ('scene', 'controller', 'plant'):
                if key not in body:
                    raise StudioError(f"run needs {key}")
            run_id = runner.start(body['scene'], body['controller'], body['plant'], body.get('d_target', 12.0),
                                  body.get('horizon', 40), body.get('seed', 0), body.get('camera', 'none'))
            self._json({'id': run_id}, 201)
            return True

        if len(api) >= 2 and api[0] == 'run':
            run_id, rest = api[1], api[2:]
            if method == 'GET' and not rest:
                self._json(runner.status(run_id))
            elif method == 'GET' and rest == ['record']:
                self._json(wd.read_record(run_id))
            elif method == 'GET' and len(rest) == 2 and rest[0] in ('frame', 'window') and rest[1].endswith('.png'):
                try:
                    k = int(rest[1][:-4])
                except ValueError:
                    raise StudioError("the decision index must be an integer") from None
                frames = wd.read_frames(run_id)
                if not 0 <= k < len(frames):
                    raise StudioError(f"decision must be in [0, {len(frames) - 1}]")
                png = frame_png(frames[k], query.get('scale', 3)) if rest[0] == 'frame' else window_png(frames, k, int(query.get('scale', 2)))
                self._send(200, 'image/png', png)
            elif method == 'POST' and rest == ['save']:
                body = self._body() or {}
                if not body.get('dataset'):
                    raise StudioError("save needs a dataset name")
                self._json(append_run(wd, run_id, _name(body['dataset'])))
            elif method == 'POST' and rest == ['cancel']:
                self._json({'cancelled': runner.cancel(run_id)})
            elif method == 'DELETE' and not rest:
                runner.delete(run_id)
                self._json({'deleted': run_id})
            else:
                return False
            return True

        if len(api) >= 2 and api[0] == 'dataset' and method == 'GET':
            from .guide import dataset as F
            name, rest = api[1], api[2:]
            ddir = wd.dataset_dir(name)
            if not (ddir / 'manifest.json').is_file():
                raise StudioError(f"no dataset named {name!r}")
            if not rest:
                man = F.read_manifest(str(ddir))
                self._json({'name': name, 'path': str(ddir), 'format': man.get('format'), 'renderer': man.get('renderer'),
                            'n_records': man.get('n_records'), 'episodes': man.get('episodes', []),
                            'meta': {k: v for k, v in man.items() if k not in ('episodes',) and isinstance(v, (str, int, float))}})
                return True
            if len(rest) == 2:
                try:
                    i = int(rest[0])
                    k = int(rest[1][:-4] if rest[1].endswith('.png') else rest[1])
                except ValueError:
                    raise StudioError("episode and frame indices must be integers") from None
                ep = self.server.cache.get(ddir, i)
                n = len(ep['action'])
                if not 0 <= k < n:
                    raise StudioError(f"frame must be in [0, {n - 1}]")
                if rest[1].endswith('.png'):
                    self._send(200, 'image/png', frame_png(ep['rgb'][k], query.get('scale', 3)))
                else:
                    self._json({'episode': i, 'k': k, 'n': n,
                                **{key: np.asarray(ep[key][k]).tolist() for key in ('ego_motion', 'actuator_state', 'action', 'd_target', 'step')
                                   if key in ep}})
                return True
            return False

        if api == ['replay'] and method == 'POST':
            body = self._body() or {}
            spec = {'plant_spec': wd.plant_spec(body.get('plant', 'public')), 'replays': body.get('replays') or []}
            tmp = wd.runs / f"replay-{secrets.token_hex(3)}"
            spec_p, out_p = tmp.with_suffix('.spec.json'), tmp.with_suffix('.out.json')
            _write_json(spec_p, spec)
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT)] + [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]))
            from . import data
            env['USVSIM_DATA'] = data.DATA
            try:
                proc = subprocess.run([sys.executable, '-m', 'usvsim.studio', '--replay', str(spec_p), str(out_p)],
                                      capture_output=True, text=True, timeout=300, env=env,
                                      cwd=spec['plant_spec'] if spec['plant_spec'] != 'public' and os.path.isdir(spec['plant_spec']) else None)
            except subprocess.TimeoutExpired:
                raise StudioError("the replay took more than 300 s") from None
            finally:
                spec_p.unlink(missing_ok=True)
            if not out_p.is_file():
                raise StudioError("the replay failed:\n" + (proc.stdout + proc.stderr)[-3000:])
            out = json.loads(out_p.read_text())
            out_p.unlink(missing_ok=True)
            if 'error' in out:
                raise StudioError(out['error'])
            self._json(out)
            return True

        if api == ['replays'] and method == 'GET':
            from . import data
            lset = data.load('released')
            self._json({'replays': [{'name': n, 'group': g} for n, g in zip(lset.names, lset.groups)]})
            return True
        return False


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, workdir, port=0, host='127.0.0.1', verbose=False):
        super().__init__((host, port), _Handler)
        self.wd = workdir
        self.runner = Runner(workdir)
        self.cache = _EpisodeCache()
        self.verbose = verbose

    @property
    def url(self):
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/"

    def close(self):
        self.runner.shutdown()
        self.server_close()


def serve(workdir, port=8766, open_browser=True):
    if not PAGE.is_file():
        raise SystemExit(f"{PAGE} is missing")
    editor_page()
    wd = Workdir(workdir)
    notes = wd.init()
    try:
        srv = StudioServer(wd, port)
    except OSError as exc:
        raise SystemExit(f"cannot listen on port {port}: {exc}; try --port 0") from None
    print(f"studio: {srv.url}")
    print(f"  work folder   {wd.path}")
    for n in notes:
        print(f"  seeded        {n}")
    print("  Ctrl-C stops it; runs in progress are cancelled.")
    sys.stdout.flush()
    if open_browser:
        import webbrowser
        webbrowser.open(srv.url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
    return 0


def add_args(p):
    p.add_argument('dir', nargs='?', default='.', help="the work folder (created if needed)")
    p.add_argument('--port', type=int, default=8766)
    p.add_argument('--no-open', action='store_true', help="do not open a browser")


def run(a):
    return serve(a.dir, a.port, not a.no_open)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['--worker']:
        return _worker_main(argv[1], argv[2])
    if argv[:1] == ['--replay']:
        return _replay_main(argv[1], argv[2])
    import argparse
    ap = argparse.ArgumentParser(prog='python -m usvsim.studio')
    add_args(ap)
    return run(ap.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
