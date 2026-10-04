# usvsim — Track 2 starter kit: USV calibration (2-1) and dataset design (2-2)

Track 2 has two subproblems that share one vessel and one set of real driving records. The organisers hold
a reference environment — the base simulator with several elements added — and treat it as reality: its
recorded runs are the **real-driving library** in `data/`. **2-1** asks for a simulator that follows those
trajectories under command sequences it has never seen. **2-2** asks for a driving dataset that teaches a
fixed policy to follow a guide vessel in that environment — in its wake, `d_target` behind it — seen through
its camera. Enter either or both; this one kit serves both.

Contest rules — limits, scoring, ranking, what is published, the deadline — are on the site's problem page and
common rules (https://kraftonai-autonomy-hackathon.com/problems/), which are authoritative. This document
describes the kit: read the section below for the subproblem you enter. `VERSION.txt` names the build you have;
quote it when you write to the organisers.

## Install

```
python -m pip install -e .
```

numpy is the package's only dependency. At scoring time a 2-1 submission may import numpy, scipy, pillow and
the Python standard library, nothing else; `docs/RUNTIME.md` pins the interpreter and the versions. Scoring
runs numpy 1.21.5, so develop and test against that version. The 2-2 modules, the dataset validator included,
need numpy only; only the fixed learner needs torch (`pip install -e '.[learner]'`, or the torch you have).
Then `usvsim --help` lists the tools:

```
usvsim replay-eval SUBMISSION           2-1: your score on the released logs, computed as the contest does
usvsim validate-submission SUBMISSION   2-1: the intake gates, in order; passing means "will be scored"
usvsim speed-check SUBMISSION           2-1: the 600 s / one core budget, extrapolated to the hidden set
usvsim validate-dataset DIR             2-2: what intake checks on a dataset directory
usvsim studio [DIR]                     the studio: draw a scene, run your controller, watch it, save the episode
```

## The studio

```
usvsim studio ~/usv             # http://127.0.0.1:8766/ on a work folder (created if needed)
```

The studio is one page and your test surface for both subproblems, built to be lived in. Your scenes are
listed on the left (a new work folder starts with a demo scene), Track 1's map editor is in the middle, your
controllers and a play button are under the scenes. Draw a scene — bodies and the water's edge as they are, a
**traffic loop** is the guide's path, **start** is where the ego begins — pick a controller (a directory under
`controllers/` with a `policy.py` defining `make_policy()`; **New** gives you one that holds still), a plant
(the shipped one, or a 2-1 submission you drop under `plants/`), a **camera** (`none` is the clean picture; the
other entries are the functions in the work folder's `camera.py`, your models of what the reference camera adds --
one deliberately simple example is seeded there so the shape of the thing is clear), `d_target`, and press **Run**. The controller
runs in a child process through the same `run_episode` the organisers score with: your code is reloaded on
every run, and a crash is a result with its traceback rather than a dead page. Then scrub the run on the map
it was made on: the ego and the guide where they were at each decision, the station the score measures against
(a stretch of the guide's own track, `d_target` behind the guide) with the 2 m band either side of that track,
the camera's footprint, the trail; under it the per-decision score, the score and its cause,
and **the frame the fixed learner sees at that decision** (or its four-frame window). **Save** appends the run
to a dataset directory in the submission format, with the record count against the budget.

Two more views. **Frames** pages through any dataset — the kit's real-driving library first, which is where
you see what the reference camera does to the picture. **2-1 replay** runs a simulator of yours over chosen
released logs and draws its trajectory on the measured one, with the per-channel error the metric caps.
Space plays, the arrow keys step.

Everything the studio does is also a function you can call (`usvsim.guide.course` reads a scene file as an
`Episode`); the studio is for looking, the code below is for making data at scale.

## What is in here

| | |
|---|---|
| `README.md`, `docs/RULES.md`, `docs/RUNTIME.md`, `VERSION.txt` | this file; where the contest rules are (the site) and which kit document holds which contract; the scoring runtime; the build stamp |
| `usvsim/plant.py`, `params.json`, `metric.py`, `data.py` | 2-1: the public nominal simulator (read `plant.py` first — it is the model you are correcting), its five exposed dials, the scoring metric, log loading and replay |
| `usvsim/submission.py`, `docs/PACKAGING.md`, `examples/submission/` | 2-1: the loader the scorer uses; the submission directory layout; a complete submission (the five-dial parameter fit) |
| `examples/residual_model.py` | 2-1: learn the difference instead of deriving it |
| `usvsim/guide/` | 2-2: the camera (`raster`), the guide's path (`path`), the scene vocabulary (`scenes`), the closed-loop recorder (`closedloop`), the dataset format (`dataset`) and its validator (`validate`), the fixed learner (`learner`), the shipped plant behind the harness (`ego`), the measurement model (`measure`), a scene file read as an episode (`course`) |
| `usvsim/studio.py` | the studio's server and its run worker (`usvsim studio`) |
| `usvnav/` | Track 1's top-view renderer: the modules the camera draws with (`look` is the studio's picture), vendored at the stamped build |
| `docs/SCHEMA.md` | 2-2: the dataset format |
| `data/released.npz` | the real-driving library: **12 replays**, two per manoeuvre group (`straight`, `accel_decel`, `turn`, `scurve`, `brake`, `lowspeed`), 60 s each — commands at 20 Hz (segments of 0.5–8 s, mixed), measured state at 10 Hz. Each replay ran in a weak constant water current of its own (0.03–0.10 m/s, direction random, not disclosed): positions are ground-relative, velocities water-relative |
| `data/released_frames/` | **all 12 replays seen through the reference camera**, each on a day chosen by design (every weather at a nominal and at a heavy severity, two light days, two clean ones), one episode file per replay in the 2-2 dataset format: a 128 × 128 frame every 0.5 s, the log's measured motion at the same instants, the command in force (`actuator_state` is that command with the azimuth clipped; the reference's actuator feedback is not released). No guide vessel: these are calibration drives, and the directory exists to show what the reference camera adds to a picture. `manifest.json` names each episode's replay, its `day` (the weather) and `severity` |
| `data/anchors.json`, `data/manifest.json` | per-group `E_base` values (the shipped simulator's own error; `usvsim.data.anchors()` reads them), the metric's normalisation and caps, the replay lengths; column names, the measurement model, group membership, the replay ↔ frames pairing |
| `tools/studio.html`, `tools/editor.html` | the studio's page, and Track 1's map editor it embeds (vendored at the stamped build) |
| `tests/` | a check you can run yourself (`python tests/test_dataset.py`): the 2-2 path from an episode to a validated dataset |

## The real-driving library

One record, two uses. Each of the 12 replays exists as a trajectory in `released.npz` (2-1 fits on it) and as an
episode file in `released_frames/` (2-2 reads it for what the reference camera sees — the water, the light, the
weather, the lens). The two are the same run: frame
`k` was taken at `t = 0.5 k`, log sample `5k − 1`; the episode's `ego_motion` is the log's measured surge,
sway and yaw rate at those instants (zero at `t = 0`, the rest state the log does not sample). The commands are
excitation manoeuvres and there is no guide vessel in the scene: this is calibration data seen through the
reference camera, not following data. The 2-2 training set is your dataset.

## 2-1 — calibrate the simulator

The shipped simulator is a simplification and the real vessel does not behave like it. Your job is to close
that difference: submit a simulator that, replayed open-loop against command sequences you have never seen,
follows the same trajectories the real vessel did. The difference is **structural**, not a matter of
coefficients. Fitting the five dials the shipped model exposes is a reasonable first submission
(`examples/submission/` is that fit); going further means working out what the shipped model is missing and
adding it.

```
usvsim replay-eval examples/submission           # score a submission on the released logs
usvsim validate-submission my_sim/               # will it be accepted?  (a directory, or a bare .py)
usvsim speed-check my_sim/                       # will it finish in time?
```

**What you submit.** A directory with a `manifest.json` that names your entry module (`docs/PACKAGING.md`;
`examples/submission/` is one). The entry module defines:

```python
def make_sim(config, seed):
    return sim          # sim.reset(initial_state); sim.step(command, dt)
```

`command` is `[T_stern, delta, T_bow]` — stern azimuth thruster magnitude and direction, bow tunnel thruster
magnitude. `state` is `[x, y, heading, surge, sway, yaw_rate]`, body x through the bow, body y to **port**,
positive yaw rate counter-clockwise from above. Replays start from rest at the origin, one command per 0.05 s,
state recorded every second step; `reset` is called once per sequence.

**Built once.** The scorer builds your simulator once through `make_sim`, with one fixed `seed`, and replays
each sequence once (`docs/RUNTIME.md`).

**The released records.** They carry measurement noise and a constant water current of their own, so part of
the work is to tell the vessel's mechanism from the water it moved through: positions are ground-relative,
velocities water-relative. The metric — four channels (position, heading, surge speed, yaw rate), normalised,
capped per sample — is `usvsim/metric.py`; `replay-eval` applies it to the released logs.

**Two traps.** `replay-eval` runs on the logs you fit on, so it reports a training score, not an estimate of
your score on unseen sequences. Adding one correction at a time without refitting the rest — the five original
parameters absorb part of whatever you have not modelled yet; add a term and the old fit is stale, refit
everything together.

## 2-2 — design the dataset

You submit **data**: action-labelled driving records (`docs/SCHEMA.md`). The organisers train the fixed learner
(`usvsim/guide/learner.py`) on it and roll the policy out behind a moving guide in the reference environment you
do not have. The per-decision station measure — on the guide's own track, `d_target` behind the guide — is
`usvsim/guide/station.py`; `closedloop.Env` and the studio report it for every decision. The format and the
record budget the validator enforces (at most 18,000 records) are in `docs/SCHEMA.md`.

**What the kit gives you, and what it does not.**

- *The camera.* `usvsim.guide.raster` draws the exact top view as the studio shows it: 128 px at 0.5 m/px, bow
  up, the ego at the centre; the map's colours with a wet shoreline, shadows, decks and outlines, anti-aliased --
  the same geometry and colours as the reference camera; the guide is the one light hull with a black centre
  mark. What the reference camera adds to that picture is not in the kit. `data/released_frames/` is where you
  see it, and your model of it goes in **`closedloop.Env(camera=...)`**: a function `camera(frame, ctx) -> frame`
  called on every frame as it is rendered, with `ctx` carrying the time, the ego's pose, the frame's class map
  and water mask, the world position of every pixel and the episode's seed. The studio's camera selector and its
  `camera.py` are the same hook.
- *The recorder.* `closedloop.Env` and `run_episode` take an `Episode` — a guide path, the bodies on the
  water, where the ego starts, the distance to keep — and a plant of your choice; they step it at the 0.5 s
  decision period, render, measure, and record the arrays intake expects, with the alignment the scorer uses.
- *The vocabulary.* `path.GuidePath` is where the guide is at every tick (poses at 20 Hz; `from_rates`
  integrates a speed and yaw-rate profile you write). `scenes.body` is what can be on the water. How the
  organisers move their guide and compose their scenes is not in the kit.
- *The instrument.* `learner.py` is the fixed model and training recipe the organisers train on your
  data. `learner.train` reproduces it, so you can see what your data teaches — in
  whatever environment, on whatever episodes, you decide to test it.
- *The format and the gate.* `dataset.write` / `dataset.read`, and `usvsim validate-dataset`: the intake
  checks, minus the comparison against hidden frames.

Generating data — a guide path, a scene, a start, a controller, the recorder:

```python
import numpy as np
from usvsim.guide import closedloop as C, dataset as F, path as P, scenes as S

class MyController:                 # privileged = True: sees the true state and may read the guide through env
    privileged = True
    def bind(self, env): self.env = env
    def act(self, tick, state, actuator):
        gx, gy, gpsi = self.env.guide_pose()        # the guide's true pose at this tick
        return np.array([0.6, 0.0, 0.0])            # [T_stern, delta, T_bow]: your controller here

n = 60 * 20                                                              # 60 s of guide motion at 20 Hz
guide = P.GuidePath.from_rates(np.full(n, 1.2), np.full(n, 0.03))      # a gentle turn at 1.2 m/s
bodies = [S.body('buoy', 30.0, -8.0), S.body(S.KINDS[1], 55.0, 15.0, psi=0.4)]
ep = C.Episode(guide, bodies, d_target=12.0, horizon=40, seed=1)      # the ego starts on station unless you say where
r = C.run_episode(C.Env('public', ep), MyController(), record=True)   # 'public' = the shipped plant, or pass your own
F.write('my_dataset', [F.episode_from_records(r['records'])], meta=dict(notes='first try'))
```

An observation-only controller (`privileged = False`) is called as `act(obs)` with the policy's window: `rgb`
(4, 128, 128, 3), `ego_motion` (4, 3), `actuator_state` (4, 3), `history_mask` (4,), `d_target` (4,). Any
plant may drive the generation — the shipped one behind `'public'`, the one you calibrated in 2-1, anything with
`reset(state)`, `step(cmd, dt)`, `.s` and `.act` (`usvsim/guide/ego.py` is the adapter for the shipped plant).
`Episode` also takes `init_state`, `init_actuator`, a `d_schedule` of station changes, a `boundary` polygon for
the water's edge, and the episode's `horizon`. Only the recorded fields ship; edit them as you see fit.

Training the fixed learner on your data:

```python
from usvsim.guide import dataset as F, learner as L
data, manifest = F.read('my_dataset')
ds = L.Dataset(data['rgb'], data['ego_motion'], data['actuator_state'], data['action'],
               data['episode'], data['step'], d_target=data['d_target'], device='cuda')
model, loss = L.train(ds, seed=0)                               # the contest's recipe, unchanged
L.save(model, 'policy.pt', dict(frames='all', numeric='full'))
policy = L.load_policy('policy.pt')                             # an observation-only controller: run_episode drives it
```

## Submitting

Submit with the kit's `./submit.sh <submission ID> <token> <2-1|2-2> <directory>` (the ID and token are in
your entry e-mail; the script validates, zips and uploads). Details: the common rules page, section '제출과 접수'.
What a submission must contain: a 2-1 submission is a directory as `docs/PACKAGING.md` describes (check it
with `usvsim validate-submission` and `usvsim speed-check`); a 2-2 submission is a dataset directory as
`docs/SCHEMA.md` describes (check it with `usvsim validate-dataset`). The limits and the deadline are contest
rules: see the site's problem page and common rules.
