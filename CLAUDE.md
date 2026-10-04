# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Workspace for the KRAFTON AI R&D Hackathon (autonomy). It holds three organiser-supplied starter kits, each unzipped to `trackN-kit/trackN-kit/`, plus the problem statements as Korean PDFs (`공통 규정` = common rules, `문제 1/2/3` = problems 1–3). Contest rules (limits, scoring, ranking, deadline) are authoritative on the site (https://kraftonai-autonomy-hackathon.com/problems/) and in the PDFs; each kit's docs describe only the kit. Not a git repo.

The kits are the scoring code. Treat their packages (`usvnav/`, `usvsim/`, `wam/`) as read-only reference — participant work goes in submission directories, agents, courses, datasets, and `template/` (track 3). Each kit's `submit.sh` validates, packages and uploads (`./submit.sh <submission ID> <token> ...`); it needs bash/curl/python3 (WSL on Windows) and supports `--dry-run`. Never run an actual upload without the user's explicit go-ahead.

The scoring runtimes differ per track — match them when developing:

| Track | Scorer Python | Pinned packages | Where |
|---|---|---|---|
| 1 | 3.13 | numpy 2.5.3, onnxruntime-gpu 1.23.2 | `track1-kit/track1-kit/docs/RUNTIME.md` |
| 2 | 3.10.12 | numpy 1.21.5 (2-1 may import only numpy, scipy, pillow, stdlib) | `track2-kit/track2-kit/docs/RUNTIME.md` |
| 3 | 3.10 | onnxruntime-gpu 1.23.2, onnx, numpy (`runtime/scorer-requirements.txt`) | `track3-kit/track3-kit/runtime/RUNTIME.md` |

Scoring machine: RTX 5060 8 GB, no network. Anything a submission imports beyond the runtime must ship inside the submission.

## Track 1 — `usvnav` (USV autonomous driver)

Agent drives a 3×2 m boat along a waypoint chain on a river under four perception conditions: 1-1 exact object list, 1-2 200×200×3 top-view raster (0.5 m/px, bow up), 1-3 180-beam 360° LiDAR, 1-4 raster under a drift estimated from `vel`'s sway. Interface: `Agent(config_path)` with `reset(meta)` and `act(obs) -> [v_cmd, w_cmd]`. No state may carry across episodes, and the same episode run twice must produce identical actions (validator checks this). Bearing is positive to port.

```
cd track1-kit/track1-kit && python -m pip install -e .
python tests/run.py                       # all tests (no pytest; dependency-free runner)
python tests/run.py test_plant            # filter by substring of "<file>.<test_name>"
usvnav studio                             # web UI at http://127.0.0.1:8765/ — author courses, run agents, scrub traces
usvnav validate mine.json
usvnav run mine.json --agent myfile:Agent --condition 1-3 --trace t.json
usvnav observe mine.json --condition 1-2 --noise grain --out frame.png
usvnav score sets/practice --agent me=myfile:Agent
usvnav validate-submission mine [--conditions 1-1,1-3 --ticks 60]
usvnav leaderboard init board --set sets/practice; usvnav leaderboard submit board mine --episode-limit-s 600
```

- Start from `examples/submission/` (agent.py + config + `submission.json`); spec in `docs/PACKAGING.md`. `usvnav/agent.py` is the tutorial agent.
- `docs/` is **generated from code** by `usvnav docs` — don't hand-edit; `docs/RULES.md` (capability list, timing: 120 s/episode, 120 s construction) and `docs/OBSERVATION.md` (every dtype/shape/palette) are the key references; `docs/examples/` has a real observation per condition.
- Everything (studio, validator, scorer) runs agents in a child process through the same `sim.run_episode`; `score.py`/`contest.py` implement relative scoring against the best completing submission.
- The scorer renders 1-2/1-4 **without noise**; noise for robustness testing comes from the work folder's `noise.py` (`--noise`). Hidden courses aren't sampled from the practice set — author hard courses yourself.

## Track 2 — `usvsim` (Make Sim to Real)

One kit, two subproblems sharing a vessel and the real-driving library in `data/` (12 replays; `released.npz` trajectories, `released_frames/` same runs through the reference camera).

```
cd track2-kit/track2-kit && python -m pip install -e .        # add '.[learner]' for torch (2-2 learner)
python tests/test_dataset.py
usvsim replay-eval SUBMISSION        # 2-1 score on released logs (training score, not held-out)
usvsim validate-submission SUBMISSION
usvsim speed-check SUBMISSION        # 600 s / one core budget
usvsim validate-dataset DIR          # 2-2 intake checks
usvsim studio [DIR]                  # http://127.0.0.1:8766/
```

- **2-1 (calibrate the simulator):** submit a directory with `manifest.json` naming a module defining `make_sim(config, seed)` → object with `reset(initial_state)` / `step(command, dt)`. Command `[T_stern, delta, T_bow]`; state `[x, y, heading, surge, sway, yaw_rate]`, body y to port, CCW yaw positive; 20 Hz commands, state logged every 2nd step. Read `usvsim/plant.py` first (the nominal model being corrected); metric in `usvsim/metric.py`. The gap is structural, not just the five dials in `params.json`. Logs include per-replay constant water current (positions ground-relative, velocities water-relative) and noise. When adding a model term, refit all parameters jointly.
- **2-2 (design the dataset):** submit data (≤18,000 records, format in `docs/SCHEMA.md`); organisers train the fixed learner `usvsim/guide/learner.py` on it and test guide-following at `d_target` in the hidden reference environment. Generate with `usvsim.guide.closedloop` (`Episode`, `Env`, `run_episode`), write with `guide.dataset`, reproduce training with `learner.train`. The reference camera's degradations are not in the kit — model them via `closedloop.Env(camera=fn)` using `data/released_frames/` as evidence. Station measure: `usvsim/guide/station.py`.
- `usvnav/` and `tools/editor.html` inside this kit are vendored from Track 1 (renderer).

## Track 3 — Micro World-Action Model (Dream It Yourself)

Train a world-action model of a cart double-pendulum from video (2000 episodes, 128×128 @ 25 Hz, lossless mp4 + npz; only 10% have actions; `a[t]` applies between `o[t]` and `o[t+1]`) and export it as a set of ONNX graphs. The kit has no model — only a reference visual stem (`stem/`), which counts toward the 500M param cap and may be retrained.

```
cd track3-kit/track3-kit
python -m pip install -r runtime/scorer-requirements.txt torch imageio-ffmpeg
python template/train.py --steps ...      # after filling model.py and the loss in train.py
python template/export.py --out my_submission --checkpoint work/ckpt.pt
python -m wam.validator my_submission     # must print OK
```

- Work happens in `template/model.py` (one module per graph, fixed signatures) and the loss in `template/train.py`; `export.py` writes `graphs/` + `manifest.json` and runs the validator.
- Graphs: `visual_encode`, `visual_decode`, `encode`, `predict_step`, `update`, `observe`, `act` — exact I/O in README and `wam/contract.py`. All state lives in `ctx` (≤4 MB/row); opset 17, standard ops only, no random ops, static shapes; batch-row independence and determinism within 1e-4. Manifest limits: `context_length ≤ 32`, `chunk_size K ≤ 8`, `prediction_steps N ≤ 32 and ≤ 4K`.
- `train.py` decodes the dataset to a uint8 memmap at `work/frames.npy` (~68 GB full); pre-create it from a subset (e.g. the manifest's `dev_subset`) via `wam.package.to_memmap(..., episode_ids=...)` to iterate quickly.
- Scored as 3-1 (predict L=32 frames after a 32-frame diagnostic segment) and 3-2 (control up to 512 steps from upright / mid-swing / hanging starts). Link lengths vary per episode (0.20–0.30 m) and are identifiable from motion, not apparent size; appearance is randomized per episode (table in README).
