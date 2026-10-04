# usvnav — Track 1 USV autonomy

A river reach, a 3 × 2 m boat, a chain of waypoints, and four ways of perceiving the
same world. This kit is the simulator you will be scored on, the studio you build and run
your test cases in, and the scoring code.

Contest rules — limits, scoring, ranking, what is published, the deadline — are on the
site's problem page and common rules (https://kraftonai-autonomy-hackathon.com/problems/),
which are authoritative. The documents here describe the kit.

Start with `docs/RULES.md` — its first section is the problem definition: the
capabilities an agent has to handle. Then `docs/OBSERVATION.md`, which is every
number: dtypes, shapes, ranges, the palette, the action contract, with one real
observation per condition in `docs/examples/`.

---

## Install

```
python -m pip install -e .
```

numpy is the only dependency. Everything else — the PNG writer, the colour maths, the
studio and its editor — is stdlib or a single HTML file, because scoring runs on a machine
with no network and every dependency has to be in the kit. At scoring time `onnxruntime`
(GPU build) is provided as well, pinned in `docs/RUNTIME.md` — as are Python 3.13 and the
numpy version; develop on the same ones (the pins and the pip lines are in that document).

`VERSION.txt` at the top of the kit names the build you have — the usvnav version, the
commit and the build time. Quote it when you write to the organisers.

## Write an agent

The whole interface:

```python
import numpy as np

class Agent:
    def __init__(self, config_path=None):
        pass                                          # a path inside your submission

    def reset(self, meta):
        # meta: boundary polygon, the full waypoint chain, arrival radii, hull size
        # (a constant), observation_mode. No obstacle geometry -- that is what you
        # have to perceive.
        self.meta = meta

    def act(self, obs):
        distance, bearing = obs["wp_polar"]           # bearing is positive to port
        return np.array([1.5, np.clip(bearing, -0.6, 0.6)])   # [v_cmd, w_cmd]
```

No state may carry across episodes; `reset` is called at the start of each one, and the
same episode driven twice must give the same run. `examples/submission/` is a complete
submission you can copy; `usvnav/agent.py` is the tutorial agent.

## Build test cases, run them, watch them

```
usvnav studio                # http://127.0.0.1:8765/ on a work folder in the current directory
usvnav studio ~/usv --port 0
```

The studio is one page and your only test surface, so it is built to be lived in. Your
courses are listed on the left (a new work folder starts with the practice courses), the
editor is in the middle, your agents and a play button are under the courses. Pick a
course, drag bodies, lanes (the moving vessels' paths), traffic loops and waypoints, **Save** — every rule is checked on save,
including the ordered connectivity the editor alone cannot check — pick an agent (a
submission directory under `agents/`; **New from example** copies the worked example), a
condition and a seed, and press **Run**. The agent runs in a child process through the same
`run_episode` the scorer uses: your code is reloaded on every run, and a crash is a result
with its traceback rather than a dead page. Then scrub the run on the map it was made on:
the hull and the traffic where they were at each tick, the trail, the clearance over time,
the outcome and its cause, the same numbers scoring uses — and **what the agent saw** at any
tick, as the object list, the raster or the LiDAR scan, rendered from the recorded state by
the perception code the run used. Space plays, the arrow keys step.

The picture the agent receives in 1-2 and 1-4 is the studio's own map, drawn by the same
code, and the scorer renders it **without noise**. Noise is yours to add: the studio's
**noise** selector applies a function from the work folder's `noise.py` (or the kit's
example, `grain`) to every run it starts and to *what the agent saw*, so you can test an
agent against renderer noise of your own design. `docs/OBSERVATION.md` §5.2 has the shape
of a noise function; `--noise` does the same from a shell.

The same things from a shell, for scripts:

```
usvnav editor                                            # the map editor alone, one HTML file
usvnav validate mine.json                                # every course rule, each problem naming its rule
usvnav run mine.json --agent myfile:Agent --condition 1-3 --trace t.json
usvnav view mine.json --out mine.png --trace t.json      # the course with the trajectory on it
usvnav observe mine.json --condition 1-2 --out frame.png # what the agent sees at the start
usvnav observe mine.json --condition 1-2 --noise grain    # the same, under the kit's example noise
```

The hidden courses are not sampled by anything you are given — `sets/practice/` has a few
composite courses to show the format — so the courses that matter are the ones you author:
imagine what a hard case is for the capability list in `docs/RULES.md`, build it, drive it,
look at the trace.

## Score a field over a set

```
usvnav score sets/practice --agent me=myfile:Agent                    # alone: 1.0 on what you complete
usvnav score sets/practice --agent me=myfile:Agent --agent other=their:Agent --out scores.json
```

A set is a directory of course files and a `manifest.json` naming each episode's course,
seed and conditions; `usvnav.contest.write_set` builds one from your own courses. `score`
runs every submission on every episode under every condition and applies the contest's
scoring procedure as implemented in `usvnav/score.py` and `usvnav/contest.py` — the same
code path the organisers score with; the procedure is defined on the problem page.

## Package and validate a submission

A submission is a **directory**: `agent.py` exposing `Agent(config_path)`, a config file,
weights if any, and a `submission.json` manifest. `examples/submission/` is a complete one;
`docs/PACKAGING.md` is the spec.

```
cp -r examples/submission mine && $EDITOR mine/agent.py mine/submission.json
usvnav validate-submission mine                       # every rule, mechanically, plus a smoke run
usvnav validate-submission mine --conditions 1-1,1-3 --ticks 60   # quick pass while iterating
```

The validator checks the manifest, the files and the size, then constructs your agent in a
child process and drives it a few hundred ticks under each condition through the same
`run_episode` the scorer uses: two finite numbers every tick, no exception, the same
actions when the same episode is driven twice, no network attempt, and the wall clock per
`act`. A submission it rejects is not run by the leaderboard.

The manifest carries no name: the contest identifies a submission by the submission ID it
is uploaded with, and a local board names its row after the submission's directory. How
the runner times a run (120 s of wall clock per episode, 120 s to construct the agent) is
in `docs/RULES.md` §4.

## Submit

Submit with the kit's `./submit.sh <submission ID> <token> <submission directory>` (the
submission ID and token are in your entry e-mail; the script validates, zips and uploads).
Details: the common rules page, section '제출과 접수'.

## A leaderboard of your own

```
usvnav leaderboard init board --set sets/practice
usvnav leaderboard submit board mine                  # validate, run, store, re-score the field
usvnav leaderboard show board                         # the ranking
usvnav leaderboard show board --name mine            # one submitter's own page
```

The same board the organisers run on the hidden set: every submission's raw items are
stored and the field is re-scored on every arrival. `leaderboard.md` is the ranking;
`teams/<team>.md` is that participant's page — every submission they made, each with its
per-episode scores, and for the counting one the condition scores, the failure-cause
distribution, submission faults on their own and the 1-2 versus 1-4 difference. Each episode gets the scoring machine's 120 s; the kit draws
the 1-2 / 1-4 picture in pure Python, more slowly than the scoring machine, so give a local
board more with `usvnav leaderboard submit ... --episode-limit-s 600`.

## The documents

`docs/` is generated from the code by `usvnav docs` (`usvnav docs --check` confirms the files
match the code), so it cannot drift from what the simulator does:

- `docs/RULES.md` — the capability list, the interface, the submission and how the kit
  runs it (contest rules are on the site);
- `docs/OBSERVATION.md` — the observation and action schema, with `docs/examples/`;
- `docs/RUNTIME.md` — what a submission meets at scoring time;
- `docs/PACKAGING.md` — the submission directory, its manifest and every validator check.

## What `obs` contains, in one screen

Six fields in every condition — `t`, `pose`, `vel`, `wp_index`, `wp_polar`,
`prev_action` — and `obs["perception"]`, one of: an exact object list (1-1), a
200 × 200 × 3 top-view raster at 0.5 m/px, bow up (1-2), 180 laser ranges over 360° with
no-return encoded as the maximum range (1-3), or the raster under a drift you have to
estimate from `vel`'s sway component (1-4). Everything else is in `docs/OBSERVATION.md`.

## Layout

```
usvnav/
  world.py       classes, sizes, the course, its lanes and traffic routes
  geometry.py    the two collision primitives and exact distances
  plant.py       vessel dynamics, the action contract, the 1-4 disturbance
  collide.py     the whole per-tick geometry, behind a broad phase
  sim.py         the episode loop, outcomes and per-episode metrics
  freespace.py   free space and connectivity
  render.py      the 1-2/1-4 raster
  colour.py      the palette checks
  lidar.py       1-3
  objects.py     1-1, the object list
  rules.py       the course rules the validator and the editor share
  coursefile.py  the course file format and its rules
  contest.py     sets of episodes, and scoring a field of submissions over one
  score.py       the relative scoring itself: items against the best completing submission
  runner.py      the scoring runner: your agent in a child process, the episode time limit
  record.py      the run record (`usvnav-run/2`) the studio and the scorer write, replay reads
  submission.py  the submission directory, its manifest, and the local validator
  board.py       the persistent leaderboard
  docs.py        the participant documents, generated from the code
  runtime.py     the scoring runtime, as data
  studio.py      the studio's server and its run worker
  figure.py      the course as a picture (`usvnav view`)
  agent.py       the tutorial agent
  cli.py         everything above, from a shell
  bundle.py      what ships in the kit, as an allow-list (organiser-side)
  png.py         a PNG writer, stdlib only
docs/            generated: capabilities and execution, schema, runtime, packaging, example dumps
examples/        the worked example submission
sets/practice/   a few practice courses
tools/
  studio.html    the studio's page (embeds the editor)
  editor.html    the map editor
  demo-course.json
tests/
  run.py         python tests/run.py     (no pytest needed)
VERSION.txt      the build you have: usvnav version, commit, build time
```
