# Track 2-2 dataset format (`t2-2-dataset/1`)

A submission is one directory. The learner reads only `episodes/`; `manifest.json` is read
for validation, never as learner input. No model weights are submitted.

Contest rules — limits, scoring, ranking, what is published, the deadline — are on the site's problem page
and common rules (https://kraftonai-autonomy-hackathon.com/problems/), which are authoritative. This document
describes the format and the validator.

```
dataset/
  manifest.json
  episodes/
    ep_00000.npz
    ep_00001.npz
    ...
```

## One file per episode

Each `ep_*.npz` is a lossless NumPy archive holding one episode: a contiguous run of decisions
from one reset. A history window never crosses an episode file. Arrays, all indexed by the
decision `t` (one decision every 0.5 s):

| array | shape | dtype | meaning |
|---|---|---|---|
| `rgb` | (T, 128, 128, 3) | uint8 | the top-view observation at decision `t` |
| `ego_motion` | (T, 3) | float32 | measured surge (m/s), sway (m/s), yaw rate (rad/s) at `t` |
| `actuator_state` | (T, 3) | float32 | `[T_stern, delta, T_bow]` as your simulator reports its actuators at `t` (the shipped plant: the command in force, azimuth clipped). In the kit's real-driving library it is the command in force; the reference environment's actuator feedback is not released. The fixed learner does not read this field |
| `d_target` | (T,) | float32 | the target distance to the guide in force at `t`, metres, in [8, 18] |
| `action` | (T, 3) | float32 | the command applied over `[t, t+1)`: `T_stern` in [-1, 1], `delta` in [-pi/2, pi/2], `T_bow` in [-1, 1] |
| `step` | (T,) | int32 | `0 .. T-1` |
| `timestamp` | (T,) | float32 | `step * 0.5` |
| `reset` | (T,) | bool | true at `step == 0` only |

**Alignment.** `record[t]`'s observation is paired with `record[t]`'s action, which is applied
over the following interval.

## Budget

18,000 decision records in total, counted over all episodes. Duplicates count. The validator
rejects a submission over the budget and warns below half of it.

## Manifest

```json
{
  "format": "t2-2-dataset/1",
  "n_episodes": 3,
  "n_records": 120,
  "decision_s": 0.5,
  "raster": {"size_px": 128, "m_per_px": 0.5},
  "renderer": "render:<hash>@<commit>",
  "episodes": [{"file": "episodes/ep_00000.npz", "n": 40, "camera": "speckle"}, ...]
}
```

Any further keys (`notes`, say) are allowed and ignored. An entry's `file` is a relative path inside
the dataset directory (`episodes/...`): an absolute path, a `..` component or a symlink leading outside
the directory is rejected. `renderer` is the stamp of the renderer
build that produced the frames; the organisers' frames carry the same stamp. An episode entry's
`camera` (optional; the studio writes it) names the camera function the frames went through --
informational, for your own bookkeeping; the learner reads pixels.

## Validation

`usvsim validate-dataset <dir>` runs what intake runs, minus the check against hidden-set frames:
format (dtypes and the full shapes of the table above), ranges, contiguity, budget, duplicate and near-duplicate counts, and coverage warnings
for a dataset with almost no turning, a constant actuator state, constant actions or a single
`d_target`. Warnings do not reject.

Episode files are opened with `allow_pickle=False`, and what a file *declares* is checked before
anything in it is decompressed: an archive whose members declare more than 1 GiB uncompressed,
an object-dtype array, or a single file declaring more records than the whole budget is rejected
unopened. Arrays outside the table above are ignored with a warning and never loaded.

## Examples

`data/released_frames/` is a directory in exactly this format: all 12 replays of the real-driving library seen
through the reference camera, each on a day chosen by design (`day` and `severity` in its manifest). There is no guide vessel in
these scenes and the actions are 2-1 excitation sequences: the directory shows the format and what the reference
camera adds to a picture, nothing about following. Open it with `usvsim validate-dataset` to see the layout pass.
