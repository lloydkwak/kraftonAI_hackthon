# Track 3 starter kit -- 문제 3 · Dream It Yourself

Everything you need to train, validate and package a Micro World-Action Model submission.  Built 2026-10-02 16:06 KST.

Contest rules -- scoring, pass conditions, limits, ranking, what is published, the deadline -- are on the site's
problem page and common rules (https://kraftonai-autonomy-hackathon.com/problems/), which are authoritative.  This
README describes the kit: its files, the data format, the submission format and the tools.

There is no model in this kit.  The only trained thing you receive is the common visual
stem's reference weights, and you may retrain those.

## Setup

The scorer runs Python 3.10, and the commands in this kit were checked with it.  From the kit's top directory,
on a machine with an NVIDIA driver:

    python -m pip install -r runtime/scorer-requirements.txt torch imageio-ffmpeg

`runtime/scorer-requirements.txt` is the exact wheel set the official scorer runs (`onnxruntime-gpu` with its CUDA
12 libraries, `onnx`, `numpy`; see `runtime/RUNTIME.md`).  Without an NVIDIA GPU,
`python -m pip install numpy onnx onnxruntime torch imageio-ffmpeg` is enough to train on the CPU and validate.
`torch` is for training and export (pick the build that suits your machine); `imageio-ffmpeg` decodes the
dataset's videos.

## Contents

| path | what |
|---|---|
| `stem/graphs/visual_encode.onnx`, `visual_decode.onnx` | the common visual stem: fixed architecture, **reference weights** you may start from or retrain |
| `stem/reference_stem.pt` | the same weights in torch, with the training recipe that produced them |
| `template/` | **the submission template**: `model.py` (one module per graph, signatures fixed, bodies TODO), `train.py` (the data path written, the loss TODO), `export.py` (graph set + manifest + validation) |
| `dataset/` | 2000 episodes / 1,390,291 frames of lossless video; actions on 200 episodes (10 %); `manifest.json`, `sha256sums.txt`, a 200-episode dev subset |
| `wam/` | `contract.py` (every tensor, limit and manifest field; `quantize`, the uint8 rounding the scorer applies to your frames), `runner.py` (the ONNX runner the scorer uses), `validator.py` (`python -m wam.validator <dir>`), `package.py` (dataset loader / memmap), `stem.py` (the stem's architecture and its recipe's loss), `export.py`, `runtime.py` |
| `runtime/RUNTIME.md`, `runtime/scorer-requirements.txt` | the scoring runtime, and the exact package versions the official scorer runs |

The stem is **940,892 parameters** and costs **217 MFLOPs** per `visual_encode` call and **465 MFLOPs** per decoded frame at 128 px by the stem's own count (the validator's normative counter, which includes norms and activations, reads about 0.5 % more).  Its parameters count toward the submission's parameter total, which the validator caps at 0.5B; FLOPs are counted and reported, not capped.

## The dataset

* 2000 episodes, 1,390,291 frames, 128x128, 25 Hz, 400 to 1000 frames each.  **200 episodes (10 %) carry the actions between their frames; the rest are video only.**  Decoded to a uint8 memmap the whole set is about 68 GB; `package.to_memmap(..., episode_ids=...)` decodes a subset.
* Added since the first release (earlier files are unchanged byte for byte -- fetch only the new ones): 1060 episodes numbered between ep_000400 and ep_001998 (106 with actions, upright starts, pushes, re-swings, imperfect control); 540 episodes numbered between ep_000407 and ep_001999 (54 with actions, expert swing-up / catch / hold with pushes and re-swings).
* Each episode: `ep_XXXXXX.mp4`, lossless H.264 in RGB (`libx264rgb -qp 0 -pix_fmt gbrp`); `ep_XXXXXX.npz`
  with `length`, `episode_id`, `labelled` and, on labelled episodes, `actions` (float32, `T-1`).
* **`a[t]` applies between `o[t]` and `o[t+1]`**.  `package.load_episode` asserts it.
* No reward, no termination flag, no state values ship.
* The rail has end stops at +-0.4 m; the cart cannot leave it and no episode ends early.
* `package.to_memmap(dir, out.npy)` decodes once into a uint8 memmap plus an index of offsets and
  actions; `template/train.py` does this for you, over the whole set.  To start on a subset, write
  `work/frames.npy` first with `package.to_memmap("dataset", "work/frames.npy", episode_ids=...)` (the manifest's
  `dev_subset`, say); `train.py` uses an existing memmap as it is.

## The submission

A directory with `manifest.json` and `graphs/`.  Every tensor has a leading batch axis `B`
(dynamic); a graph must give the same result per row whatever `B` is.  `template/export.py`
writes both from `template/model.py`.

```json
{
  "context_length": T_ctx  (<= 32),
  "chunk_size": K          (<= 8),
  "prediction_steps": N    (<= 32, and N <= 4K),
  "ctx": {"shape": [...], "dtype": "float32" | "float16"}   (<= 4 MB per row),
  "latent_dtype": "float32" | "float16",
  "input_resolution": [128, 128], "visual_grid": [16, 16, 48], "future_latent_grid": [16, 16, 48],
  "external_weights": ["..."], "notes": "..."
}
```

The manifest carries no participant name or ID (a submission is identified by the submission ID and token it
was uploaded with); `external_weights` declares external data and pretrained weights, if any.

| graph | inputs (per row) | output |
|---|---|---|
| `visual_encode` | `frame` `(3, 128, 128)` float32 in [0, 1] | `grid` `(48, 16, 16)` |
| `visual_decode` | `z_final` `(K, 48, 16, 16)` | `frames` `(K, 3, 128, 128)` in [0, 1] |
| `encode` | `grids` `(32, 48, 16, 16)`, `actions` `(31,)` | `ctx` (your declared shape) |
| `predict_step` | `z` `(K, 48, 16, 16)`, `level` `()`, `step` `()`, `actions` `(K,)`, `ctx` | `z_next` `(K, 48, 16, 16)` |
| `update` | `ctx`, `z_final` `(K, 48, 16, 16)`, `actions` `(K,)` | `ctx_next` |
| `observe` | `ctx`, `grid` `(48, 16, 16)`, `action_prev` `()` | `ctx_next` |
| `act` | `ctx` | `action` `()` in [-1, 1] |

* `encode` always receives the whole 32-frame diagnostic segment; keep what your `T_ctx` wants.
* A graph may omit inputs it ignores (a deterministic `predict_step` drops `z`, `level`, `step`);
  it may not add inputs.  Outputs must match exactly.
* The runner supplies each block's starting latent as `tanh(N(0, 1))` from a published seed
  bundle, and `level_i = i/N`, `step_i = 1/N`.  A deterministic model ignores all three.
* Frames are quantised to uint8 before scoring, exactly as the scorer does.
* Standard ONNX operators only (opset 17), no custom domains, no random operators.
* Parameters are counted over all graphs -- every stored tensor, initializers and `Constant` nodes alike, loop
  and branch bodies included -- with shared tensors counted once; the stem's weights count.
* FLOPs are counted from static shapes at `B` = 1, a loop body times its constant trip count; a graph whose
  shapes the validator cannot make static (a data-dependent shape, a trip count read from the data) is
  rejected, not under-counted.
* Caps checked on the submission: **500 M parameters** (shared tensors counted once, the stem
  included); `N` <= 4`K`; `ctx` <= 4 MB per row; the uploaded archive <= **4 GB**.  FLOPs are
  counted and reported, not capped.  Nothing holds state outside `ctx`: every graph is a function of its declared
  inputs.
* The validator also times each graph and checks `encode`'s time against the problem page's limit (a failure on
  the CUDA provider, a warning on the CPU).  The time and memory conditions a submission must meet when it is
  scored are on the problem page.
* Weights live inside the graph files, so a network used by two graphs is stored twice, and one ONNX file cannot
  exceed 2 GB (protobuf) -- a model near the parameter limit needs `float16` or a split across graphs.

Validate locally with `python -m wam.validator <submission>`.  `./submit.sh` runs the same validator before
uploading and does not upload a submission that fails it; the scorer runs the same code when the upload arrives.

## The scoring runtime

Official scoring runs ONNX Runtime **1.23.2** on the CUDA execution provider with
`cudnn_conv_algo_search = HEURISTIC` (no per-run autotuning) and TF32 off (`wam/runtime.py`;
package lock in `runtime/scorer-requirements.txt`; details in `runtime/RUNTIME.md`).  Your graphs run on the CPU
provider locally and produce the same frames to within uint8 rounding.  Determinism is checked by the validator
when your upload arrives (the same code as the kit's `wam.validator`), on random probe inputs: each graph runs twice
on the same `B` = 2 input and the two outputs must agree within **1e-4** (absolute and relative, `numpy.allclose`),
and the input's second row run alone must give the batch's second row to the same tolerance.
The runtime is fixed for the contest.  The scoring PC: NVIDIA GeForce RTX 5060 (8 GB), Intel Core i7-14700,
30 GB RAM, one submission at a time, graphs run at `B` = 1.

## Published constants

What the statement says is "published in the starter kit", in one place.

* **Evaluation episodes:** the diagnostic segment is 32 frames with 31 actions between them, of
  one of two kinds: the same fixed waveform for every episode that does not start upright, or, for an upright start,
  the organizers' balance controller holding the machine while its cart set-point goes out and back by
  **0.05 m** (`x_ref(t) = 0.05 (1 - cos(2 pi t / 31)) / 2`; the controller's weights differ per episode).
  Either way `encode` receives the 31 actions actually applied, and every dataset episode opens with one of the two.
  Each 3-1 episode then predicts **L = 32** frames; each 3-2 episode runs up to 512 control steps.  The
  runner's generation-seed bundle holds **1** seed per block (a deterministic model is unaffected).
* **3-2 start states:**
  * **upright** -- both links within **±0.03 rad** of vertical, each rate within **±0.1 rad/s**, the cart
    within **±0.02 m**; the organizers' balance controller holds it through the diagnostic segment (above) and your
    model takes over the balance at its end;
  * **mid-swing** -- a swing with part of the energy needed to stand: the state an energy pump reaches from hanging in
    10-50 steps, kept at 15-70 % of the hanging-to-upright energy with both link rates under 6 rad/s, the cart re-centred
    within **±0.02 m** at up to 0.2 m/s; then the waveform;
  * **hanging** -- the cart within **±0.02 m**, each link angle within **±0.05 rad** of hanging, each angular rate
    within **±0.05 rad/s**; then the waveform.
* **Physics drawn per episode:** the two link lengths, independently and uniformly in **0.20 – 0.30 m**.
  The rail's end stops are at **±0.4 m**.  Everything else about the machine is the same in every episode.
* **Appearance drawn per episode** (all axes continuous and independent, fixed within an episode; object colours are fixed
  in every episode and the room is plain):

| axis | what | range |
|---|---|---|
| `W` | visible half-width of the frame, m (the zoom) | 0.9 – 1.26 |
| `rig_scale` | one overall size factor of the machine over fixed proportions | 0.72 – 1.3 |
| `rail_overhang` | how far the rail runs past the travel limit before its leg, m | 0.02 – 0.19 |
| `floor_drop` | how far the floor sits below the rail, m | 0.7 – 1.15 |
| `frame_dy` | vertical framing offset, screen units | -0.1 – 0.1 |
| `light_angle` | light direction, rad, screen space | -2.4 – -0.7 |
| `light_z` | how much the light faces the camera | 0.35 – 0.85 |
| `ambient` | ambient light | 0.34 – 0.62 |
| `diffuse` | diffuse light | 0.32 – 0.68 |
| `floor_bounce` | light reflected up off the floor | 0.02 – 0.1 |
| `contrast` | contrast | 0.85 – 1.18 |
| `vignette` | vignetting | 0.08 – 0.42 |
| `shadow_offset` | cast-shadow displacement (the rig's distance from the wall), m | 0.03 – 0.12 |
| `shadow_soft` | cast-shadow blur, px | 0.8 – 2.6 |
| `shadow_alpha` | cast-shadow darkness | 0.14 – 0.46 |
| `grain` | sensor grain amplitude, fixed within an episode | 0 – 0.015 |
| `defocus` | lens defocus, Gaussian sigma in px | 0 – 0.55 |

* **What one frame tells you about the link lengths** (measured): a learned pixel regressor on a single frame reaches
  about **13 mm RMSE** on `l1`, `l2` against a prior of 29 mm -- the render's dimensions are randomized so that no pixel
  distance is a ruler; the **motion** in the waveform segment identifies them to **about 3 mm**.  The information is in
  how the machine moves, not in how big it is drawn.

## Submitting

One archive (`.zip` or `.tar.gz`, <= 4 GB) with `manifest.json` and `graphs/` at its top level or one
directory down.  Submit with the kit's `./submit.sh <submission ID> <token> <submission directory | .zip | .tar.gz>`
(the ID and token are in your entry e-mail; the script validates, zips and uploads).

## Using the template

1. `python template/train.py --steps ...` -- after filling `model.py` and the loss in `train.py`.
2. `python template/export.py --out my_submission --checkpoint work/ckpt.pt`.
3. Read the validator's verdict at the end of step 2; fix until it prints `OK`.
4. Archive `my_submission/` (<= 4 GB) and submit it (see "Submitting").
