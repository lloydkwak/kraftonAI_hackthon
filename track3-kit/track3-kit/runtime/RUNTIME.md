# Track 3 -- the scoring runtime

Generated 2026-10-02 from `wam/runtime.py` and `runtime/scorer-requirements.txt`.  The official
scorer runs exactly this.

## What runs your graphs

* **ONNX Runtime 1.23.2** (`onnxruntime-gpu`), **CUDAExecutionProvider**, one GPU.  The scorer refuses to
  start if the installed version is not 1.23.2 or the CUDA provider is not the active one -- there is no
  silent fallback to the CPU (`wam.runtime.make_session`).
* Provider options, fixed for the contest:

| option | value |
|---|---|
| `device_id` | `0` |
| `cudnn_conv_algo_search` | `HEURISTIC` |
| `use_tf32` | `0` |
| `do_copy_in_default_stream` | `1` |
| `arena_extend_strategy` | `kNextPowerOfTwo` |

  `cudnn_conv_algo_search = HEURISTIC` means no timing-based autotuning (cuDNN 9's `DEFAULT` selects its slow
  fallback engines); `use_tf32 = 0` keeps fp32 arithmetic as on the CPU path.  `device_id` is which card, not a
  numerical setting (`WAM_CUDA_DEVICE` overrides it on a multi-GPU host).
* Session options: `intra_op_num_threads` 4, `inter_op_num_threads` 1, graph optimisation `ORT_ENABLE_ALL`.
* Batch size: scoring runs rows in batches of up to 8; **every limit is measured at `B` = 1**, one episode at a
  time: generation time per frame at p95 (<= 100 ms), control step p95 (<= 40 ms), peak GPU memory of the scorer process
  (<= 6 GB), `encode` wall-clock (<= 1 s).  A graph must give the same answer per row whatever `B` is.
* Determinism: the validator runs each graph twice on the same random `B` = 2 probe input and requires the same
  output within **1e-4** (absolute and relative), and the probe's second row run alone must give the batch's second
  row to the same tolerance; `Random*` operators are rejected at graph inspection.  The runner supplies
  every block's starting latent from a published seed bundle (`contract.seed_bundle`, `initial_latent`).
* Precision is yours: the graph's own dtypes execute; lower precision must still meet the determinism tolerance.

## The environment

The scorer's Python environment (Python 3.10) is exactly the wheel set below, installed from PyPI; the CUDA 12.9
and cuDNN 9 runtime libraries come as wheels, so the host needs only a CUDA-capable NVIDIA driver.  The wheel set and the provider options above are the whole numerical content of the
pin.  To reproduce it in a container, install the wheel set with Python 3.10 on
`nvidia/cuda:12.9.1-base-ubuntu22.04` (GPU access through the NVIDIA container toolkit).

## Running the same thing locally

* Validate: `python -m wam.validator <submission_dir>` -- the same code the intake runs, on the CPU provider
  (`onnxruntime` CPU wheel is enough; frames agree with the CUDA path to within uint8 rounding).
* Reproduce the scorer's environment: `python -m pip install -r runtime/scorer-requirements.txt` on a CUDA 12 driver,
  then `python -m wam.runtime` prints the pin check (`onnxruntime_matches_pin`, `cuda_available`).
* The scoring GPU is an NVIDIA GeForce RTX 5060 (8 GB), in a PC with an Intel Core i7-14700 and 30 GB of RAM that
  scores one submission at a time; the time limits are judged on the figures measured on this card at scoring
  time.  Wall-clock figures on your card are not the scorer's.  A large graph near the parameter cap (0.5B) can
  exceed the time limits on this card, so design on the assumption that the time limits bind before the size cap.

## The wheel set (`runtime/scorer-requirements.txt`)

```
coloredlogs==15.0.1
exceptiongroup==1.3.1
flatbuffers==25.12.19
humanfriendly==10.0
iniconfig==2.3.0
ml_dtypes==0.6.0
mpmath==1.3.0
numpy==2.2.6
nvidia-cublas-cu12==12.9.2.10
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cudnn-cu12==9.25.1.1
nvidia-cufft-cu12==11.4.1.4
nvidia-curand-cu12==10.3.10.19
nvidia-nvjitlink-cu12==12.9.86
onnx==1.22.0
onnxruntime-gpu==1.23.2
packaging==26.3
pillow==12.3.0
pluggy==1.6.0
protobuf==7.36.1
Pygments==2.21.0
pytest==9.1.1
scipy==1.15.3
sympy==1.14.0
tomli==2.4.1
typing_extensions==4.16.0
```

Contract constants this runtime enforces: 500 M parameters (shared tensors counted once), `encode`
<= 1 s per call, `ctx` <= 4 MB per row, archive <= 4 GB; the
control step p95 <= 40 ms and the generation time per frame p95 <= 100 ms are measured on this runtime.  FLOPs are counted and
reported, not capped.
