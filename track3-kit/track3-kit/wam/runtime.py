"""The pinned scoring runtime.

The official scorer runs ONNX Runtime `1.23.2` on the CUDA execution provider with the options in
`CUDA_OPTIONS`: `cudnn_conv_algo_search = HEURISTIC` (cuDNN's heuristic engine choice, no
timing-based per-run autotuning; under cuDNN 9 `DEFAULT` is not the default-quality choice but the
slow fallback engine list), TF32 off so the GPU arithmetic matches a CPU run as closely as fp32
allows, copies on the default stream.  Asking for the pinned runtime on a machine without a working
CUDA provider raises; there is no silent fallback to the CPU.

The CPU provider (4 intra-op threads) is what the validator uses by default; `provider="cpu"`
selects it explicitly.  `python -m wam.runtime` prints `describe()`: the package versions of the
runtime and its CUDA libraries, the providers this process can see, and whether the pin holds.
"""

from __future__ import annotations

import importlib.metadata as md
import os

PINNED_ORT = "1.23.2"
CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"
CUDA_OPTIONS = {
    "device_id": os.environ.get("WAM_CUDA_DEVICE", "0"),   # which card; not part of the numerical pin
    "cudnn_conv_algo_search": "HEURISTIC",      # cuDNN's heuristic pick, no timing-based
                                                # autotuning.  Not DEFAULT: with cuDNN 9's
                                                # frontend that selects the slow *fallback*
                                                # engine list
    "use_tf32": "0",                            # fp32 arithmetic, as on the CPU path
    "do_copy_in_default_stream": "1",
    "arena_extend_strategy": "kNextPowerOfTwo",
}
ALIASES = {"pinned": CUDA, "cuda": CUDA, "gpu": CUDA, "cpu": CPU}
CUDA_PACKAGES = ("onnxruntime-gpu", "onnxruntime", "nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12",
                 "nvidia-cublas-cu12", "nvidia-cufft-cu12", "nvidia-curand-cu12", "nvidia-cuda-nvrtc-cu12")


def resolve(provider: str) -> tuple:
    """`provider` (an alias or an ONNX Runtime provider name) -> `(want, providers)`:
    the provider that must end up active, and the list to hand `InferenceSession`."""
    want = ALIASES.get(provider.lower(), provider)
    if want == CUDA:
        return want, [(CUDA, dict(CUDA_OPTIONS)), CPU]
    return want, [want]


def session_options(threads: int = 4):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so


def make_session(path: str, provider: str = CPU, threads: int = 4):
    """An `InferenceSession` whose *active* provider is the one asked for, or an error."""
    import onnxruntime as ort
    want, provs = resolve(provider)
    sess = ort.InferenceSession(path, session_options(threads), providers=provs)
    active = sess.get_providers()[0]
    if active != want:
        raise RuntimeError(f"{path}: wanted {want}, ONNX Runtime activated {active} "
                           f"(available: {ort.get_available_providers()}) -- no silent fallback: timings "
                           f"must come from the runtime that was asked for")
    return sess


def describe() -> dict:
    """The runtime as it is in this process: versions, providers, whether the pin holds."""
    import numpy
    import onnxruntime as ort
    pk = {}
    for name in CUDA_PACKAGES:
        try:
            pk[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    avail = ort.get_available_providers()
    return dict(onnxruntime=ort.__version__, pinned_onnxruntime=PINNED_ORT,
                onnxruntime_matches_pin=(ort.__version__ == PINNED_ORT),
                numpy=numpy.__version__, packages=pk, available_providers=avail,
                cuda_available=(CUDA in avail), cuda_options=dict(CUDA_OPTIONS), device=ort.get_device())


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=1))
