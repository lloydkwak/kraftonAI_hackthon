"""The scoring runtime, as data.

What a submission meets at scoring time is a fixed environment, and the submission's
manifest names the environment it expects (`runtime`) so the validator can check the two
agree. There is exactly one runtime, and it is this dictionary. The runtime document
(`docs/RUNTIME.md`) and the packaging spec render it rather than restating it, so the
numbers have one home.

**Pinned, not read.** `RUNTIME` declares the interpreter and the package versions, and the
scoring machine is checked against it. Reading them off the running interpreter would make
the definition whatever happened to be installed; the environment is frozen for the
contest, so this is the thing that is frozen.

**numpy and onnxruntime, nothing else.** The package ships with numpy as its one
dependency and the scoring machine has no network, so anything else a submission needs has
to be inside its directory -- pure Python, or wheels (compiled extensions included) unpacked
there. The one addition is `onnxruntime`, GPU build, pinned --
framework-neutral, and the same version Track 3 pins. The CUDA libraries it loads are pip
packages too (`CUDA`), pinned for the same reason: the scoring machine has no system CUDA,
and "GPU build" means nothing unless the libraries under it are part of the definition.
"""

from __future__ import annotations

import importlib.metadata as _md
import platform
import sys

RUNTIME_ID = "usvnav-runtime/1"

#: Distribution names (what `pip install` takes), pinned exactly.
PACKAGES = {
    "numpy": "2.5.3",
    "onnxruntime-gpu": "1.23.2",            # imports as `onnxruntime`
}

#: The CUDA libraries onnxruntime-gpu 1.23 loads, as the pip packages they ship in.
CUDA = {
    "nvidia-cuda-runtime-cu12": "12.9.79",
    "nvidia-cudnn-cu12": "9.25.1.1",
    "nvidia-cublas-cu12": "12.9.2.10",
    "nvidia-cufft-cu12": "11.4.1.4",
    "nvidia-curand-cu12": "10.3.10.19",
    "nvidia-cuda-nvrtc-cu12": "12.9.86",
    "nvidia-nvjitlink-cu12": "12.9.86",
}

#: What `onnxruntime.get_available_providers()` must contain. The GPU build reports the
#: CUDA provider whether or not a card is present; `gpu_smoke` is what proves the card.
PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")

RUNTIME = {
    "id": RUNTIME_ID,
    "python": "3.13",                       # major.minor; the patch level is not pinned
    "packages": PACKAGES,
    "cuda": CUDA,
    "providers": PROVIDERS,
    "network": False,
    "cpu": "Intel Core i7-14700 (28 threads)",
    "ram": "30 GB",
    "gpu": "NVIDIA GeForce RTX 5060 (8 GB); onnxruntime's CUDAExecutionProvider is how a "
           "submission reaches it",
    #: The scoring PC is shared: it may run several agents at the same time. No numbers are
    #: promised, so none are stated.
    "sharing": "the scoring PC may run several agents at the same time -- other episodes or other "
               "submissions -- sharing the CPU and the GPU; no CPU cores are pinned to an agent, and "
               "the time limits are wall clock under those conditions",
    #: Memory for the agent's process alone; the scorer's processes are outside it. Exceeding it
    #: kills the agent's process: that episode is a `crash`, and the process is started again for
    #: the next episode.
    "memory_cap": "6 GB",
    "submission_form": "directory",
    "agent_process": "a child of the runner, observation and action over a local pipe; one process "
                     "per condition, constructed once, running that condition's episodes in "
                     "sequence with reset() between them; a process that dies or overruns an "
                     "episode is started again for the next one",
    "wall_clock": "not real time -- the runner waits for act(); there is no per-tick limit, only "
                  "the episode's",
}


def _installed(dist: str) -> str | None:
    try:
        return _md.version(dist)
    except _md.PackageNotFoundError:
        return None


def check_environment() -> list[str]:
    """Where the running interpreter differs from `RUNTIME`. Empty means it matches."""
    out = []
    have = f"{sys.version_info.major}.{sys.version_info.minor}"
    if have != RUNTIME["python"]:
        out.append(f"python {have}, runtime pins {RUNTIME['python']}")
    for dist, want in {**PACKAGES, **CUDA}.items():
        got = _installed(dist)
        if got is None:
            out.append(f"{dist} is not installed")
        elif got != want:
            out.append(f"{dist} {got}, runtime pins {want}")
    try:
        import onnxruntime as ort
    except ImportError:
        out.append("onnxruntime is not importable")
    else:
        missing = [p for p in PROVIDERS if p not in ort.get_available_providers()]
        if missing:
            out.append(f"onnxruntime lacks {', '.join(missing)} -- not the GPU build?")
    return out


def gpu_smoke(model_path) -> str:
    """Run `model_path` once on the CUDA provider and return the provider that ran it.

    Raises whatever onnxruntime raises when the card, the driver or the CUDA libraries are
    not there. This is the runtime's own check that "GPU build" is more than a wheel name.
    """
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider"])
    used = sess.get_providers()[0]
    if used != "CUDAExecutionProvider":
        raise RuntimeError(f"onnxruntime fell back to {used}")
    name = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in name.shape]
    sess.run(None, {name.name: np.ones(shape, dtype=np.float32)})
    return used


def describe() -> str:
    lines = [f"runtime {RUNTIME['id']}",
             f"  python      {RUNTIME['python']}  ({platform.python_implementation()})"]
    for name, ver in PACKAGES.items():
        lines.append(f"  {name:<11} {ver}")
    short = {k.removeprefix("nvidia-").removesuffix("-cu12"): v for k, v in CUDA.items()}
    lines.append("  cuda        " + ", ".join(f"{k} {v}" for k, v in short.items()))
    lines.append(f"  network     {'none' if not RUNTIME['network'] else 'yes'}")
    lines.append(f"  gpu         {RUNTIME['gpu']}")
    lines.append(f"  submission  {RUNTIME['submission_form']}")
    return "\n".join(lines)
