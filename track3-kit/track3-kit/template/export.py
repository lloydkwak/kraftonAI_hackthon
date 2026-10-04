"""Export the filled template to a submission directory and validate it.

    python template/export.py --out my_submission [--checkpoint work/ckpt.pt]

Writes `<out>/manifest.json` and `<out>/graphs/*.onnx` in the contract's graph set, then
runs the same validator the intake runs (`python -m wam.validator <out>` does the same on
an existing directory).  Submit the directory as one archive (.zip or .tar.gz, <= 4 GB).
"""

from __future__ import annotations

import argparse
import os
import sys

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KIT not in sys.path:
    sys.path.insert(0, KIT)

from wam import export as EX                                          # noqa: E402
from wam import validator as V                                        # noqa: E402

import model as M                                                     # noqa: E402  (template/model.py)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--stem", default=None, help="stem checkpoint (default: ../stem/reference_stem.pt)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-validate", action="store_true")
    a = ap.parse_args(argv)

    built = M.build(a.checkpoint, a.stem)
    m = EX.export_graph_set(a.out, built["stem"], built["graphs"], M.manifest(notes=a.notes))
    print(f"exported {len(m.graphs)} graphs to {a.out} (ctx {m.ctx_bytes():,} bytes per row)")
    if a.no_validate:
        return 0
    rep = V.validate(a.out)
    print("OK" if rep.ok else "REJECTED")
    for e in rep.errors:
        print("  error:", e)
    for w in rep.warnings:
        print("  warning:", w)
    n = rep.numbers
    if "parameters" in n:
        print(f"  parameters: {n['parameters']['total_shared_once']:,} (shared tensors once)")
    for k, v in n.get("flops_per_call_b1", {}).items():
        print(f"  {k}: {v / 1e6:.1f} MFLOPs per call at B=1")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
