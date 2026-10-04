# Submission template

Three files. Fill the `TODO` blocks; keep every signature.

| file | what it is | what you do |
|---|---|---|
| `model.py` | the manifest values (`T_CTX`, `K`, `N`, `CTX_SHAPE`), the stem loader, and one `nn.Module` per graph with the contract's inputs and outputs | write the modules; `build()` returns them with your weights loaded |
| `train.py` | the dataset decoded once into a memmap, window sampling with and without actions, frames through the stem's encoder, an optimiser and a checkpoint | write the loss (and anything else you want) |
| `export.py` | `python template/export.py --out <dir>`: exports the graph set, writes `manifest.json`, runs the validator | nothing |

The graphs and their tensors are listed at the top of `model.py`; the full contract is
`wam/contract.py` and the kit `README.md`. `python -m wam.validator <dir>` is the check
`submit.sh` runs before uploading and the scorer runs on arrival; make it pass before submitting.

Nothing here is a model. The stem in `../stem/` is the only trained thing you receive, and
you may retrain it.
