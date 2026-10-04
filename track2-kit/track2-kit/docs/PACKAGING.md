# Packaging a 2-1 submission (`t2-1-submission/1`)

A contest submission is **one directory**. `usvsim validate-submission` and the other kit tools also accept
a bare `.py` for local work, but a contest submission must be a directory, because its manifest names the
entry module.

```
my_submission/
  manifest.json      required
  sim.py             the entry module (any name; named in the manifest)
  helpers.py, ...    optional: anything the entry imports, beside it
  weights.npz, ...   optional: data the entry loads, relative to its own directory
```

`manifest.json`:

```json
{"format": "t2-1-submission/1", "entry": "sim.py", "notes": "free text"}
```

- `format` and `entry` are required. Any other key (`notes`, ...) is optional and not used by
  scoring: a submission is identified by the submission ID and token it was uploaded with.
- `entry` is a path inside the directory to a Python file defining `make_sim(config, seed)`
  (README, "What you submit"). Its directory is on `sys.path` when it is loaded, so helper
  modules beside it import by name; load data files relative to `__file__`, never by an absolute path.
- Everything else is yours. There is no size cap on a 2-1 submission, but it runs under the
  constraints in `RUNTIME.md`: one core, 600 s for the whole hidden set, the runtime's packages only,
  no network, an 8 GB memory ceiling. The scorer builds it once through `make_sim`, with one fixed
  `seed`, and scores it once.

Check before you submit:

```
usvsim validate-submission my_submission/     # layout, contract, finiteness, beats the shipped plant
usvsim speed-check my_submission/             # will it finish in time?
usvsim replay-eval my_submission/             # your score on the released logs
```

`examples/submission/` is the parameter-fit example packaged this way.
