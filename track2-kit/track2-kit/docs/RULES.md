# Track 2 — Rules

Contest rules — limits, scoring, ranking, what is published, the deadline — are on the site's problem page and
common rules (https://kraftonai-autonomy-hackathon.com/problems/), which are authoritative. This document
describes the kit.

The kit's documents hold the technical contracts only:

| | |
|---|---|
| `README.md` | the tour: install, the tools, the studio, the 2-1 simulator contract (`make_sim`), the 2-2 recorder, camera hook and fixed learner, the released real-driving library, `submit.sh` |
| `docs/PACKAGING.md` | the 2-1 submission directory (`t2-1-submission/1`) and what `usvsim validate-submission` checks |
| `docs/SCHEMA.md` | the 2-2 dataset directory (`t2-2-dataset/1`) and what `usvsim validate-dataset` checks |
| `docs/RUNTIME.md` | the interpreter and package pins, the hardware, and what a submission runs under (one core, no network, the memory and time limits) |

A 2-1 submission runs in a sandbox: a child process pinned to one core, no network, only the kit on its path
(`docs/RUNTIME.md`).
