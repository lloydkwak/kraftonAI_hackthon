"""`usvsim <command>`: the kit's tools behind one entry point.

    usvsim replay-eval SUBMISSION           2-1: your score on the released logs, computed as the contest does
    usvsim validate-submission SUBMISSION   2-1: the intake gates, in order; passing means "will be scored"
    usvsim speed-check SUBMISSION           2-1: the 600 s / one core budget, extrapolated to the hidden set
    usvsim validate-dataset DIR             2-2: what intake checks on a dataset directory
    usvsim studio [DIR]                     the studio: draw a scene, run your controller, watch it, save the episode

A SUBMISSION is a `t2-1-submission/1` directory (`docs/PACKAGING.md`) or, locally, a bare `.py`.
Each 2-1 command is also a module: `python -m usvsim.replay_eval ...`.
"""
import argparse
import sys

from . import replay_eval, speed_check, validate_submission


def _validate_dataset(a):
    from .guide import validate as V                  # the 2-2 modules load only for this command
    rep = V.validate(a.dataset)
    print(rep.text(), flush=True)
    return 0 if rep.ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog='usvsim', description="Track 2 starter kit tools")
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('replay-eval', help="2-1: score a simulator on the released logs, the way the contest does")
    replay_eval.add_args(p)
    p.set_defaults(run=replay_eval.run)

    p = sub.add_parser('validate-submission', help="2-1: run the intake gates on a submission, in order")
    validate_submission.add_args(p)
    p.set_defaults(run=validate_submission.run)

    p = sub.add_parser('speed-check', help="2-1: measure a simulator against the scoring budget")
    speed_check.add_args(p)
    p.set_defaults(run=speed_check.run)

    p = sub.add_parser('validate-dataset', help="2-2: check a dataset directory the way intake will")
    p.add_argument('dataset', help="a t2-2-dataset/1 directory (docs/SCHEMA.md)")
    p.set_defaults(run=_validate_dataset)

    p = sub.add_parser('studio', help="the studio: edit scenes, run your controller, watch the run, save episodes; replay a 2-1 simulator")
    from . import studio
    studio.add_args(p)
    p.set_defaults(run=studio.run)

    a = ap.parse_args(argv)
    return a.run(a)


if __name__ == '__main__':
    sys.exit(main())
