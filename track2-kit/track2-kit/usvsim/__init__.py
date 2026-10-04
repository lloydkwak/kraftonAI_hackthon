"""usvsim -- the Track 2 starter kit's package: the public USV plant, its logs and metric
(2-1), and under `usvsim.guide` the camera, the closed-loop recorder, the dataset format and the
fixed learner (2-2).

    from usvsim import data, metric, plant          # 2-1
    from usvsim.guide import closedloop, path, scenes, dataset, validate   # 2-2

    lset = data.load('released')          # the logs you may fit on
    traj = data.replay(my_module, lset)   # your simulator over the same commands
    e    = metric.error(traj, lset.states)

`usvsim --help` lists the tools; `README.md` is the tour.
"""
from . import data, metric, plant       # noqa: F401

__version__ = '0.6.2'
__all__ = ['data', 'metric', 'plant', '__version__']
