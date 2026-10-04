"""The 2-2 path, end to end on the public plant: compose an episode from values, drive it, record,
write the dataset, read it back, validate it. What your own generator does, in miniature."""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np

from usvsim.guide import closedloop as C, dataset as F, path as P, scenes as S


class Zero:                              # any policy: observation in, command out
    privileged = False

    def act(self, obs):
        assert obs['rgb'].shape == (C.WINDOW, 128, 128, 3)
        return np.zeros(3)


n = int(round(8 * C.PER * P.DT / P.DT))  # ticks for 8 decisions
path = P.GuidePath.from_rates(np.full(n + 40, 1.0), np.zeros(n + 40))       # a straight run at 1 m/s
bodies = [S.body('buoy', 25.0, -9.0), S.body(S.KINDS[1], 40.0, 14.0, psi=0.3)]
ep = C.Episode(path, bodies, d_target=12.0, horizon=8, seed=1)
r = C.run_episode(C.Env('public', ep), Zero(), record=True)
assert r['steps'] == 8 and r['records']['rgb'].shape == (8, 128, 128, 3)
assert 0.0 <= r['score'] <= 100.0

tmp = tempfile.mkdtemp(prefix='usvsim_ds_')
try:
    F.write(tmp, [F.episode_from_records(r['records'])], meta=dict(notes='kit self-test'))
    back, man = F.read(tmp)
    assert back['rgb'].shape == (8, 128, 128, 3) and np.array_equal(back['rgb'], r['records']['rgb'])
    from usvsim.guide import validate as V
    rep = V.validate(tmp)
    assert rep.ok, rep.text()
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print(f"  one episode: {r['steps']} decisions recorded, written, read back and validated (score {r['score']:.1f})")
