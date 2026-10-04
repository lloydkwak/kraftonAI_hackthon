"""The 2-2 score's geometry: the follower belongs on the guide's own track, d behind it -- d measured
along the track or in a straight line, whichever; up to BAND metres to either side is free.

The track is the path the guide drove, extended BACK metres straight behind its start (the water it came from). At a
decision the station is a stretch of that track: from the straight-line station point (the latest track point at least
d from the guide in a straight line, interpolated to exactly d) to the along-track one (d of travel behind the guide). On a straight the two
coincide; in a turn the stretch lengthens. The follower's projection onto the recent track gives an along-track error
e_s (0 inside the stretch) and a distance n from the track:

    q_t = exp(-1/2 (e_s / SIGMA)^2) * exp(-1/2 (max(0, n - BAND) / SIGMA_N)^2)
"""
import numpy as np

SIGMA = 2.0       # m, along the track
BAND = 2.0        # m, either side of the track without loss
SIGMA_N = 2.0     # m, across the track beyond the band
BACK = 40.0       # m of straight track behind the start
_BACK_STEP = 0.1  # m


class Track:
    def __init__(self, poses):
        poses = np.asarray(poses, float)
        x0, y0, p0 = poses[0]
        t = np.arange(BACK, 0.0, -_BACK_STEP)
        back = np.stack([x0 - t * np.cos(p0), y0 - t * np.sin(p0)], 1)
        self.pts = np.vstack([back, poses[:, :2]])
        self.s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(self.pts, axis=0).T))]) - BACK
        self.n_back, self.n = len(back), len(poses)

    def _k(self, tick):
        return self.n_back + min(max(int(tick), 0), self.n - 1)

    def station(self, tick, d):
        k = self._k(tick); g = self.pts[k]; sg = self.s[k]; s_a = sg - d
        lo = int(np.searchsorted(self.s, sg - 3.0 * d - 10.0))
        r = np.hypot(*(self.pts[lo:k + 1] - g).T)
        far = np.nonzero(r >= d)[0]
        if not len(far):
            return s_a, s_a
        i = far[-1]                               # r[i] >= d > r[i + 1]: interpolate the crossing, so that
        f = (r[i] - d) / (r[i] - r[i + 1])        # on a straight the two station points coincide exactly
        s_c = float(self.s[lo + i] + f * (self.s[lo + i + 1] - self.s[lo + i]))
        return min(s_c, s_a), s_a

    def errors(self, tick, d, xy):
        k = self._k(tick); s_c, s_a = self.station(tick, d)
        lo = int(np.searchsorted(self.s, s_c - 25.0))
        dd = np.hypot(*(self.pts[lo:k + 1] - np.asarray(xy, float)[:2]).T)
        j = int(np.argmin(dd)); sp, n = float(self.s[lo + j]), float(dd[j])
        e_s = 0.0 if s_c <= sp <= s_a else min(abs(sp - s_c), abs(sp - s_a))
        return e_s, n


def q(e_s, n):
    return float(np.exp(-0.5 * (e_s / SIGMA) ** 2) * np.exp(-0.5 * (max(0.0, n - BAND) / SIGMA_N) ** 2))
