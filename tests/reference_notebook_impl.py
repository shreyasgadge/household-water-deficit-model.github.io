"""Verbatim copy of the scalar loop from the paper notebooks (Final-paper.ipynb).

Used only by the tests, as the ground truth that `supplymodel.simulate` must match.
"""
import numpy as np


def supplyx(t, s, start_day=0):
    tl = len(t)
    dt = t[1] - t[0]
    steps_per_day = int(round(1 / dt))
    num_ones = int(round((s / 7) * tl))
    num_ones = max(0, min(num_ones, tl))
    arr = np.zeros(tl, dtype=int)
    if num_ones > 0:
        start = start_day * steps_per_day
        start = min(start, tl - num_ones)
        arr[start:start + num_ones] = 1
    return arr


def derv(e, d, K): return (e - d) / K
def derc(d): return d
def derw(TW, e, n): return -(n * e) / TW
def evolve(x, dx, dt): return x + dx * dt


def extract(v, s, w, e):
    return e if (v < 1 and s == 1 and w > 0) else 0


def tapextr(v, d):
    return d if v > 0 else 0


def run_notebook_loop(e, M, s, d, ss, n, start_days, dt=0.1, t=100):
    """e, M: per-cluster lists. Returns deficit frequency per cluster."""
    nc = len(n)
    week = np.arange(0, t + 1, 1)
    wtime = np.arange(0, 7, dt)
    tw = len(week) * len(wtime)
    arr = [supplyx(wtime, s[j], start_days[j]) for j in range(nc)]
    V = np.zeros((nc, tw + 1))
    dusg = np.zeros((nc, tw + 1))
    for i in range(len(week)):
        w = 1
        for k in range(len(wtime)):
            dw_total = 0
            for j in range(nc):
                m = i * len(wtime) + k
                ex = extract(V[j][m], arr[j][k], w, e[j])
                u = tapextr(V[j][m], d[j])
                dv = derv(ex, u, ss[j])
                dw_total += derw(M[j], ex, n[j])
                v = evolve(V[j][m], dv, dt)
                v = min(v, 1)
                V[j][m + 1] = v
                dusg[j][m + 1] = u
            w = max(evolve(w, dw_total, dt), 0)
    return np.array([np.count_nonzero(dusg[j] == 0) / len(dusg[j]) for j in range(nc)])
