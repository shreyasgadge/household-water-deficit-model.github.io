"""Representative-agent model of intermittent piped supply with household storage.

Everything in this module is a direct transcription of the model code used in the
paper's notebooks. Names have been made descriptive; the mapping to the original
notebook names is given in each docstring.

Model summary (one representative household per cluster ``j``)
--------------------------------------------------------------
State variables, integrated with forward Euler at step ``dt`` (days):

* ``V_j`` - storage fill fraction in [0, 1]
* ``W``   - shared, normalised network water availability; reset to 1 at the
            start of every week and depleted while households extract
* ``use_j`` - water used from storage (tap use) at each step

Dynamics::

    extraction_j = e_j   if V_j < 1 and S_j(t) == 1 and W > 0   else 0
    use_j        = d_j   if V_j > 0                              else 0
    dV_j/dt      = (extraction_j - use_j) / K_j
    dW/dt        = - sum_j n_j * extraction_j / M

``S_j(t)`` is the binary weekly supply signal of cluster ``j`` (see
:func:`supply_signal`). The outcome reported throughout the paper is the
*frequency of running out of water*, i.e. the fraction of time steps with
``use_j == 0`` (storage empty).

Symbols <-> code
----------------
====================  =========================  ============================
paper / notebook      this module                meaning
====================  =========================  ============================
``e``                 ``extraction``             extraction rate
``M`` (``TW``/``mw``) ``municipal_water``        municipal water availability
``K`` (``ss``)        ``storage``                storage size
``d``                 ``demand``                 daily demand
``s``                 ``supply_days``            supply duration (days/week)
``n``                 ``households``             households per cluster
====================  =========================  ============================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Simulation settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SimConfig:
    """Time discretisation shared by every experiment in the paper.

    ``dt = 0.1`` days, 7-day weeks (70 steps), and ``n_weeks = 101`` weeks
    (the notebooks use ``t = 100`` and ``week = np.arange(0, t + 1)``).
    """

    dt: float = 0.1
    n_weeks: int = 101
    days_per_week: int = 7

    @property
    def week_time(self) -> np.ndarray:
        """Time grid of one week (notebook: ``wtime``)."""
        return np.arange(0, self.days_per_week, self.dt)

    @property
    def steps_per_week(self) -> int:
        return len(self.week_time)

    @property
    def n_steps(self) -> int:
        """Total number of simulated steps (notebook: ``tw``)."""
        return self.n_weeks * self.steps_per_week


DEFAULT_CONFIG = SimConfig()


# --------------------------------------------------------------------------
# Elementary model equations (notebook names in brackets)
# --------------------------------------------------------------------------


def dV_dt(extraction, use, storage):
    """Storage fill-fraction derivative ``(extraction - use) / K``  [``derv``]."""
    return (extraction - use) / storage


def dC_dt(use):
    """Cumulative-use derivative  [``derc``]. Not used for any reported outcome."""
    return use


def dW_dt(municipal_water, extraction, households):
    """Network-water derivative ``-(n * extraction) / M`` for one cluster  [``derw``]."""
    return -(households * extraction) / municipal_water


def euler_step(x, dx, dt):
    """Forward-Euler update  [``evolve``]."""
    return x + dx * dt


def extraction_rate(volume, supply_on, network_water, extraction):
    """Extraction from the network  [``extract``].

    A household extracts at rate ``extraction`` only while its storage is not
    full, piped supply is on for its cluster, and the network still has water.
    """
    if volume < 1 and supply_on == 1 and network_water > 0:
        return extraction
    return 0


def tap_use(volume, demand):
    """Water used from storage  [``tapextr``]: full demand if storage is non-empty."""
    if volume > 0:
        return demand
    return 0


# --------------------------------------------------------------------------
# Supply schedule
# --------------------------------------------------------------------------


def supply_signal(t, supply_days, start_day=0):
    """Binary supply signal for one week  [``supplyx``].

    Parameters
    ----------
    t : array
        Time grid of one week (``SimConfig.week_time``).
    supply_days : float
        Supply duration in days per week (``s``).
    start_day : int
        Day of the week on which supply starts (0 = first day). Supply that
        would run past the end of the week is clamped to end at day 7 (it does
        not wrap around).

    Returns
    -------
    np.ndarray of 0/1 with ``len(t)`` entries.
    """
    tl = len(t)
    dt = t[1] - t[0]
    steps_per_day = int(round(1 / dt))
    num_ones = int(round((supply_days / 7) * tl))
    num_ones = max(0, min(num_ones, tl))
    arr = np.zeros(tl, dtype=int)
    if num_ones > 0:
        start = int(round(start_day * steps_per_day))
        start = min(start, tl - num_ones)  # clamp so it doesn't overflow
        arr[start : start + num_ones] = 1
    return arr


def start_days_for_order(order, gap=2, slots=None, n_clusters=None):
    """Start days for a sequential supply order.

    ``order`` lists cluster indices (0-based) in the order they are supplied,
    e.g. ``(0, 2, 1)`` = cluster 1, then 3, then 2. The ``rank``-th cluster in
    the order starts on day ``slots[rank]`` (default ``rank * gap``, i.e. days
    0, 2, 4 for ``gap=2``).
    """
    n_clusters = n_clusters or len(order)
    if slots is None:
        slots = [rank * gap for rank in range(n_clusters)]
    start_days = [0] * n_clusters
    for rank, cluster in enumerate(order):
        start_days[cluster] = slots[rank]
    return start_days


def order_label(start_days, sep=" \u2192 "):
    """Human-readable supply order from start days, e.g. ``[0, 4, 2]`` -> ``'1 \u2192 3 \u2192 2'``."""
    if len(set(start_days)) == 1:
        return "Simultaneous"
    order = np.argsort(start_days, kind="stable") + 1
    return sep.join(map(str, order))


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------


def _as_batch(x, batch, n_clusters, name):
    """Broadcast scalars / (nc,) / (B,) / (B, nc) inputs to shape (B, nc)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return np.full((batch, n_clusters), float(x))
    if x.ndim == 1 and x.shape[0] == n_clusters:
        return np.tile(x, (batch, 1))
    if x.ndim == 1 and x.shape[0] == batch:
        return np.tile(x[:, None], (1, n_clusters))
    if x.shape == (batch, n_clusters):
        return x
    raise ValueError(f"{name}: cannot broadcast shape {x.shape} to ({batch}, {n_clusters})")


def _infer_batch(*arrays, n_clusters):
    batch = 1
    for a in arrays:
        a = np.asarray(a)
        if a.ndim == 2:
            batch = max(batch, a.shape[0])
        elif a.ndim == 1 and a.shape[0] != n_clusters:
            batch = max(batch, a.shape[0])
    return batch


@dataclass
class SimResult:
    """Output of :func:`simulate`.

    ``deficit_frequency`` has shape ``(B, nc)``: fraction of time steps with no
    water use (storage empty), including the initial empty state, exactly as in
    the notebooks (``count_nonzero(dusg == 0) / len(dusg)``).

    ``volume`` and ``use`` (shape ``(B, nc, n_steps + 1)``) are only filled when
    ``keep_traces=True``.
    """

    deficit_frequency: np.ndarray
    volume: np.ndarray | None = None
    use: np.ndarray | None = None


def simulate(
    *,
    extraction,
    municipal_water,
    supply_days,
    demand,
    storage,
    households,
    start_days=0,
    config: SimConfig = DEFAULT_CONFIG,
    keep_traces: bool = False,
) -> SimResult:
    """Simulate one or many parameter sets at once.

    All arguments may be a scalar, a length-``nc`` vector (one value per
    cluster), a length-``B`` vector (one value per parameter set) or a
    ``(B, nc)`` array. ``B`` (the batch size) and ``nc`` (number of clusters)
    are inferred; ``nc`` comes from ``households``. A length-``nc`` vector is
    always read as per-cluster, so when ``B == nc`` pass batch-level values as
    explicit ``(B, nc)`` arrays.

    Every arithmetic operation matches the original scalar notebook loops
    element-for-element (see ``tests/test_model.py``), so results are identical
    to the paper's code - only much faster for sweeps and Sobol sampling.

    Notes
    -----
    * ``extraction`` and ``municipal_water`` may differ between clusters. The
      common-``M`` experiments in the paper pass a scalar; the Sobol analysis
      varies them for one cluster at a time (as in the original ``run_model``).
    * ``start_days`` may be a length-``nc`` vector (shared by the whole batch)
      or a ``(B, nc)`` array.
    * The network water ``W`` is reset to 1 at the start of every week.
    """
    nc = np.asarray(households).shape[-1]  # number of clusters
    batch = _infer_batch(
        extraction, municipal_water, supply_days, demand, storage, households, n_clusters=nc
    )

    e = _as_batch(extraction, batch, nc, "extraction")
    M = _as_batch(municipal_water, batch, nc, "municipal_water")
    s = _as_batch(supply_days, batch, nc, "supply_days")
    d = _as_batch(demand, batch, nc, "demand")
    K = _as_batch(storage, batch, nc, "storage")
    n = _as_batch(households, batch, nc, "households")

    sd = np.asarray(start_days, dtype=float)
    if sd.ndim == 0:
        sd = np.full((batch, nc), float(sd))
    elif sd.ndim == 1:
        sd = np.tile(sd, (batch, 1))

    dt = config.dt
    wtime = config.week_time
    spw = config.steps_per_week
    n_steps = config.n_steps

    # Binary supply signal, one per (parameter set, cluster): shape (B, nc, steps_per_week)
    sig = np.zeros((batch, nc, spw))
    cache: dict = {}
    for b in range(batch):
        for j in range(nc):
            key = (float(s[b, j]), float(sd[b, j]))
            if key not in cache:
                cache[key] = supply_signal(wtime, key[0], key[1])
            sig[b, j] = cache[key]

    V = np.zeros((batch, nc))
    zero_count = np.ones((batch, nc))  # index 0 of the original `dusg` array is always 0
    if keep_traces:
        V_tr = np.zeros((batch, nc, n_steps + 1))
        use_tr = np.zeros((batch, nc, n_steps + 1))

    m = 0
    for _ in range(config.n_weeks):
        w = np.ones(batch)
        for k in range(spw):
            ex = np.where((V < 1) & (sig[:, :, k] == 1) & (w[:, None] > 0), e, 0.0)
            u = np.where(V > 0, d, 0.0)
            dv = (ex - u) / K
            dw_total = np.zeros(batch)
            for j in range(nc):  # sequential sum over clusters, as in the notebooks
                dw_total += -(n[:, j] * ex[:, j]) / M[:, j]
            V = np.minimum(V + dv * dt, 1)
            w = np.maximum(w + dw_total * dt, 0)
            zero_count += u == 0
            m += 1
            if keep_traces:
                V_tr[:, :, m] = V
                use_tr[:, :, m] = u

    freq = zero_count / (n_steps + 1)
    return SimResult(
        deficit_frequency=freq,
        volume=V_tr if keep_traces else None,
        use=use_tr if keep_traces else None,
    )
