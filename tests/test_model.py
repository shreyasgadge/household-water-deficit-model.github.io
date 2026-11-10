import numpy as np
import pandas as pd
import pytest

from supplymodel import (SimConfig, load_cluster_parameters, order_label, parameters_from_households,
                         simulate, start_days_for_order, supply_signal)
from reference_notebook_impl import run_notebook_loop

P = load_cluster_parameters()
SHORT = SimConfig(n_weeks=6)  # short horizon keeps the reference loop fast


@pytest.mark.parametrize("start_days", [[0, 0, 0], [0, 2, 4], [4, 0, 2]])
def test_matches_notebook_loop(start_days):
    e, M = [1.3, 0.7, 2.0], [300.0, 1200.0, 900.0]
    ref = run_notebook_loop(e, M, P.supply_days, P.demand, P.storage, P.households,
                            start_days, t=SHORT.n_weeks - 1)
    out = simulate(extraction=e, municipal_water=M, start_days=start_days, config=SHORT,
                   **P.model_kwargs()).deficit_frequency[0]
    np.testing.assert_array_equal(out, ref)


def test_batch_equals_single_runs():
    es = np.array([0.5, 1.0, 2.0, 1.5]); Ms = np.array([200.0, 1000.0, 1900.0, 700.0])
    batch = simulate(extraction=es, municipal_water=Ms, config=SHORT, **P.model_kwargs()).deficit_frequency
    for b in range(4):
        single = simulate(extraction=es[b], municipal_water=Ms[b], config=SHORT,
                          **P.model_kwargs()).deficit_frequency[0]
        np.testing.assert_array_equal(batch[b], single)


def test_supply_signal_total_and_clamp():
    t = SimConfig().week_time
    assert supply_signal(t, 3.0, 0).sum() == 30
    assert supply_signal(t, 3.0, 6).sum() == 30          # clamped, not truncated
    assert supply_signal(t, 0.0, 0).sum() == 0


def test_orders():
    assert start_days_for_order((0, 2, 1)) == [0, 4, 2]
    assert order_label([0, 4, 2]) == "1 \u2192 3 \u2192 2"
    assert order_label([0, 0, 0]) == "Simultaneous"


def test_parameters_from_households():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Supply Duration": rng.uniform(1, 6, 60), "Actual Demand": rng.uniform(.3, 1.2, 60),
                       "Storage Size": rng.uniform(2, 8, 60).astype(str), "Cluster": np.repeat([1, 2, 3], 20)})
    p = parameters_from_households(df)
    assert p.n_clusters == 3 and list(p.households) == [20, 20, 20]


def test_no_supply_means_always_dry():
    out = simulate(extraction=1.0, municipal_water=1000.0, config=SHORT,
                   **{**P.model_kwargs(), "supply_days": [0, 0, 0]}).deficit_frequency
    assert (out > 0.99).all()
