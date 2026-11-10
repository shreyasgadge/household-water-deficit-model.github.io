import numpy as np

from supplymodel import SimConfig, load_cluster_parameters
from supplymodel.sensitivity import focal_deficit, make_problem, run_sobol

P = load_cluster_parameters()
SHORT = SimConfig(n_weeks=6)


def test_focal_result_independent_of_call_history():
    """Regression test: results for one cluster must not depend on earlier calls."""
    X = np.array([[0.5, 300, 5.0, 0.7, 3.0], [1.9, 1900, 3.0, 1.1, 6.0]])
    first = focal_deficit(X, P, [0, 0, 0], 2, SHORT)
    focal_deficit(X * 0.9 + 0.01, P, [0, 0, 0], 0, SHORT)   # a different cluster in between
    focal_deficit(X * 1.1, P, [0, 0, 0], 1, SHORT)
    np.testing.assert_array_equal(first, focal_deficit(X, P, [0, 0, 0], 2, SHORT))


def test_baseline_row_matches_plain_simulation():
    from supplymodel import simulate
    x = [[1.0, 1100.0, P.storage[1], P.demand[1], P.supply_days[1]]]
    out = focal_deficit(x, P, [0, 0, 0], 1, SHORT)[0]
    ref = simulate(extraction=1.0, municipal_water=1100.0, config=SHORT, **P.model_kwargs()).deficit_frequency[0, 1]
    assert out == ref


def test_run_sobol_shapes():
    r = run_sobol(P, [0, 2, 4], N=8, config=SHORT)
    assert set(r) == {"cluster_1", "cluster_2", "cluster_3"}
    assert len(r["cluster_1"]["ST"]) == 5 and make_problem(P)["num_vars"] == 5
