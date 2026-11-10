"""Run the Sobol sensitivity analysis and save results as JSON.

  python scripts/sensitivity/run_sobol.py --mode simultaneous --N 1024
  python scripts/sensitivity/run_sobol.py --mode orders --N 1024 --n-jobs 3

``simultaneous`` -> results/sobol_simultaneous_N<N>.json  (cluster_k -> indices)
``orders``       -> results/sobol_orders_N<N>.json        (cluster_k -> order_k -> indices),
                    for the 7 supply schedules in ``supplymodel.sensitivity.ORDERS``.

Cost: N * 12 model runs per cluster and schedule (second-order indices on).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import parse_args  # noqa: E402
from supplymodel.sensitivity import ORDERS, make_problem, run_sobol  # noqa: E402


def add_args(ap):
    ap.add_argument("--mode", choices=["simultaneous", "orders"], default="simultaneous")
    ap.add_argument("--N", type=int, default=1024, help="Sobol base sample size (power of 2)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-jobs", type=int, default=1, help="processes (one per cluster is enough)")


args, P = parse_args(__doc__, add_args)
out = Path(args.out)

if args.mode == "simultaneous":
    res = run_sobol(P, [0, 0, 0], N=args.N, seed=args.seed, n_jobs=args.n_jobs)
    path = out / f"sobol_simultaneous_N{args.N}.json"
else:
    res = {f"cluster_{c + 1}": {} for c in range(P.n_clusters)}
    for o, start_days in enumerate(ORDERS):
        print(f"order_{o + 1}: start days {start_days}")
        r = run_sobol(P, start_days, N=args.N, seed=args.seed, n_jobs=args.n_jobs)
        for ck, v in r.items():
            res[ck][f"order_{o + 1}"] = v
    path = out / f"sobol_orders_N{args.N}.json"

json.dump({"problem": make_problem(P) | {"bounds": [list(map(float, b)) for b in make_problem(P)["bounds"]]},
           "results": res}, open(path, "w"), indent=1)
print("saved", path)
