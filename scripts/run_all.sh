#!/usr/bin/env bash
# Reproduce every figure. Run from the repository root:  bash scripts/run_all.sh [N]
# N = Sobol base sample size (default 1024; use 32 for a quick smoke test).
set -euo pipefail
N="${1:-1024}"

python scripts/fig_cluster_characteristics.py
python scripts/fig_supply_scenarios.py
python scripts/fig_deficit_e_vs_M.py
python scripts/fig_deficit_e_vs_supply_duration.py
python scripts/fig_sequential_orders_e_vs_M.py
python scripts/fig_sequential_relative_extraction.py
python scripts/fig_sequential_all_orders_e_vs_duration.py

python scripts/sensitivity/run_sobol.py --mode simultaneous --N "$N" --n-jobs 3
python scripts/sensitivity/plot_sobol_simultaneous.py --N "$N"
python scripts/sensitivity/run_sobol.py --mode orders --N "$N" --n-jobs 3
python scripts/sensitivity/plot_sobol_orders.py --N "$N"
