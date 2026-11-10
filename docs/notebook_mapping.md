# Mapping from the paper notebooks to this repository

Only notebook sections marked **Main Code** / **Main, supplement** were ported, together with the
data-interpretation and general-model cells they depend on.

## `Final-paper.ipynb`

| Notebook cells | Section marker | Now |
|---|---|---|
| 0-6 | imports, load data, cluster means | `supplymodel/data.py` (`parameters_from_households`, `load_cluster_parameters`) |
| 7-8 | `supplyx`, `derv`, `derc`, `derw`, `evolve`, `extract`, `tapextr` | `supplymodel/model.py` |
| 15 | Main supplementary: population deficit e vs M | `fig_deficit_e_vs_M.py` (`PS-eM-total.png`) |
| 18 | Main: water deficit e vs M | `fig_deficit_e_vs_M.py` |
| 22 | Main: water deficit e vs supply duration | `fig_deficit_e_vs_supply_duration.py` |
| 24 | Main: sequential supply, order of cluster 3, e vs M | `fig_sequential_orders_e_vs_M.py` |
| 26 | Main, supplement: relative extraction-rate permutations vs M | `fig_sequential_relative_extraction.py` |
| 28 | Main, supplement: all supply orders, e vs supply duration | `fig_sequential_all_orders_e_vs_duration.py` |
| 48 | Main: Sobol ST across supply orders (sensitivity) | `scripts/sensitivity/plot_sobol_orders.py` |
| 55 | Main: supply scenario visualisation | `fig_supply_scenarios.py` |
| 59 | Main: data plotting with mean and range | `fig_cluster_characteristics.py` |

### Not ported
| Cells | Why |
|---|---|
| 11, 13 | not marked Main (equilibrium-volume demo; per-cluster e-vs-M heatmaps, superseded by 18) |
| 16, 19, 20 | earlier versions of the figures in 18 / 22 |
| 29-31 | animation; relies on variables (`mw_range`) that the final cell 28 no longer defines |
| 32-35, 36-37 | outcome-data / alternate-source "Validation" plots; not marked Main |
| 38-41 | "Sensitivity" order-range maps (cluster and population); not marked Main |
| 42-46, 49-53 | marked "no need" |
| 56, 57 | earlier drafts of the cluster bar chart (59 is the marked one) |

## `Sensitivity_Analysis.ipynb`

| Cells | Now |
|---|---|
| 0-7 (setup, `run_model`, Sobol problem) | `supplymodel/sensitivity.py` |
| 8 (Sobol, simultaneous, N = 1024) | `scripts/sensitivity/run_sobol.py --mode simultaneous` |
| 10 (S1 / ST bar plot) | `scripts/sensitivity/plot_sobol_simultaneous.py` (cell 9 is an earlier version) |
| 11 onward | marked "don't include" - not read |

The across-orders Sobol run (`--mode orders`) that produces the input for main-notebook cell 48 was
written from the simultaneous-supply driver, because its original code sits below the
"don't include" marker. Check `N` and `seed` against what the paper used.

## Intentional differences from the notebooks

1. **Sobol `run_model` no longer shares state between calls.** The notebook version wrote the
   sampled values into module-level lists (`e_local`, `MW_local`, ...), so after sweeping cluster 1 its
   parameters stayed at the last sample while clusters 2 and 3 were swept. `focal_deficit` resets all
   non-focal clusters to baseline for every run (regression test in `tests/test_sensitivity.py`).
2. **Red dashed lines** in the e-vs-supply-duration figure use the cluster means (3.81, 2.92, 3.27);
   the notebook hard-coded `[3.9, 2.9, 3.3]`.
3. **Combined all-permutations e-vs-duration figure** flips the heatmap columns so the x-axis
   matches the plotted extent (the notebook labelled the axis in the opposite direction).
4. Cumulative use `C` is not simulated (it was never used for an outcome).
5. Sobol results are stored as JSON instead of pickle.
