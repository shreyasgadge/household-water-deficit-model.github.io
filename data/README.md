# Data

## Model input: `cluster_parameters.csv`

One row per cluster (representative agent). **This is all the model needs.**

| column | unit | meaning |
|---|---|---|
| `cluster` | - | cluster id (1, 2, 3) |
| `households` | count | households per cluster used inside the simulation (`n`) |
| `households_total` | count | household counts used as weights for the population-level deficit (`n1`) |
| `daily demand` | m³/day | cluster mean of `Actual Demand` (`d`) |
| `storage` | m³ | cluster mean of `Storage Size` (`K`) |
| `supply_days` | days/week | cluster mean of `Supply Duration` (`s`) |

The values are the cluster means reported in the paper.

## Regenerating the aggregates from household-level data

If you have the household-level table *already assigned to clusters* (output of the clustering
repository), with columns `Supply Duration`, `Actual Demand`, `Storage Size`, `Cluster`:

```python
import pandas as pd
from supplymodel import parameters_from_households

P = parameters_from_households(pd.read_csv("data/raw/modeldata.csv"))
P.to_frame().to_csv("data/cluster_parameters.csv", index=False)
```

`Storage Size` is coerced to numeric, missing values are skipped in the means. By default the
household counts are the number of rows per cluster; pass `households=[...]` and
`households_total=[...]` to use other counts. Put private survey data in `data/raw/`
(git-ignored) and pass it to `scripts/fig_cluster_characteristics.py --household-data ...` for the
SEM error bars.

## Data availability

The household survey data underlying the clustering are available on request from the corresponding author. 
The model requires only the cluster-level aggregates in cluster_parameters.csv, which are included.
