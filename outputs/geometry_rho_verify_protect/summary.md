# Dynamic rho multi-seed benchmark

- device: CPU; torch `2.13.0+cpu`; threads: `4`
- epochs: `30`; seeds: `[10, 20, 30]`
- all model comparisons use paired initialization/shuffle seeds

## Three-seed mean and standard deviation

| dataset      |   shot | model              |   accuracy_mean |   accuracy_std |   successful_runs |   elapsed_mean_sec |   rho_mean |   pressure_mean |   boundary_mean |   crowding_mean |   compactness_mean |
|:-------------|-------:|:-------------------|----------------:|---------------:|------------------:|-------------------:|-----------:|----------------:|----------------:|----------------:|-------------------:|
| Heartbeat    |      1 | cosco_geometry_rho |        0.736585 |       0.004878 |                 3 |           4.540000 |   0.110176 |        0.216941 |        0.000000 |        0.204347 |           0.000000 |
| Heartbeat    |     10 | cosco_geometry_rho |        0.730081 |       0.002816 |                 3 |          29.910000 |   0.092686 |        0.542355 |        0.560647 |        0.394469 |           0.740556 |
| RacketSports |      1 | cosco_geometry_rho |        0.480263 |       0.041086 |                 3 |           1.583333 |   0.106058 |        0.423418 |        0.000000 |        0.393183 |           0.000000 |
| RacketSports |     10 | cosco_geometry_rho |        0.802632 |       0.019737 |                 3 |           4.720000 |   0.096893 |        0.479773 |        0.222155 |        0.447468 |           0.457646 |

## Paired per-seed deltas

| dataset      |   shot |   base_seed |   cosco_geometry_rho |
|:-------------|-------:|------------:|---------------------:|
| Heartbeat    |      1 |          10 |             0.731707 |
| Heartbeat    |      1 |          20 |             0.741463 |
| Heartbeat    |      1 |          30 |             0.736585 |
| Heartbeat    |     10 |          10 |             0.726829 |
| Heartbeat    |     10 |          20 |             0.731707 |
| Heartbeat    |     10 |          30 |             0.731707 |
| RacketSports |      1 |          10 |             0.467105 |
| RacketSports |      1 |          20 |             0.447368 |
| RacketSports |      1 |          30 |             0.526316 |
| RacketSports |     10 |          10 |             0.782895 |
| RacketSports |     10 |          20 |             0.802632 |
| RacketSports |     10 |          30 |             0.822368 |
