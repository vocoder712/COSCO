# Dynamic rho multi-seed benchmark

- device: CPU; torch `2.13.0+cpu`; threads: `4`
- epochs: `30`; seeds: `[10, 20, 30]`
- all model comparisons use paired initialization/shuffle seeds

## Three-seed mean and standard deviation

| dataset      |   shot | model              |   accuracy_mean |   accuracy_std |   successful_runs |   elapsed_mean_sec |   rho_mean |   pressure_mean |   boundary_mean |   crowding_mean |   compactness_mean |
|:-------------|-------:|:-------------------|----------------:|---------------:|------------------:|-------------------:|-----------:|----------------:|----------------:|----------------:|-------------------:|
| Heartbeat    |      1 | cosco              |        0.734959 |       0.007451 |                 3 |           5.516667 | nan        |      nan        |      nan        |      nan        |         nan        |
| Heartbeat    |      1 | cosco_geometry_rho |        0.736585 |       0.004878 |                 3 |           5.790000 |   0.110176 |        0.216941 |        0.000000 |        0.204347 |           0.000000 |
| Heartbeat    |     10 | cosco              |        0.726829 |       0.008449 |                 3 |          28.233333 | nan        |      nan        |      nan        |      nan        |         nan        |
| Heartbeat    |     10 | cosco_geometry_rho |        0.728455 |       0.010154 |                 3 |          27.890000 |   0.114836 |        0.548380 |        0.580564 |        0.397994 |           0.752599 |
| RacketSports |      1 | cosco              |        0.491228 |       0.033761 |                 3 |           1.010000 | nan        |      nan        |      nan        |      nan        |         nan        |
| RacketSports |      1 | cosco_geometry_rho |        0.464912 |       0.013695 |                 3 |           1.140000 |   0.114645 |        0.432811 |        0.000000 |        0.405717 |           0.000000 |
| RacketSports |     10 | cosco              |        0.778509 |       0.016557 |                 3 |           5.346667 | nan        |      nan        |      nan        |      nan        |         nan        |
| RacketSports |     10 | cosco_geometry_rho |        0.767544 |       0.020099 |                 3 |           5.446667 |   0.112507 |        0.524231 |        0.285302 |        0.490661 |           0.524327 |

## Paired per-seed deltas

| dataset      |   shot |   base_seed |    cosco |   cosco_geometry_rho |   cosco_geometry_rho_minus_cosco |
|:-------------|-------:|------------:|---------:|---------------------:|---------------------------------:|
| Heartbeat    |      1 |          10 | 0.726829 |             0.731707 |                         0.004878 |
| Heartbeat    |      1 |          20 | 0.741463 |             0.741463 |                         0.000000 |
| Heartbeat    |      1 |          30 | 0.736585 |             0.736585 |                         0.000000 |
| Heartbeat    |     10 |          10 | 0.717073 |             0.717073 |                         0.000000 |
| Heartbeat    |     10 |          20 | 0.731707 |             0.731707 |                         0.000000 |
| Heartbeat    |     10 |          30 | 0.731707 |             0.736585 |                         0.004878 |
| RacketSports |      1 |          10 | 0.500000 |             0.460526 |                        -0.039474 |
| RacketSports |      1 |          20 | 0.453947 |             0.453947 |                         0.000000 |
| RacketSports |      1 |          30 | 0.519737 |             0.480263 |                        -0.039474 |
| RacketSports |     10 |          10 | 0.776316 |             0.763158 |                        -0.013158 |
| RacketSports |     10 |          20 | 0.763158 |             0.750000 |                        -0.013158 |
| RacketSports |     10 |          30 | 0.796053 |             0.789474 |                        -0.006579 |
