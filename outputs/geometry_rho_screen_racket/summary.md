# Dynamic rho multi-seed benchmark

- device: CPU; torch `2.13.0+cpu`; threads: `4`
- epochs: `5`; seeds: `[10, 20, 30]`
- all model comparisons use paired initialization/shuffle seeds

## Three-seed mean and standard deviation

| dataset      |   shot | model              |   accuracy_mean |   accuracy_std |   successful_runs |   elapsed_mean_sec |   rho_mean |   pressure_mean |   boundary_mean |   crowding_mean |   compactness_mean |
|:-------------|-------:|:-------------------|----------------:|---------------:|------------------:|-------------------:|-----------:|----------------:|----------------:|----------------:|-------------------:|
| RacketSports |      1 | cosco              |        0.412281 |       0.038549 |                 3 |           0.246667 | nan        |      nan        |      nan        |      nan        |         nan        |
| RacketSports |      1 | cosco_dynamic_rho  |        0.412281 |       0.038549 |                 3 |           0.290000 |   0.100000 |        0.000000 |      nan        |      nan        |         nan        |
| RacketSports |      1 | cosco_geometry_rho |        0.425439 |       0.021148 |                 3 |           0.290000 |   0.115000 |        0.470828 |        0.000000 |        0.478355 |           0.000000 |
| RacketSports |     10 | cosco              |        0.627193 |       0.050248 |                 3 |           0.830000 | nan        |      nan        |      nan        |      nan        |         nan        |
| RacketSports |     10 | cosco_dynamic_rho  |        0.625000 |       0.053851 |                 3 |           0.770000 |   0.115000 |        0.867138 |      nan        |      nan        |         nan        |
| RacketSports |     10 | cosco_geometry_rho |        0.625000 |       0.053851 |                 3 |           0.743333 |   0.115000 |        0.716024 |        0.726380 |        0.571846 |           0.833757 |

## Paired per-seed deltas

| dataset      |   shot |   base_seed |    cosco |   cosco_dynamic_rho |   cosco_geometry_rho |   cosco_dynamic_rho_minus_cosco |   cosco_geometry_rho_minus_cosco |
|:-------------|-------:|------------:|---------:|--------------------:|---------------------:|--------------------------------:|---------------------------------:|
| RacketSports |      1 |          10 | 0.427632 |            0.427632 |             0.434211 |                        0.000000 |                         0.006579 |
| RacketSports |      1 |          20 | 0.368421 |            0.368421 |             0.401316 |                        0.000000 |                         0.032895 |
| RacketSports |      1 |          30 | 0.440789 |            0.440789 |             0.440789 |                        0.000000 |                         0.000000 |
| RacketSports |     10 |          10 | 0.638158 |            0.611842 |             0.611842 |                       -0.026316 |                        -0.026316 |
| RacketSports |     10 |          20 | 0.572368 |            0.578947 |             0.578947 |                        0.006579 |                         0.006579 |
| RacketSports |     10 |          30 | 0.671053 |            0.684211 |             0.684211 |                        0.013158 |                         0.013158 |
