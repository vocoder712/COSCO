# Dynamic rho multi-seed benchmark

- device: CPU; torch `2.13.0+cpu`; threads: `4`
- epochs: `5`; seeds: `[10, 20, 30]`
- all model comparisons use paired initialization/shuffle seeds

## Three-seed mean and standard deviation

| dataset      |   shot | model              |   accuracy_mean |   accuracy_std |   successful_runs |   elapsed_mean_sec |   rho_mean |   pressure_mean |   boundary_mean |   crowding_mean |   compactness_mean |
|:-------------|-------:|:-------------------|----------------:|---------------:|------------------:|-------------------:|-----------:|----------------:|----------------:|----------------:|-------------------:|
| RacketSports |      1 | cosco              |        0.412281 |       0.038549 |                 3 |           0.213333 | nan        |      nan        |      nan        |      nan        |         nan        |
| RacketSports |      1 | cosco_geometry_rho |        0.427632 |       0.023721 |                 3 |           0.233333 |   0.114920 |        0.470914 |        0.000000 |        0.478775 |           0.000000 |
| RacketSports |     10 | cosco              |        0.627193 |       0.050248 |                 3 |           0.610000 | nan        |      nan        |      nan        |      nan        |         nan        |
| RacketSports |     10 | cosco_geometry_rho |        0.625000 |       0.047441 |                 3 |           0.643333 |   0.107544 |        0.715371 |        0.719906 |        0.572224 |           0.830265 |

## Paired per-seed deltas

| dataset      |   shot |   base_seed |    cosco |   cosco_geometry_rho |   cosco_geometry_rho_minus_cosco |
|:-------------|-------:|------------:|---------:|---------------------:|---------------------------------:|
| RacketSports |      1 |          10 | 0.427632 |             0.434211 |                         0.006579 |
| RacketSports |      1 |          20 | 0.368421 |             0.401316 |                         0.032895 |
| RacketSports |      1 |          30 | 0.440789 |             0.447368 |                         0.006579 |
| RacketSports |     10 |          10 | 0.638158 |             0.638158 |                         0.000000 |
| RacketSports |     10 |          20 | 0.572368 |             0.572368 |                         0.000000 |
| RacketSports |     10 |          30 | 0.671053 |             0.664474 |                        -0.006579 |
