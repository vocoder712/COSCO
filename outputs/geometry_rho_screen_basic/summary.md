# Dynamic rho multi-seed benchmark

- device: CPU; torch `2.13.0+cpu`; threads: `4`
- epochs: `5`; seeds: `[10, 20, 30]`
- all model comparisons use paired initialization/shuffle seeds

## Three-seed mean and standard deviation

| dataset      |   shot | model              |   accuracy_mean |   accuracy_std |   successful_runs |   elapsed_mean_sec |   rho_mean |   pressure_mean |   boundary_mean |   crowding_mean |   compactness_mean |
|:-------------|-------:|:-------------------|----------------:|---------------:|------------------:|-------------------:|-----------:|----------------:|----------------:|----------------:|-------------------:|
| BasicMotions |      1 | cosco              |        1.000000 |       0.000000 |                 3 |           0.360000 | nan        |      nan        |      nan        |      nan        |         nan        |
| BasicMotions |      1 | cosco_dynamic_rho  |        1.000000 |       0.000000 |                 3 |           0.383333 |   0.100000 |        0.000000 |      nan        |      nan        |         nan        |
| BasicMotions |      1 | cosco_geometry_rho |        1.000000 |       0.000000 |                 3 |           0.413333 |   0.115000 |        0.633527 |        0.000000 |        0.599144 |           0.000000 |
| BasicMotions |     10 | cosco              |        1.000000 |       0.000000 |                 3 |           2.026667 | nan        |      nan        |      nan        |      nan        |         nan        |
| BasicMotions |     10 | cosco_dynamic_rho  |        1.000000 |       0.000000 |                 3 |           2.033333 |   0.109559 |        0.382370 |      nan        |      nan        |         nan        |
| BasicMotions |     10 | cosco_geometry_rho |        1.000000 |       0.000000 |                 3 |           2.053333 |   0.115000 |        0.340648 |        0.091701 |        0.628485 |           0.347512 |

## Paired per-seed deltas

| dataset      |   shot |   base_seed |    cosco |   cosco_dynamic_rho |   cosco_geometry_rho |   cosco_dynamic_rho_minus_cosco |   cosco_geometry_rho_minus_cosco |
|:-------------|-------:|------------:|---------:|--------------------:|---------------------:|--------------------------------:|---------------------------------:|
| BasicMotions |      1 |          10 | 1.000000 |            1.000000 |             1.000000 |                        0.000000 |                         0.000000 |
| BasicMotions |      1 |          20 | 1.000000 |            1.000000 |             1.000000 |                        0.000000 |                         0.000000 |
| BasicMotions |      1 |          30 | 1.000000 |            1.000000 |             1.000000 |                        0.000000 |                         0.000000 |
| BasicMotions |     10 |          10 | 1.000000 |            1.000000 |             1.000000 |                        0.000000 |                         0.000000 |
| BasicMotions |     10 |          20 | 1.000000 |            1.000000 |             1.000000 |                        0.000000 |                         0.000000 |
| BasicMotions |     10 |          30 | 1.000000 |            1.000000 |             1.000000 |                        0.000000 |                         0.000000 |
