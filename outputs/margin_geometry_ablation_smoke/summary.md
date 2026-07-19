# Prototype-margin × geometry-rho multi-seed ablation

- device: CPU; torch `2.13.0+cpu`; threads: `4`
- epochs: `1`; seeds: `[10, 20, 30]`
- prototype margin: `0.0`; beta: `0.05`
- all model comparisons use paired initialization/shuffle seeds

## Three-seed mean and standard deviation

| dataset      |   shot | model                           |   accuracy_mean |   accuracy_std |   successful_runs |   elapsed_mean_sec |   rho_mean |   pressure_mean |   boundary_mean |   crowding_mean |   compactness_mean |   margin_loss_mean |   margin_active_rate |   margin_gap_mean |
|:-------------|-------:|:--------------------------------|----------------:|---------------:|------------------:|-------------------:|-----------:|----------------:|----------------:|----------------:|-------------------:|-------------------:|---------------------:|------------------:|
| BasicMotions |      1 | cosco                           |        1.000000 |       0.000000 |                 3 |           0.136667 | nan        |      nan        |      nan        |      nan        |         nan        |         nan        |           nan        |        nan        |
| BasicMotions |      1 | cosco_geometry_rho              |        1.000000 |       0.000000 |                 3 |           0.113333 |   0.080275 |        0.640401 |        0.000000 |        0.640401 |           0.000000 |         nan        |           nan        |        nan        |
| BasicMotions |      1 | cosco_proto_margin              |        1.000000 |       0.000000 |                 3 |           0.110000 | nan        |      nan        |      nan        |      nan        |         nan        |           0.000000 |             0.000000 |          3.136668 |
| BasicMotions |      1 | cosco_proto_margin_geometry_rho |        1.000000 |       0.000000 |                 3 |          55.066667 |   0.080275 |        0.640401 |        0.000000 |        0.640401 |           0.000000 |           0.000000 |             0.000000 |          3.136668 |
| BasicMotions |     10 | cosco                           |        1.000000 |       0.000000 |                 3 |           0.796667 | nan        |      nan        |      nan        |      nan        |         nan        |         nan        |           nan        |        nan        |
| BasicMotions |     10 | cosco_geometry_rho              |        1.000000 |       0.000000 |                 3 |           0.560000 |   0.112800 |        0.347417 |        0.126512 |        0.681620 |           0.398374 |         nan        |           nan        |        nan        |
| BasicMotions |     10 | cosco_proto_margin              |        1.000000 |       0.000000 |                 3 |           0.636667 | nan        |      nan        |      nan        |      nan        |         nan        |           0.015700 |             0.033333 |          1.548221 |
| BasicMotions |     10 | cosco_proto_margin_geometry_rho |        1.000000 |       0.000000 |                 3 |           0.523333 |   0.112800 |        0.347417 |        0.126512 |        0.681620 |           0.398374 |           0.015700 |             0.033333 |          1.548221 |

## Mean ablation effects

| dataset      |   shot |   cosco_proto_margin_minus_cosco |   cosco_geometry_rho_minus_cosco |   cosco_proto_margin_geometry_rho_minus_cosco |   combination_synergy |   combination_minus_best_component |
|:-------------|-------:|---------------------------------:|---------------------------------:|----------------------------------------------:|----------------------:|-----------------------------------:|
| BasicMotions |      1 |                         0.000000 |                         0.000000 |                                      0.000000 |              0.000000 |                           0.000000 |
| BasicMotions |     10 |                         0.000000 |                         0.000000 |                                      0.000000 |              0.000000 |                           0.000000 |

## Paired per-seed deltas

| dataset      |   shot |   base_seed |    cosco |   cosco_geometry_rho |   cosco_proto_margin |   cosco_proto_margin_geometry_rho |   cosco_proto_margin_minus_cosco |   cosco_geometry_rho_minus_cosco |   cosco_proto_margin_geometry_rho_minus_cosco |   combination_synergy |   combination_minus_best_component |
|:-------------|-------:|------------:|---------:|---------------------:|---------------------:|----------------------------------:|---------------------------------:|---------------------------------:|----------------------------------------------:|----------------------:|-----------------------------------:|
| BasicMotions |      1 |          10 | 1.000000 |             1.000000 |             1.000000 |                          1.000000 |                         0.000000 |                         0.000000 |                                      0.000000 |              0.000000 |                           0.000000 |
| BasicMotions |      1 |          20 | 1.000000 |             1.000000 |             1.000000 |                          1.000000 |                         0.000000 |                         0.000000 |                                      0.000000 |              0.000000 |                           0.000000 |
| BasicMotions |      1 |          30 | 1.000000 |             1.000000 |             1.000000 |                          1.000000 |                         0.000000 |                         0.000000 |                                      0.000000 |              0.000000 |                           0.000000 |
| BasicMotions |     10 |          10 | 1.000000 |             1.000000 |             1.000000 |                          1.000000 |                         0.000000 |                         0.000000 |                                      0.000000 |              0.000000 |                           0.000000 |
| BasicMotions |     10 |          20 | 1.000000 |             1.000000 |             1.000000 |                          1.000000 |                         0.000000 |                         0.000000 |                                      0.000000 |              0.000000 |                           0.000000 |
| BasicMotions |     10 |          30 | 1.000000 |             1.000000 |             1.000000 |                          1.000000 |                         0.000000 |                         0.000000 |                                      0.000000 |              0.000000 |                           0.000000 |
