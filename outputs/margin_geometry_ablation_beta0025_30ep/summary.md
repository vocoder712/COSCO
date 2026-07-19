# Prototype-margin × geometry-rho multi-seed ablation

- device: CPU; torch `2.13.0+cpu`; threads: `4`
- epochs: `30`; seeds: `[10, 20, 30]`
- prototype margin: `0.0`; beta: `0.025`
- all model comparisons use paired initialization/shuffle seeds

## Three-seed mean and standard deviation

| dataset        |   shot | model                           |   accuracy_mean |   accuracy_std |   successful_runs |   elapsed_mean_sec |   rho_mean |   pressure_mean |   boundary_mean |   crowding_mean |   compactness_mean |   margin_loss_mean |   margin_active_rate |   margin_gap_mean |
|:---------------|-------:|:--------------------------------|----------------:|---------------:|------------------:|-------------------:|-----------:|----------------:|----------------:|----------------:|-------------------:|-------------------:|---------------------:|------------------:|
| Heartbeat      |     10 | cosco_proto_margin              |        0.725203 |       0.005633 |                 3 |          28.960000 | nan        |      nan        |      nan        |      nan        |         nan        |           0.421127 |             0.384444 |          0.323313 |
| Heartbeat      |     10 | cosco_proto_margin_geometry_rho |        0.730081 |       0.007451 |                 3 |          28.320000 |   0.091547 |        0.551682 |        0.578593 |        0.402483 |           0.753893 |           0.422346 |             0.385556 |          0.338785 |
| JapaneseVowels |     10 | cosco_proto_margin              |        0.909009 |       0.016514 |                 3 |           9.456667 | nan        |      nan        |      nan        |      nan        |         nan        |           0.013078 |             0.020617 |          5.224957 |
| JapaneseVowels |     10 | cosco_proto_margin_geometry_rho |        0.906306 |       0.013869 |                 3 |           9.210000 |   0.104259 |        0.381188 |        0.126629 |        0.388714 |           0.410257 |           0.008908 |             0.016543 |          5.435639 |
| RacketSports   |     10 | cosco_proto_margin              |        0.793860 |       0.021148 |                 3 |           4.923333 | nan        |      nan        |      nan        |      nan        |         nan        |           0.070519 |             0.081389 |          2.327259 |
| RacketSports   |     10 | cosco_proto_margin_geometry_rho |        0.813596 |       0.021148 |                 3 |           4.906667 |   0.097084 |        0.475103 |        0.212766 |        0.445439 |           0.450438 |           0.068776 |             0.071389 |          2.862940 |

## Paired per-seed deltas

| dataset        |   shot |   base_seed |   cosco_proto_margin |   cosco_proto_margin_geometry_rho |
|:---------------|-------:|------------:|---------------------:|----------------------------------:|
| Heartbeat      |     10 |          10 |             0.731707 |                          0.736585 |
| Heartbeat      |     10 |          20 |             0.721951 |                          0.731707 |
| Heartbeat      |     10 |          30 |             0.721951 |                          0.721951 |
| JapaneseVowels |     10 |          10 |             0.894595 |                          0.894595 |
| JapaneseVowels |     10 |          20 |             0.905405 |                          0.902703 |
| JapaneseVowels |     10 |          30 |             0.927027 |                          0.921622 |
| RacketSports   |     10 |          10 |             0.769737 |                          0.789474 |
| RacketSports   |     10 |          20 |             0.809211 |                          0.828947 |
| RacketSports   |     10 |          30 |             0.802632 |                          0.822368 |
