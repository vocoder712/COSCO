# Dynamic rho multi-seed benchmark

- device: CPU; torch `2.13.0+cpu`; threads: `4`
- epochs: `30`; seeds: `[10, 20, 30]`
- all model comparisons use paired initialization/shuffle seeds

## Three-seed mean and standard deviation

| dataset        |   shot | model              |   accuracy_mean |   accuracy_std |   successful_runs |   elapsed_mean_sec |   rho_mean |   pressure_mean |   boundary_mean |   crowding_mean |   compactness_mean |
|:---------------|-------:|:-------------------|----------------:|---------------:|------------------:|-------------------:|-----------:|----------------:|----------------:|----------------:|-------------------:|
| JapaneseVowels |      1 | cosco              |        0.709009 |       0.051729 |                 3 |           2.333333 | nan        |      nan        |      nan        |      nan        |         nan        |
| JapaneseVowels |      1 | cosco_geometry_rho |        0.718919 |       0.067891 |                 3 |           2.666667 |   0.108026 |        0.405294 |        0.000000 |        0.379202 |           0.000000 |
| JapaneseVowels |     10 | cosco              |        0.907207 |       0.016291 |                 3 |          12.440000 | nan        |      nan        |      nan        |      nan        |         nan        |
| JapaneseVowels |     10 | cosco_geometry_rho |        0.902703 |       0.022123 |                 3 |          10.966667 |   0.104185 |        0.383222 |        0.132155 |        0.387058 |           0.412016 |

## Paired per-seed deltas

| dataset        |   shot |   base_seed |    cosco |   cosco_geometry_rho |   cosco_geometry_rho_minus_cosco |
|:---------------|-------:|------------:|---------:|---------------------:|---------------------------------:|
| JapaneseVowels |      1 |          10 | 0.654054 |             0.640541 |                        -0.013514 |
| JapaneseVowels |      1 |          20 | 0.716216 |             0.756757 |                         0.040541 |
| JapaneseVowels |      1 |          30 | 0.756757 |             0.759459 |                         0.002703 |
| JapaneseVowels |     10 |          10 | 0.891892 |             0.883784 |                        -0.008108 |
| JapaneseVowels |     10 |          20 | 0.905405 |             0.897297 |                        -0.008108 |
| JapaneseVowels |     10 |          30 | 0.924324 |             0.927027 |                         0.002703 |
