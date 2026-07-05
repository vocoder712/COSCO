# Quick COSCO Benchmark

- torch: `2.5.1+cu124` | device: `cuda (NVIDIA GeForce RTX 4060 Laptop GPU)`
- epochs for neural models: 100
- datasets: `SpokenArabicDigits, RacketSports, Heartbeat, JapaneseVowels, Libras`
- shots: `1, 10`
- base seed: `10`
- deterministic torch: `False`
- COSCO variant: `original`

- weighted prototype gamma: `1.0`
- weighted prototype distance mode: `close`

- dynamic rho alpha: `0.5`
- dynamic rho min ratio: `0.5`
- dynamic rho max ratio: `1.3`

|                            |   cosco |   cosco_dynamic_rho |   dynamic_minus_cosco |
|:---------------------------|--------:|--------------------:|----------------------:|
| ('Heartbeat', 1)           |  0.7268 |              0.7268 |                0.0000 |
| ('Heartbeat', 10)          |  0.6537 |              0.6341 |               -0.0195 |
| ('JapaneseVowels', 1)      |  0.6676 |              0.6676 |                0.0000 |
| ('JapaneseVowels', 10)     |  0.9027 |              0.9162 |                0.0135 |
| ('Libras', 1)              |  0.4222 |              0.4167 |               -0.0056 |
| ('Libras', 10)             |  0.8667 |              0.8500 |               -0.0167 |
| ('RacketSports', 1)        |  0.4934 |              0.4803 |               -0.0132 |
| ('RacketSports', 10)       |  0.7961 |              0.8026 |                0.0066 |
| ('SpokenArabicDigits', 1)  |  0.3356 |              0.3415 |                0.0059 |
| ('SpokenArabicDigits', 10) |  0.7549 |              0.7490 |               -0.0059 |

## Dynamic rho effect

| dataset            |   shot |    cosco |   cosco_dynamic_rho |   dynamic_minus_cosco |   rho_min |   rho_mean |   rho_max |   rho_final |   proto_stress_mean |   proto_stress_final |   rho_mean_ratio |   rho_max_ratio |
|:-------------------|-------:|---------:|--------------------:|----------------------:|----------:|-----------:|----------:|------------:|--------------------:|---------------------:|-----------------:|----------------:|
| Heartbeat          |      1 | 0.726829 |            0.726829 |              0.000000 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |         1.000000 |        1.000000 |
| Heartbeat          |     10 | 0.653659 |            0.634146 |             -0.019512 |  0.114100 |   0.124570 |  0.130000 |    0.114100 |            0.620626 |             0.282007 |         1.245699 |        1.300000 |
| JapaneseVowels     |      1 | 0.667568 |            0.667568 |              0.000000 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |         1.000000 |        1.000000 |
| JapaneseVowels     |     10 | 0.902703 |            0.916216 |              0.013514 |  0.110466 |   0.114858 |  0.130000 |    0.110466 |            0.302755 |             0.209318 |         1.148581 |        1.300000 |
| Libras             |      1 | 0.422222 |            0.416667 |             -0.005556 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |         1.000000 |        1.000000 |
| Libras             |     10 | 0.866667 |            0.850000 |             -0.016667 |  0.118130 |   0.125466 |  0.130000 |    0.118130 |            0.604552 |             0.362600 |         1.254657 |        1.300000 |
| RacketSports       |      1 | 0.493421 |            0.480263 |             -0.013158 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |         1.000000 |        1.000000 |
| RacketSports       |     10 | 0.796053 |            0.802632 |              0.006579 |  0.108557 |   0.114167 |  0.130000 |    0.108557 |            0.305494 |             0.171142 |         1.141666 |        1.300000 |
| SpokenArabicDigits |      1 | 0.335607 |            0.341519 |              0.005912 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |         1.000000 |        1.000000 |
| SpokenArabicDigits |     10 | 0.754889 |            0.748977 |             -0.005912 |  0.112909 |   0.119366 |  0.130000 |    0.112909 |            0.427288 |             0.258177 |         1.193661 |        1.300000 |

## Dynamic rho summary

| dataset            |   shot |   rho_min |   rho_mean |   rho_max |   rho_final |   proto_stress_mean |   proto_stress_final |
|:-------------------|-------:|----------:|-----------:|----------:|------------:|--------------------:|---------------------:|
| SpokenArabicDigits |      1 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |
| SpokenArabicDigits |     10 |  0.112909 |   0.119366 |  0.130000 |    0.112909 |            0.427288 |             0.258177 |
| RacketSports       |      1 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |
| RacketSports       |     10 |  0.108557 |   0.114167 |  0.130000 |    0.108557 |            0.305494 |             0.171142 |
| Heartbeat          |      1 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |
| Heartbeat          |     10 |  0.114100 |   0.124570 |  0.130000 |    0.114100 |            0.620626 |             0.282007 |
| JapaneseVowels     |      1 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |
| JapaneseVowels     |     10 |  0.110466 |   0.114858 |  0.130000 |    0.110466 |            0.302755 |             0.209318 |
| Libras             |      1 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |
| Libras             |     10 |  0.118130 |   0.125466 |  0.130000 |    0.118130 |            0.604552 |             0.362600 |

## Full rows

| model             | dataset            |   shot |   accuracy |   elapsed_sec |   seed | status   | cosco_variant                        |   rho_min |   rho_mean |   rho_max |   rho_final |   proto_stress_mean |   proto_stress_final |
|:------------------|:-------------------|-------:|-----------:|--------------:|-------:|:---------|:-------------------------------------|----------:|-----------:|----------:|------------:|--------------------:|---------------------:|
| cosco             | SpokenArabicDigits |      1 |     0.3356 |        2.1000 |  19509 | ok       |                                      |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | SpokenArabicDigits |      1 |     0.3415 |        1.9000 |  19509 | ok       | dynamic_rho_proto_geometry_alpha=0.5 |    0.1000 |     0.1000 |    0.1000 |      0.1000 |              0.0000 |               0.0000 |
| cosco             | SpokenArabicDigits |     10 |     0.7549 |        2.2000 |  20517 | ok       |                                      |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | SpokenArabicDigits |     10 |     0.7490 |        2.6000 |  20517 | ok       | dynamic_rho_proto_geometry_alpha=0.5 |    0.1129 |     0.1194 |    0.1300 |      0.1129 |              0.4273 |               0.2582 |
| cosco             | RacketSports       |      1 |     0.4934 |        1.2000 |   9924 | ok       |                                      |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | RacketSports       |      1 |     0.4803 |        1.3000 |   9924 | ok       | dynamic_rho_proto_geometry_alpha=0.5 |    0.1000 |     0.1000 |    0.1000 |      0.1000 |              0.0000 |               0.0000 |
| cosco             | RacketSports       |     10 |     0.7961 |        1.3000 |  10644 | ok       |                                      |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | RacketSports       |     10 |     0.8026 |        1.4000 |  10644 | ok       | dynamic_rho_proto_geometry_alpha=0.5 |    0.1086 |     0.1142 |    0.1300 |      0.1086 |              0.3055 |               0.1711 |
| cosco             | Heartbeat          |      1 |     0.7268 |        1.1000 |   5845 | ok       |                                      |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | Heartbeat          |      1 |     0.7268 |        1.2000 |   5845 | ok       | dynamic_rho_proto_geometry_alpha=0.5 |    0.1000 |     0.1000 |    0.1000 |      0.1000 |              0.0000 |               0.0000 |
| cosco             | Heartbeat          |     10 |     0.6537 |        1.8000 |   6421 | ok       |                                      |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | Heartbeat          |     10 |     0.6341 |        2.2000 |   6421 | ok       | dynamic_rho_proto_geometry_alpha=0.5 |    0.1141 |     0.1246 |    0.1300 |      0.1141 |              0.6206 |               0.2820 |
| cosco             | JapaneseVowels     |      1 |     0.6676 |        1.7000 |  12844 | ok       |                                      |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | JapaneseVowels     |      1 |     0.6676 |        1.9000 |  12844 | ok       | dynamic_rho_proto_geometry_alpha=0.5 |    0.1000 |     0.1000 |    0.1000 |      0.1000 |              0.0000 |               0.0000 |
| cosco             | JapaneseVowels     |     10 |     0.9027 |        1.8000 |  13660 | ok       |                                      |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | JapaneseVowels     |     10 |     0.9162 |        2.0000 |  13660 | ok       | dynamic_rho_proto_geometry_alpha=0.5 |    0.1105 |     0.1149 |    0.1300 |      0.1105 |              0.3028 |               0.2093 |
| cosco             | Libras             |      1 |     0.4222 |        2.1000 |   3019 | ok       |                                      |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | Libras             |      1 |     0.4167 |        2.4000 |   3019 | ok       | dynamic_rho_proto_geometry_alpha=0.5 |    0.1000 |     0.1000 |    0.1000 |      0.1000 |              0.0000 |               0.0000 |
| cosco             | Libras             |     10 |     0.8667 |        2.6000 |   3451 | ok       |                                      |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | Libras             |     10 |     0.8500 |        3.1000 |   3451 | ok       | dynamic_rho_proto_geometry_alpha=0.5 |    0.1181 |     0.1255 |    0.1300 |      0.1181 |              0.6046 |               0.3626 |
