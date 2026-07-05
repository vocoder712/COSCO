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

- dynamic rho alpha: `0.25`
- dynamic rho min ratio: `0.5`
- dynamic rho max ratio: `1.15`

|                            |   cosco |   cosco_dynamic_rho |   dynamic_minus_cosco |
|:---------------------------|--------:|--------------------:|----------------------:|
| ('Heartbeat', 1)           |  0.7268 |              0.7268 |                0.0000 |
| ('Heartbeat', 10)          |  0.6537 |              0.6537 |                0.0000 |
| ('JapaneseVowels', 1)      |  0.6676 |              0.6676 |                0.0000 |
| ('JapaneseVowels', 10)     |  0.9027 |              0.9000 |               -0.0027 |
| ('Libras', 1)              |  0.4222 |              0.4167 |               -0.0056 |
| ('Libras', 10)             |  0.8667 |              0.8611 |               -0.0056 |
| ('RacketSports', 1)        |  0.4803 |              0.5000 |                0.0197 |
| ('RacketSports', 10)       |  0.7961 |              0.8092 |                0.0132 |
| ('SpokenArabicDigits', 1)  |  0.3415 |              0.3392 |               -0.0023 |
| ('SpokenArabicDigits', 10) |  0.7549 |              0.7567 |                0.0018 |

## Dynamic rho effect

| dataset            |   shot |    cosco |   cosco_dynamic_rho |   dynamic_minus_cosco |   rho_min |   rho_mean |   rho_max |   rho_final |   proto_stress_mean |   proto_stress_final |   rho_mean_ratio |   rho_max_ratio |
|:-------------------|-------:|---------:|--------------------:|----------------------:|----------:|-----------:|----------:|------------:|--------------------:|---------------------:|-----------------:|----------------:|
| Heartbeat          |      1 | 0.726829 |            0.726829 |              0.000000 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |         1.000000 |        1.000000 |
| Heartbeat          |     10 | 0.653659 |            0.653659 |              0.000000 |  0.107806 |   0.112753 |  0.115000 |    0.107919 |            0.642828 |             0.316760 |         1.127528 |        1.150000 |
| JapaneseVowels     |      1 | 0.667568 |            0.667568 |              0.000000 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |         1.000000 |        1.000000 |
| JapaneseVowels     |     10 | 0.902703 |            0.900000 |             -0.002703 |  0.105830 |   0.107296 |  0.115000 |    0.105834 |            0.296829 |             0.233359 |         1.072964 |        1.150000 |
| Libras             |      1 | 0.422222 |            0.416667 |             -0.005556 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |         1.000000 |        1.000000 |
| Libras             |     10 | 0.866667 |            0.861111 |             -0.005556 |  0.108799 |   0.112546 |  0.115000 |    0.108799 |            0.589692 |             0.351940 |         1.125458 |        1.150000 |
| RacketSports       |      1 | 0.480263 |            0.500000 |              0.019737 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |         1.000000 |        1.000000 |
| RacketSports       |     10 | 0.796053 |            0.809211 |              0.013158 |  0.104254 |   0.106815 |  0.115000 |    0.104254 |            0.292085 |             0.170156 |         1.068147 |        1.150000 |
| SpokenArabicDigits |      1 | 0.341519 |            0.339245 |             -0.002274 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |         1.000000 |        1.000000 |
| SpokenArabicDigits |     10 | 0.754889 |            0.756708 |              0.001819 |  0.106383 |   0.109441 |  0.115000 |    0.106383 |            0.413975 |             0.255329 |         1.094409 |        1.150000 |

## Dynamic rho summary

| dataset            |   shot |   rho_min |   rho_mean |   rho_max |   rho_final |   proto_stress_mean |   proto_stress_final |
|:-------------------|-------:|----------:|-----------:|----------:|------------:|--------------------:|---------------------:|
| SpokenArabicDigits |      1 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |
| SpokenArabicDigits |     10 |  0.106383 |   0.109441 |  0.115000 |    0.106383 |            0.413975 |             0.255329 |
| RacketSports       |      1 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |
| RacketSports       |     10 |  0.104254 |   0.106815 |  0.115000 |    0.104254 |            0.292085 |             0.170156 |
| Heartbeat          |      1 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |
| Heartbeat          |     10 |  0.107806 |   0.112753 |  0.115000 |    0.107919 |            0.642828 |             0.316760 |
| JapaneseVowels     |      1 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |
| JapaneseVowels     |     10 |  0.105830 |   0.107296 |  0.115000 |    0.105834 |            0.296829 |             0.233359 |
| Libras             |      1 |  0.100000 |   0.100000 |  0.100000 |    0.100000 |            0.000000 |             0.000000 |
| Libras             |     10 |  0.108799 |   0.112546 |  0.115000 |    0.108799 |            0.589692 |             0.351940 |

## Full rows

| model             | dataset            |   shot |   accuracy |   elapsed_sec |   seed | status   | cosco_variant                         |   rho_min |   rho_mean |   rho_max |   rho_final |   proto_stress_mean |   proto_stress_final |
|:------------------|:-------------------|-------:|-----------:|--------------:|-------:|:---------|:--------------------------------------|----------:|-----------:|----------:|------------:|--------------------:|---------------------:|
| cosco             | SpokenArabicDigits |      1 |     0.3415 |        2.1000 |  19509 | ok       |                                       |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | SpokenArabicDigits |      1 |     0.3392 |        2.0000 |  19509 | ok       | dynamic_rho_proto_geometry_alpha=0.25 |    0.1000 |     0.1000 |    0.1000 |      0.1000 |              0.0000 |               0.0000 |
| cosco             | SpokenArabicDigits |     10 |     0.7549 |        2.3000 |  20517 | ok       |                                       |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | SpokenArabicDigits |     10 |     0.7567 |        2.7000 |  20517 | ok       | dynamic_rho_proto_geometry_alpha=0.25 |    0.1064 |     0.1094 |    0.1150 |      0.1064 |              0.4140 |               0.2553 |
| cosco             | RacketSports       |      1 |     0.4803 |        1.3000 |   9924 | ok       |                                       |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | RacketSports       |      1 |     0.5000 |        1.4000 |   9924 | ok       | dynamic_rho_proto_geometry_alpha=0.25 |    0.1000 |     0.1000 |    0.1000 |      0.1000 |              0.0000 |               0.0000 |
| cosco             | RacketSports       |     10 |     0.7961 |        1.5000 |  10644 | ok       |                                       |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | RacketSports       |     10 |     0.8092 |        1.7000 |  10644 | ok       | dynamic_rho_proto_geometry_alpha=0.25 |    0.1043 |     0.1068 |    0.1150 |      0.1043 |              0.2921 |               0.1702 |
| cosco             | Heartbeat          |      1 |     0.7268 |        1.2000 |   5845 | ok       |                                       |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | Heartbeat          |      1 |     0.7268 |        1.4000 |   5845 | ok       | dynamic_rho_proto_geometry_alpha=0.25 |    0.1000 |     0.1000 |    0.1000 |      0.1000 |              0.0000 |               0.0000 |
| cosco             | Heartbeat          |     10 |     0.6537 |        1.8000 |   6421 | ok       |                                       |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | Heartbeat          |     10 |     0.6537 |        2.0000 |   6421 | ok       | dynamic_rho_proto_geometry_alpha=0.25 |    0.1078 |     0.1128 |    0.1150 |      0.1079 |              0.6428 |               0.3168 |
| cosco             | JapaneseVowels     |      1 |     0.6676 |        1.4000 |  12844 | ok       |                                       |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | JapaneseVowels     |      1 |     0.6676 |        1.7000 |  12844 | ok       | dynamic_rho_proto_geometry_alpha=0.25 |    0.1000 |     0.1000 |    0.1000 |      0.1000 |              0.0000 |               0.0000 |
| cosco             | JapaneseVowels     |     10 |     0.9027 |        1.6000 |  13660 | ok       |                                       |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | JapaneseVowels     |     10 |     0.9000 |        1.8000 |  13660 | ok       | dynamic_rho_proto_geometry_alpha=0.25 |    0.1058 |     0.1073 |    0.1150 |      0.1058 |              0.2968 |               0.2334 |
| cosco             | Libras             |      1 |     0.4222 |        1.8000 |   3019 | ok       |                                       |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | Libras             |      1 |     0.4167 |        2.2000 |   3019 | ok       | dynamic_rho_proto_geometry_alpha=0.25 |    0.1000 |     0.1000 |    0.1000 |      0.1000 |              0.0000 |               0.0000 |
| cosco             | Libras             |     10 |     0.8667 |        2.5000 |   3451 | ok       |                                       |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | Libras             |     10 |     0.8611 |        2.8000 |   3451 | ok       | dynamic_rho_proto_geometry_alpha=0.25 |    0.1088 |     0.1125 |    0.1150 |      0.1088 |              0.5897 |               0.3519 |
