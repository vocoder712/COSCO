# Quick COSCO Benchmark

- torch: `2.5.1+cu124` | device: `cuda (NVIDIA GeForce RTX 4060 Laptop GPU)`
- epochs for neural models: 100
- datasets: `SpokenArabicDigits, RacketSports`
- shots: `1, 10`
- COSCO variant: `original`

- weighted prototype gamma: `1.0`
- weighted prototype distance mode: `close`

|                            |   cosco |   cosco_weighted |
|:---------------------------|--------:|-----------------:|
| ('RacketSports', 1)        |  0.4474 |           0.3947 |
| ('RacketSports', 10)       |  0.8158 |           0.8355 |
| ('SpokenArabicDigits', 1)  |  0.3815 |           0.3734 |
| ('SpokenArabicDigits', 10) |  0.7603 |           0.7735 |

## Full rows

| model          | dataset            |   shot |   accuracy |   elapsed_sec | status   | cosco_variant            |
|:---------------|:-------------------|-------:|-----------:|--------------:|:---------|:-------------------------|
| cosco          | SpokenArabicDigits |      1 |     0.3815 |        2.1000 | ok       |                          |
| cosco_weighted | SpokenArabicDigits |      1 |     0.3734 |        2.1000 | ok       | weighted_close_gamma=1.0 |
| cosco          | SpokenArabicDigits |     10 |     0.7603 |        2.3000 | ok       |                          |
| cosco_weighted | SpokenArabicDigits |     10 |     0.7735 |        2.7000 | ok       | weighted_close_gamma=1.0 |
| cosco          | RacketSports       |      1 |     0.4474 |        1.3000 | ok       |                          |
| cosco_weighted | RacketSports       |      1 |     0.3947 |        1.5000 | ok       | weighted_close_gamma=1.0 |
| cosco          | RacketSports       |     10 |     0.8158 |        1.3000 | ok       |                          |
| cosco_weighted | RacketSports       |     10 |     0.8355 |        1.5000 | ok       | weighted_close_gamma=1.0 |
