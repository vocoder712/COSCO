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

|                            |   cosco |   cosco_weighted |
|:---------------------------|--------:|-----------------:|
| ('Heartbeat', 1)           |  0.7268 |           0.7268 |
| ('Heartbeat', 10)          |  0.6537 |           0.6488 |
| ('JapaneseVowels', 1)      |  0.6676 |           0.6676 |
| ('JapaneseVowels', 10)     |  0.9027 |           0.9000 |
| ('Libras', 1)              |  0.4222 |           0.4222 |
| ('Libras', 10)             |  0.8667 |           0.8111 |
| ('RacketSports', 1)        |  0.5000 |           0.4934 |
| ('RacketSports', 10)       |  0.7961 |           0.7961 |
| ('SpokenArabicDigits', 1)  |  0.3420 |           0.3338 |
| ('SpokenArabicDigits', 10) |  0.7549 |           0.7240 |

## Full rows

| model          | dataset            |   shot |   accuracy |   elapsed_sec |   seed | status   | cosco_variant            |
|:---------------|:-------------------|-------:|-----------:|--------------:|-------:|:---------|:-------------------------|
| cosco          | SpokenArabicDigits |      1 |     0.3420 |        3.9000 |  19509 | ok       |                          |
| cosco_weighted | SpokenArabicDigits |      1 |     0.3338 |        2.1000 |  19509 | ok       | weighted_close_gamma=1.0 |
| cosco          | SpokenArabicDigits |     10 |     0.7549 |        2.2000 |  20517 | ok       |                          |
| cosco_weighted | SpokenArabicDigits |     10 |     0.7240 |        2.7000 |  20517 | ok       | weighted_close_gamma=1.0 |
| cosco          | RacketSports       |      1 |     0.5000 |        1.3000 |   9924 | ok       |                          |
| cosco_weighted | RacketSports       |      1 |     0.4934 |        1.6000 |   9924 | ok       | weighted_close_gamma=1.0 |
| cosco          | RacketSports       |     10 |     0.7961 |        1.6000 |  10644 | ok       |                          |
| cosco_weighted | RacketSports       |     10 |     0.7961 |        1.7000 |  10644 | ok       | weighted_close_gamma=1.0 |
| cosco          | Heartbeat          |      1 |     0.7268 |        1.3000 |   5845 | ok       |                          |
| cosco_weighted | Heartbeat          |      1 |     0.7268 |        1.4000 |   5845 | ok       | weighted_close_gamma=1.0 |
| cosco          | Heartbeat          |     10 |     0.6537 |        1.9000 |   6421 | ok       |                          |
| cosco_weighted | Heartbeat          |     10 |     0.6488 |        2.0000 |   6421 | ok       | weighted_close_gamma=1.0 |
| cosco          | JapaneseVowels     |      1 |     0.6676 |        1.7000 |  12844 | ok       |                          |
| cosco_weighted | JapaneseVowels     |      1 |     0.6676 |        2.0000 |  12844 | ok       | weighted_close_gamma=1.0 |
| cosco          | JapaneseVowels     |     10 |     0.9027 |        1.9000 |  13660 | ok       |                          |
| cosco_weighted | JapaneseVowels     |     10 |     0.9000 |        2.3000 |  13660 | ok       | weighted_close_gamma=1.0 |
| cosco          | Libras             |      1 |     0.4222 |        2.0000 |   3019 | ok       |                          |
| cosco_weighted | Libras             |      1 |     0.4222 |        2.7000 |   3019 | ok       | weighted_close_gamma=1.0 |
| cosco          | Libras             |     10 |     0.8667 |        2.7000 |   3451 | ok       |                          |
| cosco_weighted | Libras             |     10 |     0.8111 |        3.1000 |   3451 | ok       | weighted_close_gamma=1.0 |
