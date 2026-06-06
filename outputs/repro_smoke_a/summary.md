# Quick COSCO Benchmark

- torch: `2.5.1+cu124` | device: `cuda (NVIDIA GeForce RTX 4060 Laptop GPU)`
- epochs for neural models: 1
- datasets: `BasicMotions`
- shots: `1`
- base seed: `123`
- deterministic torch: `False`
- COSCO variant: `original`

- weighted prototype gamma: `1.0`
- weighted prototype distance mode: `close`

|                     |   cosco |   cosco_weighted |
|:--------------------|--------:|-----------------:|
| ('BasicMotions', 1) |  1.0000 |           1.0000 |

## Full rows

| model          | dataset      |   shot |   accuracy |   elapsed_sec |   seed | status   | cosco_variant            |
|:---------------|:-------------|-------:|-----------:|--------------:|-------:|:---------|:-------------------------|
| cosco          | BasicMotions |      1 |     1.0000 |        0.4000 |   9895 | ok       |                          |
| cosco_weighted | BasicMotions |      1 |     1.0000 |        0.0000 |   9895 | ok       | weighted_close_gamma=1.0 |
