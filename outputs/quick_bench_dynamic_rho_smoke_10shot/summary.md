# Quick COSCO Benchmark

- torch: `2.5.1+cu124` | device: `cuda (NVIDIA GeForce RTX 4060 Laptop GPU)`
- epochs for neural models: 2
- datasets: `BasicMotions`
- shots: `10`
- base seed: `10`
- deterministic torch: `False`
- COSCO variant: `original`

- weighted prototype gamma: `1.0`
- weighted prototype distance mode: `close`

- dynamic rho alpha: `1.0`
- dynamic rho min ratio: `0.5`
- dynamic rho max ratio: `2.0`

|                      |   cosco |   cosco_dynamic_rho |
|:---------------------|--------:|--------------------:|
| ('BasicMotions', 10) |  1.0000 |              1.0000 |

## Full rows

| model             | dataset      |   shot |   accuracy |   elapsed_sec |   seed | status   | cosco_variant                        |   rho_min |   rho_mean |   rho_max |   rho_final |   proto_stress_mean |   proto_stress_final |
|:------------------|:-------------|-------:|-----------:|--------------:|-------:|:---------|:-------------------------------------|----------:|-----------:|----------:|------------:|--------------------:|---------------------:|
| cosco             | BasicMotions |     10 |     1.0000 |        0.5000 |  10502 | ok       |                                      |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | BasicMotions |     10 |     1.0000 |        0.1000 |  10502 | ok       | dynamic_rho_proto_geometry_alpha=1.0 |    0.1451 |     0.1455 |    0.1458 |      0.1458 |              0.4546 |               0.4582 |
