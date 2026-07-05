# Quick COSCO Benchmark

- torch: `2.5.1+cu124` | device: `cuda (NVIDIA GeForce RTX 4060 Laptop GPU)`
- epochs for neural models: 1
- datasets: `BasicMotions`
- shots: `10`
- base seed: `10`
- deterministic torch: `False`
- COSCO variant: `original`

- weighted prototype gamma: `1.0`
- weighted prototype distance mode: `close`

- dynamic rho alpha: `0.5`
- dynamic rho min ratio: `0.5`
- dynamic rho max ratio: `1.3`

|                      |   cosco |   cosco_dynamic_rho |   dynamic_minus_cosco |
|:---------------------|--------:|--------------------:|----------------------:|
| ('BasicMotions', 10) |  1.0000 |              1.0000 |                0.0000 |

## Dynamic rho effect

| dataset      |   shot |    cosco |   cosco_dynamic_rho |   dynamic_minus_cosco |   rho_min |   rho_mean |   rho_max |   rho_final |   proto_stress_mean |   proto_stress_final |   rho_mean_ratio |   rho_max_ratio |
|:-------------|-------:|---------:|--------------------:|----------------------:|----------:|-----------:|----------:|------------:|--------------------:|---------------------:|-----------------:|----------------:|
| BasicMotions |     10 | 1.000000 |            1.000000 |              0.000000 |  0.122550 |   0.122550 |  0.122550 |    0.122550 |            0.450992 |             0.450992 |         1.225496 |        1.225496 |

## Dynamic rho summary

| dataset      |   shot |   rho_min |   rho_mean |   rho_max |   rho_final |   proto_stress_mean |   proto_stress_final |
|:-------------|-------:|----------:|-----------:|----------:|------------:|--------------------:|---------------------:|
| BasicMotions |     10 |  0.122550 |   0.122550 |  0.122550 |    0.122550 |            0.450992 |             0.450992 |

## Full rows

| model             | dataset      |   shot |   accuracy |   elapsed_sec |   seed | status   | cosco_variant                        |   rho_min |   rho_mean |   rho_max |   rho_final |   proto_stress_mean |   proto_stress_final |
|:------------------|:-------------|-------:|-----------:|--------------:|-------:|:---------|:-------------------------------------|----------:|-----------:|----------:|------------:|--------------------:|---------------------:|
| cosco             | BasicMotions |     10 |     1.0000 |        0.4000 |  10502 | ok       |                                      |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |
| cosco_dynamic_rho | BasicMotions |     10 |     1.0000 |        0.0000 |  10502 | ok       | dynamic_rho_proto_geometry_alpha=0.5 |    0.1225 |     0.1225 |    0.1225 |      0.1225 |              0.4510 |               0.4510 |
