# Quick COSCO Benchmark

- torch: `2.5.1+cu124` | device: `cuda (NVIDIA GeForce RTX 4060 Laptop GPU)`
- epochs for neural models: 2
- datasets: `BasicMotions`
- shots: `1`
- base seed: `10`
- deterministic torch: `False`
- COSCO variant: `original`

- weighted prototype gamma: `1.0`
- weighted prototype distance mode: `close`

- dynamic rho alpha: `0.25`
- dynamic rho min ratio: `0.5`
- dynamic rho max ratio: `1.15`

- FFT regularization lambdas: `0.1, 0.3, 0.5`

- prototype margin values: `0.0, 0.1`
- prototype margin betas: `0.1`

|                     |   cosco |   cosco_proto_margin_m0.1_b0.1 |   cosco_proto_margin_m0_b0.1 |
|:--------------------|--------:|-------------------------------:|-----------------------------:|
| ('BasicMotions', 1) |  1.0000 |                         1.0000 |                       1.0000 |

## Prototype margin effect by hyperparameter

|   proto_margin_value |   proto_margin_beta |   cosco_mean |   cosco_proto_margin_mean |   mean_delta |   wins |   ties |   losses |   base_loss_mean |   margin_loss_mean |   total_loss_mean |   positive_rate_mean |   gap_mean | gate_pass_mean_delta   |
|---------------------:|--------------------:|-------------:|--------------------------:|-------------:|-------:|-------:|---------:|-----------------:|-------------------:|------------------:|---------------------:|-----------:|:-----------------------|
|             0.000000 |            0.100000 |     1.000000 |                  1.000000 |     0.000000 |      0 |      1 |        0 |         0.795321 |           0.000000 |          0.795321 |             0.000000 |   3.154173 | False                  |
|             0.100000 |            0.100000 |     1.000000 |                  1.000000 |     0.000000 |      0 |      1 |        0 |         0.795321 |           0.000000 |          0.795321 |             0.000000 |   3.154173 | False                  |

## Prototype margin effect

| dataset      |   shot |   cosco_proto_margin |   proto_margin_value |   proto_margin_beta |   proto_margin_base_loss_mean |   proto_margin_loss_mean |   proto_margin_total_loss_mean |   proto_margin_positive_rate_mean |   proto_margin_gap_mean |    cosco |   proto_margin_minus_cosco |
|:-------------|-------:|---------------------:|---------------------:|--------------------:|------------------------------:|-------------------------:|-------------------------------:|----------------------------------:|------------------------:|---------:|---------------------------:|
| BasicMotions |      1 |             1.000000 |             0.000000 |            0.100000 |                      0.795321 |                 0.000000 |                       0.795321 |                          0.000000 |                3.154173 | 1.000000 |                   0.000000 |
| BasicMotions |      1 |             1.000000 |             0.100000 |            0.100000 |                      0.795321 |                 0.000000 |                       0.795321 |                          0.000000 |                3.154173 | 1.000000 |                   0.000000 |

## Full rows

| model              | model_key                    | dataset      |   shot |   accuracy |   elapsed_sec |   seed | status   | cosco_variant   |   rho_min |   rho_mean |   rho_max |   rho_final |   proto_stress_mean |   proto_stress_final |   fft_reg_lambda |   fft_reg_loss_time_mean |   fft_reg_loss_freq_mean |   fft_reg_loss_total_mean |   fft_reg_lambda_min |   fft_reg_lambda_mean |   fft_reg_lambda_max |   fft_reg_lambda_final |   proto_margin_value |   proto_margin_beta |   proto_margin_base_loss_mean |   proto_margin_loss_mean |   proto_margin_total_loss_mean |   proto_margin_positive_rate_mean |   proto_margin_gap_mean |
|:-------------------|:-----------------------------|:-------------|-------:|-----------:|--------------:|-------:|:---------|:----------------|----------:|-----------:|----------:|------------:|--------------------:|---------------------:|-----------------:|-------------------------:|-------------------------:|--------------------------:|---------------------:|----------------------:|---------------------:|-----------------------:|---------------------:|--------------------:|------------------------------:|-------------------------:|-------------------------------:|----------------------------------:|------------------------:|
| cosco              | cosco                        | BasicMotions |      1 |     1.0000 |        0.5000 |   9782 | ok       |                 |       nan |        nan |       nan |         nan |                 nan |                  nan |              nan |                      nan |                      nan |                       nan |                  nan |                   nan |                  nan |                    nan |             nan      |            nan      |                      nan      |                 nan      |                       nan      |                          nan      |                nan      |
| cosco_proto_margin | cosco_proto_margin_m0_b0.1   | BasicMotions |      1 |     1.0000 |        0.1000 |   9782 | ok       | m0_b0.1         |       nan |        nan |       nan |         nan |                 nan |                  nan |              nan |                      nan |                      nan |                       nan |                  nan |                   nan |                  nan |                    nan |               0.0000 |              0.1000 |                        0.7953 |                   0.0000 |                         0.7953 |                            0.0000 |                  3.1542 |
| cosco_proto_margin | cosco_proto_margin_m0.1_b0.1 | BasicMotions |      1 |     1.0000 |        0.0000 |   9782 | ok       | m0.1_b0.1       |       nan |        nan |       nan |         nan |                 nan |                  nan |              nan |                      nan |                      nan |                       nan |                  nan |                   nan |                  nan |                    nan |               0.1000 |              0.1000 |                        0.7953 |                   0.0000 |                         0.7953 |                            0.0000 |                  3.1542 |
