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

- FFT regularization lambdas: `0.1`

- prototype margin values: `0.0`
- prototype margin betas: `0.05`

|                     |   cosco |   cosco_proto_margin_fft_reg_l0.1 |   cosco_proto_margin_m0_b0.05 |
|:--------------------|--------:|----------------------------------:|------------------------------:|
| ('BasicMotions', 1) |  1.0000 |                            1.0000 |                        1.0000 |

## Prototype margin effect by hyperparameter

| model                      |   proto_margin_value |   proto_margin_beta |   cosco_mean |   cosco_proto_margin_mean |   mean_delta |   wins |   ties |   losses |   base_loss_mean |   margin_loss_mean |   total_loss_mean |   positive_rate_mean |   gap_mean | gate_pass_mean_delta   |
|:---------------------------|---------------------:|--------------------:|-------------:|--------------------------:|-------------:|-------:|-------:|---------:|-----------------:|-------------------:|------------------:|---------------------:|-----------:|:-----------------------|
| cosco_proto_margin         |             0.000000 |            0.050000 |     1.000000 |                  1.000000 |     0.000000 |      0 |      1 |        0 |         0.795321 |           0.000000 |          0.795321 |             0.000000 |   3.154173 | False                  |
| cosco_proto_margin_fft_reg |             0.000000 |            0.050000 |     1.000000 |                  1.000000 |     0.000000 |      0 |      1 |        0 |         0.795568 |           0.000000 |          0.795568 |             0.000000 |   3.152084 | False                  |

## Prototype margin effect

| model                      | dataset      |   shot |   cosco_proto_margin |   proto_margin_value |   proto_margin_beta |   proto_margin_base_loss_mean |   proto_margin_loss_mean |   proto_margin_total_loss_mean |   proto_margin_positive_rate_mean |   proto_margin_gap_mean |    cosco |   proto_margin_minus_cosco |
|:---------------------------|:-------------|-------:|---------------------:|---------------------:|--------------------:|------------------------------:|-------------------------:|-------------------------------:|----------------------------------:|------------------------:|---------:|---------------------------:|
| cosco_proto_margin         | BasicMotions |      1 |             1.000000 |             0.000000 |            0.050000 |                      0.795321 |                 0.000000 |                       0.795321 |                          0.000000 |                3.154173 | 1.000000 |                   0.000000 |
| cosco_proto_margin_fft_reg | BasicMotions |      1 |             1.000000 |             0.000000 |            0.050000 |                      0.795568 |                 0.000000 |                       0.795568 |                          0.000000 |                3.152084 | 1.000000 |                   0.000000 |

## FFT regularization effect by lambda

| model                      |   fft_reg_lambda |   cosco_mean |   cosco_fft_reg_mean |   mean_delta |   wins |   ties |   losses |   loss_time_mean |   loss_freq_mean |   loss_total_mean |   effective_lambda_mean |   effective_lambda_min |   effective_lambda_max | gate_pass_mean_delta   |
|:---------------------------|-----------------:|-------------:|---------------------:|-------------:|-------:|-------:|---------:|-----------------:|-----------------:|------------------:|------------------------:|-----------------------:|-----------------------:|:-----------------------|
| cosco_proto_margin_fft_reg |         0.100000 |     1.000000 |             1.000000 |     0.000000 |      0 |      1 |        0 |         0.795568 |         0.958401 |          0.891408 |                0.100000 |               0.100000 |               0.100000 | False                  |

## FFT regularization effect

| model                      | dataset      |   shot |   cosco_fft_reg_accuracy |   fft_reg_lambda |   fft_reg_loss_time_mean |   fft_reg_loss_freq_mean |   fft_reg_loss_total_mean |   fft_reg_lambda_min |   fft_reg_lambda_mean |   fft_reg_lambda_max |   fft_reg_lambda_final |    cosco |   fft_reg_minus_cosco |
|:---------------------------|:-------------|-------:|-------------------------:|-----------------:|-------------------------:|-------------------------:|--------------------------:|---------------------:|----------------------:|---------------------:|-----------------------:|---------:|----------------------:|
| cosco_proto_margin_fft_reg | BasicMotions |      1 |                 1.000000 |         0.100000 |                 0.795568 |                 0.958401 |                  0.891408 |             0.100000 |              0.100000 |             0.100000 |               0.100000 | 1.000000 |              0.000000 |

## Full rows

| model                      | model_key                       | dataset      |   shot |   accuracy |   elapsed_sec |   seed | status   | cosco_variant      |   rho_min |   rho_mean |   rho_max |   rho_final |   proto_stress_mean |   proto_stress_final |   fft_reg_lambda |   fft_reg_loss_time_mean |   fft_reg_loss_freq_mean |   fft_reg_loss_total_mean |   fft_reg_lambda_min |   fft_reg_lambda_mean |   fft_reg_lambda_max |   fft_reg_lambda_final |   proto_margin_value |   proto_margin_beta |   proto_margin_base_loss_mean |   proto_margin_loss_mean |   proto_margin_total_loss_mean |   proto_margin_positive_rate_mean |   proto_margin_gap_mean |
|:---------------------------|:--------------------------------|:-------------|-------:|-----------:|--------------:|-------:|:---------|:-------------------|----------:|-----------:|----------:|------------:|--------------------:|---------------------:|-----------------:|-------------------------:|-------------------------:|--------------------------:|---------------------:|----------------------:|---------------------:|-----------------------:|---------------------:|--------------------:|------------------------------:|-------------------------:|-------------------------------:|----------------------------------:|------------------------:|
| cosco                      | cosco                           | BasicMotions |      1 |     1.0000 |        0.5000 |   9782 | ok       |                    |       nan |        nan |       nan |         nan |                 nan |                  nan |         nan      |                 nan      |                 nan      |                  nan      |             nan      |              nan      |             nan      |               nan      |             nan      |            nan      |                      nan      |                 nan      |                       nan      |                          nan      |                nan      |
| cosco_proto_margin         | cosco_proto_margin_m0_b0.05     | BasicMotions |      1 |     1.0000 |        0.1000 |   9782 | ok       | m0_b0.05           |       nan |        nan |       nan |         nan |                 nan |                  nan |         nan      |                 nan      |                 nan      |                  nan      |             nan      |              nan      |             nan      |               nan      |               0.0000 |              0.0500 |                        0.7953 |                   0.0000 |                         0.7953 |                            0.0000 |                  3.1542 |
| cosco_proto_margin_fft_reg | cosco_proto_margin_fft_reg_l0.1 | BasicMotions |      1 |     1.0000 |        0.2000 |   9782 | ok       | fft_reg_lambda=0.1 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.1000 |                   0.7956 |                   0.9584 |                    0.8914 |               0.1000 |                0.1000 |               0.1000 |                 0.1000 |               0.0000 |              0.0500 |                        0.7956 |                   0.0000 |                         0.7956 |                            0.0000 |                  3.1521 |
