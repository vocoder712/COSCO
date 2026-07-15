# Quick COSCO Benchmark

- torch: `2.13.0+cpu` | device: `cpu`
- epochs for neural models: 1
- datasets: `BasicMotions`
- shots: `1`
- base seed: `10`
- deterministic torch: `True`
- COSCO variant: `original`

- weighted prototype gamma: `1.0`
- weighted prototype distance mode: `close`

- dynamic rho alpha: `0.25`
- dynamic rho min ratio: `0.5`
- dynamic rho max ratio: `1.15`

- FFT regularization lambdas: `0.1`

- prototype margin values: `0.0`
- prototype margin betas: `0.05`

|                     |   cosco |   cosco_geometry_rho |
|:--------------------|--------:|---------------------:|
| ('BasicMotions', 1) |  1.0000 |               1.0000 |

## Full rows

| model              | model_key          | dataset      |   shot |   accuracy |   elapsed_sec |   seed | status   | cosco_variant                      |   rho_min |   rho_mean |   rho_max |   rho_final |   proto_stress_mean |   proto_stress_final |   geometry_boundary_mean |   geometry_crowding_mean |   geometry_compactness_mean |   fft_reg_lambda |   fft_reg_loss_time_mean |   fft_reg_loss_freq_mean |   fft_reg_loss_total_mean |   fft_reg_lambda_min |   fft_reg_lambda_mean |   fft_reg_lambda_max |   fft_reg_lambda_final |   proto_margin_value |   proto_margin_beta |   proto_margin_base_loss_mean |   proto_margin_loss_mean |   proto_margin_total_loss_mean |   proto_margin_positive_rate_mean |   proto_margin_gap_mean |
|:-------------------|:-------------------|:-------------|-------:|-----------:|--------------:|-------:|:---------|:-----------------------------------|----------:|-----------:|----------:|------------:|--------------------:|---------------------:|-------------------------:|-------------------------:|----------------------------:|-----------------:|-------------------------:|-------------------------:|--------------------------:|---------------------:|----------------------:|---------------------:|-----------------------:|---------------------:|--------------------:|------------------------------:|-------------------------:|-------------------------------:|----------------------------------:|------------------------:|
| cosco              | cosco              | BasicMotions |      1 |     1.0000 |        0.3000 |   9782 | ok       |                                    |  nan      |   nan      |  nan      |    nan      |            nan      |             nan      |                 nan      |                 nan      |                    nan      |              nan |                      nan |                      nan |                       nan |                  nan |                   nan |                  nan |                    nan |                  nan |                 nan |                           nan |                      nan |                            nan |                               nan |                     nan |
| cosco_geometry_rho | cosco_geometry_rho | BasicMotions |      1 |     1.0000 |        0.1000 |   9782 | ok       | dynamic_rho_geometry_v2_alpha=0.15 |    0.0811 |     0.0811 |    0.0811 |      0.0811 |              0.6348 |               0.6348 |                   0.0000 |                   0.6348 |                      0.0000 |              nan |                      nan |                      nan |                       nan |                  nan |                   nan |                  nan |                    nan |                  nan |                 nan |                           nan |                      nan |                            nan |                               nan |                     nan |
