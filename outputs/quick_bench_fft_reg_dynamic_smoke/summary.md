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

- FFT dynamic lambda min ratio: `0.0`
- FFT dynamic lambda max ratio: `1.0`

|                     |   cosco |   cosco_fft_reg_dynamic_l0.1 |   cosco_fft_reg_l0.1 |
|:--------------------|--------:|-----------------------------:|---------------------:|
| ('BasicMotions', 1) |  1.0000 |                       1.0000 |               1.0000 |

## FFT regularization effect by lambda

| model                 |   fft_reg_lambda |   cosco_mean |   cosco_fft_reg_mean |   mean_delta |   wins |   ties |   losses |   loss_time_mean |   loss_freq_mean |   loss_total_mean |   effective_lambda_mean |   effective_lambda_min |   effective_lambda_max | gate_pass_mean_delta   |
|:----------------------|-----------------:|-------------:|---------------------:|-------------:|-------:|-------:|---------:|-----------------:|-----------------:|------------------:|------------------------:|-----------------------:|-----------------------:|:-----------------------|
| cosco_fft_reg         |         0.100000 |     1.000000 |             1.000000 |     0.000000 |      0 |      1 |        0 |         0.795568 |         0.958401 |          0.891408 |                0.100000 |               0.100000 |               0.100000 | False                  |
| cosco_fft_reg_dynamic |         0.100000 |     1.000000 |             1.000000 |     0.000000 |      0 |      1 |        0 |         0.795519 |         0.959459 |          0.875071 |                0.082917 |               0.082222 |               0.083613 | False                  |

## FFT regularization effect

| model                 | dataset      |   shot |   cosco_fft_reg_accuracy |   fft_reg_lambda |   fft_reg_loss_time_mean |   fft_reg_loss_freq_mean |   fft_reg_loss_total_mean |   fft_reg_lambda_min |   fft_reg_lambda_mean |   fft_reg_lambda_max |   fft_reg_lambda_final |    cosco |   fft_reg_minus_cosco |
|:----------------------|:-------------|-------:|-------------------------:|-----------------:|-------------------------:|-------------------------:|--------------------------:|---------------------:|----------------------:|---------------------:|-----------------------:|---------:|----------------------:|
| cosco_fft_reg         | BasicMotions |      1 |                 1.000000 |         0.100000 |                 0.795568 |                 0.958401 |                  0.891408 |             0.100000 |              0.100000 |             0.100000 |               0.100000 | 1.000000 |              0.000000 |
| cosco_fft_reg_dynamic | BasicMotions |      1 |                 1.000000 |         0.100000 |                 0.795519 |                 0.959459 |                  0.875071 |             0.082222 |              0.082917 |             0.083613 |               0.083613 | 1.000000 |              0.000000 |

## Full rows

| model                 | model_key                  | dataset      |   shot |   accuracy |   elapsed_sec |   seed | status   | cosco_variant      |   rho_min |   rho_mean |   rho_max |   rho_final |   proto_stress_mean |   proto_stress_final |   fft_reg_lambda |   fft_reg_loss_time_mean |   fft_reg_loss_freq_mean |   fft_reg_loss_total_mean |   fft_reg_lambda_min |   fft_reg_lambda_mean |   fft_reg_lambda_max |   fft_reg_lambda_final |
|:----------------------|:---------------------------|:-------------|-------:|-----------:|--------------:|-------:|:---------|:-------------------|----------:|-----------:|----------:|------------:|--------------------:|---------------------:|-----------------:|-------------------------:|-------------------------:|--------------------------:|---------------------:|----------------------:|---------------------:|-----------------------:|
| cosco                 | cosco                      | BasicMotions |      1 |     1.0000 |        0.6000 |   9782 | ok       |                    |       nan |        nan |       nan |         nan |                 nan |                  nan |         nan      |                 nan      |                 nan      |                  nan      |             nan      |              nan      |             nan      |               nan      |
| cosco_fft_reg         | cosco_fft_reg_l0.1         | BasicMotions |      1 |     1.0000 |        0.2000 |   9782 | ok       | fft_reg_lambda=0.1 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.1000 |                   0.7956 |                   0.9584 |                    0.8914 |               0.1000 |                0.1000 |               0.1000 |                 0.1000 |
| cosco_fft_reg_dynamic | cosco_fft_reg_dynamic_l0.1 | BasicMotions |      1 |     1.0000 |        0.1000 |   9782 | ok       | fft_reg_lambda=0.1 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.1000 |                   0.7955 |                   0.9595 |                    0.8751 |               0.0822 |                0.0829 |               0.0836 |                 0.0836 |
