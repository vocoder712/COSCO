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

- FFT regularization lambdas: `0.3`

|                     |   cosco |   cosco_fft_reg_l0.3 |
|:--------------------|--------:|---------------------:|
| ('BasicMotions', 1) |  1.0000 |               1.0000 |

## FFT regularization effect by lambda

|   fft_reg_lambda |   cosco_mean |   cosco_fft_reg_mean |   mean_delta |   wins |   ties |   losses |   loss_time_mean |   loss_freq_mean |   loss_total_mean | gate_pass_mean_delta   |
|-----------------:|-------------:|---------------------:|-------------:|-------:|-------:|---------:|-----------------:|-----------------:|------------------:|:-----------------------|
|         0.300000 |     1.000000 |             1.000000 |     0.000000 |      0 |      1 |        0 |         0.796326 |         0.954425 |          1.082654 | False                  |

## FFT regularization effect

| dataset      |   shot |   cosco_fft_reg |   fft_reg_lambda |   fft_reg_loss_time_mean |   fft_reg_loss_freq_mean |   fft_reg_loss_total_mean |    cosco |   fft_reg_minus_cosco |
|:-------------|-------:|----------------:|-----------------:|-------------------------:|-------------------------:|--------------------------:|---------:|----------------------:|
| BasicMotions |      1 |        1.000000 |         0.300000 |                 0.796326 |                 0.954425 |                  1.082654 | 1.000000 |              0.000000 |

## Full rows

| model         | model_key          | dataset      |   shot |   accuracy |   elapsed_sec |   seed | status   | cosco_variant      |   rho_min |   rho_mean |   rho_max |   rho_final |   proto_stress_mean |   proto_stress_final |   fft_reg_lambda |   fft_reg_loss_time_mean |   fft_reg_loss_freq_mean |   fft_reg_loss_total_mean |
|:--------------|:-------------------|:-------------|-------:|-----------:|--------------:|-------:|:---------|:-------------------|----------:|-----------:|----------:|------------:|--------------------:|---------------------:|-----------------:|-------------------------:|-------------------------:|--------------------------:|
| cosco         | cosco              | BasicMotions |      1 |     1.0000 |        1.0000 |   9782 | ok       |                    |       nan |        nan |       nan |         nan |                 nan |                  nan |         nan      |                 nan      |                 nan      |                  nan      |
| cosco_fft_reg | cosco_fft_reg_l0.3 | BasicMotions |      1 |     1.0000 |        0.4000 |   9782 | ok       | fft_reg_lambda=0.3 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.3000 |                   0.7963 |                   0.9544 |                    1.0827 |
