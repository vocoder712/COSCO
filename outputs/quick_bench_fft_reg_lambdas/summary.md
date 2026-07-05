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

- dynamic rho alpha: `0.25`
- dynamic rho min ratio: `0.5`
- dynamic rho max ratio: `1.15`

- FFT regularization lambdas: `0.1, 0.3, 0.5`

|                            |   cosco |   cosco_fft_reg_l0.1 |   cosco_fft_reg_l0.3 |   cosco_fft_reg_l0.5 |
|:---------------------------|--------:|---------------------:|---------------------:|---------------------:|
| ('Heartbeat', 1)           |  0.7268 |               0.7366 |               0.7317 |               0.7366 |
| ('Heartbeat', 10)          |  0.6537 |               0.6732 |               0.6537 |               0.6585 |
| ('JapaneseVowels', 1)      |  0.6676 |               0.6649 |               0.6865 |               0.6649 |
| ('JapaneseVowels', 10)     |  0.9027 |               0.9000 |               0.8892 |               0.8838 |
| ('Libras', 1)              |  0.4222 |               0.3889 |               0.3611 |               0.3556 |
| ('Libras', 10)             |  0.8667 |               0.8611 |               0.8667 |               0.8722 |
| ('RacketSports', 1)        |  0.4934 |               0.5132 |               0.4934 |               0.5000 |
| ('RacketSports', 10)       |  0.7961 |               0.7961 |               0.7961 |               0.8092 |
| ('SpokenArabicDigits', 1)  |  0.3342 |               0.3629 |               0.3679 |               0.3661 |
| ('SpokenArabicDigits', 10) |  0.7549 |               0.7390 |               0.7440 |               0.7412 |

## FFT regularization effect by lambda

|   fft_reg_lambda |   cosco_mean |   cosco_fft_reg_mean |   mean_delta |   wins |   ties |   losses |   loss_time_mean |   loss_freq_mean |   loss_total_mean | gate_pass_mean_delta   |
|-----------------:|-------------:|---------------------:|-------------:|-------:|-------:|---------:|-----------------:|-----------------:|------------------:|:-----------------------|
|         0.100000 |     0.661825 |             0.663570 |     0.001744 |      4 |      1 |        5 |         1.211338 |         1.333259 |          1.344664 | False                  |
|         0.300000 |     0.661825 |             0.659016 |    -0.002809 |      3 |      4 |        3 |         1.207010 |         1.263179 |          1.585964 | False                  |
|         0.500000 |     0.661825 |             0.658808 |    -0.003017 |      6 |      0 |        4 |         1.203593 |         1.241969 |          1.824578 | False                  |

## FFT regularization effect

| dataset            |   shot |   cosco_fft_reg |   fft_reg_lambda |   fft_reg_loss_time_mean |   fft_reg_loss_freq_mean |   fft_reg_loss_total_mean |    cosco |   fft_reg_minus_cosco |
|:-------------------|-------:|----------------:|-----------------:|-------------------------:|-------------------------:|--------------------------:|---------:|----------------------:|
| SpokenArabicDigits |      1 |        0.362892 |         0.100000 |                 1.464932 |                 1.527607 |                  1.617692 | 0.334243 |              0.028649 |
| SpokenArabicDigits |      1 |        0.367894 |         0.300000 |                 1.464675 |                 1.501565 |                  1.915144 | 0.334243 |              0.033652 |
| SpokenArabicDigits |      1 |        0.366075 |         0.500000 |                 1.464446 |                 1.493134 |                  2.211013 | 0.334243 |              0.031833 |
| SpokenArabicDigits |     10 |        0.738972 |         0.100000 |                 1.590860 |                 2.128566 |                  1.803717 | 0.754889 |             -0.015916 |
| SpokenArabicDigits |     10 |        0.743975 |         0.300000 |                 1.588851 |                 1.878581 |                  2.152426 | 0.754889 |             -0.010914 |
| SpokenArabicDigits |     10 |        0.741246 |         0.500000 |                 1.588237 |                 1.782869 |                  2.479671 | 0.754889 |             -0.013643 |
| RacketSports       |      1 |        0.513158 |         0.100000 |                 0.748149 |                 0.749418 |                  0.823091 | 0.493421 |              0.019737 |
| RacketSports       |      1 |        0.493421 |         0.300000 |                 0.748125 |                 0.746838 |                  0.972177 | 0.493421 |              0.000000 |
| RacketSports       |      1 |        0.500000 |         0.500000 |                 0.747883 |                 0.746324 |                  1.121045 | 0.493421 |              0.006579 |
| RacketSports       |     10 |        0.796053 |         0.100000 |                 0.820844 |                 0.916308 |                  0.912475 | 0.796053 |              0.000000 |
| RacketSports       |     10 |        0.796053 |         0.300000 |                 0.819058 |                 0.831898 |                  1.068628 | 0.796053 |              0.000000 |
| RacketSports       |     10 |        0.809211 |         0.500000 |                 0.816339 |                 0.819271 |                  1.225975 | 0.796053 |              0.013158 |
| Heartbeat          |      1 |        0.736585 |         0.100000 |                 0.313693 |                 0.375958 |                  0.351288 | 0.726829 |              0.009756 |
| Heartbeat          |      1 |        0.731707 |         0.300000 |                 0.313682 |                 0.335303 |                  0.414273 | 0.726829 |              0.004878 |
| Heartbeat          |      1 |        0.736585 |         0.500000 |                 0.313742 |                 0.326779 |                  0.477132 | 0.726829 |              0.009756 |
| Heartbeat          |     10 |        0.673171 |         0.100000 |                 0.510665 |                 0.426657 |                  0.553331 | 0.653659 |              0.019512 |
| Heartbeat          |     10 |        0.653659 |         0.300000 |                 0.481351 |                 0.395706 |                  0.600063 | 0.653659 |              0.000000 |
| Heartbeat          |     10 |        0.658537 |         0.500000 |                 0.464588 |                 0.392186 |                  0.660681 | 0.653659 |              0.004878 |
| JapaneseVowels     |      1 |        0.664865 |         0.100000 |                 1.373630 |                 1.394270 |                  1.513057 | 0.667568 |             -0.002703 |
| JapaneseVowels     |      1 |        0.686486 |         0.300000 |                 1.373078 |                 1.389190 |                  1.789835 | 0.667568 |              0.018919 |
| JapaneseVowels     |      1 |        0.664865 |         0.500000 |                 1.372935 |                 1.385691 |                  2.065780 | 0.667568 |             -0.002703 |
| JapaneseVowels     |     10 |        0.900000 |         0.100000 |                 1.402665 |                 1.697446 |                  1.572409 | 0.902703 |             -0.002703 |
| JapaneseVowels     |     10 |        0.889189 |         0.300000 |                 1.400748 |                 1.556879 |                  1.867812 | 0.902703 |             -0.013514 |
| JapaneseVowels     |     10 |        0.883784 |         0.500000 |                 1.399854 |                 1.513557 |                  2.156632 | 0.902703 |             -0.018919 |
| Libras             |      1 |        0.388889 |         0.100000 |                 1.825963 |                 1.896944 |                  2.015658 | 0.422222 |             -0.033333 |
| Libras             |      1 |        0.361111 |         0.300000 |                 1.825212 |                 1.870600 |                  2.386392 | 0.422222 |             -0.061111 |
| Libras             |      1 |        0.355556 |         0.500000 |                 1.824697 |                 1.862694 |                  2.756044 | 0.422222 |             -0.066667 |
| Libras             |     10 |        0.861111 |         0.100000 |                 2.061976 |                 2.219419 |                  2.283918 | 0.866667 |             -0.005556 |
| Libras             |     10 |        0.866667 |         0.300000 |                 2.055323 |                 2.125228 |                  2.692891 | 0.866667 |              0.000000 |
| Libras             |     10 |        0.872222 |         0.500000 |                 2.043212 |                 2.097181 |                  3.091802 | 0.866667 |              0.005556 |

## Full rows

| model         | model_key          | dataset            |   shot |   accuracy |   elapsed_sec |   seed | status   | cosco_variant      |   rho_min |   rho_mean |   rho_max |   rho_final |   proto_stress_mean |   proto_stress_final |   fft_reg_lambda |   fft_reg_loss_time_mean |   fft_reg_loss_freq_mean |   fft_reg_loss_total_mean |
|:--------------|:-------------------|:-------------------|-------:|-----------:|--------------:|-------:|:---------|:-------------------|----------:|-----------:|----------:|------------:|--------------------:|---------------------:|-----------------:|-------------------------:|-------------------------:|--------------------------:|
| cosco         | cosco              | SpokenArabicDigits |      1 |     0.3342 |        4.6000 |  19509 | ok       |                    |       nan |        nan |       nan |         nan |                 nan |                  nan |         nan      |                 nan      |                 nan      |                  nan      |
| cosco_fft_reg | cosco_fft_reg_l0.1 | SpokenArabicDigits |      1 |     0.3629 |        7.5000 |  19509 | ok       | fft_reg_lambda=0.1 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.1000 |                   1.4649 |                   1.5276 |                    1.6177 |
| cosco_fft_reg | cosco_fft_reg_l0.3 | SpokenArabicDigits |      1 |     0.3679 |        7.2000 |  19509 | ok       | fft_reg_lambda=0.3 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.3000 |                   1.4647 |                   1.5016 |                    1.9151 |
| cosco_fft_reg | cosco_fft_reg_l0.5 | SpokenArabicDigits |      1 |     0.3661 |        7.2000 |  19509 | ok       | fft_reg_lambda=0.5 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.5000 |                   1.4644 |                   1.4931 |                    2.2110 |
| cosco         | cosco              | SpokenArabicDigits |     10 |     0.7549 |        4.2000 |  20517 | ok       |                    |       nan |        nan |       nan |         nan |                 nan |                  nan |         nan      |                 nan      |                 nan      |                  nan      |
| cosco_fft_reg | cosco_fft_reg_l0.1 | SpokenArabicDigits |     10 |     0.7390 |        7.2000 |  20517 | ok       | fft_reg_lambda=0.1 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.1000 |                   1.5909 |                   2.1286 |                    1.8037 |
| cosco_fft_reg | cosco_fft_reg_l0.3 | SpokenArabicDigits |     10 |     0.7440 |        7.2000 |  20517 | ok       | fft_reg_lambda=0.3 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.3000 |                   1.5889 |                   1.8786 |                    2.1524 |
| cosco_fft_reg | cosco_fft_reg_l0.5 | SpokenArabicDigits |     10 |     0.7412 |        7.1000 |  20517 | ok       | fft_reg_lambda=0.5 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.5000 |                   1.5882 |                   1.7829 |                    2.4797 |
| cosco         | cosco              | RacketSports       |      1 |     0.4934 |        3.0000 |   9924 | ok       |                    |       nan |        nan |       nan |         nan |                 nan |                  nan |         nan      |                 nan      |                 nan      |                  nan      |
| cosco_fft_reg | cosco_fft_reg_l0.1 | RacketSports       |      1 |     0.5132 |        5.4000 |   9924 | ok       | fft_reg_lambda=0.1 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.1000 |                   0.7481 |                   0.7494 |                    0.8231 |
| cosco_fft_reg | cosco_fft_reg_l0.3 | RacketSports       |      1 |     0.4934 |        5.3000 |   9924 | ok       | fft_reg_lambda=0.3 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.3000 |                   0.7481 |                   0.7468 |                    0.9722 |
| cosco_fft_reg | cosco_fft_reg_l0.5 | RacketSports       |      1 |     0.5000 |        5.3000 |   9924 | ok       | fft_reg_lambda=0.5 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.5000 |                   0.7479 |                   0.7463 |                    1.1210 |
| cosco         | cosco              | RacketSports       |     10 |     0.7961 |        3.3000 |  10644 | ok       |                    |       nan |        nan |       nan |         nan |                 nan |                  nan |         nan      |                 nan      |                 nan      |                  nan      |
| cosco_fft_reg | cosco_fft_reg_l0.1 | RacketSports       |     10 |     0.7961 |        5.5000 |  10644 | ok       | fft_reg_lambda=0.1 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.1000 |                   0.8208 |                   0.9163 |                    0.9125 |
| cosco_fft_reg | cosco_fft_reg_l0.3 | RacketSports       |     10 |     0.7961 |        5.5000 |  10644 | ok       | fft_reg_lambda=0.3 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.3000 |                   0.8191 |                   0.8319 |                    1.0686 |
| cosco_fft_reg | cosco_fft_reg_l0.5 | RacketSports       |     10 |     0.8092 |        5.7000 |  10644 | ok       | fft_reg_lambda=0.5 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.5000 |                   0.8163 |                   0.8193 |                    1.2260 |
| cosco         | cosco              | Heartbeat          |      1 |     0.7268 |        3.1000 |   5845 | ok       |                    |       nan |        nan |       nan |         nan |                 nan |                  nan |         nan      |                 nan      |                 nan      |                  nan      |
| cosco_fft_reg | cosco_fft_reg_l0.1 | Heartbeat          |      1 |     0.7366 |        5.0000 |   5845 | ok       | fft_reg_lambda=0.1 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.1000 |                   0.3137 |                   0.3760 |                    0.3513 |
| cosco_fft_reg | cosco_fft_reg_l0.3 | Heartbeat          |      1 |     0.7317 |        4.8000 |   5845 | ok       | fft_reg_lambda=0.3 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.3000 |                   0.3137 |                   0.3353 |                    0.4143 |
| cosco_fft_reg | cosco_fft_reg_l0.5 | Heartbeat          |      1 |     0.7366 |        5.0000 |   5845 | ok       | fft_reg_lambda=0.5 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.5000 |                   0.3137 |                   0.3268 |                    0.4771 |
| cosco         | cosco              | Heartbeat          |     10 |     0.6537 |        3.2000 |   6421 | ok       |                    |       nan |        nan |       nan |         nan |                 nan |                  nan |         nan      |                 nan      |                 nan      |                  nan      |
| cosco_fft_reg | cosco_fft_reg_l0.1 | Heartbeat          |     10 |     0.6732 |        4.8000 |   6421 | ok       | fft_reg_lambda=0.1 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.1000 |                   0.5107 |                   0.4267 |                    0.5533 |
| cosco_fft_reg | cosco_fft_reg_l0.3 | Heartbeat          |     10 |     0.6537 |        4.7000 |   6421 | ok       | fft_reg_lambda=0.3 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.3000 |                   0.4814 |                   0.3957 |                    0.6001 |
| cosco_fft_reg | cosco_fft_reg_l0.5 | Heartbeat          |     10 |     0.6585 |        4.8000 |   6421 | ok       | fft_reg_lambda=0.5 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.5000 |                   0.4646 |                   0.3922 |                    0.6607 |
| cosco         | cosco              | JapaneseVowels     |      1 |     0.6676 |        3.7000 |  12844 | ok       |                    |       nan |        nan |       nan |         nan |                 nan |                  nan |         nan      |                 nan      |                 nan      |                  nan      |
| cosco_fft_reg | cosco_fft_reg_l0.1 | JapaneseVowels     |      1 |     0.6649 |        6.9000 |  12844 | ok       | fft_reg_lambda=0.1 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.1000 |                   1.3736 |                   1.3943 |                    1.5131 |
| cosco_fft_reg | cosco_fft_reg_l0.3 | JapaneseVowels     |      1 |     0.6865 |        7.0000 |  12844 | ok       | fft_reg_lambda=0.3 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.3000 |                   1.3731 |                   1.3892 |                    1.7898 |
| cosco_fft_reg | cosco_fft_reg_l0.5 | JapaneseVowels     |      1 |     0.6649 |        7.0000 |  12844 | ok       | fft_reg_lambda=0.5 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.5000 |                   1.3729 |                   1.3857 |                    2.0658 |
| cosco         | cosco              | JapaneseVowels     |     10 |     0.9027 |        4.2000 |  13660 | ok       |                    |       nan |        nan |       nan |         nan |                 nan |                  nan |         nan      |                 nan      |                 nan      |                  nan      |
| cosco_fft_reg | cosco_fft_reg_l0.1 | JapaneseVowels     |     10 |     0.9000 |        4.0000 |  13660 | ok       | fft_reg_lambda=0.1 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.1000 |                   1.4027 |                   1.6974 |                    1.5724 |
| cosco_fft_reg | cosco_fft_reg_l0.3 | JapaneseVowels     |     10 |     0.8892 |        3.2000 |  13660 | ok       | fft_reg_lambda=0.3 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.3000 |                   1.4007 |                   1.5569 |                    1.8678 |
| cosco_fft_reg | cosco_fft_reg_l0.5 | JapaneseVowels     |     10 |     0.8838 |        3.2000 |  13660 | ok       | fft_reg_lambda=0.5 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.5000 |                   1.3999 |                   1.5136 |                    2.1566 |
| cosco         | cosco              | Libras             |      1 |     0.4222 |        2.1000 |   3019 | ok       |                    |       nan |        nan |       nan |         nan |                 nan |                  nan |         nan      |                 nan      |                 nan      |                  nan      |
| cosco_fft_reg | cosco_fft_reg_l0.1 | Libras             |      1 |     0.3889 |        3.9000 |   3019 | ok       | fft_reg_lambda=0.1 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.1000 |                   1.8260 |                   1.8969 |                    2.0157 |
| cosco_fft_reg | cosco_fft_reg_l0.3 | Libras             |      1 |     0.3611 |        4.1000 |   3019 | ok       | fft_reg_lambda=0.3 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.3000 |                   1.8252 |                   1.8706 |                    2.3864 |
| cosco_fft_reg | cosco_fft_reg_l0.5 | Libras             |      1 |     0.3556 |        4.1000 |   3019 | ok       | fft_reg_lambda=0.5 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.5000 |                   1.8247 |                   1.8627 |                    2.7560 |
| cosco         | cosco              | Libras             |     10 |     0.8667 |        2.7000 |   3451 | ok       |                    |       nan |        nan |       nan |         nan |                 nan |                  nan |         nan      |                 nan      |                 nan      |                  nan      |
| cosco_fft_reg | cosco_fft_reg_l0.1 | Libras             |     10 |     0.8611 |        4.4000 |   3451 | ok       | fft_reg_lambda=0.1 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.1000 |                   2.0620 |                   2.2194 |                    2.2839 |
| cosco_fft_reg | cosco_fft_reg_l0.3 | Libras             |     10 |     0.8667 |        4.4000 |   3451 | ok       | fft_reg_lambda=0.3 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.3000 |                   2.0553 |                   2.1252 |                    2.6929 |
| cosco_fft_reg | cosco_fft_reg_l0.5 | Libras             |     10 |     0.8722 |        4.4000 |   3451 | ok       | fft_reg_lambda=0.5 |       nan |        nan |       nan |         nan |                 nan |                  nan |           0.5000 |                   2.0432 |                   2.0972 |                    3.0918 |
