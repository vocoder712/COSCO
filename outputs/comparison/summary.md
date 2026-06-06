# COSCO vs TapNet — accuracy comparison

- torch: `2.5.1+cu124` | device: `cuda (NVIDIA GeForce RTX 4060 Laptop GPU)`
- epochs: 100

|                            |   resnet |   tapnet |
|:---------------------------|---------:|---------:|
| ('SpokenArabicDigits', 1)  |   0.3861 |   0.1314 |
| ('SpokenArabicDigits', 10) |   0.7995 |   0.7412 |
