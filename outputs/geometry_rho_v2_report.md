# Prototype geometry pressure v2 — CPU three-seed report

## Setup

- Environment: Windows Duet tablet, Python 3.14.3 venv, PyTorch 2.13.0+cpu.
- Training: 30 epochs, 4 CPU threads, paired base seeds 10/20/30.
- Every dataset/shot/model cell is the mean of exactly three runs.
- Geometry-v2 defaults: alpha 0.15, EMA beta 0.9, normalized margin target
  0.35, protection threshold 0.35, protection strength 0.75, rho range
  [0.075, 0.115] for base rho 0.1.

## Results

| dataset        | shot | COSCO mean ± sd | geometry-v2 mean ± sd | mean rho | mean pressure | delta |
|:---------------|-----:|:----------------|:----------------------|---------:|--------------:|------:|
| Heartbeat      | 1    | 0.734959 ± 0.007451 | 0.736585 ± 0.004878 | 0.110176 | 0.216941 | +0.001626 |
| Heartbeat      | 10   | 0.726829 ± 0.008449 | 0.730081 ± 0.002816 | 0.092686 | 0.542355 | +0.003252 |
| RacketSports   | 1    | 0.491228 ± 0.033761 | 0.480263 ± 0.041086 | 0.106058 | 0.423418 | -0.010965 |
| RacketSports   | 10   | 0.778509 ± 0.016557 | 0.802632 ± 0.019737 | 0.096893 | 0.479773 | +0.024123 |
| JapaneseVowels | 1    | 0.709009 ± 0.051729 | 0.718919 ± 0.067891 | 0.108026 | 0.405294 | +0.009910 |
| JapaneseVowels | 10   | 0.907207 ± 0.016291 | 0.902703 ± 0.022123 | 0.104185 | 0.383222 | -0.004505 |

- Macro mean: original COSCO 0.724624; geometry-v2 0.728531.
- Macro delta: +0.003907 (about +0.39 percentage points).
- Task wins/losses: 4/2.

## Interpretation

The v2 pressure fixes the legacy method's 1-shot degeneracy and the protective
high-pressure branch materially improves RacketSports 10-shot. It is a useful
positive baseline, but the current six-task/30-epoch evidence is not a large
accuracy improvement and is not a paper-quality full benchmark. The two
remaining negative tasks show that absolute prototype pressure alone does not
fully predict the best SAM radius. A next iteration should combine pressure
with its temporal trend or an observed SAM sharpness gap, while retaining the
three-seed paired protocol.

Raw artifacts:

- `outputs/geometry_rho_verify_30ep/` (paired original COSCO references)
- `outputs/geometry_rho_verify_protect/` (protected geometry-v2)
- `outputs/geometry_rho_verify_japanese_30ep/` (JapaneseVowels external check)
