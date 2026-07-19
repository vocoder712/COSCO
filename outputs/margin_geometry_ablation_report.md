# Prototype-margin × geometry-rho ablation

## Experimental protocol

- CPU: Python 3.14.3 venv, PyTorch 2.13.0+cpu, four threads.
- Datasets: RacketSports, Heartbeat, JapaneseVowels; 1-shot and 10-shot.
- Training: 30 epochs; paired base seeds 10, 20, 30.
- Every model uses the same initialization and DataLoader shuffle seed for a
  given dataset/shot/base-seed cell.
- Geometry-v2 uses alpha 0.15, pressure EMA 0.9, protection threshold 0.35,
  protection strength 0.75, and rho ratio range [0.75, 1.15].
- Prototype boundary loss is `beta * relu(margin + d_own - d_wrong)`.

## Orthogonal ablation at historical beta 0.05

| scope | COSCO | margin only | geometry only | combination |
|:------|------:|------------:|--------------:|------------:|
| all six tasks | 0.724624 | 0.728282 | 0.728531 | 0.728958 |
| three 10-shot tasks | 0.804182 | 0.811498 | 0.811805 | 0.812659 |

At beta 0.05 the combination is only slightly above the best individual macro
mean. Its mean 10-shot additive synergy is -0.006462. Heartbeat has a high
margin active rate (~0.382), and the combined model follows the margin-only
result instead of retaining the geometry-only improvement. This beta is too
strong for the combination.

## Refined combination at beta 0.025

The 1-shot boundary term is exactly zero when margin is zero because each
sample is its own prototype. Therefore the 1-shot margin-only model equals
COSCO and the 1-shot combined model equals geometry-v2. The refinement run
focuses on 10-shot.

| dataset | COSCO | margin only | geometry only | combination | combination − COSCO |
|:--------|------:|------------:|--------------:|------------:|--------------------:|
| Heartbeat | 0.726829 | 0.725203 | 0.730081 | 0.730081 | +0.003252 |
| JapaneseVowels | 0.907207 | 0.909009 | 0.902703 | 0.906306 | -0.000901 |
| RacketSports | 0.778509 | 0.793860 | 0.802632 | 0.813596 | +0.035088 |
| **10-shot macro** | **0.804182** | **0.809357** | **0.811805** | **0.816661** | **+0.012480** |

- Combination improvement over geometry-only macro: +0.004856.
- Combination improvement over margin-only macro: +0.007304.
- Mean strict additive synergy:
  `combination - margin - geometry + COSCO = -0.000319`.
- Mean paired combination advantage over the better individual model: +0.001481.

Including the three 1-shot tasks, the six-task macro means are:

| COSCO | margin only | geometry only | combination |
|------:|------------:|--------------:|------------:|
| 0.724624 | 0.727211 | 0.728531 | **0.730959** |

The selected combination improves the six-task macro over COSCO by +0.006335
(about +0.63 percentage points) and over geometry-v2 by +0.002428.

## Conclusion

The two methods can be stacked at beta 0.025 in the macro-average sense: the
combination is better than either individual method across the tested 10-shot
suite. The gains are not strictly super-additive, so the methods share much of
their mechanism. The boundary loss directly repairs nearest-prototype errors,
while geometry-v2 changes the SAM neighbourhood using related boundary and
compactness signals.

The improvement is concentrated in RacketSports 10-shot. JapaneseVowels still
has a small negative result and the current evidence covers three datasets at
30 epochs, so this is a promising combined baseline rather than a final broad
accuracy claim.

Raw outputs:

- `outputs/margin_geometry_ablation_30ep/` — full beta 0.05 orthogonal ablation.
- `outputs/margin_geometry_ablation_beta0025_30ep/` — beta 0.025 refinement.
- `outputs/margin_geometry_ablation_smoke/` — wiring smoke test.

## Five-dataset external reassessment

Epilepsy and NATOPS were selected before running based on CPU feasibility and
structural diversity; neither was used to select the geometry or combination
parameters. The same 30-epoch, three-paired-seed protocol and beta 0.025 were
used without retuning.

| dataset | shot | COSCO | margin | geometry | combination | combination delta |
|:--------|-----:|------:|-------:|---------:|------------:|------------------:|
| Epilepsy | 1 | 0.548309 | 0.548309 | 0.565217 | 0.565217 | +0.016908 |
| Epilepsy | 10 | 0.920290 | 0.915459 | 0.917874 | 0.922705 | +0.002415 |
| Heartbeat | 1 | 0.734959 | 0.734959 | 0.736585 | 0.736585 | +0.001626 |
| Heartbeat | 10 | 0.726829 | 0.725203 | 0.730081 | 0.730081 | +0.003252 |
| JapaneseVowels | 1 | 0.709009 | 0.709009 | 0.718919 | 0.718919 | +0.009910 |
| JapaneseVowels | 10 | 0.907207 | 0.909009 | 0.902703 | 0.906306 | -0.000901 |
| NATOPS | 1 | 0.659259 | 0.659259 | 0.648148 | 0.648148 | -0.011111 |
| NATOPS | 10 | 0.914815 | 0.916667 | 0.896296 | 0.892593 | -0.022222 |
| RacketSports | 1 | 0.491228 | 0.491228 | 0.480263 | 0.480263 | -0.010965 |
| RacketSports | 10 | 0.778509 | 0.793860 | 0.802632 | 0.813596 | +0.035088 |

Five-dataset / ten-task macro means:

| COSCO | margin only | geometry only | combination |
|------:|------------:|--------------:|------------:|
| 0.739041 | 0.740296 | 0.739872 | **0.741441** |

- Combination macro delta over COSCO: +0.002400 (+0.24 percentage points).
- Task wins/ties/losses: 6/0/4.
- Dataset-level wins/losses after averaging both shots: 4/1; NATOPS is the loss.
- Strict synergy averaged across ten tasks: +0.000315, effectively near zero.
- Mean paired difference from the better individual component: -0.005026.

The combination remains the best fixed method in macro average, but the
external tasks reduce the earlier six-task gain from +0.006335 to +0.002400.
NATOPS is a clear counterexample: margin-only is slightly positive at 10-shot,
while geometry and the combination are negative. Therefore the current method
is a weak positive aggregate result, not a broadly stable improvement.

Additional raw output:

- `outputs/margin_geometry_external_2datasets_30ep/`
- `outputs/margin_geometry_5datasets_summary.csv`
