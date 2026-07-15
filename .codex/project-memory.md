# COSCO Project Memory

Last updated: 2026-07-15

This file is persistent local memory for future Codex conversations in this
repository. Read it before making changes or answering project-specific
questions.

## Project Identity

- Repository: `E:\proj\COSCO`
- Upstream topic: COSCO, "A Sharpness-Aware Training Framework for Few-Shot
  Multivariate Time Series Classification" (CIKM 2024).
- Task domain: few-shot multivariate time-series classification on UEA/UCR
  benchmark datasets.
- Main method: ResNet backbone + Prototypical Loss + SAM optimization.
- Main comparison baseline: TapNet through AEON's `TapNetClassifier`.

## Important Files

- `README.md`: upstream paper-oriented reproduction notes.
- `USAGE.md`: local engineering guide for Windows + RTX 4060 + CUDA 12.x.
- `run.py`: single experiment CLI entry point.
- `compare_models.py`: added comparison runner for multiple datasets, shots,
  and models; writes `outputs/comparison/summary.csv` and `.md`.
- `compare_quick_bench.py`: quick smoke benchmark for COSCO improvement
  iterations. It compares `ed_1nn`, `dtw_1nn`, `tapnet`, supervised `resnet`,
  `cosco`, `cosco_weighted`, and `cosco_dynamic_rho`; includes a
  `--dummy_cosco_improvement` no-op hook. Default quick comparison is now
  `cosco` vs `cosco_dynamic_rho`.
- `utils/load_data.py`: loads `Datasets/<dataset>/{1,10}-shot/*.npy` and full
  test splits; wraps arrays in a torch `Dataset`.
- `utils/proto_model.py`: core COSCO training/evaluation loop.
- `Prototypical_Loss.py`: prototypical loss variants and nearest-centroid
  inference. Also contains `WeightedPrototypicalLoss`, a COSCO improvement
  variant that computes mean centroids first, then L2 distances to that mean,
  then softmax distance weights, then a weighted centroid.
- `SAM.py`: SAM optimizer implementation.
- `Baselines/ResNet.py`: ResNet backbone. `forward()` returns `(log_probs,
  embedding)`.
- `Baselines/TapNet.py`: lazy AEON TapNet wrapper.
- `utils/save.py`: appends run results to CSV and writes per-run text results.
- `train_centroids.pt`: overwritten by COSCO/ResNet training; stores final
  class centroids only, not the ResNet weights.

## Runtime Environment

- Recommended conda env: `cosco`, Python 3.11.x.
- PyTorch: `2.5.1+cu124`, torchvision `0.20.1+cu124`.
- NumPy: `1.26.4`; pandas: `2.0.3`; scikit-learn: `1.5.2`.
- AEON must be pinned to `0.11.x`; AEON 1.4+ removed `TapNetClassifier`.
- TensorFlow should be `2.15.x` on Windows + Python 3.11. TensorFlow 2.18 has
  DLL load issues here, and 2.10 does not support Python 3.11.
- `keras_self_attention==0.51.0` is required for AEON TapNet but is not pulled
  automatically.
- Do not blindly run `pip install -r requirements.txt` if GPU torch is needed;
  install torch/torchvision from PyTorch's CUDA index first, then install the
  rest.

## Data Layout And Shapes

- Data root: `Datasets/`
- Dataset layout:
  - `X_test.npy`, `y_test.npy`
  - `1-shot/X_train.npy`, `1-shot/y_train.npy`
  - `10-shot/X_train.npy`, `10-shot/y_train.npy`
  - each shot folder also has `original_indices.npy`.
- Arrays are stored as `(N, T, C)` = samples, time, channels.
- ResNet expects `(N, C, T)`, so training/evaluation calls use
  `.transpose(1, 2)`.
- Labels are usually shaped `(N, 1)` in numpy. TapNet needs flat 1D labels.
- All 21 datasets have both 1-shot and 10-shot splits:
  `ArticularyWordRecognition`, `BasicMotions`, `CharacterTrajectories`,
  `EigenWorms`, `Epilepsy`, `EthanolConcentration`, `FaceDetection`,
  `FingerMovements`, `HandMovementDirection`, `Heartbeat`, `JapaneseVowels`,
  `Libras`, `MotorImagery`, `NATOPS`, `PEMS-SF`, `PenDigits`,
  `RacketSports`, `SelfRegulationSCP1`, `SelfRegulationSCP2`,
  `SpokenArabicDigits`, `UWaveGestureLibrary`.
- Example inspected 1-shot shapes:
  - `BasicMotions`: train `(4,100,6)`, test `(40,100,6)`, 4 classes.
  - `SpokenArabicDigits`: train `(10,65,13)`, test `(2199,65,13)`, 10 classes.
  - `PEMS-SF`: train `(7,144,963)`, test `(173,144,963)`, 7 classes.
  - `EigenWorms`: train `(5,17984,6)`, test `(131,17984,6)`, 5 classes.

## Commands

- Single COSCO run:
  `conda run -n cosco --no-capture-output python run.py --dataset BasicMotions --model resnet --shot 1 --nEpoch 100 --save_dir outputs/ --save_name cosco_basicmotions_1shot.csv`
- Single TapNet run:
  `conda run -n cosco --no-capture-output python run.py --dataset BasicMotions --model tapnet --shot 1 --nEpoch 100 --save_dir outputs/ --save_name tapnet_basicmotions_1shot.csv`
- Comparison sweep:
  `conda run -n cosco --no-capture-output python compare_models.py --datasets BasicMotions Epilepsy --shots 1 10 --models resnet tapnet --nEpoch 100 --out_dir outputs/comparison/`
- Quick improvement benchmark:
  `conda run -n cosco --no-capture-output python compare_quick_bench.py --datasets BasicMotions RacketSports --shots 1 10 --models ed_1nn dtw_1nn tapnet resnet cosco --nEpoch 5 --dummy_cosco_improvement --out_dir outputs/quick_bench/`
- Quick original-vs-weighted COSCO smoke:
  `conda run -n cosco --no-capture-output python compare_quick_bench.py --datasets BasicMotions --shots 1 10 --models cosco cosco_weighted --nEpoch 2 --weighted_proto_gamma 1.0 --weighted_proto_mode close --out_dir outputs/quick_bench_weighted_smoke/`
- Quick dynamic-rho COSCO benchmark:
  `conda run -n cosco --no-capture-output python compare_quick_bench.py --log_every 0 --out_dir outputs/quick_bench_dynamic_rho_a025_m115/`
- GPU sanity check:
  `conda run -n cosco --no-capture-output python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"`

## Behavioral Notes

- `run.py` initializes an empty aggregate CSV at `args.save_dir + args.save_name`
  and then calls `full_training(args)`.
- `compare_models.py` calls `load_data()` directly, runs each model/dataset/shot
  combination, catches per-combination failures, and records `status`.
- `compare_quick_bench.py` resets random seeds for every dataset/shot run via
  `stable_run_seed(base_seed, dataset, shot)`. This makes `cosco` and
  `cosco_weighted` share the same ResNet initialization and DataLoader shuffle
  for the same task, and makes results independent of model order. Use
  `--deterministic_torch` to request deterministic PyTorch kernels where
  available.
- `compare_quick_bench.py` also supports `cosco_dynamic_rho`, which uses the
  same per-dataset/shot seed as `cosco`. It writes accuracy deltas plus
  `rho_min/rho_mean/rho_max/rho_final` and prototype-stress statistics into
  the CSV/Markdown summaries. Use `--log_every 0` to suppress epoch logs during
  sweeps.
- `utils/proto_model.py::proto_neg_train_model`:
  - selects CUDA if available;
  - constructs `ResNet(input_size=train_data.shape[-1], nb_classes=len(unique labels))`;
  - uses `PrototypicalLoss(flag='neg')` regardless of `args.prototypical_loss_type`;
  - wraps SGD with SAM when `args.sam` is true;
  - disables BatchNorm running-stat updates during SAM's second step;
  - collects final-epoch embeddings to compute train centroids;
  - saves centroids to root `train_centroids.pt`;
  - evaluates by nearest-centroid classification on test embeddings.
- `utils/proto_model.py::weighted_proto_neg_train_model` preserves original
  COSCO by wrapping the same training loop with `WeightedPrototypicalLoss`.
  It writes centroids to `train_centroids_weighted.pt`.
- Dynamic SAM-rho details:
  - `cosco_dynamic_rho` keeps the original mean-prototype loss but adapts SAM
    `rho` before `optimizer.first_step()`;
  - prototype geometry pressure is
    `mean(distance_to_own_prototype) / mean(distance_to_nearest_other_prototype)`;
  - dynamic rho uses
    `rho_base * clamp(1 + alpha * stress, min_ratio, max_ratio)`;
  - current default dynamic-rho parameters are `alpha=0.25`,
    `min_ratio=0.5`, `max_ratio=1.15`, with base `rho=0.1`;
  - `1-shot` usually has zero prototype stress because each class has one
    support sample, so dynamic rho stays at the base value.
- Weighted prototype details:
  - original mean centroid is computed per class;
  - per-sample L2 distance to that mean is computed;
  - `--weighted_proto_gamma` controls softmax temperature;
  - `--weighted_proto_mode close` uses `softmax(-distance / gamma)`, giving
    closer samples larger weights;
  - `--weighted_proto_mode far` uses the literal `softmax(distance / gamma)`,
    giving farther samples larger weights.
- Windows-specific fix: DataLoader uses `num_workers=0` in the main COSCO path
  to avoid spawn/pickling issues.
- TapNet wrapper lazily imports `TapNetClassifier`, so importing
  `Baselines.TapNet` does not fail unless `train_tapnet()` is actually called.
- `Baselines/ResNet.py` still has an older `ResNetTrainer` with `num_workers=1`;
  the active COSCO path does not use that trainer.

## Current Outputs

### 2026-07-15 Duet CPU / geometry-rho v2 iteration

- Current workspace is `D:\proj\COSCO` on a Duet tablet with no discrete GPU
  and no Conda. The working environment is `.venv`, Python 3.14.3,
  PyTorch 2.13.0+cpu, NumPy 2.5.1, pandas 3.0.3, scikit-learn 1.9.0.
- `requirements-cpu-py314.txt` and the new top section in `USAGE.md` document
  the lightweight venv installation. Install the PyTorch wheel from its
  official CPU index; the PyPI large-file endpoint stalled on this machine.
- Removed unused torchvision imports from `Baselines/ResNet.py`,
  `Prototypical_Loss.py`, and `SAM.py`; torchvision is not needed by COSCO.
- Added `cosco_geometry_rho`, which combines normalized nearest-wrong boundary
  pressure, prototype crowding, and within-class compactness. Prototype
  crowding remains nonzero for 1-shot. Pressure is EMA-smoothed.
- Geometry-v2 rho is protective: moderate/low pressure can boost rho, while
  high pressure shrinks it. Selected CPU-screen defaults are alpha 0.15,
  EMA beta 0.9, margin target 0.35, protect threshold 0.35, protect strength
  0.75, min/max rho ratios 0.75/1.15.
- Added `compare_dynamic_rho_multiseed.py`. It requires exactly three seeds,
  uses paired model initialization/shuffle seeds, and writes raw runs,
  mean/std summaries, and paired deltas.
- 30-epoch three-seed verification covered Heartbeat, RacketSports, and
  JapaneseVowels at 1/10-shot (six tasks, 18 runs per method). Macro accuracy:
  COSCO 0.724624, geometry-v2 0.728531, delta +0.003907, wins/losses 4/2.
  Biggest gain was RacketSports 10-shot +0.024123; remaining drops were
  RacketSports 1-shot -0.010965 and JapaneseVowels 10-shot -0.004505.
- Full report: `outputs/geometry_rho_v2_report.md`. This is a positive research
  baseline, not yet the requested large improvement and not a 100-epoch/full
  21-dataset paper-quality benchmark.
- Next recommended method step: combine geometry pressure with its temporal
  trend or a measured SAM sharpness gap, because absolute pressure alone does
  not predict the best rho on the two remaining negative tasks.

- Current `outputs/comparison/summary.md` is for `SpokenArabicDigits`, 100
  epochs, torch `2.5.1+cu124`, device `NVIDIA GeForce RTX 4060 Laptop GPU`.
- Current measured accuracies:
  - ResNet/COSCO, 1-shot: `0.3860845839017735`
  - TapNet, 1-shot: `0.13142337426102774`
  - ResNet/COSCO, 10-shot: `0.7994542974079127`
  - TapNet, 10-shot: `0.7412460209185994`
- `USAGE.md` also contains a separate BasicMotions example result where COSCO
  scored `1.0000` for both 1-shot and 10-shot and TapNet scored `0.5750` /
  `0.7000`. Treat that as documented example output, not the currently opened
  `outputs/comparison` state.
- Current `outputs/quick_bench/summary.md` is a quick smoke benchmark, not a
  final paper-quality benchmark. It used 5 epochs for neural models, datasets
  `BasicMotions` and `RacketSports`, both 1-shot and 10-shot, and the no-op
  `dummy_noop` COSCO hook.
- Quick benchmark accuracies:
  - `BasicMotions` 1-shot: COSCO `1.0000`, DTW-1NN `0.8000`, ED-1NN `0.4250`,
    ResNet `0.3250`, TapNet `0.3500`.
  - `BasicMotions` 10-shot: COSCO `1.0000`, DTW-1NN `0.9750`, ED-1NN `0.6000`,
    ResNet `0.2500`, TapNet `0.3500`.
  - `RacketSports` 1-shot: COSCO `0.3553`, DTW-1NN `0.5197`, ED-1NN `0.3553`,
    ResNet `0.3026`, TapNet `0.2763`.
  - `RacketSports` 10-shot: COSCO `0.7237`, DTW-1NN `0.8487`, ED-1NN `0.6908`,
    ResNet `0.2632`, TapNet `0.2961`.
- Dynamic-rho iteration report:
  - Weekly report path:
    `outputs/quick_bench_dynamic_rho_a025_m115/dynamic_rho_weekly_report.md`.
  - Best current dynamic-rho quick-bench output:
    `outputs/quick_bench_dynamic_rho_a025_m115/summary.md`.
  - Tested datasets: `SpokenArabicDigits`, `RacketSports`, `Heartbeat`,
    `JapaneseVowels`, `Libras`; shots: `1` and `10`; neural epochs: `100`.
  - Mean accuracy over the 10 dataset/shot tasks:
    `cosco = 0.661237`, `cosco_dynamic_rho = 0.663100`,
    mean delta `+0.001863`.
  - Wins/ties/losses by task: `3 / 3 / 4`.
  - Biggest gains: `RacketSports` 1-shot `+0.0197`, `RacketSports` 10-shot
    `+0.0132`, `SpokenArabicDigits` 10-shot `+0.0018`.
  - Remaining drops: `Libras` 1-shot/10-shot `-0.0056`,
    `JapaneseVowels` 10-shot `-0.0027`, `SpokenArabicDigits` 1-shot `-0.0023`.
  - Interpretation: dynamic rho is a weak positive baseline, not a strong
    stable improvement. Monotonically increasing rho with prototype stress was
    harmful when too aggressive (`alpha=1.0,max_ratio=2.0`) and still risky for
    high-stress datasets. The current conservative default only barely clears
    original COSCO on average.
  - Suggested next work: try window/protective dynamic rho, prototype-margin
    driven rho, SAM sharpness-gap driven rho, or shift to DTW-distance
    distillation / pairwise metric loss / augmentation consistency for stronger
    embedding improvements.

## Known Pitfalls

- Running from a subdirectory breaks `load_data()` because paths are relative to
  `./Datasets/`; run commands from the repository root.
- `train_centroids.pt` is a generated artifact and is overwritten by each COSCO
  ResNet training run. It does not include the trained ResNet `state_dict`.
- `compare_quick_bench.py` uses `train_centroids_quick_bench.pt` for original
  COSCO and `train_centroids_dynamic_rho.pt` for dynamic rho to avoid further
  overwriting root `train_centroids.pt` during quick-bench sweeps. Older runs
  may already have modified `train_centroids.pt`.
- For long-lived inference, add explicit checkpoint saving in
  `utils/proto_model.py::proto_neg_train_model`, for example
  `torch.save(model_resnet.state_dict(), 'resnet.pt')`.
- Boolean argparse flags currently use `type=bool`; strings like `"False"` can
  behave unexpectedly in Python CLIs. Prefer passing defaults or change to
  `store_true` / `store_false` if editing CLI behavior.
- `PrototypicalLoss` applies `softmax()` before `F.cross_entropy()`, which is
  mathematically unusual because cross entropy expects logits. This is upstream
  behavior and should be changed only intentionally.
- `args.optimizer` is read but the COSCO SAM path always uses SGD as the base
  optimizer.
- `args.prototypical_loss_type` is exposed by CLI but the active training path
  always uses `flag='neg'`.
- Some imports are redundant or stale (`torchvision`, duplicated `torch.nn`,
  private `_BatchNorm` imports), but they are mostly harmless.

## Working Tree Snapshot

- At memory creation time, `git status --short` showed:
  - modified: `train_centroids.pt`
  - untracked: `outputs/`
- This memory file and `.codex/` were added after that snapshot.
- Do not revert user/generated artifacts unless explicitly asked.
