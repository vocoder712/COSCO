# FFT View Diagnostic

- datasets: `BasicMotions`
- shots: `1`
- normalize time data: `False`
- FFT view: `rFFT -> log1p(abs(.)) -> per-sample/channel z-score`
- fusion: `row_zscore(time_distance) + row_zscore(fft_distance)`
- elapsed seconds: `0.0`

## Follow-Up Gate

- mean `time_fft_proto - time_proto`: `0.050000`
- wins/ties/losses: `1/0/0`
- threshold: `0.005`
- gate pass: `True`

## Accuracy Pivot

|                     |   time_ed_1nn |   fft_ed_1nn |   time_fft_ed_1nn |   time_proto |   fft_proto |   time_fft_proto |
|:--------------------|--------------:|-------------:|------------------:|-------------:|------------:|-----------------:|
| ('BasicMotions', 1) |        0.4250 |       0.6000 |            0.4750 |       0.4250 |      0.6000 |           0.4750 |

## Accuracy Deltas

| dataset      |   shot |   fft_ed_1nn |   fft_proto |   time_ed_1nn |   time_fft_ed_1nn |   time_fft_proto |   time_proto |   fft_ed_minus_time_ed |   time_fft_ed_minus_time_ed |   fft_proto_minus_time_proto |   time_fft_proto_minus_time_proto |
|:-------------|-------:|-------------:|------------:|--------------:|------------------:|-----------------:|-------------:|-----------------------:|----------------------------:|-----------------------------:|----------------------------------:|
| BasicMotions |      1 |     0.600000 |    0.600000 |      0.425000 |          0.475000 |         0.475000 |     0.425000 |               0.175000 |                    0.050000 |                     0.175000 |                          0.050000 |

## Complementarity

| dataset      |   shot |   both_correct_rate |   time_only_correct_rate |   fft_only_correct_rate |   neither_correct_rate |   disagreement_rate | pair               |
|:-------------|-------:|--------------------:|-------------------------:|------------------------:|-----------------------:|--------------------:|:-------------------|
| BasicMotions |      1 |            0.400000 |                 0.025000 |                0.200000 |               0.375000 |            0.325000 | time_vs_fft_ed_1nn |
| BasicMotions |      1 |            0.400000 |                 0.025000 |                0.200000 |               0.375000 |            0.325000 | time_vs_fft_proto  |

## Prototype Margins

| model          | dataset      |   shot |   proto_margin_mean |   proto_margin_median |   proto_margin_positive_rate |
|:---------------|:-------------|-------:|--------------------:|----------------------:|-----------------------------:|
| time_proto     | BasicMotions |      1 |        -4857.744202 |           -926.211304 |                     0.425000 |
| fft_proto      | BasicMotions |      1 |           10.390137 |             17.231964 |                     0.600000 |
| time_fft_proto | BasicMotions |      1 |           -0.628466 |             -0.232215 |                     0.475000 |

## Data Shapes

| dataset      |   shot |   train_n |   test_n |   classes |   time_length |   fft_bins |   channels |
|:-------------|-------:|----------:|---------:|----------:|--------------:|-----------:|-----------:|
| BasicMotions |      1 |         4 |       40 |         4 |           100 |         51 |          6 |

## Full Rows

| model           | dataset      |   shot |   accuracy |   train_n |   test_n |   classes |   time_length |   fft_bins |   channels | normalize_time   | fft_transform                            | distance_fusion   |   both_correct_rate |   time_only_correct_rate |   fft_only_correct_rate |   neither_correct_rate |   disagreement_rate |   proto_margin_mean |   proto_margin_median |   proto_margin_positive_rate |
|:----------------|:-------------|-------:|-----------:|----------:|---------:|----------:|--------------:|-----------:|-----------:|:-----------------|:-----------------------------------------|:------------------|--------------------:|-------------------------:|------------------------:|-----------------------:|--------------------:|--------------------:|----------------------:|-----------------------------:|
| time_ed_1nn     | BasicMotions |      1 |   0.425000 |         4 |       40 |         4 |           100 |         51 |          6 | False            | rfft_log1p_abs_per_sample_channel_zscore |                   |            0.400000 |                 0.025000 |                0.200000 |               0.375000 |            0.325000 |          nan        |            nan        |                   nan        |
| fft_ed_1nn      | BasicMotions |      1 |   0.600000 |         4 |       40 |         4 |           100 |         51 |          6 | False            | rfft_log1p_abs_per_sample_channel_zscore |                   |            0.400000 |                 0.025000 |                0.200000 |               0.375000 |            0.325000 |          nan        |            nan        |                   nan        |
| time_fft_ed_1nn | BasicMotions |      1 |   0.475000 |         4 |       40 |         4 |           100 |         51 |          6 | False            | rfft_log1p_abs_per_sample_channel_zscore | row_zscore_sum    |            0.400000 |                 0.025000 |                0.200000 |               0.375000 |            0.325000 |          nan        |            nan        |                   nan        |
| time_proto      | BasicMotions |      1 |   0.425000 |         4 |       40 |         4 |           100 |         51 |          6 | False            | rfft_log1p_abs_per_sample_channel_zscore |                   |            0.400000 |                 0.025000 |                0.200000 |               0.375000 |            0.325000 |        -4857.744202 |           -926.211304 |                     0.425000 |
| fft_proto       | BasicMotions |      1 |   0.600000 |         4 |       40 |         4 |           100 |         51 |          6 | False            | rfft_log1p_abs_per_sample_channel_zscore |                   |            0.400000 |                 0.025000 |                0.200000 |               0.375000 |            0.325000 |           10.390137 |             17.231964 |                     0.600000 |
| time_fft_proto  | BasicMotions |      1 |   0.475000 |         4 |       40 |         4 |           100 |         51 |          6 | False            | rfft_log1p_abs_per_sample_channel_zscore | row_zscore_sum    |            0.400000 |                 0.025000 |                0.200000 |               0.375000 |            0.325000 |           -0.628466 |             -0.232215 |                     0.475000 |
