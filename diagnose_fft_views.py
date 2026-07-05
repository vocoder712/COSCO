"""
FFT-view diagnostic benchmark for COSCO.

This script does not train neural models. It checks whether a log-magnitude
frequency view provides useful and complementary nearest-neighbour/prototype
signals before adding a full multi-view COSCO training variant.
"""

import argparse
import os
import time
from argparse import Namespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from utils.load_data import load_data


DEFAULT_DATASETS = [
    "SpokenArabicDigits",
    "RacketSports",
    "Heartbeat",
    "JapaneseVowels",
    "Libras",
]

MODEL_ORDER = [
    "time_ed_1nn",
    "fft_ed_1nn",
    "time_fft_ed_1nn",
    "time_proto",
    "fft_proto",
    "time_fft_proto",
]


def flatten_labels(y: np.ndarray) -> np.ndarray:
    return y.reshape(-1)


def fft_logmag_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Convert (N, T, C) time series to log-magnitude rFFT view (N, F, C).

    The result is z-scored per sample and per channel along the frequency axis.
    This keeps the channel count unchanged and prevents absolute energy scale
    from dominating Euclidean/prototype distances.
    """
    spectrum = np.fft.rfft(x.astype(np.float32), axis=1)
    logmag = np.log1p(np.abs(spectrum)).astype(np.float32)
    mean = logmag.mean(axis=1, keepdims=True)
    std = logmag.std(axis=1, keepdims=True)
    zscored = (logmag - mean) / (std + eps)
    if not np.isfinite(zscored).all():
        raise ValueError("FFT log-magnitude z-score produced non-finite values")
    return zscored.astype(np.float32, copy=False)


def flatten_view(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1).astype(np.float32, copy=False)


def squared_l2_distances(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared L2 distances without materialising a 3D broadcast.
    """
    query = query.astype(np.float32, copy=False)
    candidates = candidates.astype(np.float32, copy=False)
    query_norm = np.sum(query * query, axis=1, keepdims=True)
    candidate_norm = np.sum(candidates * candidates, axis=1, keepdims=True).T
    distances = query_norm + candidate_norm - 2.0 * np.matmul(query, candidates.T)
    return np.maximum(distances, 0.0).astype(np.float32, copy=False)


def row_zscore(distances: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = distances.mean(axis=1, keepdims=True)
    std = distances.std(axis=1, keepdims=True)
    return (distances - mean) / (std + eps)


def predict_from_distances(distances: np.ndarray, candidate_labels: np.ndarray) -> np.ndarray:
    nearest = np.argmin(distances, axis=1)
    return candidate_labels[nearest]


def run_1nn(train_x: np.ndarray, train_y: np.ndarray,
            test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    distances = squared_l2_distances(flatten_view(test_x), flatten_view(train_x))
    pred = predict_from_distances(distances, train_y)
    return pred, distances


def run_fused_1nn(time_train_x: np.ndarray, fft_train_x: np.ndarray,
                  train_y: np.ndarray, time_test_x: np.ndarray,
                  fft_test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    time_dist = squared_l2_distances(flatten_view(time_test_x), flatten_view(time_train_x))
    fft_dist = squared_l2_distances(flatten_view(fft_test_x), flatten_view(fft_train_x))
    fused_dist = row_zscore(time_dist) + row_zscore(fft_dist)
    pred = predict_from_distances(fused_dist, train_y)
    return pred, fused_dist


def compute_class_centroids(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat_x = flatten_view(x)
    class_labels = np.unique(y)
    centroids = []
    for class_label in class_labels:
        centroids.append(flat_x[y == class_label].mean(axis=0))
    return class_labels, np.vstack(centroids).astype(np.float32)


def run_proto(train_x: np.ndarray, train_y: np.ndarray,
              test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_labels, centroids = compute_class_centroids(train_x, train_y)
    distances = squared_l2_distances(flatten_view(test_x), centroids)
    pred = predict_from_distances(distances, class_labels)
    return pred, distances, class_labels


def run_fused_proto(time_train_x: np.ndarray, fft_train_x: np.ndarray,
                    train_y: np.ndarray, time_test_x: np.ndarray,
                    fft_test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_labels, time_centroids = compute_class_centroids(time_train_x, train_y)
    fft_labels, fft_centroids = compute_class_centroids(fft_train_x, train_y)
    if not np.array_equal(time_labels, fft_labels):
        raise ValueError("Time and FFT prototype labels differ")

    time_dist = squared_l2_distances(flatten_view(time_test_x), time_centroids)
    fft_dist = squared_l2_distances(flatten_view(fft_test_x), fft_centroids)
    fused_dist = row_zscore(time_dist) + row_zscore(fft_dist)
    pred = predict_from_distances(fused_dist, time_labels)
    return pred, fused_dist, time_labels


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def complementarity_stats(y_true: np.ndarray, time_pred: np.ndarray,
                          fft_pred: np.ndarray) -> Dict[str, float]:
    time_correct = time_pred == y_true
    fft_correct = fft_pred == y_true
    total = float(len(y_true))
    return {
        "both_correct_rate": float(np.mean(time_correct & fft_correct)),
        "time_only_correct_rate": float(np.mean(time_correct & ~fft_correct)),
        "fft_only_correct_rate": float(np.mean(~time_correct & fft_correct)),
        "neither_correct_rate": float(np.mean(~time_correct & ~fft_correct)),
        "disagreement_rate": float(np.mean(time_pred != fft_pred)),
        "count": total,
    }


def prototype_margin_stats(y_true: np.ndarray, distances: np.ndarray,
                           class_labels: np.ndarray) -> Dict[str, float]:
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}
    margins = []
    for row, label in zip(distances, y_true):
        true_index = label_to_index.get(label)
        if true_index is None or len(row) < 2:
            continue
        true_distance = row[true_index]
        other_distances = np.delete(row, true_index)
        nearest_other = float(np.min(other_distances))
        margins.append(nearest_other - float(true_distance))

    if not margins:
        return {
            "proto_margin_mean": np.nan,
            "proto_margin_median": np.nan,
            "proto_margin_positive_rate": np.nan,
        }

    margins_arr = np.asarray(margins, dtype=np.float64)
    return {
        "proto_margin_mean": float(np.mean(margins_arr)),
        "proto_margin_median": float(np.median(margins_arr)),
        "proto_margin_positive_rate": float(np.mean(margins_arr > 0)),
    }


def build_run_args(args: argparse.Namespace, dataset: str, shot: int) -> Namespace:
    return Namespace(dataset=dataset, shot=shot, normalize=args.normalize)


def diagnose_one_task(args: argparse.Namespace, dataset: str,
                      shot: int) -> List[Dict[str, object]]:
    run_args = build_run_args(args, dataset, shot)
    train_data, train_label, test_data, test_label = load_data(run_args)
    train_y = flatten_labels(train_label)
    test_y = flatten_labels(test_label)

    fft_train = fft_logmag_zscore(train_data)
    fft_test = fft_logmag_zscore(test_data)

    predictions: Dict[str, np.ndarray] = {}
    proto_distances: Dict[str, np.ndarray] = {}
    proto_labels: Dict[str, np.ndarray] = {}

    predictions["time_ed_1nn"], _ = run_1nn(train_data, train_y, test_data)
    predictions["fft_ed_1nn"], _ = run_1nn(fft_train, train_y, fft_test)
    predictions["time_fft_ed_1nn"], _ = run_fused_1nn(
        train_data, fft_train, train_y, test_data, fft_test
    )
    predictions["time_proto"], proto_distances["time_proto"], proto_labels["time_proto"] = run_proto(
        train_data, train_y, test_data
    )
    predictions["fft_proto"], proto_distances["fft_proto"], proto_labels["fft_proto"] = run_proto(
        fft_train, train_y, fft_test
    )
    predictions["time_fft_proto"], proto_distances["time_fft_proto"], proto_labels["time_fft_proto"] = run_fused_proto(
        train_data, fft_train, train_y, test_data, fft_test
    )

    ed_comp = complementarity_stats(
        test_y, predictions["time_ed_1nn"], predictions["fft_ed_1nn"]
    )
    proto_comp = complementarity_stats(
        test_y, predictions["time_proto"], predictions["fft_proto"]
    )

    rows = []
    for model in MODEL_ORDER:
        model_kind = "proto" if "proto" in model else "ed_1nn"
        comp = proto_comp if model_kind == "proto" else ed_comp
        row: Dict[str, object] = {
            "model": model,
            "dataset": dataset,
            "shot": shot,
            "accuracy": accuracy(test_y, predictions[model]),
            "train_n": int(train_data.shape[0]),
            "test_n": int(test_data.shape[0]),
            "classes": int(len(np.unique(train_y))),
            "time_length": int(train_data.shape[1]),
            "fft_bins": int(fft_train.shape[1]),
            "channels": int(train_data.shape[2]),
            "normalize_time": bool(args.normalize),
            "fft_transform": "rfft_log1p_abs_per_sample_channel_zscore",
            "distance_fusion": "row_zscore_sum" if model.startswith("time_fft") else "",
            "both_correct_rate": comp["both_correct_rate"],
            "time_only_correct_rate": comp["time_only_correct_rate"],
            "fft_only_correct_rate": comp["fft_only_correct_rate"],
            "neither_correct_rate": comp["neither_correct_rate"],
            "disagreement_rate": comp["disagreement_rate"],
        }
        if model in proto_distances:
            row.update(prototype_margin_stats(
                test_y, proto_distances[model], proto_labels[model]
            ))
        else:
            row.update({
                "proto_margin_mean": np.nan,
                "proto_margin_median": np.nan,
                "proto_margin_positive_rate": np.nan,
            })
        rows.append(row)
    return rows


def win_tie_loss(delta: pd.Series, tolerance: float = 1e-12) -> Tuple[int, int, int]:
    wins = int((delta > tolerance).sum())
    ties = int((delta.abs() <= tolerance).sum())
    losses = int((delta < -tolerance).sum())
    return wins, ties, losses


def build_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["dataset", "shot"],
        columns="model",
        values="accuracy",
        aggfunc="first",
    )
    delta = pivot.copy()
    if {"time_ed_1nn", "fft_ed_1nn", "time_fft_ed_1nn"}.issubset(delta.columns):
        delta["fft_ed_minus_time_ed"] = delta["fft_ed_1nn"] - delta["time_ed_1nn"]
        delta["time_fft_ed_minus_time_ed"] = (
            delta["time_fft_ed_1nn"] - delta["time_ed_1nn"]
        )
    if {"time_proto", "fft_proto", "time_fft_proto"}.issubset(delta.columns):
        delta["fft_proto_minus_time_proto"] = delta["fft_proto"] - delta["time_proto"]
        delta["time_fft_proto_minus_time_proto"] = (
            delta["time_fft_proto"] - delta["time_proto"]
        )
    return delta.reset_index()


def write_outputs(df: pd.DataFrame, args: argparse.Namespace,
                  elapsed_sec: float) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    summary_md = os.path.join(args.out_dir, "summary.md")
    df.to_csv(summary_csv, index=False)

    accuracy_view = df.pivot_table(
        index=["dataset", "shot"],
        columns="model",
        values="accuracy",
        aggfunc="first",
    )[MODEL_ORDER]
    delta_table = build_delta_table(df)

    proto_delta_col = "time_fft_proto_minus_time_proto"
    gate_delta = (
        delta_table[proto_delta_col]
        if proto_delta_col in delta_table.columns
        else pd.Series(dtype=float)
    )
    gate_mean = float(gate_delta.mean()) if not gate_delta.empty else np.nan
    wins, ties, losses = win_tie_loss(gate_delta) if not gate_delta.empty else (0, 0, 0)
    gate_pass = bool(gate_mean >= args.followup_delta_threshold and wins >= losses)

    complementarity = df.loc[
        df["model"].isin(["time_ed_1nn", "time_proto"]),
        [
            "model",
            "dataset",
            "shot",
            "both_correct_rate",
            "time_only_correct_rate",
            "fft_only_correct_rate",
            "neither_correct_rate",
            "disagreement_rate",
        ],
    ].copy()
    complementarity["pair"] = complementarity["model"].map({
        "time_ed_1nn": "time_vs_fft_ed_1nn",
        "time_proto": "time_vs_fft_proto",
    })
    complementarity = complementarity.drop(columns=["model"])

    margin_rows = df.loc[
        df["model"].str.contains("proto"),
        [
            "model",
            "dataset",
            "shot",
            "proto_margin_mean",
            "proto_margin_median",
            "proto_margin_positive_rate",
        ],
    ].copy()

    shape_rows = df.drop_duplicates(["dataset", "shot"])[
        [
            "dataset",
            "shot",
            "train_n",
            "test_n",
            "classes",
            "time_length",
            "fft_bins",
            "channels",
        ]
    ].copy()

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# FFT View Diagnostic\n\n")
        f.write(f"- datasets: `{', '.join(args.datasets)}`\n")
        f.write(f"- shots: `{', '.join(map(str, args.shots))}`\n")
        f.write(f"- normalize time data: `{args.normalize}`\n")
        f.write("- FFT view: `rFFT -> log1p(abs(.)) -> per-sample/channel z-score`\n")
        f.write("- fusion: `row_zscore(time_distance) + row_zscore(fft_distance)`\n")
        f.write(f"- elapsed seconds: `{elapsed_sec:.1f}`\n\n")

        f.write("## Follow-Up Gate\n\n")
        f.write(f"- mean `time_fft_proto - time_proto`: `{gate_mean:.6f}`\n")
        f.write(f"- wins/ties/losses: `{wins}/{ties}/{losses}`\n")
        f.write(f"- threshold: `{args.followup_delta_threshold}`\n")
        f.write(f"- gate pass: `{gate_pass}`\n\n")

        f.write("## Accuracy Pivot\n\n")
        f.write(accuracy_view.to_markdown(floatfmt=".4f"))
        f.write("\n\n## Accuracy Deltas\n\n")
        f.write(delta_table.to_markdown(index=False, floatfmt=".6f"))
        f.write("\n\n## Complementarity\n\n")
        f.write(complementarity.to_markdown(index=False, floatfmt=".6f"))
        f.write("\n\n## Prototype Margins\n\n")
        f.write(margin_rows.to_markdown(index=False, floatfmt=".6f"))
        f.write("\n\n## Data Shapes\n\n")
        f.write(shape_rows.to_markdown(index=False))
        f.write("\n\n## Full Rows\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".6f"))
        f.write("\n")

    print(f"[saved] {summary_csv}")
    print(f"[saved] {summary_md}")
    print("\n=== Accuracy pivot ===")
    print(accuracy_view.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== Follow-up gate ===")
    print(
        f"mean_delta={gate_mean:.6f} wins/ties/losses={wins}/{ties}/{losses} "
        f"pass={gate_pass}"
    )


def run_transform_smoke_tests() -> None:
    cases = [
        np.random.default_rng(0).normal(size=(3, 8, 2)).astype(np.float32),
        np.random.default_rng(1).normal(size=(2, 9, 3)).astype(np.float32),
        np.random.default_rng(2).normal(size=(1, 17984, 6)).astype(np.float32),
        np.random.default_rng(3).normal(size=(4, 144, 963)).astype(np.float32),
    ]
    for case in cases:
        transformed = fft_logmag_zscore(case)
        expected_bins = case.shape[1] // 2 + 1
        expected_shape = (case.shape[0], expected_bins, case.shape[2])
        if transformed.shape != expected_shape:
            raise AssertionError(f"Expected {expected_shape}, got {transformed.shape}")
        if not np.isfinite(transformed).all():
            raise AssertionError("FFT transform smoke test produced non-finite values")
    print("[ok] FFT transform smoke tests passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose time-vs-FFT nearest-neighbour/prototype signals."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--shots", nargs="+", type=int, choices=[1, 10], default=[1, 10])
    parser.add_argument("--normalize", action="store_true",
                        help="Apply existing per-series time z-score before FFT diagnostics.")
    parser.add_argument("--out_dir", default="outputs/fft_view_diagnostic_logmag_zscore/")
    parser.add_argument("--followup_delta_threshold", type=float, default=0.005)
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run FFT transform smoke tests and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        run_transform_smoke_tests()
        return

    t0 = time.time()
    rows: List[Dict[str, object]] = []
    for dataset in args.datasets:
        for shot in args.shots:
            tag = f"{dataset} | {shot}-shot"
            print(f"[run] {tag}")
            task_t0 = time.time()
            rows.extend(diagnose_one_task(args, dataset, shot))
            print(f"[done] {tag} ({time.time() - task_t0:.1f}s)")

    write_outputs(pd.DataFrame(rows), args, time.time() - t0)


if __name__ == "__main__":
    main()
