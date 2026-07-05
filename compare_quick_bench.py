"""
Quick benchmark for COSCO improvement iterations.

This script is intentionally separate from `compare_models.py`. The old script
keeps the COSCO-vs-TapNet reproduction path; this one is a fast smoke benchmark
for future COSCO changes. It compares raw 1NN baselines, TapNet, supervised
ResNet, and COSCO on a small set of datasets and both 1-shot / 10-shot splits.
"""

import argparse
import os
import random
import sys
import time
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from Baselines.ResNet import ResNet
from Baselines.TapNet import train_tapnet
from utils.load_data import Dataset, load_data
from utils.proto_model import (
    fft_regularized_proto_neg_train_model,
    margin_fft_regularized_proto_neg_train_model,
    margin_proto_neg_train_model,
    proto_neg_train_model,
    weighted_proto_neg_train_model,
)


MODEL_CHOICES = [
    "ed_1nn",
    "dtw_1nn",
    "tapnet",
    "resnet",
    "cosco",
    "cosco_weighted",
    "cosco_dynamic_rho",
    "cosco_fft_reg",
    "cosco_proto_margin",
    "cosco_proto_margin_fft_reg",
]


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic_torch
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)


def stable_run_seed(base_seed: int, dataset: str, shot: int) -> int:
    """
    Derive a model-order-independent seed for one dataset/shot task.

    COSCO and COSCO-weighted intentionally share the same seed for the same
    dataset/shot, so their ResNet initialization and DataLoader shuffle are
    comparable.
    """
    key = f"{dataset}:{shot}"
    key_value = sum((i + 1) * ord(ch) for i, ch in enumerate(key))
    return (base_seed + key_value) % (2 ** 31 - 1)


def make_run_args(args: argparse.Namespace, dataset: str, shot: int,
                  model: str, fft_reg_lambda: float = np.nan,
                  proto_margin_value: float = np.nan,
                  proto_margin_beta: float = np.nan) -> Namespace:
    return Namespace(
        lr=args.lr,
        rho=args.rho,
        nEpoch=args.nEpoch,
        dataset=dataset,
        shot=shot,
        normalize=args.normalize,
        model=model,
        sam=True,
        optimizer="adam",
        prototypical_loss=True,
        prototypical_loss_type="neg",
        save_dir=args.out_dir,
        save_name=f"{model}_{dataset}_{shot}shot.csv",
        dummy_cosco_improvement=args.dummy_cosco_improvement,
        weighted_proto_gamma=args.weighted_proto_gamma,
        weighted_proto_mode=args.weighted_proto_mode,
        dynamic_rho=(model == "cosco_dynamic_rho"),
        dynamic_rho_alpha=args.dynamic_rho_alpha,
        dynamic_rho_min_ratio=args.dynamic_rho_min_ratio,
        dynamic_rho_max_ratio=args.dynamic_rho_max_ratio,
        dynamic_rho_summary={},
        log_every=args.log_every,
        seed=args.seed,
        fft_reg_lambda=fft_reg_lambda,
        fft_reg_summary={},
        proto_margin_value=proto_margin_value,
        proto_margin_beta=proto_margin_beta,
        proto_margin_summary={},
    )


def flatten_labels(y: np.ndarray) -> np.ndarray:
    return y.reshape(-1)


def fft_lambda_key(value: float) -> str:
    return f"{float(value):g}"


def proto_margin_key(margin: float, beta: float) -> str:
    return f"m{float(margin):g}_b{float(beta):g}"


def run_ed_1nn(train_data: np.ndarray, train_label: np.ndarray,
               test_data: np.ndarray, test_label: np.ndarray) -> float:
    x_train = train_data.reshape(train_data.shape[0], -1).astype(np.float32)
    x_test = test_data.reshape(test_data.shape[0], -1).astype(np.float32)
    y_train = flatten_labels(train_label)
    y_test = flatten_labels(test_label)

    pred = []
    for x in x_test:
        distances = np.sum((x_train - x) ** 2, axis=1)
        pred.append(y_train[int(np.argmin(distances))])
    return float(accuracy_score(y_test, np.asarray(pred)))


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Multivariate DTW with squared Euclidean local cost."""
    n, m = a.shape[0], b.shape[0]
    prev = np.full(m + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0

    for i in range(1, n + 1):
        cur = np.full(m + 1, np.inf, dtype=np.float64)
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = float(np.sum((ai - b[j - 1]) ** 2))
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur

    return float(np.sqrt(prev[m]))


def run_dtw_1nn(train_data: np.ndarray, train_label: np.ndarray,
                test_data: np.ndarray, test_label: np.ndarray) -> float:
    y_train = flatten_labels(train_label)
    y_test = flatten_labels(test_label)

    pred = []
    for x in test_data.astype(np.float32):
        distances = [dtw_distance(x, x_train) for x_train in train_data.astype(np.float32)]
        pred.append(y_train[int(np.argmin(distances))])
    return float(accuracy_score(y_test, np.asarray(pred)))


def run_tapnet_baseline(train_data: np.ndarray, train_label: np.ndarray,
                        test_data: np.ndarray, test_label: np.ndarray,
                        input_size: int, args: Namespace) -> float:
    # AEON deep-learning classifiers expect (n_cases, n_channels, n_timepoints).
    train_x = train_data.transpose(0, 2, 1)
    test_x = test_data.transpose(0, 2, 1)
    return float(train_tapnet(
        train_x,
        flatten_labels(train_label),
        test_x,
        flatten_labels(test_label),
        input_size,
        args,
    ))


def run_resnet_supervised(train_data: np.ndarray, train_label: np.ndarray,
                          test_data: np.ndarray, test_label: np.ndarray,
                          input_size: int, args: Namespace) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_train_flat = flatten_labels(train_label)
    n_classes = len(np.unique(y_train_flat))

    model = ResNet(input_size=input_size, nb_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(
        Dataset(train_data, train_label),
        batch_size=min(1024, len(train_data)),
        shuffle=True,
        num_workers=0,
    )

    model.train()
    for _ in range(args.nEpoch):
        for inputs, labels in loader:
            inputs = inputs.to(device).float().transpose(1, 2)
            labels = labels.to(device).long().squeeze()
            optimizer.zero_grad()
            log_probs, _ = model(inputs)
            loss = F.nll_loss(log_probs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_test = torch.from_numpy(test_data.astype(np.float32)).to(device).transpose(1, 2)
        log_probs, _ = model(x_test)
        pred = torch.argmax(log_probs, dim=1).cpu().numpy()

    return float(accuracy_score(flatten_labels(test_label), pred))


def apply_dummy_cosco_improvement(args: Namespace) -> Namespace:
    """
    No-op placeholder for future COSCO changes.

    Keeping this hook in the benchmark lets us prove the experiment plumbing
    works before adding a real method change.
    """
    args.cosco_variant = "dummy_noop"
    return args


def run_cosco(train_data: np.ndarray, train_label: np.ndarray,
              test_data: np.ndarray, test_label: np.ndarray,
              input_size: int, args: Namespace) -> float:
    if getattr(args, "dummy_cosco_improvement", False):
        args = apply_dummy_cosco_improvement(args)

    trainloader = DataLoader(
        Dataset(train_data, train_label),
        batch_size=1024,
        shuffle=True,
        num_workers=0,
    )
    return float(proto_neg_train_model(
        trainloader,
        train_label,
        test_data,
        test_label,
        input_size,
        args,
        centroid_path="train_centroids_quick_bench.pt",
    ))


def run_cosco_weighted(train_data: np.ndarray, train_label: np.ndarray,
                       test_data: np.ndarray, test_label: np.ndarray,
                       input_size: int, args: Namespace) -> float:
    trainloader = DataLoader(
        Dataset(train_data, train_label),
        batch_size=1024,
        shuffle=True,
        num_workers=0,
    )
    return float(weighted_proto_neg_train_model(
        trainloader,
        train_label,
        test_data,
        test_label,
        input_size,
        args,
    ))


def run_cosco_dynamic_rho(train_data: np.ndarray, train_label: np.ndarray,
                          test_data: np.ndarray, test_label: np.ndarray,
                          input_size: int, args: Namespace) -> float:
    args.dynamic_rho = True
    trainloader = DataLoader(
        Dataset(train_data, train_label),
        batch_size=1024,
        shuffle=True,
        num_workers=0,
    )
    return float(proto_neg_train_model(
        trainloader,
        train_label,
        test_data,
        test_label,
        input_size,
        args,
        centroid_path="train_centroids_dynamic_rho.pt",
    ))


def run_cosco_fft_reg(train_data: np.ndarray, train_label: np.ndarray,
                      test_data: np.ndarray, test_label: np.ndarray,
                      input_size: int, args: Namespace) -> float:
    trainloader = DataLoader(
        Dataset(train_data, train_label),
        batch_size=1024,
        shuffle=True,
        num_workers=0,
    )
    return float(fft_regularized_proto_neg_train_model(
        trainloader,
        train_label,
        test_data,
        test_label,
        input_size,
        args,
    ))


def run_cosco_proto_margin(train_data: np.ndarray, train_label: np.ndarray,
                           test_data: np.ndarray, test_label: np.ndarray,
                           input_size: int, args: Namespace) -> float:
    trainloader = DataLoader(
        Dataset(train_data, train_label),
        batch_size=1024,
        shuffle=True,
        num_workers=0,
    )
    return float(margin_proto_neg_train_model(
        trainloader,
        train_label,
        test_data,
        test_label,
        input_size,
        args,
    ))


def run_cosco_proto_margin_fft_reg(train_data: np.ndarray, train_label: np.ndarray,
                                   test_data: np.ndarray, test_label: np.ndarray,
                                   input_size: int, args: Namespace) -> float:
    trainloader = DataLoader(
        Dataset(train_data, train_label),
        batch_size=1024,
        shuffle=True,
        num_workers=0,
    )
    return float(margin_fft_regularized_proto_neg_train_model(
        trainloader,
        train_label,
        test_data,
        test_label,
        input_size,
        args,
    ))


def run_single(run_args: Namespace) -> float:
    train_data, train_label, test_data, test_label = load_data(run_args)
    input_size = train_data.shape[-1]

    if run_args.model == "ed_1nn":
        return run_ed_1nn(train_data, train_label, test_data, test_label)
    if run_args.model == "dtw_1nn":
        return run_dtw_1nn(train_data, train_label, test_data, test_label)
    if run_args.model == "tapnet":
        return run_tapnet_baseline(train_data, train_label, test_data,
                                   test_label, input_size, run_args)
    if run_args.model == "resnet":
        return run_resnet_supervised(train_data, train_label, test_data,
                                     test_label, input_size, run_args)
    if run_args.model == "cosco":
        return run_cosco(train_data, train_label, test_data, test_label,
                         input_size, run_args)
    if run_args.model == "cosco_weighted":
        return run_cosco_weighted(train_data, train_label, test_data, test_label,
                                  input_size, run_args)
    if run_args.model == "cosco_dynamic_rho":
        return run_cosco_dynamic_rho(train_data, train_label, test_data, test_label,
                                     input_size, run_args)
    if run_args.model == "cosco_fft_reg":
        return run_cosco_fft_reg(train_data, train_label, test_data, test_label,
                                 input_size, run_args)
    if run_args.model == "cosco_proto_margin":
        return run_cosco_proto_margin(train_data, train_label, test_data, test_label,
                                      input_size, run_args)
    if run_args.model == "cosco_proto_margin_fft_reg":
        return run_cosco_proto_margin_fft_reg(train_data, train_label, test_data,
                                              test_label, input_size, run_args)
    raise ValueError(f"Unknown model: {run_args.model}")


def write_outputs(df: pd.DataFrame, args: argparse.Namespace,
                  device_str: str) -> None:
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    summary_md = os.path.join(args.out_dir, "summary.md")
    df.to_csv(summary_csv, index=False)

    pivot_df = df.copy()
    if "model_key" not in pivot_df.columns:
        pivot_df["model_key"] = pivot_df["model"]
    pivot = pivot_df.pivot_table(
        index=["dataset", "shot"],
        columns="model_key",
        values="accuracy",
        aggfunc="first",
    )
    accuracy_view = pivot.copy()
    if {"cosco", "cosco_dynamic_rho"}.issubset(set(accuracy_view.columns)):
        accuracy_view["dynamic_minus_cosco"] = (
            accuracy_view["cosco_dynamic_rho"] - accuracy_view["cosco"]
        )

    dynamic_rho_cols = [
        "dataset",
        "shot",
        "rho_min",
        "rho_mean",
        "rho_max",
        "rho_final",
        "proto_stress_mean",
        "proto_stress_final",
    ]
    dynamic_rho_rows = pd.DataFrame()
    if set(dynamic_rho_cols).issubset(df.columns):
        dynamic_rho_rows = df.loc[
            df["model"] == "cosco_dynamic_rho",
            dynamic_rho_cols,
        ].copy()
    dynamic_effect_rows = pd.DataFrame()
    if (
        not dynamic_rho_rows.empty
        and {"cosco", "cosco_dynamic_rho", "dynamic_minus_cosco"}.issubset(set(accuracy_view.columns))
    ):
        dynamic_effect_rows = accuracy_view.reset_index()[
            ["dataset", "shot", "cosco", "cosco_dynamic_rho", "dynamic_minus_cosco"]
        ].merge(dynamic_rho_rows, on=["dataset", "shot"], how="inner")
        dynamic_effect_rows["rho_mean_ratio"] = dynamic_effect_rows["rho_mean"] / args.rho
        dynamic_effect_rows["rho_max_ratio"] = dynamic_effect_rows["rho_max"] / args.rho

    fft_effect_rows = pd.DataFrame()
    fft_lambda_summary = pd.DataFrame()
    fft_reg_models = ["cosco_fft_reg", "cosco_proto_margin_fft_reg"]
    if {"model", "fft_reg_lambda"}.issubset(df.columns):
        cosco_rows = df.loc[
            df["model"] == "cosco",
            ["dataset", "shot", "accuracy"],
        ].rename(columns={"accuracy": "cosco"})
        fft_rows = df.loc[
            df["model"].isin(fft_reg_models),
            [
                "model",
                "dataset",
                "shot",
                "accuracy",
                "fft_reg_lambda",
                "fft_reg_loss_time_mean",
                "fft_reg_loss_freq_mean",
                "fft_reg_loss_total_mean",
                "fft_reg_lambda_min",
                "fft_reg_lambda_mean",
                "fft_reg_lambda_max",
                "fft_reg_lambda_final",
            ],
        ].rename(columns={"accuracy": "cosco_fft_reg_accuracy"})
        if not cosco_rows.empty and not fft_rows.empty:
            fft_effect_rows = fft_rows.merge(cosco_rows, on=["dataset", "shot"], how="inner")
            fft_effect_rows["fft_reg_minus_cosco"] = (
                fft_effect_rows["cosco_fft_reg_accuracy"] - fft_effect_rows["cosco"]
            )

            def _wins(delta: pd.Series) -> int:
                return int((delta > 1e-12).sum())

            def _ties(delta: pd.Series) -> int:
                return int((delta.abs() <= 1e-12).sum())

            def _losses(delta: pd.Series) -> int:
                return int((delta < -1e-12).sum())

            fft_lambda_summary = fft_effect_rows.groupby(["model", "fft_reg_lambda"]).agg(
                cosco_mean=("cosco", "mean"),
                cosco_fft_reg_mean=("cosco_fft_reg_accuracy", "mean"),
                mean_delta=("fft_reg_minus_cosco", "mean"),
                wins=("fft_reg_minus_cosco", _wins),
                ties=("fft_reg_minus_cosco", _ties),
                losses=("fft_reg_minus_cosco", _losses),
                loss_time_mean=("fft_reg_loss_time_mean", "mean"),
                loss_freq_mean=("fft_reg_loss_freq_mean", "mean"),
                loss_total_mean=("fft_reg_loss_total_mean", "mean"),
                effective_lambda_mean=("fft_reg_lambda_mean", "mean"),
                effective_lambda_min=("fft_reg_lambda_min", "min"),
                effective_lambda_max=("fft_reg_lambda_max", "max"),
            ).reset_index()
            fft_lambda_summary["gate_pass_mean_delta"] = (
                fft_lambda_summary["mean_delta"] >= 0.005
            )

    proto_margin_effect_rows = pd.DataFrame()
    proto_margin_summary = pd.DataFrame()
    if {"model", "proto_margin_value", "proto_margin_beta"}.issubset(df.columns):
        cosco_rows = df.loc[
            df["model"] == "cosco",
            ["dataset", "shot", "accuracy"],
        ].rename(columns={"accuracy": "cosco"})
        margin_rows = df.loc[
            df["model"].isin(["cosco_proto_margin", "cosco_proto_margin_fft_reg"]),
            [
                "model",
                "dataset",
                "shot",
                "accuracy",
                "proto_margin_value",
                "proto_margin_beta",
                "proto_margin_base_loss_mean",
                "proto_margin_loss_mean",
                "proto_margin_total_loss_mean",
                "proto_margin_positive_rate_mean",
                "proto_margin_gap_mean",
            ],
        ].rename(columns={"accuracy": "cosco_proto_margin"})
        if not cosco_rows.empty and not margin_rows.empty:
            proto_margin_effect_rows = margin_rows.merge(
                cosco_rows,
                on=["dataset", "shot"],
                how="inner",
            )
            proto_margin_effect_rows["proto_margin_minus_cosco"] = (
                proto_margin_effect_rows["cosco_proto_margin"]
                - proto_margin_effect_rows["cosco"]
            )

            def _wins(delta: pd.Series) -> int:
                return int((delta > 1e-12).sum())

            def _ties(delta: pd.Series) -> int:
                return int((delta.abs() <= 1e-12).sum())

            def _losses(delta: pd.Series) -> int:
                return int((delta < -1e-12).sum())

            proto_margin_summary = proto_margin_effect_rows.groupby(
                ["model", "proto_margin_value", "proto_margin_beta"]
            ).agg(
                cosco_mean=("cosco", "mean"),
                cosco_proto_margin_mean=("cosco_proto_margin", "mean"),
                mean_delta=("proto_margin_minus_cosco", "mean"),
                wins=("proto_margin_minus_cosco", _wins),
                ties=("proto_margin_minus_cosco", _ties),
                losses=("proto_margin_minus_cosco", _losses),
                base_loss_mean=("proto_margin_base_loss_mean", "mean"),
                margin_loss_mean=("proto_margin_loss_mean", "mean"),
                total_loss_mean=("proto_margin_total_loss_mean", "mean"),
                positive_rate_mean=("proto_margin_positive_rate_mean", "mean"),
                gap_mean=("proto_margin_gap_mean", "mean"),
            ).reset_index()
            proto_margin_summary["gate_pass_mean_delta"] = (
                proto_margin_summary["mean_delta"] >= 0.005
            )
            proto_margin_summary = proto_margin_summary.sort_values(
                ["mean_delta", "wins"],
                ascending=[False, False],
            )

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Quick COSCO Benchmark\n\n")
        f.write(f"- torch: `{torch.__version__}` | device: `{device_str}`\n")
        f.write(f"- epochs for neural models: {args.nEpoch}\n")
        f.write(f"- datasets: `{', '.join(args.datasets)}`\n")
        f.write(f"- shots: `{', '.join(map(str, args.shots))}`\n")
        f.write(f"- base seed: `{args.seed}`\n")
        f.write(f"- deterministic torch: `{args.deterministic_torch}`\n")
        f.write(f"- COSCO variant: `{'dummy_noop' if args.dummy_cosco_improvement else 'original'}`\n\n")
        f.write(f"- weighted prototype gamma: `{args.weighted_proto_gamma}`\n")
        f.write(f"- weighted prototype distance mode: `{args.weighted_proto_mode}`\n\n")
        f.write(f"- dynamic rho alpha: `{args.dynamic_rho_alpha}`\n")
        f.write(f"- dynamic rho min ratio: `{args.dynamic_rho_min_ratio}`\n")
        f.write(f"- dynamic rho max ratio: `{args.dynamic_rho_max_ratio}`\n\n")
        f.write(f"- FFT regularization lambdas: `{', '.join(map(str, args.fft_reg_lambdas))}`\n\n")
        f.write(f"- prototype margin values: `{', '.join(map(str, args.proto_margin_values))}`\n")
        f.write(f"- prototype margin betas: `{', '.join(map(str, args.proto_margin_betas))}`\n\n")
        f.write(accuracy_view.to_markdown(floatfmt=".4f"))
        if not proto_margin_summary.empty:
            f.write("\n\n## Prototype margin effect by hyperparameter\n\n")
            f.write(proto_margin_summary.to_markdown(index=False, floatfmt=".6f"))
        if not proto_margin_effect_rows.empty:
            f.write("\n\n## Prototype margin effect\n\n")
            f.write(proto_margin_effect_rows.to_markdown(index=False, floatfmt=".6f"))
        if not fft_lambda_summary.empty:
            f.write("\n\n## FFT regularization effect by lambda\n\n")
            f.write(fft_lambda_summary.to_markdown(index=False, floatfmt=".6f"))
        if not fft_effect_rows.empty:
            f.write("\n\n## FFT regularization effect\n\n")
            f.write(fft_effect_rows.to_markdown(index=False, floatfmt=".6f"))
        if not dynamic_effect_rows.empty:
            f.write("\n\n## Dynamic rho effect\n\n")
            f.write(dynamic_effect_rows.to_markdown(index=False, floatfmt=".6f"))
        if not dynamic_rho_rows.empty:
            f.write("\n\n## Dynamic rho summary\n\n")
            f.write(dynamic_rho_rows.to_markdown(index=False, floatfmt=".6f"))
        f.write("\n\n## Full rows\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    print(f"\n[saved] {summary_csv}")
    print(f"[saved] {summary_md}")
    print("\n=== Accuracy comparison ===")
    print(accuracy_view.to_string(float_format=lambda x: f"{x:.4f}"))
    if not fft_lambda_summary.empty:
        print("\n=== FFT regularization effect by lambda ===")
        print(fft_lambda_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        ))
    if not proto_margin_summary.empty:
        print("\n=== Prototype margin effect by hyperparameter ===")
        print(proto_margin_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        ))
    if not dynamic_effect_rows.empty:
        print("\n=== Dynamic rho effect ===")
        print(dynamic_effect_rows.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        ))
    if not dynamic_rho_rows.empty:
        print("\n=== Dynamic rho summary ===")
        print(dynamic_rho_rows.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        ))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick benchmark for COSCO improvement iterations."
    )
    parser.add_argument("--datasets", nargs="+",
                        default=["SpokenArabicDigits", "RacketSports", "Heartbeat", "JapaneseVowels", "Libras"],)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 10],
                        choices=[1, 10])
    parser.add_argument("--models", nargs="+",
                        default=["cosco", "cosco_proto_margin"],
                        choices=MODEL_CHOICES)
    parser.add_argument("--nEpoch", type=int, default=100,
                        help="Epochs for TapNet, supervised ResNet, and COSCO.")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--deterministic_torch", action="store_true",
                        help="Request deterministic PyTorch kernels where available.")
    parser.add_argument("--out_dir", default="outputs/quick_bench/")
    parser.add_argument("--dummy_cosco_improvement", action="store_true",
                        help="Enable the no-op COSCO improvement hook.")
    parser.add_argument("--weighted_proto_gamma", type=float, default=1.0,
                        help="Temperature gamma for weighted prototype softmax.")
    parser.add_argument("--weighted_proto_mode", choices=["close", "far"],
                        default="close",
                        help="'close' uses softmax(-distance/gamma); 'far' uses softmax(distance/gamma).")
    parser.add_argument("--dynamic_rho_alpha", type=float, default=0.25,
                        help="Prototype-geometry stress multiplier for dynamic SAM rho.")
    parser.add_argument("--dynamic_rho_min_ratio", type=float, default=0.5,
                        help="Minimum dynamic rho as a ratio of --rho.")
    parser.add_argument("--dynamic_rho_max_ratio", type=float, default=1.15,
                        help="Maximum dynamic rho as a ratio of --rho.")
    parser.add_argument("--fft_reg_lambdas", nargs="+", type=float,
                        default=[0.1],
                        help="Auxiliary FFT prototypical-loss weights for cosco_fft_reg.")
    parser.add_argument("--proto_margin_values", nargs="+", type=float,
                        default=[0.0],
                        help="Minimum correct-vs-nearest-wrong prototype distance gaps.")
    parser.add_argument("--proto_margin_betas", nargs="+", type=float,
                        default=[0.05],
                        help="Weights for the auxiliary prototype margin loss.")
    parser.add_argument("--log_every", type=int, default=1,
                        help="Print COSCO epoch logs every N epochs; 0 disables epoch logs.")
    args = parser.parse_args()

    set_seed(args.seed, deterministic_torch=args.deterministic_torch)
    os.makedirs(args.out_dir, exist_ok=True)

    device_str = (
        f"cuda ({torch.cuda.get_device_name(0)})"
        if torch.cuda.is_available() else "cpu"
    )
    print(f"[info] torch {torch.__version__} | device = {device_str}")

    rows = []
    for dataset in args.datasets:
        for shot in args.shots:
            for model in args.models:
                is_fft_reg_model = model in {"cosco_fft_reg", "cosco_proto_margin_fft_reg"}
                is_proto_margin_model = model in {"cosco_proto_margin", "cosco_proto_margin_fft_reg"}
                lambda_values = args.fft_reg_lambdas if is_fft_reg_model else [np.nan]
                margin_values = args.proto_margin_values if is_proto_margin_model else [np.nan]
                beta_values = args.proto_margin_betas if is_proto_margin_model else [np.nan]
                run_specs = [
                    (fft_reg_lambda, proto_margin_value, proto_margin_beta)
                    for fft_reg_lambda in lambda_values
                    for proto_margin_value in margin_values
                    for proto_margin_beta in beta_values
                ]
                for fft_reg_lambda, proto_margin_value, proto_margin_beta in run_specs:
                    lambda_suffix = (
                        f" | lambda={fft_lambda_key(fft_reg_lambda)}"
                        if is_fft_reg_model
                        else ""
                    )
                    margin_suffix = (
                        f" | {proto_margin_key(proto_margin_value, proto_margin_beta)}"
                        if is_proto_margin_model
                        else ""
                    )
                    tag = f"{model:<18} | {dataset:<14} | {shot}-shot{lambda_suffix}{margin_suffix}"
                    print(f"\n[run] {tag}")
                    t0 = time.time()
                    run_seed = stable_run_seed(args.seed, dataset, shot)
                    set_seed(run_seed, deterministic_torch=args.deterministic_torch)
                    run_args = make_run_args(
                        args,
                        dataset,
                        shot,
                        model,
                        fft_reg_lambda,
                        proto_margin_value,
                        proto_margin_beta,
                    )
                    try:
                        acc = run_single(run_args)
                        status = "ok"
                    except Exception as exc:  # noqa: BLE001
                        acc = float("nan")
                        status = f"failed: {type(exc).__name__}: {exc}"
                        print(f"[err] {tag}: {status}", file=sys.stderr)

                    elapsed = round(time.time() - t0, 1)
                    dynamic_summary = getattr(run_args, "dynamic_rho_summary", {}) or {}
                    fft_summary = getattr(run_args, "fft_reg_summary", {}) or {}
                    model_key = (
                        f"{model}_l{fft_lambda_key(fft_reg_lambda)}"
                        if is_fft_reg_model
                        else (
                            f"{model}_{proto_margin_key(proto_margin_value, proto_margin_beta)}"
                            if is_proto_margin_model
                            else model
                        )
                    )
                    rows.append({
                        "model": model,
                        "model_key": model_key,
                        "dataset": dataset,
                        "shot": shot,
                        "accuracy": acc,
                        "elapsed_sec": elapsed,
                        "seed": run_seed,
                        "status": status,
                        "cosco_variant": (
                            "dummy_noop"
                            if model == "cosco" and args.dummy_cosco_improvement
                            else (
                                f"weighted_{args.weighted_proto_mode}_gamma={args.weighted_proto_gamma}"
                                if model == "cosco_weighted"
                                else (
                                    f"dynamic_rho_proto_geometry_alpha={args.dynamic_rho_alpha}"
                                    if model == "cosco_dynamic_rho"
                                    else (
                                        f"fft_reg_lambda={fft_lambda_key(fft_reg_lambda)}"
                                        if is_fft_reg_model
                                        else (
                                            proto_margin_key(proto_margin_value, proto_margin_beta)
                                            if is_proto_margin_model
                                            else ""
                                        )
                                    )
                                )
                            )
                        ),
                        "rho_min": dynamic_summary.get("rho_min", np.nan),
                        "rho_mean": dynamic_summary.get("rho_mean", np.nan),
                        "rho_max": dynamic_summary.get("rho_max", np.nan),
                        "rho_final": dynamic_summary.get("rho_final", np.nan),
                        "proto_stress_mean": dynamic_summary.get("proto_stress_mean", np.nan),
                        "proto_stress_final": dynamic_summary.get("proto_stress_final", np.nan),
                        "fft_reg_lambda": (
                            float(fft_reg_lambda)
                            if is_fft_reg_model
                            else np.nan
                        ),
                        "fft_reg_loss_time_mean": fft_summary.get("fft_reg_loss_time_mean", np.nan),
                        "fft_reg_loss_freq_mean": fft_summary.get("fft_reg_loss_freq_mean", np.nan),
                        "fft_reg_loss_total_mean": fft_summary.get("fft_reg_loss_total_mean", np.nan),
                        "fft_reg_lambda_min": fft_summary.get("fft_reg_lambda_min", np.nan),
                        "fft_reg_lambda_mean": fft_summary.get("fft_reg_lambda_mean", np.nan),
                        "fft_reg_lambda_max": fft_summary.get("fft_reg_lambda_max", np.nan),
                        "fft_reg_lambda_final": fft_summary.get("fft_reg_lambda_final", np.nan),
                        "proto_margin_value": (
                            float(proto_margin_value)
                            if is_proto_margin_model
                            else np.nan
                        ),
                        "proto_margin_beta": (
                            float(proto_margin_beta)
                            if is_proto_margin_model
                            else np.nan
                        ),
                        "proto_margin_base_loss_mean": getattr(
                            run_args, "proto_margin_summary", {}
                        ).get("proto_margin_base_loss_mean", np.nan),
                        "proto_margin_loss_mean": getattr(
                            run_args, "proto_margin_summary", {}
                        ).get("proto_margin_loss_mean", np.nan),
                        "proto_margin_total_loss_mean": getattr(
                            run_args, "proto_margin_summary", {}
                        ).get("proto_margin_total_loss_mean", np.nan),
                        "proto_margin_positive_rate_mean": getattr(
                            run_args, "proto_margin_summary", {}
                        ).get("proto_margin_positive_rate_mean", np.nan),
                        "proto_margin_gap_mean": getattr(
                            run_args, "proto_margin_summary", {}
                        ).get("proto_margin_gap_mean", np.nan),
                    })
                    print(f"[done] {tag} -> acc={acc:.4f} ({elapsed}s, {status})")

                    partial_df = pd.DataFrame(rows)
                    partial_df.to_csv(os.path.join(args.out_dir, "summary_partial.csv"),
                                      index=False)

    write_outputs(pd.DataFrame(rows), args, device_str)


if __name__ == "__main__":
    main()
