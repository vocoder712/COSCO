"""
Quick benchmark for COSCO improvement iterations.

This script is intentionally separate from `compare_models.py`. The old script
keeps the COSCO-vs-TapNet reproduction path; this one is a fast smoke benchmark
for future COSCO changes. It compares raw 1NN baselines, TapNet, supervised
ResNet, and COSCO on a small set of datasets and both 1-shot / 10-shot splits.
"""

import argparse
import os
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
from utils.proto_model import proto_neg_train_model, weighted_proto_neg_train_model


MODEL_CHOICES = ["ed_1nn", "dtw_1nn", "tapnet", "resnet", "cosco", "cosco_weighted"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_args(args: argparse.Namespace, dataset: str, shot: int,
                  model: str) -> Namespace:
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
    )


def flatten_labels(y: np.ndarray) -> np.ndarray:
    return y.reshape(-1)


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

    raise ValueError(f"Unknown model: {run_args.model}")


def write_outputs(df: pd.DataFrame, args: argparse.Namespace,
                  device_str: str) -> None:
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    summary_md = os.path.join(args.out_dir, "summary.md")
    df.to_csv(summary_csv, index=False)

    pivot = df.pivot_table(
        index=["dataset", "shot"],
        columns="model",
        values="accuracy",
        aggfunc="first",
    )

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Quick COSCO Benchmark\n\n")
        f.write(f"- torch: `{torch.__version__}` | device: `{device_str}`\n")
        f.write(f"- epochs for neural models: {args.nEpoch}\n")
        f.write(f"- datasets: `{', '.join(args.datasets)}`\n")
        f.write(f"- shots: `{', '.join(map(str, args.shots))}`\n")
        f.write(f"- COSCO variant: `{'dummy_noop' if args.dummy_cosco_improvement else 'original'}`\n\n")
        f.write(f"- weighted prototype gamma: `{args.weighted_proto_gamma}`\n")
        f.write(f"- weighted prototype distance mode: `{args.weighted_proto_mode}`\n\n")
        f.write(pivot.to_markdown(floatfmt=".4f"))
        f.write("\n\n## Full rows\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    print(f"\n[saved] {summary_csv}")
    print(f"[saved] {summary_md}")
    print("\n=== Accuracy comparison ===")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick benchmark for COSCO improvement iterations."
    )
    parser.add_argument("--datasets", nargs="+",
                        default=["SpokenArabicDigits", "RacketSports"])
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 10],
                        choices=[1, 10])
    parser.add_argument("--models", nargs="+", default=["cosco", "cosco_weighted"],
                        choices=MODEL_CHOICES)
    parser.add_argument("--nEpoch", type=int, default=100,
                        help="Epochs for TapNet, supervised ResNet, and COSCO.")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="outputs/quick_bench/")
    parser.add_argument("--dummy_cosco_improvement", action="store_true",
                        help="Enable the no-op COSCO improvement hook.")
    parser.add_argument("--weighted_proto_gamma", type=float, default=1.0,
                        help="Temperature gamma for weighted prototype softmax.")
    parser.add_argument("--weighted_proto_mode", choices=["close", "far"],
                        default="close",
                        help="'close' uses softmax(-distance/gamma); 'far' uses softmax(distance/gamma).")
    args = parser.parse_args()

    if len(args.datasets) > 3:
        raise ValueError("Use at most 3 datasets for the quick benchmark.")

    set_seed(args.seed)
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
                tag = f"{model:<8} | {dataset:<14} | {shot}-shot"
                print(f"\n[run] {tag}")
                t0 = time.time()
                run_args = make_run_args(args, dataset, shot, model)
                try:
                    acc = run_single(run_args)
                    status = "ok"
                except Exception as exc:  # noqa: BLE001
                    acc = float("nan")
                    status = f"failed: {type(exc).__name__}: {exc}"
                    print(f"[err] {tag}: {status}", file=sys.stderr)

                elapsed = round(time.time() - t0, 1)
                rows.append({
                    "model": model,
                    "dataset": dataset,
                    "shot": shot,
                    "accuracy": acc,
                    "elapsed_sec": elapsed,
                    "status": status,
                    "cosco_variant": (
                        "dummy_noop"
                        if model == "cosco" and args.dummy_cosco_improvement
                        else (
                            f"weighted_{args.weighted_proto_mode}_gamma={args.weighted_proto_gamma}"
                            if model == "cosco_weighted"
                            else ""
                        )
                    ),
                })
                print(f"[done] {tag} -> acc={acc:.4f} ({elapsed}s, {status})")

                partial_df = pd.DataFrame(rows)
                partial_df.to_csv(os.path.join(args.out_dir, "summary_partial.csv"),
                                  index=False)

    write_outputs(pd.DataFrame(rows), args, device_str)


if __name__ == "__main__":
    main()
