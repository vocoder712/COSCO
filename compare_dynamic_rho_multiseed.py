"""CPU-friendly paired three-seed benchmark for COSCO dynamic-rho variants."""

import argparse
import os
import time
from argparse import Namespace

import numpy as np
import pandas as pd
import torch

from compare_quick_bench import make_run_args, run_single, set_seed, stable_run_seed


MODELS = ["cosco", "cosco_dynamic_rho", "cosco_geometry_rho"]


def build_shared_args(cli):
    return Namespace(
        lr=cli.lr,
        rho=cli.rho,
        nEpoch=cli.nEpoch,
        normalize=cli.normalize,
        out_dir=cli.out_dir,
        dummy_cosco_improvement=False,
        weighted_proto_gamma=1.0,
        weighted_proto_mode="close",
        dynamic_rho_alpha=cli.dynamic_rho_alpha,
        geometry_rho_alpha=cli.dynamic_rho_alpha,
        geometry_rho_min_ratio=cli.dynamic_rho_min_ratio,
        dynamic_rho_min_ratio=cli.dynamic_rho_min_ratio,
        dynamic_rho_max_ratio=cli.dynamic_rho_max_ratio,
        geometry_ema_beta=cli.geometry_ema_beta,
        geometry_margin_target=cli.geometry_margin_target,
        geometry_protect_threshold=cli.geometry_protect_threshold,
        geometry_protect_strength=cli.geometry_protect_strength,
        fft_reg_lambdas=[0.1],
        proto_margin_values=[0.0],
        proto_margin_betas=[0.05],
        log_every=cli.log_every,
        seed=cli.seeds[0],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Paired multi-seed benchmark for dynamic SAM rho."
    )
    parser.add_argument("--datasets", nargs="+", default=["BasicMotions", "RacketSports"])
    parser.add_argument("--shots", nargs="+", type=int, choices=[1, 10], default=[1, 10])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--seeds", nargs=3, type=int, default=[10, 20, 30],
                        metavar=("SEED1", "SEED2", "SEED3"))
    parser.add_argument("--nEpoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--dynamic_rho_alpha", type=float, default=0.15)
    parser.add_argument("--dynamic_rho_min_ratio", type=float, default=0.75)
    parser.add_argument("--dynamic_rho_max_ratio", type=float, default=1.15)
    parser.add_argument("--geometry_ema_beta", type=float, default=0.9)
    parser.add_argument("--geometry_margin_target", type=float, default=0.35)
    parser.add_argument("--geometry_protect_threshold", type=float, default=0.35)
    parser.add_argument("--geometry_protect_strength", type=float, default=0.75)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--deterministic_torch", action="store_true", default=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=0)
    parser.add_argument("--out_dir", default="outputs/dynamic_rho_multiseed/")
    cli = parser.parse_args()

    os.makedirs(cli.out_dir, exist_ok=True)
    torch.set_num_threads(max(1, cli.threads))
    shared = build_shared_args(cli)
    rows = []

    for dataset in cli.datasets:
        for shot in cli.shots:
            for base_seed in cli.seeds:
                paired_seed = stable_run_seed(base_seed, dataset, shot)
                for model in cli.models:
                    set_seed(paired_seed, cli.deterministic_torch)
                    shared.seed = base_seed
                    run_args = make_run_args(shared, dataset, shot, model)
                    started = time.time()
                    try:
                        accuracy = run_single(run_args)
                        status = "ok"
                    except Exception as exc:  # keep the other paired runs alive
                        accuracy = float("nan")
                        status = f"failed: {type(exc).__name__}: {exc}"
                    geometry = getattr(run_args, "dynamic_rho_summary", {}) or {}
                    rows.append({
                        "dataset": dataset,
                        "shot": shot,
                        "base_seed": base_seed,
                        "seed": paired_seed,
                        "model": model,
                        "accuracy": accuracy,
                        "elapsed_sec": round(time.time() - started, 2),
                        "status": status,
                        "rho_mean": geometry.get("rho_mean", np.nan),
                        "rho_min": geometry.get("rho_min", np.nan),
                        "rho_max": geometry.get("rho_max", np.nan),
                        "pressure_mean": geometry.get("proto_stress_mean", np.nan),
                        "boundary_mean": geometry.get("geometry_boundary_mean", np.nan),
                        "crowding_mean": geometry.get("geometry_crowding_mean", np.nan),
                        "compactness_mean": geometry.get("geometry_compactness_mean", np.nan),
                    })
                    print(
                        f"[done] {dataset} {shot}-shot seed={base_seed} {model}: "
                        f"acc={accuracy:.4f} ({status})"
                    )
                    pd.DataFrame(rows).to_csv(
                        os.path.join(cli.out_dir, "runs_partial.csv"), index=False
                    )

    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(cli.out_dir, "runs.csv"), index=False)
    summary = runs.groupby(["dataset", "shot", "model"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        successful_runs=("accuracy", "count"),
        elapsed_mean_sec=("elapsed_sec", "mean"),
        rho_mean=("rho_mean", "mean"),
        pressure_mean=("pressure_mean", "mean"),
        boundary_mean=("boundary_mean", "mean"),
        crowding_mean=("crowding_mean", "mean"),
        compactness_mean=("compactness_mean", "mean"),
    )
    summary.to_csv(os.path.join(cli.out_dir, "summary.csv"), index=False)

    paired = runs.pivot_table(
        index=["dataset", "shot", "base_seed"],
        columns="model",
        values="accuracy",
        aggfunc="first",
    ).reset_index()
    for model in cli.models:
        if model != "cosco" and {"cosco", model}.issubset(paired.columns):
            paired[f"{model}_minus_cosco"] = paired[model] - paired["cosco"]
    paired.to_csv(os.path.join(cli.out_dir, "paired_deltas.csv"), index=False)

    with open(os.path.join(cli.out_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write("# Dynamic rho multi-seed benchmark\n\n")
        handle.write(f"- device: CPU; torch `{torch.__version__}`; threads: `{cli.threads}`\n")
        handle.write(f"- epochs: `{cli.nEpoch}`; seeds: `{cli.seeds}`\n")
        handle.write("- all model comparisons use paired initialization/shuffle seeds\n\n")
        handle.write("## Three-seed mean and standard deviation\n\n")
        handle.write(summary.to_markdown(index=False, floatfmt=".6f"))
        handle.write("\n\n## Paired per-seed deltas\n\n")
        handle.write(paired.to_markdown(index=False, floatfmt=".6f"))
        handle.write("\n")

    print("\n=== three-seed summary ===")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
