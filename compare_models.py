"""
COSCO vs TapNet 对比实验入口 / COSCO vs TapNet comparison runner.

中文说明:
    本脚本依次跑两种模型 (COSCO = ResNet+SAM+ProtoLoss, TapNet 基线) 在
    同一组数据集 / shot 上的实验, 并把结果汇总到一个 CSV 与 markdown 表
    中, 便于直接比较.

    默认设置 (可用命令行参数覆盖):
        - 数据集 : BasicMotions
        - shot   : 1 与 10
        - 训练轮数: 100
        - 输出目录: outputs/comparison/
        - 结果文件: outputs/comparison/summary.csv
        - markdown: outputs/comparison/summary.md

English:
    Runs COSCO (ResNet + SAM + ProtoLoss) and the TapNet baseline back to
    back on the same datasets / shot settings, then aggregates the results
    into a CSV and a markdown table for easy comparison.

    Default configuration (override via CLI flags):
        - datasets : BasicMotions
        - shots    : 1 and 10
        - epochs   : 100
        - out dir  : outputs/comparison/
        - csv file : outputs/comparison/summary.csv
        - markdown : outputs/comparison/summary.md
"""

import argparse
import os
import sys
import time
from argparse import Namespace

import numpy as np
import pandas as pd
import torch

# 引入项目内模块 / project-local imports
from utils.load_data import load_data, Dataset
from utils.proto_model import proto_neg_train_model
from Baselines.TapNet import train_tapnet
from torch.utils.data import DataLoader


def make_run_args(dataset: str, shot: int, model: str, epochs: int,
                  save_dir: str) -> Namespace:
    """
    构造一个与 run.py 等价的 args 对象, 供训练函数使用.

    Build an argparse.Namespace that mirrors `run.py`'s CLI defaults so
    the training functions can be called programmatically.
    """
    return Namespace(
        lr=0.01,
        rho=0.1,
        nEpoch=epochs,
        dataset=dataset,
        shot=shot,
        normalize=False,
        model=model,
        sam=True,
        optimizer='adam',
        prototypical_loss=True,
        prototypical_loss_type='neg',
        save_dir=save_dir,
        save_name=f'{model}_{dataset}_{shot}shot.csv',
    )


def run_single(args: Namespace) -> float:
    """
    跑一次单模型 / 单数据集 / 单 shot 的训练 + 评估.

    Run one (model, dataset, shot) training + evaluation pass and return
    the test-set accuracy.
    """
    # 数据加载 / load data
    train_data, train_label, test_data, test_label = load_data(args)

    input_size = train_data.shape[-1]

    if args.model == 'tapnet':
        # TapNet 需要 1D 标签 / TapNet expects 1D labels
        flat_train_label = train_label.reshape(-1)
        flat_test_label = test_label.reshape(-1)
        return train_tapnet(train_data, flat_train_label,
                            test_data, flat_test_label,
                            input_size, args)

    # COSCO / ResNet 路径
    traindata = Dataset(train_data, train_label)
    trainloader = DataLoader(traindata, batch_size=1024,
                             shuffle=True, num_workers=0)
    return proto_neg_train_model(trainloader, train_label,
                                 test_data, test_label,
                                 input_size, args)


def main():
    parser = argparse.ArgumentParser(
        description="COSCO vs TapNet 对比实验 / comparison experiment"
    )
    parser.add_argument('--datasets', nargs='+', default=['BasicMotions'],
                        help='待对比的数据集名 / dataset names to evaluate')
    parser.add_argument('--shots', nargs='+', type=int, default=[1, 10],
                        choices=[1, 10],
                        help='shot 设置 / shot settings')
    parser.add_argument('--models', nargs='+', default=['resnet', 'tapnet'],
                        choices=['resnet', 'tapnet'],
                        help='要对比的模型 / models to compare')
    parser.add_argument('--nEpoch', type=int, default=100,
                        help='训练轮数 / number of epochs')
    parser.add_argument('--out_dir', type=str, default='outputs/comparison/',
                        help='输出目录 / output directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 报告设备信息 / report device info
    device_str = (
        f"cuda ({torch.cuda.get_device_name(0)})"
        if torch.cuda.is_available() else "cpu"
    )
    print(f"[info] torch {torch.__version__} | device = {device_str}")

    rows = []  # 收集所有结果 / collect every result row

    for dataset in args.datasets:
        for shot in args.shots:
            for model in args.models:
                tag = f"{model:<6} | {dataset:<22} | {shot}-shot"
                print(f"\n[run] {tag} ...")
                t0 = time.time()

                run_args = make_run_args(dataset, shot, model,
                                         args.nEpoch, args.out_dir)
                try:
                    acc = float(run_single(run_args))
                    status = 'ok'
                except Exception as e:  # noqa: BLE001
                    # 任一组合失败都不要影响其他组合.
                    # Never let a single combo break the whole sweep.
                    acc = float('nan')
                    status = f'failed: {type(e).__name__}: {e}'
                    print(f"[err] {tag}: {status}", file=sys.stderr)

                elapsed = time.time() - t0
                print(f"[done] {tag} -> acc={acc:.4f}  ({elapsed:.1f}s, {status})")

                rows.append({
                    'model': model,
                    'dataset': dataset,
                    'shot': shot,
                    'accuracy': acc,
                    'elapsed_sec': round(elapsed, 1),
                    'status': status,
                })

    # 写汇总 CSV / write summary CSV
    summary_csv = os.path.join(args.out_dir, 'summary.csv')
    df = pd.DataFrame(rows)
    df.to_csv(summary_csv, index=False)
    print(f"\n[saved] summary CSV: {summary_csv}")

    # 透视为对比表: 行=数据集+shot, 列=model.
    # Pivot into a comparison table: rows = dataset+shot, cols = model.
    pivot = df.pivot_table(index=['dataset', 'shot'],
                           columns='model', values='accuracy',
                           aggfunc='first')
    summary_md = os.path.join(args.out_dir, 'summary.md')
    with open(summary_md, 'w', encoding='utf-8') as f:
        f.write("# COSCO vs TapNet — accuracy comparison\n\n")
        f.write(f"- torch: `{torch.__version__}` | device: `{device_str}`\n")
        f.write(f"- epochs: {args.nEpoch}\n\n")
        f.write(pivot.to_markdown(floatfmt='.4f'))
        f.write("\n")
    print(f"[saved] summary markdown: {summary_md}")

    # 在控制台打印对比表 / print to console
    print("\n=== Accuracy comparison ===")
    print(pivot.to_string(float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
