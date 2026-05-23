"""
COSCO 框架训练入口脚本 / COSCO framework training entry script.

中文说明:
    本脚本是 COSCO 论文 (CIKM'24) 的官方训练入口, 用于在小样本多变量时间
    序列分类任务上对比/复现 COSCO (ResNet + SAM + Prototypical Loss) 以及
    TapNet 基线模型. 直接运行 `python run.py --help` 可查看所有可调参数.

English:
    This script is the official training entry point of the COSCO paper
    (CIKM'24). It runs few-shot multivariate time-series classification with
    either the COSCO framework (ResNet backbone + SAM + Prototypical Loss)
    or the TapNet baseline. Run `python run.py --help` to list all flags.
"""

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision import datasets, transforms, models
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import uuid
import argparse

# 项目内部模块 / project-local modules
from Prototypical_Loss import PrototypicalLoss
from Prototypical_Loss import prototypical_testing as ptest
from Baselines.ResNet import *
from Baselines.TapNet import *
from SAM import SAM

from utils.load_data import *
from utils.save import *
from utils.proto_model import *

if __name__ == '__main__':
    # 解析命令行参数 / parse CLI arguments
    parser = argparse.ArgumentParser(
        description="COSCO few-shot MTSC trainer / COSCO 小样本多变量时序分类训练器"
    )

    # 超参数 / Hyper-parameters
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率 / learning rate')
    parser.add_argument('--rho', type=float, default=0.1,
                        help='SAM 邻域半径 / SAM neighbourhood radius rho')
    parser.add_argument('--nEpoch', type=int, default=100,
                        help='训练轮数 / number of epochs')

    # 数据加载 / Data loading
    parser.add_argument('--dataset', type=str, default='CharacterTrajectories',
                        help='UEA 数据集名 / UEA dataset name')
    parser.add_argument('--shot', type=int, default=1, choices=[1, 10],
                        help='每类样本数 / number of support samples per class')
    parser.add_argument('--normalize', type=bool, default=False,
                        help='是否按时间维标准化 / whether to z-normalise per series')

    # 基线模型 / Baseline model (default ResNet)
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['resnet', 'tapnet'],
                        help='主干模型 / backbone choice')

    # SAM 配置 / SAM configuration
    parser.add_argument('--sam', type=bool, default=True,
                        help='是否启用 SAM / enable Sharpness-Aware Minimisation')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='SAM 之下的基础优化器 / base optimiser used inside SAM')

    # Prototypical Loss 原型损失配置
    parser.add_argument('--prototypical_loss', type=bool, default=True,
                        help='是否使用原型损失 / use prototypical loss')

    # 其他参数 / Other parameters
    parser.add_argument('--prototypical_loss_type', type=str, default='neg',
                        choices=['neg', 'sim', 'cos', 'negexp'],
                        help='原型损失变体 / variant of prototypical loss')

    # 结果保存 / Result-saving configuration
    parser.add_argument('--save_dir', type=str, default='content/classification_data/',
                        help='结果输出目录 / output directory for results')
    parser.add_argument('--save_name', type=str, default='results.csv',
                        help='汇总 CSV 文件名 / aggregated CSV file name')

    args = parser.parse_args()

    # 结果 DataFrame 的列定义 / columns of the result DataFrame
    columns = ["Dataset", "Shots", "Normalization", "Result"]

    # 初始化空 DataFrame / construct empty DataFrame
    df = pd.DataFrame(columns=columns)

    # 汇总 CSV 路径 / aggregated CSV filepath
    filepath = args.save_dir + args.save_name

    # 创建保存目录并写入空 CSV / create save directory and write empty CSV
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

    # 训练循环 / training loop
    # 注: 这里保留原作者的二重循环写法, 便于将来扩展到多数据集/多 shot 设置.
    # Note: keep the original nested loop so it is trivial to extend to
    # multiple datasets / shot settings later on.
    for dataset_name in [args.dataset]:
        for shot_dir in [args.shot]:
            full_training(args)
