"""
数据加载工具 / data-loading utilities for COSCO.

中文说明:
    `Datasets/` 目录按 UEA 多变量时间序列分类基准组织, 每个数据集下含:
        X_test.npy / y_test.npy           完整测试集
        {1,10}-shot/X_train.npy           小样本支持集 (按类别采样得到)
        {1,10}-shot/y_train.npy           对应支持集标签
    本模块负责按 args.dataset / args.shot 加载, 并可选做按时间维 z-score
    归一化.

English:
    The `Datasets/` directory follows the UEA multivariate TSC benchmark
    layout. Each dataset folder contains:
        X_test.npy / y_test.npy           full test split
        {1,10}-shot/X_train.npy           few-shot support set
        {1,10}-shot/y_train.npy           support-set labels
    This module loads those numpy arrays based on args.dataset / args.shot
    and optionally performs per-series z-score normalisation along the
    temporal axis.
"""

import torch
import argparse
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
    """
    极简 torch Dataset, 把 numpy 数组打包成 (X, y).

    Minimal torch Dataset that wraps numpy arrays into (X, y) tensors.
    """

    def __init__(self, X_train, y_train):
        # 必须把 float64 转为 float32, 否则 nn.Conv1d 会报
        # "expected scalar type Double but found Float".
        # Convert float64 -> float32, otherwise nn.Conv1d errors with
        # "expected scalar type Double but found Float".
        self.X = torch.from_numpy(X_train.astype(np.float32))

        # 标签必须转 Long, 否则 cross_entropy 会报
        # "expected scalar type Long but found Float".
        # Labels must be Long; otherwise cross_entropy raises
        # "expected scalar type Long but found Float".
        self.y = torch.from_numpy(y_train).type(torch.LongTensor)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


def load_data(args):
    """
    根据命令行参数加载 train/test 数据.

    Load train/test arrays based on CLI args.

    Parameters
    ----------
    args : argparse.Namespace
        必须包含 `dataset`, `shot`, `normalize` 字段.

    Returns
    -------
    train_data, train_label, test_data, test_label : np.ndarray
        分别为训练特征 / 训练标签 / 测试特征 / 测试标签.
    """
    # 仓库相对路径; 若把 run.py 迁出根目录, 请同步修改这里.
    # Repository-relative path; keep in sync if `run.py` is moved.
    base_path = "./Datasets/"
    dataset = args.dataset
    shot = args.shot
    normalize_data = args.normalize
    epsilon = 1e-8  # 数值稳定项 / numerical-stability epsilon

    # 读取完整测试集 / load full test split
    test_data = np.load(base_path + dataset + '/X_test.npy')
    test_label = np.load(base_path + dataset + '/y_test.npy')

    # 读取对应 shot 的训练支持集 / load few-shot support set
    train_data = np.load(base_path + dataset + '/' + str(shot) + '-shot/X_train.npy')
    train_label = np.load(base_path + dataset + '/' + str(shot) + '-shot/y_train.npy')

    # 可选的逐序列 z-score 归一化 / optional per-series z-score normalisation
    if normalize_data:
        train_data = (train_data - train_data.mean(axis=1)[:, None]) \
                     / (train_data.std(axis=1)[:, None] + epsilon)
        test_data = (test_data - test_data.mean(axis=1)[:, None]) \
                    / (test_data.std(axis=1)[:, None] + epsilon)

    return train_data, train_label, test_data, test_label
