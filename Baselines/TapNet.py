# TapNet baseline wrapper / TapNet 基线模型封装.
#
# 中文说明:
#   TapNet 是 IJCAI'20 提出的多变量时间序列分类网络. 此处通过 AEON 工具箱
#   提供的实现来训练并评估. 由于 aeon >= 1.4 已经移除 TapNetClassifier,
#   请确保运行环境中的 aeon 版本为 0.11.x.
#
# English:
#   TapNet (IJCAI'20) is a deep network for multivariate time-series
#   classification. We rely on the AEON toolkit's implementation here.
#   Note: aeon >= 1.4 has dropped TapNetClassifier, so the runtime
#   environment must pin aeon to a 0.11.x release.

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def train_tapnet(train_data, train_label, test_data, test_label, input_size, args):
    """
    使用 AEON 提供的 TapNet 对支持集进行训练, 并在测试集上评估准确率.

    Train TapNet (from AEON) on the support set and evaluate accuracy on
    the test set.

    Parameters
    ----------
    train_data : np.ndarray
        训练特征 / training tensor, shape (n_samples, n_channels, seq_len).
    train_label : np.ndarray
        训练标签 (1D) / 1D label array.
    test_data : np.ndarray
        测试特征 / test tensor.
    test_label : np.ndarray
        测试标签 (1D) / 1D label array.
    input_size : int
        通道数, 未直接使用 / number of channels (kept for signature symmetry).
    args : argparse.Namespace
        运行参数, 仅使用 `nEpoch` / runtime args, only `nEpoch` is consumed.

    Returns
    -------
    acc : float
        测试集 top-1 准确率 / top-1 accuracy on the test set.
    """
    # 延迟导入, 让 aeon >= 1.4 也能 import 本模块而不报错 (此时只有调用
    # train_tapnet 才会抛 ImportError, 让 ResNet/COSCO 路径可单独运行).
    # Lazy import so that `from Baselines.TapNet import *` still works on
    # aeon >= 1.4. The ImportError only surfaces when this function is
    # actually called, leaving the COSCO/ResNet path unaffected.
    from aeon.classification.deep_learning import TapNetClassifier

    # 构造 TapNet 分类器 / build TapNet classifier
    tapnet = TapNetClassifier(n_epochs=args.nEpoch, batch_size=8)

    # 拟合训练集 / fit on training data
    tapnet.fit(train_data, train_label)

    # 在测试集上预测 / predict on test set
    y_pred = tapnet.predict(test_data)

    # 准确率 / accuracy
    acc = accuracy_score(test_label, y_pred)
    print("Final Accuracy: ", acc)
    return acc


import torch  # 保留向下兼容的占位导入 / kept for backward-compat with original layout
