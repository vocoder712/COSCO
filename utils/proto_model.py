"""
COSCO 训练与评估核心 / core training-and-evaluation routines for COSCO.

中文说明:
    本文件提供两个对外可用函数:
    - `proto_neg_train_model`: 用 ResNet + SAM + 原型损失 (negative-distance
       变体) 训练模型, 并在测试集上以原型距离做最近原型分类.
    - `full_training`: 入口函数, 负责加载数据 -> 选择模型 (resnet / tapnet)
       -> 训练 -> 评估 -> 保存结果. 由 `run.py` 调用.

English:
    This file exposes two public functions:
    - `proto_neg_train_model`: trains a ResNet backbone with SAM + the
       negative-distance Prototypical Loss, then performs nearest-prototype
       classification on the test set.
    - `full_training`: high-level entry that loads data, dispatches to the
       requested baseline (resnet / tapnet), trains, evaluates, and saves
       the result. Called by `run.py`.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from Prototypical_Loss import PrototypicalLoss
from Prototypical_Loss import prototypical_testing as ptest
from Baselines.ResNet import *
from Baselines.TapNet import train_tapnet  # 让 full_training 能在 tapnet 分支调用
from SAM import SAM

from utils.load_data import *
from utils.save import *

# 直接导入具体的 BatchNorm 类, 避免依赖 torch 内部的 _BatchNorm 私有 API.
# Import the concrete BatchNorm classes instead of the private _BatchNorm API.
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d


def disable_running_stats(model):
    """
    在 SAM 的第二步前临时关闭 BatchNorm running stats 的更新.

    Temporarily freezes the running-stats update of every BatchNorm layer
    before SAM's second forward-backward step, so that the perturbed
    forward pass does not corrupt the BN statistics.
    """
    def _disable(module):
        if isinstance(module, (BatchNorm1d, BatchNorm2d, BatchNorm3d)):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    """
    恢复 BatchNorm running stats 的正常更新.

    Restore the running-stats momentum that was saved by
    `disable_running_stats`, so that the next "clean" forward pass updates
    BN statistics normally.
    """
    def _enable(module):
        if isinstance(module, (BatchNorm1d, BatchNorm2d, BatchNorm3d)) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def proto_neg_train_model(trainloader, train_label, test_data, test_label, input_size, args):
    """
    使用 ResNet + SAM + Prototypical Loss(neg) 完成 COSCO 主流程.

    Train the COSCO main pipeline: ResNet backbone optimised with SAM and
    the negative-distance variant of Prototypical Loss, then evaluate on
    the test set via nearest-centroid classification.

    Parameters
    ----------
    trainloader : DataLoader
        训练集 DataLoader / training-set DataLoader.
    train_label : np.ndarray
        训练标签, 仅用于推断类别数 / training labels, used only to infer
        the number of classes.
    test_data : np.ndarray
        测试特征, shape (n, t, c) / test tensor, shape (n, t, c).
    test_label : np.ndarray
        测试标签 / test labels.
    input_size : int
        通道数 / number of input channels.
    args : argparse.Namespace
        命令行参数 / parsed CLI arguments.

    Returns
    -------
    acc : float
        测试集 top-1 准确率 / top-1 accuracy on the test set.
    """
    # 自动选择 GPU / CPU 设备 / auto-pick GPU when available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构造 ResNet 主干 / build the ResNet backbone
    model_resnet = ResNet(input_size=input_size, nb_classes=len(np.unique(train_label)))

    # CE 损失只是占位, 真正用到的是后面 PrototypicalLoss
    # CE loss is a placeholder; the real objective is PrototypicalLoss below.
    criterion = nn.CrossEntropyLoss()
    lr = args.lr
    rho = args.rho
    nEpoch = args.nEpoch

    runSAM = args.sam
    optimizer = args.optimizer

    # 选择原型损失的 negative-distance 变体 / pick the neg-distance variant
    criterion = PrototypicalLoss(flag='neg')

    # 是否启用 Sharpness-Aware Minimization
    # Decide whether to wrap the base optimiser with SAM.
    if runSAM is False:
        optimizer = torch.optim.SGD(model_resnet.parameters(), lr=lr, momentum=0.9)
    else:
        base_optimizer = torch.optim.SGD  # 论文中 SAM 内部使用的基础优化器
        optimizer = SAM(model_resnet.parameters(), base_optimizer,
                        lr=lr, momentum=0.9, rho=rho)

    # 模型搬到目标设备 / move model to target device
    model_resnet = model_resnet.to(device)

    # 早停相关变量 (原作者保留但实际未启用) / early-stopping bookkeeping
    # (kept from upstream; not actively used).
    best_loss = 10000
    max_limit = 20
    counter = 0

    # 训练主循环 / main training loop
    for epoch in range(nEpoch):
        running_loss = 0.0
        val_running_loss = 0.0
        all_embeddings = []
        all_labels = []

        for i, data in enumerate(trainloader, 0):
            # 取一个 batch / fetch one batch (inputs, labels)
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # ---------- SAM 第一次 forward-backward / first SAM step ----------
            enable_running_stats(model_resnet)  # 允许 BN 统计正常更新
            optimizer.zero_grad()

            # ResNet 输出: (logits, embedding)
            # `transpose(1, 2)` 把 (B, T, C) 转成 (B, C, T) 以适配 Conv1d.
            outputs1 = model_resnet(torch.tensor(inputs).transpose(1, 2))
            outputs = outputs1[0]   # logits, 这里未使用 / unused here
            embed = outputs1[1]     # 嵌入向量, 输入原型损失 / embedding fed to ProtoLoss

            # 标签去掉多余的维度 / squeeze trailing label dim
            labels = torch.squeeze(labels, dim=1)

            # 原型损失 / prototypical loss
            loss = criterion(embed, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # ---------- SAM 第二次 forward-backward / second SAM step ----------
            disable_running_stats(model_resnet)  # 冻结 BN 统计
            tmp = criterion(
                model_resnet(torch.tensor(inputs).transpose(1, 2).float())[1],
                labels,
            )
            tmp.backward()
            optimizer.second_step(zero_grad=True)

            optimizer.zero_grad()

            # 累计 loss / accumulate loss
            running_loss += loss.item()

            # 末轮收集 embedding/label, 用于计算训练集类原型
            # On the last epoch, gather embeddings/labels to build training
            # centroids that will be used at inference time.
            if epoch == nEpoch - 1:
                all_embeddings.append(embed.detach().cpu())
                all_labels.append(labels.detach().cpu())

        # 末轮: 计算每类的类原型 / on the last epoch, compute per-class centroids
        if epoch == nEpoch - 1:
            all_embeddings = torch.cat(all_embeddings)
            print(all_embeddings.size())
            all_labels = torch.cat(all_labels)
            train_centroids = criterion._compute_class_centroid(all_labels, all_embeddings)

        print("Epoch:", epoch + 1, "-->", running_loss, loss.item(), tmp.item())

    print('Finished Training')

    # 保存训练集原型, 后续推理 / 复现都依赖它
    # Persist the training centroids; both inference and reproduction rely on them.
    torch.save(train_centroids, 'train_centroids.pt')

    # ---------- 测试阶段 / inference & evaluation ----------
    test_data = torch.from_numpy(test_data).float()
    test_data = test_data.to(device)

    # 前向获取测试集嵌入 / forward pass to obtain test embeddings
    pred, embed = model_resnet(test_data.transpose(1, 2).float())

    # 加载训练集原型 / reload the saved centroids
    train_centroids = torch.load('train_centroids.pt')

    # 最近原型分类 / nearest-centroid prediction
    predicted_test_labels = ptest(embed, train_centroids)

    correct = 0
    total = 0
    labels = torch.squeeze(torch.from_numpy(test_label), dim=1)
    total = labels.size(0)
    correct = (predicted_test_labels.to(device) == labels.to(device)).sum().item()
    acc = correct / total

    print("Final Accuracy: ", acc)
    return acc


def full_training(args):
    """
    入口函数: 加载数据 -> 训练所选模型 -> 评估 -> 保存结果.

    High-level entry: load data, train the chosen backbone, evaluate,
    persist the result.

    Parameters
    ----------
    args : argparse.Namespace
        来自 `run.py` 的参数对象 / argument namespace from `run.py`.

    Returns
    -------
    acc : np.ndarray
        每次重复实验的准确率数组 / array of per-run accuracies.
    """
    # 读取训练 / 测试数据 / load train/test splits from disk
    train_data, train_label, test_data, test_label = load_data(args)

    # 构造 torch Dataset (供 ResNet 路径使用) / wrap into torch Dataset for ResNet
    traindata = Dataset(train_data, train_label)

    # 通道维度 (UEA 数据为 (N, T, C)) / number of channels (UEA is (N, T, C))
    input_size = train_data.shape[-1]

    # 不同模型期望的输入形式不同 / different backbones expect different label shapes
    if args.model == "tapnet":
        # TapNet 来自 sklearn 风格的 API, 标签必须是 1D
        # TapNet uses the sklearn-style API and needs flat 1D labels.
        test_label = test_label.reshape(-1)
        train_label = train_label.reshape(-1)
    elif args.model == "resnet":
        # ResNet 走 torch DataLoader 路径; Windows 下必须 num_workers=0,
        # 否则会触发 spawn / pickling 错误.
        # ResNet uses a torch DataLoader; on Windows num_workers must be 0
        # or DataLoader spawn/pickling will crash on the inline Dataset class.
        batch_size = 1024
        trainloader = DataLoader(traindata, batch_size=batch_size,
                                 shuffle=True, num_workers=0)

    acc = []

    # 多次重复以获得平均值 (默认 1 次, 与上游一致)
    # Repeat several runs and average; default is one repeat (matches upstream).
    for i in range(1):
        if args.model == 'tapnet':
            acc_tmp = train_tapnet(train_data, train_label, test_data, test_label, input_size, args)
        elif args.model == 'resnet':
            acc_tmp = proto_neg_train_model(trainloader, train_label, test_data, test_label, input_size, args)
        print(i)
        acc.append(acc_tmp)

    acc = np.array(acc)

    # 写入文本结果 / write text result file
    save_to_file_directory(acc, args)

    # 追加到汇总 CSV / append to the aggregated CSV
    save_to_dataframe(acc, args)

    return acc

