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

from Prototypical_Loss import PrototypicalLoss, WeightedPrototypicalLoss
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


def _prototype_geometry_stress(embed, labels, eps=1e-12):
    """
    Compute a batch-level prototype-geometry pressure score.

    The score is the mean distance to the sample's own class prototype divided
    by the mean distance to the nearest other class prototype. It is small when
    classes are compact and well separated, and larger when within-class spread
    is high relative to between-class separation.
    """
    with torch.no_grad():
        labels = labels.squeeze().long()
        unique_labels = torch.unique(labels, sorted=True)
        if unique_labels.numel() < 2:
            return embed.new_tensor(0.0)

        centroids = []
        target_indices = torch.empty_like(labels, dtype=torch.long)
        for class_index, class_label in enumerate(unique_labels):
            mask = labels == class_label
            if not torch.any(mask):
                continue
            centroids.append(embed[mask].mean(dim=0))
            target_indices[mask] = class_index

        if len(centroids) < 2:
            return embed.new_tensor(0.0)

        centroids = torch.stack(centroids, dim=0)
        distances = torch.cdist(embed, centroids, p=2)
        own_dist = distances.gather(1, target_indices.view(-1, 1)).squeeze(1)

        other_distances = distances.clone()
        other_distances.scatter_(1, target_indices.view(-1, 1), float("inf"))
        nearest_other_dist = other_distances.min(dim=1).values
        finite_mask = torch.isfinite(nearest_other_dist)
        if not torch.any(finite_mask):
            return embed.new_tensor(0.0)

        own_mean = own_dist[finite_mask].mean()
        other_mean = nearest_other_dist[finite_mask].mean()
        stress = own_mean / (other_mean + eps)
        return torch.nan_to_num(stress, nan=0.0, posinf=1e6, neginf=0.0)


def _set_sam_rho(optimizer, rho):
    """Update rho for every SAM parameter group."""
    for group in optimizer.param_groups:
        group["rho"] = rho


def _compute_dynamic_rho(base_rho, stress, alpha, min_ratio, max_ratio):
    scaled_rho = base_rho * (1.0 + alpha * stress)
    min_rho = base_rho * min_ratio
    max_rho = base_rho * max_ratio
    return max(min(scaled_rho, max_rho), min_rho)


def proto_neg_train_model(trainloader, train_label, test_data, test_label, input_size, args,
                          criterion_override=None, centroid_path='train_centroids.pt'):
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
    dynamic_rho = bool(getattr(args, 'dynamic_rho', False))
    dynamic_rho_alpha = float(getattr(args, 'dynamic_rho_alpha', 0.25))
    dynamic_rho_min_ratio = float(getattr(args, 'dynamic_rho_min_ratio', 0.5))
    dynamic_rho_max_ratio = float(getattr(args, 'dynamic_rho_max_ratio', 1.15))
    log_every = int(getattr(args, 'log_every', 1))

    runSAM = args.sam
    optimizer = args.optimizer

    # 选择原型损失的 negative-distance 变体 / pick the neg-distance variant.
    # `criterion_override` is used only by experimental COSCO variants; the
    # original COSCO call path keeps the mean-centroid PrototypicalLoss.
    criterion = criterion_override or PrototypicalLoss(flag='neg')

    # 是否启用 Sharpness-Aware Minimization
    # Decide whether to wrap the base optimiser with SAM.
    if runSAM is False:
        optimizer = torch.optim.SGD(model_resnet.parameters(), lr=lr, momentum=0.9)
    else:
        base_optimizer = torch.optim.SGD  # 论文中 SAM 内部使用的基础优化器
        optimizer = SAM(model_resnet.parameters(), base_optimizer,
                        lr=lr, momentum=0.9, rho=rho)

    if dynamic_rho and runSAM is False:
        dynamic_rho = False
    if dynamic_rho:
        if rho < 0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        if dynamic_rho_alpha < 0:
            raise ValueError(f"dynamic_rho_alpha must be non-negative, got {dynamic_rho_alpha}")
        if dynamic_rho_min_ratio < 0:
            raise ValueError(f"dynamic_rho_min_ratio must be non-negative, got {dynamic_rho_min_ratio}")
        if dynamic_rho_min_ratio > dynamic_rho_max_ratio:
            raise ValueError(
                "dynamic_rho_min_ratio must be <= dynamic_rho_max_ratio, "
                f"got {dynamic_rho_min_ratio} > {dynamic_rho_max_ratio}"
            )

    # 模型搬到目标设备 / move model to target device
    model_resnet = model_resnet.to(device)
    args.dynamic_rho_summary = {}
    rho_trace = []
    stress_trace = []

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
            if dynamic_rho:
                stress = float(_prototype_geometry_stress(embed.detach(), labels.detach()).item())
                current_rho = _compute_dynamic_rho(
                    rho,
                    stress,
                    dynamic_rho_alpha,
                    dynamic_rho_min_ratio,
                    dynamic_rho_max_ratio,
                )
                _set_sam_rho(optimizer, current_rho)
                rho_trace.append(current_rho)
                stress_trace.append(stress)
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

        should_log_epoch = (
            log_every > 0
            and (
                epoch == 0
                or epoch == nEpoch - 1
                or (epoch + 1) % log_every == 0
            )
        )
        if should_log_epoch:
            epoch_msg = f"Epoch: {epoch + 1} --> {running_loss} {loss.item()} {tmp.item()}"
            if dynamic_rho and rho_trace:
                epoch_msg += f" rho={rho_trace[-1]:.6f} proto_stress={stress_trace[-1]:.6f}"
            print(epoch_msg)

    print('Finished Training')
    if dynamic_rho and rho_trace:
        args.dynamic_rho_summary = {
            "rho_min": float(np.min(rho_trace)),
            "rho_mean": float(np.mean(rho_trace)),
            "rho_max": float(np.max(rho_trace)),
            "rho_final": float(rho_trace[-1]),
            "proto_stress_mean": float(np.mean(stress_trace)),
            "proto_stress_final": float(stress_trace[-1]),
        }

    # 保存训练集原型, 后续推理 / 复现都依赖它
    # Persist the training centroids; both inference and reproduction rely on them.
    torch.save(train_centroids, centroid_path)

    # ---------- 测试阶段 / inference & evaluation ----------
    test_data = torch.from_numpy(test_data).float()
    test_data = test_data.to(device)

    # 前向获取测试集嵌入 / forward pass to obtain test embeddings
    pred, embed = model_resnet(test_data.transpose(1, 2).float())

    # 加载训练集原型 / reload the saved centroids
    train_centroids = torch.load(centroid_path)

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


def weighted_proto_neg_train_model(trainloader, train_label, test_data, test_label, input_size, args):
    """
    COSCO variant using weighted prototypical centroids.

    The original COSCO implementation is preserved in `proto_neg_train_model`.
    This wrapper swaps only the centroid computation inside the prototypical
    loss: mean centroid -> distance-softmax weighted centroid.
    """
    gamma = getattr(args, 'weighted_proto_gamma', 1.0)
    distance_mode = getattr(args, 'weighted_proto_mode', 'close')
    criterion = WeightedPrototypicalLoss(
        flag='neg',
        gamma=gamma,
        distance_mode=distance_mode,
    )
    return proto_neg_train_model(
        trainloader,
        train_label,
        test_data,
        test_label,
        input_size,
        args,
        criterion_override=criterion,
        centroid_path='train_centroids_weighted.pt',
    )


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
