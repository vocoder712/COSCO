"""
原型损失实现 / Prototypical Loss for COSCO.

中文说明:
    本文件实现 COSCO 论文中使用的 "原型损失" 及其四种变体:
        - neg:     基于负距离的 softmax 交叉熵 (论文默认)
        - sim:     基于 1/欧氏距离 的相似度
        - cos:     基于余弦相似度
        - negexp:  基于 exp(-alpha*距离) 的相似度
    训练时按当前 mini-batch 计算每个类的均值嵌入作为该类原型,
    然后用 嵌入->原型 的距离/相似度 作为分类 logits.
    推理时则用最近邻原型直接分类 (`prototypical_testing`).

English:
    Implementation of the prototypical loss used by COSCO, plus four
    variants (neg / sim / cos / negexp). At training time, class
    centroids are computed from the current mini-batch embeddings and
    the embedding-to-centroid distance/similarity is used as the logit.
    At inference time, `prototypical_testing` performs nearest-centroid
    classification.
"""

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.nn.modules.batchnorm import _BatchNorm
import pandas as pd
import torch.nn as nn
import os
import uuid
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class PrototypicalLoss:
    """
    Prototypical Loss 与其变体. 通过 flag 选择使用哪种距离/相似度.

    Prototypical Loss with selectable distance/similarity variant.
    """

    def __init__(self, flag='neg'):
        # flag in {neg, sim, cos, negexp}
        self.flag = flag

    # ---------- 类原型计算 / class-centroid computation ----------

    def _compute_per_class_centroid(self, i, label, data):
        """
        计算第 i 类的均值嵌入 / mean embedding of class i.
        """
        label1d = label.squeeze()
        data_class = data[label1d == i, :]
        return torch.mean(data_class, 0, True)

    def _compute_class_centroid(self, label, data):
        """
        计算每个类的原型 (堆叠返回) / stack centroids for every class.
        """
        unique_labels = label.unique().squeeze()
        centroids = self._compute_per_class_centroid(unique_labels[0], label, data)
        for i in range(1, len(unique_labels)):
            index = unique_labels[i]
            centroids = torch.cat(
                (centroids, self._compute_per_class_centroid(index, label, data)),
                dim=0,
            )
        return centroids

    # ---------- 距离/相似度 / distance & similarity ----------

    def _cosine_similarity(self, data, centroid):
        """余弦相似度 / cosine similarity."""
        data_norm = torch.nn.functional.normalize(data, dim=1)
        centroid_norm = torch.nn.functional.normalize(centroid, dim=1)
        similarity = torch.mm(data_norm, centroid_norm.t())
        return similarity

    def _similarity_matrix(self, data, centroid):
        """基于 1/L2 距离的相似度 / similarity via 1 / L2-distance."""
        epsilon = 1e-6
        similarity = 1 / (torch.cdist(data, centroid, p=2) + epsilon)
        return similarity

    def _distance_matrix(self, data, centroid):
        """L2 距离矩阵 / L2 distance matrix."""
        distance = torch.cdist(data, centroid, p=2)
        return distance

    # ---------- 损失 / loss heads ----------

    def _prototypical_loss_sim(self, S, labels, alpha=0.01):
        """相似度->softmax->CE / similarity -> softmax -> CE loss."""
        softmax = torch.nn.Softmax(dim=1)
        o = softmax(S / alpha)
        labels = labels.squeeze().long()
        loss = F.cross_entropy(o, labels, reduction='mean')
        return loss

    def _prototypical_loss_neg(self, D, labels):
        """负距离 (论文默认) / negative-distance variant (default)."""
        softmax = torch.nn.Softmax(dim=1)
        o = softmax(-D)
        labels = labels.squeeze().long()
        loss = F.cross_entropy(o, labels, reduction='mean')
        return loss

    def _prototypical_loss_negexp(self, D, labels, alpha=0.1):
        """exp(-alpha*距离) 变体 / exp(-alpha*distance) variant."""
        softmax = torch.nn.Softmax(dim=1)
        o = softmax(-alpha * torch.exp(D))
        labels = labels.squeeze().long()
        loss = F.cross_entropy(o, labels, reduction='mean')
        return loss

    # ---------- 仿函数入口 / functor entry ----------

    def __call__(self, data, label):
        """根据 self.flag 计算损失 / dispatch on self.flag."""
        if self.flag == 'neg':
            centroids = self._compute_class_centroid(label, data)
            distance = self._distance_matrix(data, centroids)
            return self._prototypical_loss_neg(distance, label)
        elif self.flag == 'sim':
            centroids = self._compute_class_centroid(label, data)
            similarity = self._similarity_matrix(data, centroids)
            return self._prototypical_loss_sim(similarity, label)
        elif self.flag == 'cos':
            centroids = self._compute_class_centroid(label, data)
            similarity = self._cosine_similarity(data.detach(), centroids)
            return self._prototypical_loss_sim(similarity, label)
        elif self.flag == 'negexp':
            centroids = self._compute_class_centroid(label, data)
            distance = self._distance_matrix(data, centroids)
            return self._prototypical_loss_negexp(distance, label)


def prototypical_testing(test_embed, train_centroids):
    """
    最近原型分类 / nearest-centroid classification at inference time.

    Parameters
    ----------
    test_embed : Tensor
        测试样本嵌入 / test-set embeddings, shape (n, d).
    train_centroids : Tensor
        训练集每类原型 / per-class centroids from training set, shape (k, d).

    Returns
    -------
    test_label : Tensor
        预测类别 (argmin over centroids) / predicted class index per sample.
    """
    # 一律放到 CPU 计算距离, 避免 device 不一致问题.
    # Move both tensors to CPU so the distance computation never crashes
    # on a device mismatch.
    test_embed = test_embed.cpu()
    train_centroids = train_centroids.cpu()

    # 计算到每个原型的 L2 距离 / pairwise L2 distance to every centroid.
    cdist = torch.cdist(test_embed, train_centroids)

    # 取距离最小的类作为预测 / pick the closest centroid as the predicted class.
    test_label = torch.argmin(cdist, dim=1)

    return test_label
