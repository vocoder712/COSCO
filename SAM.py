"""
Sharpness-Aware Minimization (SAM) 优化器.

中文说明:
    SAM 是 Foret et al. (2021) 提出的优化策略, 用于减小模型在 loss 邻域
    内的最坏情况损失, 从而提升泛化能力. COSCO 的核心创新点之一就是把 SAM
    应用到小样本时序分类任务上.

    使用方式 (与 Foret 原版一致, 需两次前向反向):
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        loss2 = criterion(model(x), y)
        loss2.backward()
        optimizer.second_step(zero_grad=True)

English:
    SAM (Foret et al. 2021) minimises the worst-case loss within a small
    neighbourhood of the current weights, which improves generalisation.
    A core contribution of COSCO is plugging SAM into few-shot multivariate
    time-series classification.

    Usage (two forward-backward passes per update, matching the original
    paper):

        loss = criterion(model(x), y)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        loss2 = criterion(model(x), y)
        loss2.backward()
        optimizer.second_step(zero_grad=True)
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


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization 优化器封装.

    Wraps a base optimiser (e.g. SGD) to implement SAM. Use `first_step` /
    `second_step` for explicit two-pass training (recommended for SAM +
    BatchNorm-momentum freezing).
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Parameters
        ----------
        params : iterable
            模型参数 / model parameters.
        base_optimizer : callable
            构造基础优化器的类, 例如 `torch.optim.SGD`.
            Constructor of the base optimiser, e.g. `torch.optim.SGD`.
        rho : float
            邻域半径 / neighbourhood radius (controls how far we climb the loss).
        adaptive : bool
            是否使用 ASAM 的参数自适应缩放 / whether to use ASAM-style scaling.
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # 实例化基础优化器, 并与 self.param_groups 对齐
        # Instantiate the base optimiser and share param_groups with self.
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        第一步: 沿梯度方向 "爬升" 到 w + e(w), 找到邻域内损失最大处.

        Step 1: ascend along the gradient to reach w + e(w), i.e. the
        approximate worst-case point inside the rho-ball.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                # 备份原参数, 第二步要回滚 / cache original w, restored in step 2
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        第二步: 回滚到原参数 w, 再用 w+e(w) 处算得的梯度做实际更新.

        Step 2: revert to the original w, then apply the base optimiser
        using gradients computed at w + e(w).
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        闭包形式一次性完成两步; COSCO 训练里没有使用这个入口.

        Closure-style API that performs both steps in one call. COSCO's
        training loop uses the explicit `first_step` / `second_step`
        sequence instead.
        """
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # 闭包必须完整执行 forward + backward.
        # The closure must run a full forward-backward pass.
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        """
        所有参数梯度的二范数, 用于 first_step 的缩放.

        Global L2 norm of all parameter gradients used to scale the ascent
        step. Tolerates model parallelism by moving everything onto a
        shared device first.
        """
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # 确保基础优化器与 self 共享同一份 param_groups.
        # Keep base optimiser's param_groups aligned with self after reload.
        self.base_optimizer.param_groups = self.param_groups
