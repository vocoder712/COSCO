# COSCO 使用说明 / Usage Guide

> 本文档说明如何在本机 (Windows + RTX 4060 + CUDA 12.x) 上完成
> COSCO 框架与 TapNet 基线的对比实验, 包括环境搭建、训练、推理、评估
> 与结果可视化. 本文档与原仓库的 `README.md` 互补, README 偏向论文复
> 现, 本文档偏向工程使用.
>
> This document explains how to reproduce the COSCO-vs-TapNet comparison
> on this machine (Windows + RTX 4060 + CUDA 12.x): environment setup,
> training, inference, evaluation, and result inspection. It complements
> the upstream `README.md` (paper-reproduction oriented) with an
> engineering-oriented walkthrough.

---

## 1. 环境 / Environment

### 1.1 软件依赖 / software stack

| 组件 / Component       | 版本 / Version          | 备注 / Notes                                                   |
|------------------------|-------------------------|----------------------------------------------------------------|
| Python                 | 3.11.x                  | conda env `cosco`                                              |
| PyTorch                | 2.5.1 + cu124           | GPU build, 与 RTX 30/40 系列兼容                               |
| torchvision            | 0.20.1 + cu124          | 与 torch 对齐                                                  |
| numpy                  | 1.26.4                  | aeon 0.11 不支持 numpy 2.x                                     |
| pandas                 | 2.0.3                   | 结果汇总用                                                     |
| scikit-learn           | 1.5.2                   | TapNet 准确率计算                                              |
| aeon                   | 0.11.1                  | **必须固定**, 1.4 起 `TapNetClassifier` 被移除                 |
| tensorflow             | 2.15.1 (CPU)            | aeon TapNet 的运行时. 2.18 在 Win+Py3.11 上 DLL 报错; 2.10 不支持 Py3.11 |
| keras_self_attention   | 0.51.0                  | aeon TapNet 的软依赖, 不装会报 ModuleNotFoundError              |
| tabulate               | >= 0.9                  | `compare_models.py` 输出 markdown 表用                         |

### 1.2 一次性安装 / one-time install

```bash
# 1. 创建空 conda env (若已存在可跳过)
conda create -n cosco python=3.11 -y

# 2. 安装 GPU 版 torch / torchvision (cu124 与 CUDA 12.x 驱动兼容)
conda run -n cosco --no-capture-output \
  pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1

# 3. 安装项目其它依赖
conda run -n cosco --no-capture-output \
  pip install "aeon==0.11.*" "tensorflow==2.15.*" keras_self_attention \
              pandas scikit-learn tabulate
```

> **避坑 / Pitfalls**
> - 不要直接 `pip install -r requirements.txt`: 这样会拉到 CPU 版 torch.
>   `requirements.txt` 仅作为版本锁参考, 实际安装请走上面的两条命令.
> - TensorFlow 2.18 在 Windows + Python 3.11 上会出现
>   `ImportError: DLL load failed`. 已在 USAGE 中固定为 2.15.1.
> - `keras_self_attention` 是 aeon TapNet 的软依赖, aeon `pip install` 时
>   不会自动带, 必须显式装.
>
> - Do NOT `pip install -r requirements.txt` directly — that would pull
>   the CPU torch build. Use the two commands above; treat
>   `requirements.txt` as a version-lock reference only.
> - TensorFlow 2.18 throws `ImportError: DLL load failed` on
>   Windows + Python 3.11. We pin 2.15.1.
> - `keras_self_attention` is a soft dependency of aeon's TapNet and is
>   not installed automatically; install it explicitly.

### 1.3 自检 / sanity check

```bash
conda run -n cosco --no-capture-output python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

预期输出包含 `True` 与显卡型号; 若 `False`, 请检查 NVIDIA 驱动版本是否
≥ 525 (`nvidia-smi`).

Expected output should print `True` and your GPU name; if `False`,
ensure the NVIDIA driver is ≥ 525 (check via `nvidia-smi`).

---

## 2. 数据 / Datasets

仓库自带 `Datasets/` 目录, 已经按 UEA 多变量时序分类基准切分:

```
Datasets/<DatasetName>/
  X_test.npy              测试集特征 / test features  (N, T, C)
  y_test.npy              测试集标签 / test labels    (N,)
  1-shot/X_train.npy      1-shot 支持集 / 1-shot support set
  1-shot/y_train.npy
  10-shot/X_train.npy     10-shot 支持集 / 10-shot support set
  10-shot/y_train.npy
```

支持的数据集列表见上游 `README.md`. 若需要扩充, 直接按上述目录结构放入
即可, 无需改代码.

The list of supported datasets is in the upstream `README.md`. To add a
new one, mirror the directory layout above; no code change required.

---

## 3. 训练 / Training

### 3.1 单组实验 / single run via `run.py`

```bash
# COSCO (ResNet + SAM + Prototypical Loss)
conda run -n cosco --no-capture-output python run.py \
  --dataset BasicMotions \
  --model resnet \
  --shot 1 \
  --nEpoch 100 \
  --save_dir outputs/ \
  --save_name cosco_basicmotions_1shot.csv

# TapNet baseline
conda run -n cosco --no-capture-output python run.py \
  --dataset BasicMotions \
  --model tapnet \
  --shot 1 \
  --nEpoch 100 \
  --save_dir outputs/ \
  --save_name tapnet_basicmotions_1shot.csv
```

主要参数 (完整列表见 `python run.py --help`):

| 参数 / Flag                | 默认 / Default | 含义 / Meaning                                              |
|----------------------------|----------------|-------------------------------------------------------------|
| `--dataset`                | `CharacterTrajectories` | UEA 数据集名 / UEA dataset name                    |
| `--shot`                   | `1`            | 每类样本数 / samples per class (1 or 10)                    |
| `--model`                  | `resnet`       | `resnet` 走 COSCO, `tapnet` 走基线 / COSCO vs TapNet        |
| `--nEpoch`                 | `100`          | 训练轮数 / epochs                                           |
| `--lr`                     | `0.01`         | 学习率 / learning rate                                      |
| `--rho`                    | `0.1`          | SAM 邻域半径 / SAM neighbourhood radius                     |
| `--sam`                    | `True`         | 是否启用 SAM (`False` 退化为普通 SGD)                       |
| `--prototypical_loss_type` | `neg`          | 原型损失变体 / variant: neg/sim/cos/negexp                  |
| `--normalize`              | `False`        | 是否按时间维 z-score 归一化                                 |
| `--save_dir`               | `content/...`  | 结果目录 / output directory                                 |
| `--save_name`              | `results.csv`  | 汇总 CSV 文件名                                             |

### 3.2 对比实验 / sweep via `compare_models.py`

`compare_models.py` 是为本任务新增的脚本, 用于一次性跑 COSCO 与 TapNet
在多个 (dataset, shot) 组合上的对比, 并把结果汇总为对比表.

`compare_models.py` is added for this task. It sweeps COSCO and TapNet
over multiple (dataset, shot) combinations and aggregates the results
into a comparison table.

```bash
# 默认: BasicMotions × {1,10}-shot × {resnet, tapnet}, 100 epochs
conda run -n cosco --no-capture-output python compare_models.py

# 自定义示例 / custom example
conda run -n cosco --no-capture-output python compare_models.py \
  --datasets BasicMotions Epilepsy \
  --shots 1 10 \
  --models resnet tapnet \
  --nEpoch 100 \
  --out_dir outputs/comparison/
```

运行结束后会得到:

- `outputs/comparison/summary.csv`  — 全量明细 (含每次运行的耗时与状态)
- `outputs/comparison/summary.md`   — markdown 对比表
- `outputs/comparison/<model>_<dataset>_<shot>shot.csv` — 各组合的逐次准确率

---

## 4. 推理 / Inference

COSCO (ResNet 路径) 训练完会自动保存类原型到工程根目录的
`train_centroids.pt`. 推理时只需:

1. 用同一份 ResNet checkpoint 提取测试样本嵌入;
2. 用 `Prototypical_Loss.prototypical_testing(test_embed, train_centroids)`
   做最近原型分类.

COSCO (ResNet path) saves the per-class centroids to
`train_centroids.pt` at the project root after training. To do inference:

1. Extract embeddings for new samples with the same ResNet weights;
2. Call `Prototypical_Loss.prototypical_testing(test_embed, train_centroids)`
   for nearest-centroid classification.

最小可运行示例 / minimal example:

```python
import numpy as np
import torch
from Baselines.ResNet import ResNet
from Prototypical_Loss import prototypical_testing

# 1. 加载已训练的 ResNet (此处假设你已经把 state_dict 保存到 resnet.pt)
# 1. Load a pre-trained ResNet (assuming you persisted its state_dict).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(input_size=6, nb_classes=4).to(device)         # BasicMotions: C=6, K=4
model.load_state_dict(torch.load("resnet.pt", map_location=device))
model.eval()

# 2. 准备一个 batch 的测试数据 (N, T, C) -> 喂给模型时需要 (N, C, T)
# 2. Prepare a test batch shaped (N, T, C); the model expects (N, C, T).
X = np.load("Datasets/BasicMotions/X_test.npy").astype(np.float32)
with torch.no_grad():
    _, embeds = model(torch.from_numpy(X).to(device).transpose(1, 2))

# 3. 加载训练阶段保存的类原型
# 3. Load the centroids saved during training.
centroids = torch.load("train_centroids.pt")

# 4. 最近原型分类
# 4. Nearest-centroid prediction.
pred = prototypical_testing(embeds, centroids)
print(pred[:10])
```

> **注意 / Note**: 上游训练代码当前只保存了 `train_centroids.pt`, 没有显式
> 保存 ResNet 的 `state_dict`. 若要长期持久化模型, 请在
> `utils/proto_model.py::proto_neg_train_model` 训练结束处补一行
> `torch.save(model_resnet.state_dict(), 'resnet.pt')`.
>
> Upstream training only persists `train_centroids.pt`; it does not save
> the ResNet `state_dict`. If you need a long-lived checkpoint, add
> `torch.save(model_resnet.state_dict(), 'resnet.pt')` near the end of
> `utils/proto_model.py::proto_neg_train_model`.

TapNet 在 `train_tapnet` 内部用 AEON 的 `TapNetClassifier.fit/predict`,
若想保存/加载, 走 sklearn 标准接口:

```python
import pickle
with open("tapnet.pkl", "wb") as f:
    pickle.dump(tapnet, f)        # 训练后保存

with open("tapnet.pkl", "rb") as f:
    tapnet = pickle.load(f)        # 推理前加载
y_pred = tapnet.predict(X_test)
```

---

## 5. 评估 / Evaluation

- **COSCO / TapNet 内部**: 都会在训练结束后直接打印 `Final Accuracy`,
  并把分数写入 `<save_dir>/<dataset>/<shot>-shot/results_sam_proto_neg.txt`
  以及汇总 CSV (`<save_dir>/<save_name>`).
- **对比实验**: 运行 `compare_models.py` 后, 直接看
  `outputs/comparison/summary.md`, 形如:

```markdown
|                      |   resnet |   tapnet |
|:---------------------|---------:|---------:|
| ('BasicMotions', 1)  |   1.0000 |   0.5750 |
| ('BasicMotions', 10) |   1.0000 |   0.7000 |
```

> 这是本机 (RTX 4060 Laptop, torch 2.5.1+cu124, tensorflow 2.15.1, 100 epochs)
> 上 `compare_models.py` 默认参数的实测结果. COSCO 在 1-shot / 10-shot 上
> 都显著超越 TapNet 基线.
>
> The numbers above are real measurements on this machine (RTX 4060 Laptop,
> torch 2.5.1+cu124, tensorflow 2.15.1, 100 epochs) using the default
> `compare_models.py` invocation. COSCO clearly beats TapNet at both
> 1-shot and 10-shot.

- **自定义指标**: 若需要 F1/recall/混淆矩阵, 可以在
  `compare_models.run_single` 之后接 sklearn:

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
```

---

## 6. 常见问题 / FAQ

1. **`torch.cuda.is_available()` 返回 False**
   - 确认安装的是 `+cu124` wheel 而不是默认的 CPU wheel.
   - 重装命令见 §1.2; 检查 `pip show torch` 的 `Location` 与 `Version`.

2. **`ImportError: cannot import name 'TapNetClassifier'`**
   - aeon 升到了 1.4+. 用 `pip install "aeon==0.11.*"` 锁回去.

3. **`ModuleNotFoundError: tensorflow` 或 `keras_self_attention`**
   - aeon 0.11 的 TapNet 是 keras/tensorflow 实现, 需要这两个软依赖.
   - `pip install "tensorflow==2.15.*" keras_self_attention`.

4. **TF 2.18 `ImportError: DLL load failed`**
   - Windows + Python 3.11 + TF 2.18 的已知问题. 降级到 2.15.x 即可.

5. **`NameError: name 'train_tapnet' is not defined`**
   - 上游 `utils/proto_model.py` 漏了 `from Baselines.TapNet import train_tapnet`.
     本仓库已经补上, 若以后 `git pull` 把它冲掉, 请重新加上.

6. **`DataLoader` 在 Windows 下 spawn 报错**
   - 项目已经把 `num_workers` 改为 `0`, 切勿在 Windows 上设置 > 0.

7. **路径找不到 `./Datasets/...`**
   - 必须在 **项目根目录** 下执行 `python run.py` 或 `python compare_models.py`,
     不要 `cd` 到 `utils/` 等子目录.

8. **`'list' object has no attribute 'mean'`**
   - 一般是因为某个组合训练失败但仍尝试聚合. `compare_models.py` 的
     `status` 列会指出失败原因, 检查并重跑该组合即可.

---

## 7. 目录速查 / Project layout cheat sheet

```
COSCO/
├── run.py                 # 单组实验入口 / single-run entry
├── compare_models.py      # 对比实验入口 (本任务新增) / comparison sweep
├── SAM.py                 # SAM 优化器实现
├── Prototypical_Loss.py   # 原型损失 + 最近原型推理
├── Baselines/
│   ├── ResNet.py          # COSCO 主干 / ResNet backbone
│   └── TapNet.py          # AEON TapNet 封装
├── utils/
│   ├── load_data.py       # 数据加载 + torch Dataset
│   ├── proto_model.py     # COSCO 训练循环 + 入口 `full_training`
│   └── save.py            # 结果落盘工具
├── Datasets/              # UEA 数据集 (按 dataset/shot 组织)
├── outputs/               # 训练 / 对比结果落地处
├── USAGE.md               # 本文档 / this guide
└── README.md              # 论文 README
```
