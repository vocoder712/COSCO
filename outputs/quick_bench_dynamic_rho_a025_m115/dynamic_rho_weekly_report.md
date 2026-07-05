# COSCO 动态扰动半径周报

## 本周工作

本周围绕 COSCO 的 SAM 扰动半径 `rho` 做了动态自适应改进。核心想法是：
原始 COSCO 使用固定 `rho=0.1`，不区分 dataset、shot、训练阶段和原型几何状态；
本次改为根据当前 batch embedding 的原型几何压力动态调整 SAM 半径。

实现上新增了 `cosco_dynamic_rho`：

- 在 `utils/proto_model.py` 中计算原型几何压力：

  \[
  stress =
  \frac{
  mean(d_{own})
  }{
  mean(d_{nearest\_other}) + \epsilon
  }
  \]

- 用该压力调整 SAM 半径：

  \[
  rho_t =
  rho_{base}
  \cdot
  clamp(1 + \alpha \cdot stress, min\_ratio, max\_ratio)
  \]

- 在 `compare_quick_bench.py` 中新增 `cosco_dynamic_rho` 模型，并增加
  `Dynamic rho effect` 输出表，把 accuracy delta、rho 统计、prototype stress
  放在同一张表中，便于判断扰动是否真的产生作用。
- 加入 `--log_every`，用于关闭或降低 COSCO epoch 日志频率，方便做多轮快速迭代。

## 实验过程

第一版参数较激进：

- `dynamic_rho_alpha=1.0`
- `dynamic_rho_max_ratio=2.0`

结果显示，10-shot 下 `rho` 经常被放大到 `0.16` 到 `0.19`，部分数据集准确率明显下降。
典型下降包括 `Heartbeat`、`Libras`、`SpokenArabicDigits`。

随后改为保守配置：

- `dynamic_rho_alpha=0.5`
- `dynamic_rho_max_ratio=1.3`

该配置能明显减小下降幅度，但总体平均仍略低于原始 COSCO。

最终胜出的配置为：

- `dynamic_rho_alpha=0.25`
- `dynamic_rho_min_ratio=0.5`
- `dynamic_rho_max_ratio=1.15`
- `rho=0.1`

该配置已设为当前默认动态 rho 参数。

## 当前效果

评估脚本：

```powershell
conda run -n cosco --no-capture-output python compare_quick_bench.py --dynamic_rho_alpha 0.25 --dynamic_rho_max_ratio 1.15 --log_every 0 --out_dir outputs/quick_bench_dynamic_rho_a025_m115/
```

数据集：

- `SpokenArabicDigits`
- `RacketSports`
- `Heartbeat`
- `JapaneseVowels`
- `Libras`

shots：

- `1`
- `10`

总体平均：

| model | mean accuracy |
|---|---:|
| `cosco` | `0.661237` |
| `cosco_dynamic_rho` | `0.663100` |
| delta | `+0.001863` |

逐项结果：

| dataset | shot | cosco | cosco_dynamic_rho | delta |
|---|---:|---:|---:|---:|
| Heartbeat | 1 | 0.7268 | 0.7268 | 0.0000 |
| Heartbeat | 10 | 0.6537 | 0.6537 | 0.0000 |
| JapaneseVowels | 1 | 0.6676 | 0.6676 | 0.0000 |
| JapaneseVowels | 10 | 0.9027 | 0.9000 | -0.0027 |
| Libras | 1 | 0.4222 | 0.4167 | -0.0056 |
| Libras | 10 | 0.8667 | 0.8611 | -0.0056 |
| RacketSports | 1 | 0.4803 | 0.5000 | +0.0197 |
| RacketSports | 10 | 0.7961 | 0.8092 | +0.0132 |
| SpokenArabicDigits | 1 | 0.3415 | 0.3392 | -0.0023 |
| SpokenArabicDigits | 10 | 0.7549 | 0.7567 | +0.0018 |

结果判断：

- 当前动态 rho 已经让总体平均准确率略微超过原始 COSCO。
- 主要收益来自 `RacketSports`，其次是 `SpokenArabicDigits` 10-shot。
- `Heartbeat` 基本持平。
- `JapaneseVowels`、`Libras` 有小幅下降。
- 提升幅度很小，不应过度解读为稳定强改进。

## 问题与结论

本次工作说明：动态 SAM 半径是一个可用的改进接口，但“stress 越大 rho 越大”
不是稳定策略。

观察到的规律：

- 高 stress 数据集如果继续增大 `rho`，容易过度正则化，导致准确率下降。
- 中低 stress 数据集小幅增大 `rho` 可能有帮助，例如 `RacketSports`。
- 1-shot 下类内散度为 0，当前几何压力无法真正解释 1-shot 改动；1-shot 的小幅差异更可能来自 CUDA 非完全确定性或训练轨迹微扰。

因此，当前 `cosco_dynamic_rho` 更适合作为弱正收益 baseline，而不是最终主线结论。

## 下周计划

建议继续保留动态 rho，但不要把它作为唯一主线。

优先尝试：

1. 窗口型动态 rho。

   不再单调增加 `rho`。只在 prototype stress 处于中等区间时略微增大，
   高 stress 时保持原始 `rho` 或减小 `rho`。

2. margin 驱动动态 rho。

   用正确原型距离和最近错误原型距离的 margin 作为依据，比单纯类内散度更适合 1-shot。

3. loss / sharpness gap 驱动动态 rho。

   直接估计 SAM 两次前向之间的 loss gap，用优化层面的 sharpness 信号调节扰动半径。

4. 从别的角度继续改进 embedding。

   当前 quick bench 多次显示 DTW-1NN 在 `RacketSports` 上强于 COSCO，
   说明 ResNet embedding 没有充分吸收时序弹性对齐信息。下周可尝试：

   - DTW 距离蒸馏到 embedding 距离；
   - pairwise metric loss；
   - 时序增强一致性约束。

建议路线：

- 短期继续做动态 rho 的窗口型/保护型版本，因为改动小、已有弱正收益。
- 中期转向 margin loss 或 DTW 蒸馏，因为它们更可能带来实质性提升。
