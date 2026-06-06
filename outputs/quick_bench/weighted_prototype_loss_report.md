# COSCO 原型损失加权改进小结

## 原始 COSCO 原型损失

设一个 mini-batch 的 embedding 为

$$
\mathbf{z}_i=f_\theta(\mathbf{x}_i), \quad y_i \in \{1,\dots,C\}
$$

对第 \(c\) 类，原始 COSCO 使用类内 embedding 的算术均值作为原型：

$$
\mathbf{p}_c
=
\frac{1}{|\mathcal{S}_c|}
\sum_{i \in \mathcal{S}_c}
\mathbf{z}_i
$$

其中：

$$
\mathcal{S}_c=\{i \mid y_i=c\}
$$

样本 \(\mathbf{z}_i\) 到所有类原型的欧氏距离为：

$$
d_{ic}=\|\mathbf{z}_i-\mathbf{p}_c\|_2
$$

当前代码中的 `neg` 原型损失使用负距离做分类分数：

$$
s_{ic}=-d_{ic}
$$

然后计算交叉熵：

$$
\mathcal{L}_{proto}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathrm{CE}
\left(
\mathrm{softmax}(\mathbf{s}_i),
y_i
\right)
$$

直观上，原始方法默认类内每个支持样本对原型贡献相同。

## 加权均值原型

本次改进只修改原型计算方式，其他部分保持不变：仍然使用 ResNet embedding、SAM 训练、负距离原型分类。

对第 \(c\) 类，先按原方法计算临时均值原型：

$$
\boldsymbol{\mu}_c
=
\frac{1}{|\mathcal{S}_c|}
\sum_{i \in \mathcal{S}_c}
\mathbf{z}_i
$$

再计算类内每个样本到该均值原型的距离：

$$
r_i
=
\|\mathbf{z}_i-\boldsymbol{\mu}_c\|_2,
\quad i \in \mathcal{S}_c
$$

使用温度参数 \(\gamma>0\) 计算权重。当前默认实现使用 `close` 模式：

$$
\alpha_i
=
\frac{
\exp(-r_i/\gamma)
}{
\sum_{j \in \mathcal{S}_c}
\exp(-r_j/\gamma)
},
\quad i \in \mathcal{S}_c
$$

也就是说，离原均值原型越近的样本权重越大。

新的加权原型为：

$$
\tilde{\mathbf{p}}_c
=
\sum_{i \in \mathcal{S}_c}
\alpha_i \mathbf{z}_i
$$

随后仍然用负欧氏距离作为分类分数：

$$
\tilde{d}_{ic}
=
\|\mathbf{z}_i-\tilde{\mathbf{p}}_c\|_2
$$

$$
\tilde{s}_{ic}
=
-\tilde{d}_{ic}
$$

$$
\mathcal{L}_{weighted}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathrm{CE}
\left(
\mathrm{softmax}(\tilde{\mathbf{s}}_i),
y_i
\right)
$$

代码中也保留了 `far` 模式：

$$
\alpha_i
=
\frac{
\exp(r_i/\gamma)
}{
\sum_{j \in \mathcal{S}_c}
\exp(r_j/\gamma)
}
$$

该模式会让远离均值的样本权重更大，目前不作为默认设置。

## 当前实验结果

实验设置：原始 COSCO 与 `cosco_weighted` 对比，数据集不超过 5 个，包含 1-shot 和 10-shot。

| dataset | shot | cosco | cosco_weighted | delta |
|---|---:|---:|---:|---:|
| Heartbeat | 1 | 0.7268 | 0.7268 | 0.0000 |
| Heartbeat | 10 | 0.6537 | 0.6488 | -0.0049 |
| JapaneseVowels | 1 | 0.6676 | 0.6676 | 0.0000 |
| JapaneseVowels | 10 | 0.9027 | 0.9000 | -0.0027 |
| Libras | 1 | 0.4222 | 0.4222 | 0.0000 |
| Libras | 10 | 0.8667 | 0.8111 | -0.0556 |
| RacketSports | 1 | 0.5000 | 0.4934 | -0.0066 |
| RacketSports | 10 | 0.7961 | 0.7961 | 0.0000 |
| SpokenArabicDigits | 1 | 0.3420 | 0.3338 | -0.0082 |
| SpokenArabicDigits | 10 | 0.7549 | 0.7240 | -0.0309 |

## 结果判断

这个加权原型方案目前没有显示出正收益。

现象很明确：

- 1-shot 下基本不可能产生差异。每类只有 1 个样本时，均值原型就是该样本本身，加权均值仍然等于它。
- 10-shot 下多数数据集下降，尤其 `Libras` 和 `SpokenArabicDigits` 降幅明显。
- 少数结果持平，但没有稳定提升。

主要问题可能是：当前权重机制会进一步收缩到类内中心点，削弱边界样本的作用。few-shot 场景里，支持集本来就很小，10-shot 也不足以可靠估计类内分布。把离中心远的样本降权，可能会丢掉类内变化模式，导致测试样本靠近边界时分类变差。

因此，这个方案目前困难较大，可能行不通。至少在当前实现和当前数据集上，它不像是一个有希望的主线改进。

## 关于论文结果的疑点

当前代码复现实验中，COSCO 的准确率与论文报告相比存在明显差距：论文结果普遍偏高，而本地按公开代码、固定数据划分和可复现 seed 跑出来的结果没有达到论文水平。

这说明存在复现疑点，但不能仅凭当前结果直接断定原作者“参水”。更稳妥的解释空间包括：

- 论文使用的预处理、归一化、随机种子或数据划分与仓库代码不完全一致。
- 论文可能做了多次运行后取最优或平均，但代码文档没有完整说明。
- 当前公开实现可能不是论文最终实验代码。
- 超参数可能针对每个数据集单独调过，而仓库默认参数是统一设置。
- 也不能排除论文实验报告存在选择性汇报或不透明处理。

结论：论文结果需要谨慎看待。后续对比应以本地固定 seed、固定 split、固定脚本的结果为准，不要把论文表格直接当作可靠上界。

## 下一步改进方向

1. 不再继续主推“中心距离 softmax 加权均值”。

   它在 1-shot 没有效果，在 10-shot 倾向下降。

2. 尝试保留边界样本信息。

   当前 `close` 模式强调中心样本，但 few-shot 分类可能更需要边界覆盖。可以测试：

   $$
   \alpha_i \propto \exp(r_i/\gamma)
   $$

   即 `far` 模式，但需要防止异常点支配原型。

3. 使用混合原型。

   不直接替换均值原型，而是插值：

   $$
   \mathbf{p}^{mix}_c
   =
   (1-\lambda)\mathbf{p}^{mean}_c
   +
   \lambda\mathbf{p}^{weighted}_c
   $$

   其中 \(\lambda\) 可固定，也可按 shot 或类内方差自适应。

4. 引入多原型而不是单原型。

   每类一个均值原型过于粗糙。可以考虑每类 \(K\) 个 prototype：

   $$
   \{\mathbf{p}_{c,1}, \dots, \mathbf{p}_{c,K}\}
   $$

   分类时取最近子原型：

   $$
   d_{ic}=\min_k \|\mathbf{z}_i-\mathbf{p}_{c,k}\|_2
   $$

5. 用 DTW/ED 结果指导 embedding。

   之前快速实验里，DTW-1NN 在 `RacketSports` 上明显强于 COSCO，说明原始时序的弹性对齐信息没有被 ResNet embedding 充分吸收。可以考虑：

   - DTW 距离蒸馏到 embedding 距离；
   - 加入 pairwise metric loss；
   - 对时序增强后保持 embedding 一致性。

6. 先修复评估可信度。

   后续所有改进都应固定：

   - dataset split；
   - seed；
   - model initialization；
   - DataLoader shuffle；
   - epoch；
   - summary 输出。

   否则微小改动很容易被随机波动掩盖。

