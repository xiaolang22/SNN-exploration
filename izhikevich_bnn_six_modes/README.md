# Izhikevich 脉冲神经网络六模式电刺激实验

基于 Izhikevich 脉冲神经元模型构建生物神经网络储备池（reservoir），模拟体外微电极阵列（MEA）实验中六种电刺激模式下的网络响应、突触可塑性演化与模式分类。

## 项目概述

本项目验证了一个核心假设：**通过 STDP 可塑性驱动的突触重组，脉冲神经网络能够逐步学会区分不同的电刺激模式**。

实验设计：
- 2 个刺激区域 × 3 种电压（100/300/500 mV）= 6 种刺激模式
- 每轮训练中 6 种模式按固定顺序重复刺激并切换
- 通过分类正确率的逐轮提升来量化网络的学习效果

## 方法

### 神经元模型

采用 Izhikevich (2003) 简化脉冲神经元模型：

```
dv/dt = 0.04v² + 5v + 140 - u + I
du/dt = a(bv - u)
if v >= 30 mV: v ← c, u ← u + d
```

- 兴奋性神经元 (80%)：Regular Spiking (RS) 参数
- 抑制性神经元 (20%)：Fast Spiking (FS) 参数

### 网络拓扑

- 1000 个神经元均匀随机分布在 3.85 × 2.10 mm 平面
- 距离依赖的连接概率：`P(i→j) = p × (0.35 + 0.65 × exp(-d/λ))`
- 对数正态分布初始化突触权重
- 100 个记录通道（模拟 MEA 电极）

### 突触可塑性

主实验只保留 **STDP**（Spike-Timing-Dependent Plasticity）：基于 pre/post spike 时序调节突触权重，仅作用于兴奋性突触。

### 刺激协议

- 双相脉冲：正相 1ms (+1.0) → 负相 1ms (-1.0) → 恢复期 3ms
- 高斯空间扩散场：`I(d) = α × V × exp(-d²/2σ²) × gain`
- 自动校准：网格搜索最优 α 和 σ 以匹配目标招募数

### 解码

- **原始计数特征**：刺激后 5-55ms 窗口内各记录通道的 spike count 向量
- **V100 特征**：取 spike count 最高的前 100 个通道索引
- 分类器：RandomForest (100 trees)，10-fold 交叉验证

## 安装

### 环境要求

- Python >= 3.10
- NumPy, Matplotlib, scikit-learn

### 安装依赖

```bash
pip install -r requirements.txt
```

## 运行

```bash
cd izhikevich_bnn_six_modes_new
python run.py
```

运行时间约 3-5 分钟（取决于硬件）。

## 参数配置

所有参数在 `src/config.py` 中以 dataclass 形式定义，可直接修改默认值：

| 参数组 | 关键参数 | 默认值 | 说明 |
|--------|----------|--------|------|
| 网络 | `n_total` | 1000 | 神经元总数 |
| 网络 | `connection_prob` | 0.05 | 基础连接概率 |
| 网络 | `stimulus_current_gain` | 40.0 | 刺激电流增益 |
| STDP | `a_plus / a_minus` | 0.004 / 0.005 | LTP/LTD 学习率 |
| STDP | `tau_pre / tau_post` | 20 ms | 迹衰减时间常数 |
| 刺激 | `target_counts` | (20, 80, 200) | 三种电压的目标招募数 |
| 实验 | `n_rounds` | 3 | 训练轮数 |
| 实验 | `reps_per_mode` | 20 | 每模式每轮重复次数 |

## 输出

运行完成后在 `output/` 目录生成：

```
output/
├── figures/
│   ├── learning_curve.pdf       # 分类正确率学习曲线（主图）
│   ├── firing_rate_evolution.pdf # 发放率随 trial 演化
│   ├── channel_heatmap.pdf      # 通道活动热力图
│   ├── weight_distribution.pdf  # 突触权重分布变化
│   ├── spatial_activity.pdf     # 6 模式空间响应对比
│   └── confusion_matrix.pdf     # 最后一轮 10-fold out-of-fold 混淆矩阵
└── results.json                 # 数值结果汇总
```

## 预期结果

- 原始计数特征分类正确率：从 ~78% 逐轮提升至 ~90%+
- V100 特征分类正确率：从 ~37% 逐轮提升至 ~68%
- 随机基线：16.7%（6 分类）

学习曲线的上升趋势表明 STDP 可塑性成功增强了网络对不同刺激模式的区分能力。

## 项目结构

```
izhikevich_bnn_six_modes_new/
├── run.py              # 入口脚本
├── requirements.txt    # 依赖
├── src/
│   ├── config.py       # 参数定义
│   ├── network.py      # Izhikevich 网络模型与可塑性
│   ├── stimulation.py  # 热点选择、刺激校准
│   ├── experiment.py   # 实验流程编排
│   ├── decoding.py     # V100 特征提取与分类
│   └── visualize.py    # 科研绘图
└── output/             # 运行结果（自动生成）
```

## 参考文献

1. Izhikevich, E.M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks*, 14(6), 1569-1572.
2. Bi, G.Q. & Poo, M.M. (1998). Synaptic modifications in cultured hippocampal neurons. *Journal of Neuroscience*, 18(24), 10464-10472.
3. Tsodyks, M. & Markram, H. (1997). The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability. *PNAS*, 94(2), 719-723.
4. Maass, W., Natschläger, T. & Markram, H. (2002). Real-time computing without stable states: A new framework for neural computation based on perturbations. *Neural Computation*, 14(11), 2531-2560.
