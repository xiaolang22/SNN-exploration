# ARTP 机制说明

## 1. 机制命名

当前这套拓扑可塑性机制可命名为：

```text
ARTP: Activity-Reinforced Topology Plasticity
```

中文可写为：

```text
活动强化型拓扑可塑性
```

这个命名强调两点：

1. `Activity-Reinforced`
   结构变化主要由神经活动统计驱动，常用传播对更容易获得结构强化。
2. `Topology Plasticity`
   改变的是网络连边结构，而不是突触权重值。

需要特别说明的是，`ARTP` 更接近一种`基于活动统计的结构用进废退机制`，而不是严格意义上的“长链捷径生成算法”。

---

## 2. 机制核心思想

### 2.1 自然语言描述

ARTP 的基本思想是：

```text
在训练过程中，网络会记录哪些神经元对经常以稳定时延顺序共同参与信息传播。
如果某个未直接连接的神经元对反复表现出“前者先放电、后者后放电”的关系，
则说明它们之间存在稳定的功能耦合，这样的神经元对更值得新增一条直接结构连接。

与此同时，若某些现有连接长期使用率低，或者连接到低活跃节点，
则这些连接更可能被删除。

因此，ARTP 会把有限的连接预算逐步从低贡献区域转移到高活动传播区域，
从而得到一种受活动驱动的结构重组过程。
```

用更直白的话说：

```text
常用的传播关系更容易“长出边”，
少用的旧边更容易“被剪掉”。
```

### 2.2 机制本质

从原理上看，ARTP 不是显式寻找“多跳长链”再做路径压缩，而是：

1. 观察活动驱动下的神经元时序共现
2. 强化高频、稳定、相对局部的传播对
3. 剪除低利用率连接
4. 保持总边数近似恒定

因此它本质上属于：

```text
activity-dependent structural reinforcement and pruning
```

即：

```text
活动依赖的结构强化与结构剪枝
```

---

## 3. 当前代码中的整体流程

### 3.1 入口位置

ARTP 的主入口位于：

- [src/topology_plasticity.py](</D:/workspace/SNN/Simulation_Modeling_of_In_Vitro_Neural_Networks_and_Experimental_Comparative_Research/explore1/02_random_topology_plasticity/src/topology_plasticity.py:1>)
- [src/experiment.py](</D:/workspace/SNN/Simulation_Modeling_of_In_Vitro_Neural_Networks_and_Experimental_Comparative_Research/explore1/02_random_topology_plasticity/src/experiment.py:1>)
- [src/config.py](</D:/workspace/SNN/Simulation_Modeling_of_In_Vitro_Neural_Networks_and_Experimental_Comparative_Research/explore1/02_random_topology_plasticity/src/config.py:1>)

其中：

1. `Experiment.run_rounds()` 负责在每一轮训练结束后调用结构更新
2. `OnlineTopologyPlasticity.update()` 负责一次完整的拓扑重连
3. `ReservoirModel.rewire_edges()` 负责真正修改连边

### 3.2 训练中的触发时机

在 [experiment.py](</D:/workspace/SNN/Simulation_Modeling_of_In_Vitro_Neural_Networks_and_Experimental_Comparative_Research/explore1/02_random_topology_plasticity/src/experiment.py:114>) 中，网络每完成一轮训练后会：

1. 收集这一轮所有 trial 的 `spike history`
2. 计算本轮的功能指标和拓扑指标
3. 若满足 `update_interval_rounds`，调用：

```python
update_summary = self.topology_plasticity.update(round_spike_records, round_mode_labels)
```

也就是说，当前实现采用的是：

```text
轮末更新，而不是逐 spike 更新
```

这使得结构可塑性的时标慢于神经动力学时标。

---

## 4. 算法步骤

### 4.1 第一步：记录每个 trial 中的 spike 历史

在 [network.py](</D:/workspace/SNN/Simulation_Modeling_of_In_Vitro_Neural_Networks_and_Experimental_Comparative_Research/explore1/02_random_topology_plasticity/src/network.py:178>) 中，`run_segment_with_spike_history()` 会返回：

1. 记录通道的 spike count
2. 每个时间步的全网络发放神经元列表

伪代码可以写成：

```text
for each time step t:
    fired_t = model.step(...)
    spike_history.append(fired_t)
```

这一步是后面做活动统计的基础。

### 4.2 第二步：提取首发放时序关系

在 [topology_plasticity.py](</D:/workspace/SNN/Simulation_Modeling_of_In_Vitro_Neural_Networks_and_Experimental_Comparative_Research/explore1/02_random_topology_plasticity/src/topology_plasticity.py:48>) 的 `_collect_pair_stats()` 中，对每个 trial 会先构造：

```python
first_spike_time[neuron] = neuron 在该 trial 中首次发放的时间步
```

这样每个 trial 只保留每个神经元的首次激活时刻，而不是完整脉冲序列。

目的在于用一个低复杂度近似来表示：

```text
哪个神经元先参与该次刺激传播，哪个神经元后参与。
```

### 4.3 第三步：构造候选传播对

仍在 `_collect_pair_stats()` 中，算法会遍历按首发放时间排序后的神经元对，并检查：

1. 源节点必须是兴奋性神经元
2. 目标节点在时序上晚于源节点
3. 二者时间差必须落在 `delay_window_ms`
4. 当前二者不能已经直接连边
5. 空间距离不能超过 `max_candidate_distance_mm`

若满足这些条件，则把 `(src, tgt)` 视为一个结构强化候选。

对应的代码逻辑是：

```python
if adjacency[src, tgt]:
    continue
if dist > params.max_candidate_distance_mm:
    continue
if not (params.delay_window_ms[0] <= delay_ms <= params.delay_window_ms[1]):
    continue
```

这里的含义是：

```text
只有“未直连、时延合适、距离不过远”的时序传播对，
才有资格长出新边。
```

### 4.4 第四步：累计候选对统计量

对于每个候选对 `(src, tgt)`，代码会积累：

1. `count`
   该候选对在多少次 trial 中出现
2. `delays`
   每次出现时的时延
3. `labels`
   它分别在各刺激模式下出现多少次
4. `distance_mm`
   该对的空间距离

因此，一个候选对不仅知道自己“出现了多少次”，还知道：

```text
出现得稳不稳，
是不是偏向某类刺激，
以及它在空间上离得多远。
```

### 4.5 第五步：筛掉低共现候选

在 `_collect_pair_stats()` 的返回前，代码会执行：

```python
if value["count"] >= params.min_pair_cooccurrence
```

这一步的作用是去掉偶然出现的噪声对，只保留跨 trial 重复出现的功能耦合对。

### 4.6 第六步：给候选对打分

在 `_select_shortcuts()` 中，对每个候选对会计算：

```text
score =
  pair_count_weight * count
+ consistency_weight * consistency
+ class_specificity_weight * specificity
- distance_penalty * distance_mm
```

其中：

1. `count`
   出现次数，越多说明该功能关系越常用
2. `consistency`
   用时延标准差的倒数近似，越稳定越高
3. `specificity`
   该候选对是否更多出现在某一类刺激中
4. `distance_mm`
   距离越远惩罚越大

代码实现见 [topology_plasticity.py](</D:/workspace/SNN/Simulation_Modeling_of_In_Vitro_Neural_Networks_and_Experimental_Comparative_Research/explore1/02_random_topology_plasticity/src/topology_plasticity.py:97>)。

### 4.7 第七步：选出待新增边

候选对按分数降序排序后，取前若干条作为新增边：

```python
selected = scored[:limit]
add_edges = [(item["src"], item["tgt"]) for item in selected]
```

其中 `limit` 同时受以下参数限制：

1. `max_add_per_update`
2. `max_edge_fraction_per_update`
3. 候选总数

因此，每次结构更新都只做小步修改，而不是全图重构。

### 4.8 第八步：给现有边打删除分数

在 `_select_edges_to_prune()` 中，代码对现有边做反向评估。

它先统计：

1. 每个神经元在本轮的总发放次数 `spike_counts`
2. 每条已有方向对 `(src, tgt)` 的时序使用次数 `pair_counts`

然后只对满足以下条件的边考虑删除：

1. 该边是兴奋性边
2. 删掉它不会让源节点出度低于保护阈值
3. 删掉它不会让目标节点入度低于保护阈值

对应删除分数：

```text
prune_score = 1 / (1 + usage) + inactive_bonus
```

也就是：

1. 用得越少，越容易被删
2. 若连到低活跃节点，则额外加删除倾向

这就是当前实现中的“结构用进废退”部分。

### 4.9 第九步：保持总边数近似恒定

在 `update()` 中，如果 `keep_total_edges_constant = True`，代码会执行：

```python
n_ops = min(len(add_edges), len(remove_edge_ids))
```

这会强制新增和删除数量对齐。

因此当前机制不是无约束地不断加边，而是：

```text
在固定结构预算下做边的重新分配。
```

### 4.10 第十步：真正改图

在 [network.py](</D:/workspace/SNN/Simulation_Modeling_of_In_Vitro_Neural_Networks_and_Experimental_Comparative_Research/explore1/02_random_topology_plasticity/src/network.py:253>) 的 `rewire_edges()` 中，模型会：

1. 删除被选中的旧边
2. 为新增边分配新的兴奋性权重
3. 重新构建 `edge_src / edge_tgt / edge_w / edge_is_exc`
4. 刷新入度、出度和邻接索引

因此结构更新会真实进入下一轮训练过程，而不是只做离线统计。

---

## 5. 当前机制的流程图描述

### 5.1 总流程图

可以用如下流程表示：

```text
开始一轮训练
  ->
运行全部 trial，并记录每个时间步的 spike history
  ->
提取每个 trial 中各神经元的首次发放时间
  ->
枚举满足时延窗口、距离上限、未直连条件的候选传播对
  ->
累计候选对的 count / delay / label 统计
  ->
对候选对打分，选出高分新增边
  ->
统计现有边的 usage 和低活跃性
  ->
对现有边打删除分数，选出低贡献旧边
  ->
按总边数守恒原则执行 add + prune
  ->
刷新网络连接索引
  ->
进入下一轮训练
```

### 5.2 候选新增边子流程

```text
给定 trial 的 spike history
  ->
提取 first_spike_time
  ->
按首发放时间排序
  ->
遍历 src, tgt
  ->
是否为 excitatory src?
  ->
是否在 delay_window 内?
  ->
当前是否未直连?
  ->
距离是否不超过 max_candidate_distance_mm?
  ->
满足则累计为候选对
```

### 5.3 候选删除边子流程

```text
统计本轮各神经元 spike_counts
  ->
统计现有边 usage
  ->
只保留 excitatory 边
  ->
检查入度/出度保护
  ->
计算 prune_score
  ->
选择最高 prune_score 的若干条边删除
```

---

## 6. 重点参数说明

当前参数定义见 [config.py](</D:/workspace/SNN/Simulation_Modeling_of_In_Vitro_Neural_Networks_and_Experimental_Comparative_Research/explore1/02_random_topology_plasticity/src/config.py:45>)。

### 6.1 更新频率参数

- `update_interval_rounds`
  含义：每隔多少轮触发一次结构更新
  当前值：`1`
  解释：每轮训练后都更新一次

### 6.2 结构改变量参数

- `max_add_per_update`
  含义：每次最多新增多少条边
  当前值：`48`

- `max_prune_per_update`
  含义：每次最多删除多少条边
  当前值：`48`

- `max_edge_fraction_per_update`
  含义：每次更新允许改动的边数占总边数的比例上限
  当前值：`0.002`

这三者共同控制：

```text
结构重连到底是小步调整还是激进重构。
```

### 6.3 候选对筛选参数

- `min_pair_cooccurrence`
  含义：候选传播对最少需要在多少次 trial 中出现
  当前值：`3`

- `delay_window_ms`
  含义：允许的前后放电时延窗口
  当前值：`(1.0, 20.0)`

- `max_candidate_distance_mm`
  含义：候选新增边允许的最大空间距离
  当前值：`1.5`

这些参数决定：

```text
什么样的神经元对被认为是有意义的功能传播对。
```

### 6.4 候选对打分参数

- `pair_count_weight`
  含义：共现次数的权重
  当前值：`0.9`

- `consistency_weight`
  含义：时延一致性的权重
  当前值：`0.55`

- `class_specificity_weight`
  含义：类别特异性的权重
  当前值：`0.4`

- `distance_penalty`
  含义：空间距离惩罚
  当前值：`0.1`

这些参数决定：

```text
算法更偏向“高频传播对”，
还是更偏向“稳定且类别相关的传播对”。
```

### 6.5 删除边参数

- `inactive_quantile`
  含义：低活跃节点判定分位数
  当前值：`0.2`

- `protect_min_in_degree`
  含义：最小入度保护
  当前值：`1`

- `protect_min_out_degree`
  含义：最小出度保护
  当前值：`1`

这些参数决定：

```text
哪些旧边可以被安全删除，
以及结构剪枝会不会把网络剪塌。
```

### 6.6 结构预算参数

- `keep_total_edges_constant`
  含义：是否保持总边数守恒
  当前值：`True`

这让 ARTP 更像资源重分配，而不是无界增长。

---

## 7. 当前实现为什么更像“局部 hub 强化”

虽然 ARTP 被设计为结构可塑性机制，但当前实现从行为上更接近：

```text
高活动区域结构强化 + 低活动区域结构剪枝
```

原因主要有三点：

1. 候选对是按 `src, tgt` 的活动统计打分，而不是按整条多跳路径打分
2. 分数里没有显式的 `path_gain` 项，即没有直接度量“加这条边后压缩了多少路径长度”
3. 高 `pair_count` 的源节点容易反复与多个目标形成高分 pair，因此会逐步变成局部 hub

因此当前 ARTP 更准确的定位应当是：

```text
活动强化型结构重分配机制
```

而不是：

```text
显式长链捷径生成机制
```

---

## 8. 当前机制的优点与局限

### 8.1 优点

1. 实现简单，可直接嵌入现有训练循环
2. 只依赖 spike history，不依赖反向传播
3. 可以在固定结构预算下在线运行
4. 能提供明确的可解释统计量，如新增边候选分数、删除边 usage 等

### 8.2 局限

1. 只看首发放时间，忽略了完整脉冲序列信息
2. 只对神经元对打分，不对多跳路径打分
3. 没有显式“捷径收益”项
4. 候选边仍偏向局部短程传播强化
5. 目前不直接约束 burst、能耗或沉默节点比例

---

## 9. 一句话概括

ARTP 可以概括为：

```text
一种基于 trial 级首发放时序统计的在线结构重连机制，
它通过强化高频稳定传播对、剪除低利用率兴奋性边，
在固定连接预算下实现活动驱动的拓扑重分配。
```

如果写得更短一点，可以表述为：

```text
ARTP 是一种活动强化型拓扑可塑性机制，
核心规则是“常用传播对增连，低贡献旧边减连”。
```
