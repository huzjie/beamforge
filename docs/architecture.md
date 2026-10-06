# 架构设计

## 三段式训练工厂

```
pretrain (23.8T tokens 的稀疏 MoE)
   -> midtrain (扩上下文 1M + 增强推理)
   -> RL (GRPO，在 1.3B 沙箱里调 reasoning/coding/agent)
```

每一段都可独立运行：`PretrainRunner` / `MidtrainRunner` / `RLRunner`，由 `TrainingFactory.run_all`
串起来。这是 Beam 的「AI factory」思路：把训练看成可编排、可容错的工业流水线，而不是一次性脚本。

## 稀疏 MoE 设计

- 路由：top-K + 只对选中专家做 softmax（`router.py`）。
- 专家：SwiGLU（W1/W3 gate/up，W2 down），维度契约见 `experts.py`。
- 负载均衡：`aux_load_balance_loss = sum f_j^2 - 1/N`，鼓励 token 均匀分布。
- 稀疏记账：`SparseAccountant` 把「总参数 vs 活跃参数 vs 通信成本」变成可报告的数字。

## 自愈编排设计

`FaultTolerantOrchestrator` 把注入器、分类器、检查点、恢复策略、MTTR 追踪串成闭环：

1. 注入器在确定性步数投放故障；
2. 分类器把错误归到 5 类之一；
3. 恢复策略选择 retry / resample / restart / rescale；
4. 检查点按 `checkpoint_every` 分片落盘；
5. MTTR 追踪中位/均值/p90。

关键洞察：**自愈不是零故障，而是让恢复变得便宜且可预测**（Beam 的 8 分钟中位恢复就是证明）。
