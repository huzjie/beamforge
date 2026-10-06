# 模型卡

## beamforge-mock（可训练参考后端）

- 参数量：零（标量 skill）
- 能力：确定性问答 + 单调可训练
- 用途：冒烟、基准、端到端演示
- 许可：Apache-2.0

## 目标对齐：Reflection Beam

| 属性 | Beam | 本仓库对应 |
|------|------|-----------|
| 总参数 | 501B | `SparseAccountant.total` |
| 活跃参数 | 23B / token | `SparseAccountant.active` |
| 稀疏比 | ~21.8× | `sparse_ratio` |
| 上下文 | 1M | `midtrain.ContextExtender` |
| 预训练 | 23.8T token | `data.SynthTokens` |
| RL 沙箱 | 1.3B | `sandbox.SandboxFactory` |
| 故障恢复 | 8 min 中位 | `fault.MTTRTracker` |
