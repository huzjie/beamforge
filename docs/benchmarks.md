# 基准

## throughput
报告 `comm_per_token = top_k * dim` 与 `speedup_vs_dense = (n_experts*dim) / (top_k*dim)`，
证明稀疏 MoE 通信与专家总数解耦。

## fault_tolerance
注入 71 次故障，报告完成步数、故障分类分布、MTTR（中位/均值/p90）、恢复动作统计。

## scaling
报告 total_params / active_params / sparse_ratio / comm_per_token，随专家数扫描 8..512。

## sandbox
报告沙箱工厂吞吐（eps/sec）与池容量、累计创建数。

## 运行

```bash
python -m beamforge bench
python -m beamforge report   # 落盘 beamforge_report.json
```
