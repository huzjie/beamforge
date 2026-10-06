# 容错与弹性训练

## 为什么中位数而不是均值

一次灾难性事件（如整机架断电）会拉爆均值，但中位数反映「典型故障」的恢复速度。
`MTTRTracker` 同时报 median / mean / p90，工程上以 median 为 SLO。

## 五类故障与恢复动作

| 故障类 | 关键词 | 恢复动作 | 预估耗时 |
|--------|--------|---------|---------|
| HARDWARE | ecc / xid / gpu | rescale（弹性）或 restart | 6 min |
| NETWORK | nccl / timeout | retry | 0.5 min |
| OOM | out of memory | rescale | 6 min |
| DATA | corrupt / shard | resample | 1 min |
| SOFTWARE | assert / traceback | restart | 4 min |

## 复用要点

- 分片检查点（`Checkpointer` 每个 rank 一个 shard + manifest），只恢复受影响的子集。
- 弹性伸缩（`RecoveryManager(elastic=True)`）对 HARDWARE/OOM 选 rescale，扩容同时小幅回血 skill。
- 注入器用 md5 种子保证故障流可复现，MTTR 基准跨 run 稳定。
