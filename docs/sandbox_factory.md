# RL 沙箱工厂

## 为什么能跑到 1.3B 沙箱

不是同时常驻 1.3B 沙箱，而是**池化 + 弹性伸缩**：沙箱是廉价可挂起/可创建的资源，
`SandboxPool` 维护一个活跃池，`ElasticScheduler` 按队列深度与利用率扩缩容。

## 三类沙箱

- `code`：代码生成；
- `web`：网页搜索；
- `agent`：通用智能体任务。

每个 `SandboxEnv` 是确定性 MDP（observe -> act -> reward），同一 (kind, id) 永远产出同一轨迹，
因此 rollout 可复现、可缓存。

## 复用要点

- `SandboxFactory.run_episodes` 直接给出吞吐（eps/sec）。
- 调度器目标利用率 0.8，队列积压则 `scale(+8)`，空闲则 `scale(-8)`。
- 策略函数签名为 `policy(observation) -> action`，可替换为任意真实模型。
