# 性能调优

- 稀疏比 = n_experts / top_k，调大专家数不增通信（`bench throughput` 验证）。
- 负载均衡损失系数 `aux_loss_coef` 平衡路由均衡与专家利用率。
- 沙箱吞吐靠 `ElasticScheduler` 的目标利用率（默认 0.8）与步长（默认 8）。
- 真实吞吐交给 vllm 后端，mock/cpu 仅作参考与冒烟。
