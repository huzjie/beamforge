# FAQ

**Q: 为什么纯 Python，性能不是很慢吗？**
A: 这是参考实现，目标是「零依赖、可跑、可验证思路」，不是生产吞吐。真实吞吐交给
openai/vllm/transformers 后端。

**Q: mock 后端真的在学吗？**
A: 是。skill 标量真实参与打分（`correct = skill > threshold`），训练信号恒正，
skill 从 0.5 单调爬到 1.0，基准可复现。

**Q: MTTR 为什么用中位数？**
A: 一次灾难性事件会拉爆均值，中位数反映典型故障恢复速度，更适合做 SLO。

**Q: 如何接真实模型？**
A: 配 `configs/config.yaml` 的 `backend.name=openai/vllm/transformers`，填 api_key/base_url。
