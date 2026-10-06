# 术语

- **MoE**：Mixture-of-Experts，混合专家，稀疏激活。
- **top-K**：每个 token 只路由到得分最高的 K 个专家。
- **SwiGLU**：Swish 门控线性单元，专家 FFN 常用激活。
- **MTTR**：Mean/Median Time To Recovery，恢复耗时。
- **GRPO**：Group Relative Policy Optimization，组内相对优势强化学习。
- **midtrain**：预训练与后训练之间的中间阶段（扩上下文 + 增强推理）。
- **弹性伸缩**：按负载动态增减计算节点/沙箱。
