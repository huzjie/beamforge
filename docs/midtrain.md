# Midtraining：上下文扩展与推理增强

## RoPE 频率缩放

扩上下文到 1M token 不必从头重训。`RoPEScaler` 把旋转频率按 `target/base` 缩放，
让长距离位置保持良态。`ContextExtender.scale` 直接给出缩放比（1M / 2048 = 512）。

## 推理增强

`MidtrainPipeline` 模拟 midtrain 过程中推理准确率单调爬升，输出 context_scale 与
reasoning_skill_final，用于量化这一阶段的收益。

## 复用要点

- 旋转频率 `theta = base^(-2i/dim)`，缩放时除以 scale 即可，无需改权重。
- midtrain 是独立阶段，与 pretrain / RL 解耦，可单独调优。
