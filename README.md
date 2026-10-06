# beamforge

**稀疏前沿模型训练工厂（Sparse Frontier-Model Training Factory）** —— 一个零依赖、可直接运行的参考实现，把
[Reflection AI Beam](https://reflection.ai)（2026-10-05 发布）背后的三套工程手法拆解成可复用的代码：

- **501B 参数 / 每 token 仅激活 23B** 的稀疏 MoE 路由（≈21.8× 稀疏比）；
- **自愈式弹性训练编排**（71 次故障、8 分钟中位恢复）；
- **海量 RL 沙箱工厂**（1.3B 沙箱的池化 + 弹性伸缩）；
- **midtrain**（把上下文扩到 1M token + 增强推理的独立阶段）。

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

---

## 一句话原理

前沿 MoE 的算力与通信都按**活跃参数**（top-K 个专家）计，而不是总参数。Beam 的 501B 只是「存储量」，
真正决定训练/推理成本的是 23B 活跃量。所以它用 1/3~1/4 的硬件跑出接近 GLM-5.2 的效果，靠的不是更聪明，
而是把「稀疏」做成了一等公民——本仓库把这套思路落成 `moe/`、`fault/`、`sandbox/`、`midtrain/` 四个可跑模块。

## 核心能力

| 模块 | 做了什么 | 可复用点 |
|------|---------|---------|
| `moe/` | top-K 路由 + SwiGLU 专家 + 负载均衡损失 | 通信量随 top_K 走，与专家总数解耦 |
| `fault/` | 故障分类 + 确定性注入 + 检查点 + 弹性恢复 | 自愈 =「便宜的确定性恢复」而非「零故障」 |
| `sandbox/` | 沙箱池 + 弹性调度 + rollout 采集 | 1.3B 沙箱靠池化复用，不靠同时常驻 |
| `rl/` | GRPO 组内相对优势 + 长度惩罚奖励 | 去掉价值网络，组内标准化优势 |
| `midtrain/` | RoPE 频率缩放扩上下文 + 推理增强 | 扩上下文不必从头重训 |
| `backends/` | mock / cpu / openai / vllm / transformers | 可训练 mock 后端，skill 单调趋 1 |

## 快速开始（零依赖，Python 3.9+）

```bash
python -m beamforge doctor        # 冒烟：所有组件 import + 实例化
python -m beamforge train         # 三段式训练：pretrain -> midtrain -> RL
python -m beamforge bench         # 跑全部基准（含 71 次故障的 MTTR）
python -m beamforge serve         # 起 OpenAI 兼容 HTTP 服务 :8899
python -m beamforge report        # 生成完整 JSON 报告
python -m unittest discover -s tests -t .   # 单测
```

## 可复用的干货块（直接抄）

**① 稀疏 MoE 的通信成本怎么算？**

```python
from beamforge.moe.sparse import SparseAccountant

# 501B 总量 / 23B 活跃 ≈ 21.8× 稀疏比，通信只随 top_k 走
acc = SparseAccountant(n_experts=512, top_k=2, expert_params=4_194_304, dim=8192)
print(acc.report())
# comm_per_token = top_k * dim = 16384，与 n_experts=512 无关
```

**② 自愈集群的故障分类词表（先把「什么错了」说清楚）**

```python
from beamforge.fault.taxonomy import ErrorTaxonomy
print(ErrorTaxonomy.classify("NCCL timeout"))   # ErrorClass.NETWORK -> retry
print(ErrorTaxonomy.classify("CUDA out of memory"))  # ErrorClass.OOM -> rescale
```

**③ GRPO 奖励函数（一句话搞定长度惩罚）**

```python
from beamforge.rl.reward import shaped_reward
r = shaped_reward(is_correct=True, length=80, ref_length=50, lam=0.5)
# r = 1.0 - 0.5 * (80/50) = 0.2 ：答对但啰嗦会被扣分
```

**④ 踩坑：别把小维度 MoE 实例化进 doctor 用大维度**

MoE 每专家 3 个 `hidden×dim` 矩阵，`MoELayer(128, 6, 4096, 2048)` 会实例化 ~3.2B 浮点，
纯 Python 直接卡死。冒烟/示例一律用小维度（如 `dim=16, hidden=16`），成本模型用 `SparseAccountant`
计数而不是真实分配。

**⑤ 提示词模板（训练工厂主控智能体）**

```
你是 beamforge 训练工厂的主控智能体，负责编排 pretrain -> midtrain -> RL 三段式流程。
每段结束都要输出 {阶段, 指标, 是否达标, 下一步}。遇到故障按错误类别选择
重试/重启/重采样/弹性伸缩，并预估恢复耗时。
```

## 基准结果口径

| 基准 | 指标 | 说明 |
|------|------|------|
| `throughput` | comm_per_token / speedup_vs_dense | 随 top_k 走，不随专家数 |
| `fault_tolerance` | MTTR 中位/均值/p90 | 71 次注入故障，目标中位 8 分钟 |
| `scaling` | sparse_ratio = n_experts/top_k | 501B/23B ≈ 21.8× |
| `sandbox` | eps/sec | 沙箱工厂吞吐 |

## 架构

```
beamforge/
├── core/        # 零依赖纯 Python 张量内核（自动微分）
├── moe/         # 稀疏路由 / 专家 / 负载均衡 / 稀疏记账
├── fault/       # 故障分类 / 注入 / 检查点 / 恢复 / 编排 / MTTR
├── sandbox/     # 沙箱环境 / 池 / 调度 / rollout / 工厂
├── rl/          # GRPO / 奖励 / 可训练策略
├── midtrain/    # 上下文扩展 / 推理增强
├── model/       # 稀疏 Transformer / 引擎门面
├── backends/    # mock / cpu / openai / vllm / transformers
├── train/       # pretrain / midtrain / RL 三段式
├── data/        # 确定性合成语料
├── bench/       # 四组基准
├── serving/     # stdlib HTTP 服务 + 客户端
└── cli/         # doctor / train / bench / serve / report
```

更多细节见 [docs/](docs/)。许可证 Apache-2.0。
