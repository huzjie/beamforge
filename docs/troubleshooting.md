# 排障

| 现象 | 原因 | 解法 |
|------|------|------|
| `unknown backend 'mock'` | 后端注册装饰器缺失 | 检查 `@register_backend` 与 `__init__.py` 显式 import |
| doctor 卡死 | 大维度 MoE 实例化 | 冒烟用小维度，成本用 `SparseAccountant` 计数 |
| 训练 skill 不涨 | 信号未参与打分 / delta 非恒正 | 保证 `correct = skill > threshold` + 恒正梯度 |
| `JSONDecodeError` on bench | benchmark 参数不匹配 | 各基准只收各自参数 |
