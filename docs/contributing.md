# 贡献指南

1. 保持核心零依赖（`beamforge/core`、`moe`、`fault`、`sandbox` 不 import 第三方库）。
2. 后端新增请用 `@register_backend("name")` 并在 `backends/__init__.py` 显式 import。
3. 基准新增请用 `@register_benchmark("name")` 并在 `bench/__init__.py` 显式 import。
4. 提交前跑 `python -m compileall -q beamforge && python -m unittest discover -s tests -t .`。
