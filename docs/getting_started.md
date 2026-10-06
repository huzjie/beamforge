# 上手

```bash
git clone https://github.com/huzjie/beamforge
cd beamforge
python -m beamforge doctor   # 零依赖，直接跑
python -m beamforge train
python -m beamforge bench
```

可选依赖：

```bash
pip install openai        # openai 后端
pip install vllm          # vllm 后端
pip install transformers  # transformers 后端
```
