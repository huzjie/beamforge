# 模块 API 速查

- `MoELayer(dim, hidden, n_experts, top_k)` -> `forward(x)` 返回 `(out, chosen, gates)`。
- `SparseAccountant(n_experts, top_k, expert_params, dim)` -> `report()`。
- `FaultTolerantOrchestrator(n_faults, seed)` -> `run(total_steps, train_step_fn)` -> `report()`。
- `SandboxFactory(kinds, capacity)` -> `run_episodes(n_episodes, policy_fn)`。
- `GRPOTrainer(policy, reward, lr)` -> `step(prompt, k)`。
- `BeamEngine(backend)` -> `answer(prompt)` / `skill()` / `train_step(correct)`。
- `get_backend(name, cfg)` / `list_backends()`。
- `run_benchmark(name)` / `run_all()`。
