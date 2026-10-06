"""A sandboxed RL environment.

Mirrors the three sandbox *kinds* Beam was optimized for: code generation, web
search, and agent tasks. Each is a tiny deterministic MDP: observe -> act ->
reward, seeded so the same (kind, id) always yields the same trajectory.
"""


class SandboxEnv:
    KINDS = ("code", "web", "agent")

    def __init__(self, kind, env_id, max_steps=8):
        assert kind in self.KINDS
        self.kind = kind
        self.env_id = env_id
        self.max_steps = max_steps
        self.step = 0

    def observe(self):
        from ..utils.stable import stable_vector
        return stable_vector(f"{self.kind}:{self.env_id}:obs:{self.step}", 8)

    def step_env(self, action):
        """action: int 0..3. Returns (reward, done)."""
        from ..utils.stable import stable_float
        # skill-independent baseline; the reward of a good policy is modeled by
        # the policy's own correctness, injected by the caller via reward shaping.
        base = stable_float(f"{self.kind}:{self.env_id}:rew:{self.step}:{action}", 0.0, 1.0)
        self.step += 1
        done = self.step >= self.max_steps
        return base, done

    def reset(self):
        self.step = 0


def make_env(kind, env_id, **kw):
    return SandboxEnv(kind, env_id, **kw)
