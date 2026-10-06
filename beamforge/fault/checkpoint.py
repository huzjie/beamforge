"""Sharded checkpoint/restore with a durable manifest.

The second pillar of fast recovery: you can only recover in minutes if the
checkpoint is *sharded* (resume a subset) and the manifest is cheap to verify.
"""
import json
import os
import time


class Checkpointer:
    def __init__(self, root, every=50, world_size=1):
        self.root = root
        self.every = every
        self.world_size = world_size
        os.makedirs(root, exist_ok=True)

    def save(self, step, state):
        # state: dict -> one shard file per rank + manifest
        for rank in range(self.world_size):
            shard = {"step": step, "rank": rank, "state": state.get(str(rank), state)}
            p = os.path.join(self.root, f"ckpt_{step:06d}_r{rank}.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(shard, f)
        manifest = {"step": step, "world_size": self.world_size, "ts": time.time()}
        with open(os.path.join(self.root, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        return step

    def latest_step(self):
        mf = os.path.join(self.root, "manifest.json")
        if not os.path.exists(mf):
            return None
        with open(mf, "r", encoding="utf-8") as f:
            return json.load(f).get("step")

    def load(self, step=None):
        step = step or self.latest_step()
        if step is None:
            return None
        state = {}
        for rank in range(self.world_size):
            p = os.path.join(self.root, f"ckpt_{step:06d}_r{rank}.json")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    state[str(rank)] = json.load(f).get("state")
        return state

    def should_checkpoint(self, step):
        return step % self.every == 0
