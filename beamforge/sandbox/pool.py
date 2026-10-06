"""A pool of sandboxes with spin-up / tear-down accounting.

Key idea: sandboxes are *cheap to create* and *cheap to suspend*; the factory
elastically scales the pool to the RL demand, which is how you run 1.3B of them
without provisioning 1.3B simultaneously.
"""


class SandboxPool:
    def __init__(self, kinds, capacity=64):
        self.kinds = kinds
        self.capacity = capacity
        self.live = 0
        self.total_created = 0
        self.total_torn_down = 0

    def acquire(self):
        if self.live < self.capacity:
            self.live += 1
            self.total_created += 1
            return self.live - 1
        return None

    def release(self, handle):
        if handle is not None and self.live > 0:
            self.live -= 1
            self.total_torn_down += 1

    def scale(self, delta):
        if delta > 0:
            self.capacity += delta
        else:
            self.capacity = max(0, self.capacity + delta)

    def utilization(self):
        return self.live / self.capacity if self.capacity else 0.0
