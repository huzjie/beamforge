"""Distributed-world abstraction: real vs mock device grid.

The mock world lets every "parallel" primitive run on a single CPU while still
reporting the *communication cost* it would incur on a real cluster, which is
what the throughput / fault benchmarks measure.
"""


class DistWorld:
    def __init__(self, world_size=1, rank=0):
        self.world_size = world_size
        self.rank = rank

    def all_reduce_cost(self, n_bytes):
        # ring all-reduce ~ 2*(p-1)/p * bytes
        p = max(self.world_size, 1)
        return 2 * (p - 1) / p * n_bytes

    def all_gather_cost(self, n_bytes):
        p = max(self.world_size, 1)
        return (p - 1) / p * n_bytes


class MockWorld(DistWorld):
    """A device grid emulated on one CPU; cost model kept intact."""

    def __init__(self, n_devices=8):
        super().__init__(world_size=n_devices, rank=0)
        self.n_devices = n_devices

    def barrier(self):
        return 0
