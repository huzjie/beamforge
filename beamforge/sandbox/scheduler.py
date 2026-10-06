"""Elastic scheduler: match live sandboxes to the RL workload.

Scales the pool up when the queue is deep and down when idle, which is the
sandbox-factory analogue of cluster autoscaling.
"""


class ElasticScheduler:
    def __init__(self, pool, target_utilization=0.8, step=8):
        self.pool = pool
        self.target = target_utilization
        self.step = step

    def tick(self, queued):
        util = self.pool.utilization()
        if queued > 0 and util >= self.target:
            self.pool.scale(self.step)
        elif queued == 0 and util < 0.2:
            self.pool.scale(-self.step)
        return self.pool.capacity
