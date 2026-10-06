"""A tiny metrics registry (counters / gauges / histograms)."""
import statistics


class MetricsRegistry:
    def __init__(self):
        self.counters = {}
        self.gauges = {}
        self.histograms = {}

    def incr(self, name, n=1):
        self.counters[name] = self.counters.get(name, 0) + n

    def set(self, name, value):
        self.gauges[name] = value

    def observe(self, name, value):
        self.histograms.setdefault(name, []).append(value)

    def snapshot(self):
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {k: {"n": len(v), "mean": round(statistics.mean(v), 3),
                               "median": round(statistics.median(v), 3)}
                           for k, v in self.histograms.items()},
        }
