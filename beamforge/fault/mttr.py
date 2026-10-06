"""Mean/median time-to-recovery tracker (the headline metric).

Beam's claim is a *median* recovery of 8 minutes across 71 errors. Median (not
mean) matters because a single catastrophic event would skew the mean.
"""
import statistics


class MTTRTracker:
    def __init__(self):
        self.recoveries = []

    def record(self, minutes):
        self.recoveries.append(minutes)

    @property
    def median(self):
        if not self.recoveries:
            return 0.0
        return statistics.median(self.recoveries)

    @property
    def mean(self):
        if not self.recoveries:
            return 0.0
        return statistics.mean(self.recoveries)

    @property
    def p90(self):
        if not self.recoveries:
            return 0.0
        s = sorted(self.recoveries)
        idx = int(0.9 * len(s))
        return s[min(idx, len(s) - 1)]

    def report(self):
        return {
            "n": len(self.recoveries),
            "median_minutes": round(self.median, 2),
            "mean_minutes": round(self.mean, 2),
            "p90_minutes": round(self.p90, 2),
        }
