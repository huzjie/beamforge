"""A text dashboard rendering metrics + traces + MTTR."""
from ..fault.mttr import MTTRTracker


class TextDashboard:
    def __init__(self):
        self.mttr = MTTRTracker()

    def render(self, metrics_snapshot, traces_report):
        lines = ["=" * 48, "beamforge dashboard", "=" * 48]
        for k, v in metrics_snapshot.get("counters", {}).items():
            lines.append(f"counter {k:24s} {v}")
        for k, v in metrics_snapshot.get("gauges", {}).items():
            lines.append(f"gauge   {k:24s} {v}")
        lines.append("-" * 48)
        if traces_report:
            for t in traces_report:
                lines.append(f"span    {t['name']:24s} {t['duration_ms']} ms")
        if self.mttr.recoveries:
            lines.append("-" * 48)
            lines.append(f"MTTR median {self.mttr.median} min / mean {self.mttr.mean} min")
        lines.append("=" * 48)
        return "\n".join(lines)
