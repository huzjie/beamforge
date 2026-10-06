"""Example script (run from repo root or examples/)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamforge.observability.metrics import MetricsRegistry
from beamforge.observability.traces import Tracer
from beamforge.observability.dashboard import TextDashboard

m = MetricsRegistry()
t = Tracer()
dash = TextDashboard()

with t:
    s = t.start("pretrain")
    m.incr("steps", 50)
    m.set("skill", 1.0)
    m.observe("loss", 0.02)
    t.end(s)

print(dash.render(m.snapshot(), t.report()))
