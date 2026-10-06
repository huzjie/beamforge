"""Minimal span/trace tracking for a training step."""
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class Span:
    name: str
    start: float
    end: float = 0.0
    attrs: dict = field(default_factory=dict)

    @property
    def duration_ms(self):
        return round((self.end - self.start) * 1000, 3)


class Tracer:
    def __init__(self):
        self.spans: List[Span] = []

    def start(self, name, **attrs):
        span = Span(name=name, start=time.time(), attrs=attrs)
        self.spans.append(span)
        return span

    def end(self, span):
        span.end = time.time()
        return span

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def report(self):
        return [{"name": s.name, "duration_ms": s.duration_ms, "attrs": s.attrs}
                for s in self.spans]
