"""Benchmarks."""
_BENCHMARKS = {}


def register_benchmark(name, fn=None):
    def _wrap(f):
        _BENCHMARKS[name] = f
        return f
    if fn is not None:
        return _wrap(fn)
    return _wrap


def list_benchmarks():
    return sorted(_BENCHMARKS)


def run_benchmark(name, **kw):
    if name not in _BENCHMARKS:
        raise KeyError(f"unknown benchmark '{name}'; available={sorted(_BENCHMARKS)}")
    return _BENCHMARKS[name](**kw)


def run_all(**kw):
    out = {}
    for name, fn in sorted(_BENCHMARKS.items()):
        if name == "fault_tolerance" and "n_faults" in kw:
            out[name] = fn(n_faults=kw["n_faults"])
        else:
            out[name] = fn()
    return out


from . import throughput, fault_tolerance, scaling, sandbox  # noqa: E402,F401
