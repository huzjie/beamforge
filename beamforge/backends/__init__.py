"""Backend registry."""
_BACKENDS = {}


def register_backend(name, cls=None):
    def _wrap(c):
        _BACKENDS[name] = c
        return c
    if cls is not None:
        return _wrap(cls)
    return _wrap


def get_backend(name, cfg=None):
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend '{name}'; available={sorted(_BACKENDS)}")
    return _BACKENDS[name](cfg)


def list_backends():
    return sorted(_BACKENDS)


# explicit imports so the registration decorators actually run
from . import mock, cpu, openai, vllm, transformers  # noqa: E402,F401
