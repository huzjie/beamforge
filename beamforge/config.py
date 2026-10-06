"""Configuration loader with a YAML fallback for bare Python."""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

try:
    import yaml as _yaml  # type: ignore
except Exception:  # pragma: no cover
    _yaml = None

from .utils.yamlish import parse as _yamlish_parse


@dataclass
class BackendConfig:
    name: str = "mock"
    model: str = "beamforge-mock"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_tokens: int = 256


@dataclass
class MoEConfig:
    n_experts: int = 32
    top_k: int = 2
    dim: int = 64
    hidden: int = 64
    capacity_factor: float = 1.25
    aux_loss_coef: float = 0.01


@dataclass
class FaultConfig:
    inject: bool = False
    n_faults: int = 71
    mttr_target_minutes: float = 8.0
    checkpoint_every: int = 50
    elastic: bool = True


@dataclass
class SandboxConfig:
    n_sandboxes: int = 64
    kinds: List[str] = field(default_factory=lambda: ["code", "web", "agent"])
    max_concurrency: int = 16


@dataclass
class TrainConfig:
    epochs: int = 3
    lr: float = 0.1
    batch_size: int = 16


@dataclass
class Config:
    backend: BackendConfig = field(default_factory=BackendConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        d = d or {}
        return cls(
            backend=BackendConfig(**{k: v for k, v in d.get("backend", {}).items()
                                     if k in BackendConfig.__dataclass_fields__}),
            moe=MoEConfig(**{k: v for k, v in d.get("moe", {}).items()
                             if k in MoEConfig.__dataclass_fields__}),
            fault=FaultConfig(**{k: v for k, v in d.get("fault", {}).items()
                                if k in FaultConfig.__dataclass_fields__}),
            sandbox=SandboxConfig(**{k: v for k, v in d.get("sandbox", {}).items()
                                     if k in SandboxConfig.__dataclass_fields__}),
            train=TrainConfig(**{k: v for k, v in d.get("train", {}).items()
                                 if k in TrainConfig.__dataclass_fields__}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path=None) -> Config:
    if path is None:
        return Config()
    import os
    text = open(path, "r", encoding="utf-8").read()
    if _yaml is not None:
        data = _yaml.safe_load(text) or {}
    else:
        data = _yamlish_parse(text)
    return Config.from_dict(data)
