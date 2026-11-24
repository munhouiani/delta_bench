"""config.yaml as dataclasses. A missing or unknown setting fails at start-up."""

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]  # the repository; the Modal images keep its layout


@dataclass(frozen=True)
class Model:
    name: str
    weights: str
    revision: str
    gpu: str
    image: str | None
    vllm: dict[str, Any]
    tool_call_parser: str
    reasoning_parser: str
    chat_template_kwargs: dict[str, Any]
    sampling: dict[str, Any]
    scoring_engine: str


@dataclass(frozen=True)
class Serving:
    vllm_version: str
    port: int
    max_model_len: int
    max_concurrent_requests: int
    timeout_s: int
    startup_timeout_s: int


@dataclass(frozen=True)
class Evaluation:
    splits: list[str]
    per_family: int
    samples: int
    concurrency: int
    episodes_per_simulator: int
    read_timeout_s: float
    episode_timeout_s: float
    cpu: float
    memory_mb: int
    timeout_s: int


@dataclass(frozen=True)
class Scoring:
    split: str
    timeout_s: int
    bootstrap_resamples: int


@dataclass(frozen=True)
class Modal:
    hf_cache_volume: str
    api_key_secret: str


@dataclass(frozen=True)
class Config:
    models: dict[str, Model]
    serving: Serving
    evaluation: Evaluation
    scoring: Scoring
    modal: Modal


SECTIONS = {"serving": Serving, "evaluation": Evaluation, "scoring": Scoring, "modal": Modal}


def parse(raw: dict) -> Config:
    models = {name: Model(name=name, **model) for name, model in raw["models"].items()}
    sections = {name: SECTIONS[name](**section) for name, section in raw.items() if name != "models"}
    return Config(models=models, **sections)


@cache
def load() -> Config:
    return parse(yaml.safe_load((ROOT / "config.yaml").read_text()))
