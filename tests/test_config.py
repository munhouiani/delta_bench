"""A typo in config.yaml fails at start-up."""

import pytest
import yaml

from delta_bench.config import ROOT, parse


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def test_the_file_parses(raw):
    config = parse(raw)
    assert all(m.name == name and m.scoring_engine in ("vllm", "transformers") for name, m in config.models.items())


def test_an_unknown_setting_is_refused(raw):
    raw["evaluation"]["sample"] = 4
    with pytest.raises(TypeError):
        parse(raw)


def test_a_missing_setting_is_refused(raw):
    del raw["models"]["qwen3.8-27b"]["revision"]
    with pytest.raises(TypeError):
        parse(raw)
