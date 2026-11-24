"""What each model is launched with."""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from delta_bench import harness, serving
from delta_bench.config import load

MODELS = load().models


def test_a_models_engine_arguments_become_server_flags():
    command = serving.serve_command(MODELS["nemotron-3.5-lightning"], "key")
    assert "--enforce-eager" in command and "--no-enable-prefix-caching" in command
    assert command[command.index("--mamba-ssm-cache-dtype") + 1] == "float32"
    assert command[command.index("--served-model-name") + 1] == "nemotron-3.5-lightning"


@pytest.mark.parametrize("model", MODELS.values(), ids=list(MODELS))
def test_scoring_keeps_every_prompt_log_prob(model):
    assert serving.engine_args(model)["enable_prefix_caching"] is False


@pytest.mark.parametrize("model", MODELS.values(), ids=list(MODELS))
def test_every_request_carries_the_models_sampling_and_delta_worlds_token_limit(model):
    assert harness.settings(model)["sampling"] == {**model.sampling, "max_tokens": 1024}


def test_a_part_starts_at_its_own_seed_and_a_split_cannot_be_overrun():
    model = MODELS["qwen3.8-27b"]
    args = ("url", "spark", Path("s.json"), Path("out"))
    command = harness.command(model, "dev", 24, 16, 4, *args)
    assert command[command.index("--env.taskset.seed-start") + 1] == "100024"
    assert command[command.index("-n") + 1] == "80"
    with pytest.raises(ValueError):
        harness.command(model, "dev", 24, 17, 4, *args)


@given(st.integers(1, 400), st.integers(1, 8))
def test_parts_cover_every_instance_once_within_a_simulators_limit(per_family, samples):
    parts = harness.parts(per_family, samples)
    assert [offset for offset, _ in parts] == [sum(size for _, size in parts[:k]) for k in range(len(parts))]
    assert sum(size for _, size in parts) == per_family
    assert all(size * 5 * samples <= load().evaluation.episodes_per_simulator for _, size in parts)
