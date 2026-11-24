"""The tables, from a data directory."""

import json
import math

import pytest

from delta_bench import files, report

MODELS = ["inkling-2-mini", "qwen3.8-27b"]


def record(scorer: str, instance: str, obs: int, nll: float, nbytes: int = 8, origin: str = "qwen3.8-27b") -> dict:
    return {"scorer": scorer, "episode": f"{origin}/{instance}", "origin": origin, "instance_id": instance, "obs": obs,
            "nll_pred": nll, "bytes_pred": nbytes, "tokens_pred": 2, "top1_pred": 1,
            "nll_body": nll, "bytes_body": 8, "tokens_body": 2, "top1_body": 1,
            "obs_bytes": 8, "span_bytes_pred": nbytes, "covered_bytes_pred": nbytes}


@pytest.fixture
def root(tmp_path):
    def write(records_of: dict[str, list[dict]]):
        for model, records in records_of.items():
            path = files.likelihoods(tmp_path, model)
            path.parent.mkdir(exist_ok=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return tmp_path
    return write


def test_models_are_compared_on_the_observations_both_scored(root):
    a, b = MODELS
    summary, pairs = report.predicting_tables(root({
        a: [record(a, "A-s1", 0, 1.0), record(a, "A-s1", 1, 100.0)], b: [record(b, "A-s1", 0, 1.0)]}))
    assert [r["observations"] for r in summary] == [1, 1]
    assert all(p["b_minus_a"] == 0 for p in pairs)


def test_episodes_of_a_model_outside_the_config_are_left_out(root):
    [a, _] = MODELS
    summary, _ = report.predicting_tables(root({
        a: [record(a, "A-s1", 0, 1.0), record(a, "A-s1", 0, 100.0, origin="someone-else")]}))
    assert (summary[0]["episodes"], summary[0]["observations"]) == (1, 1)


def test_models_scored_on_different_pools_are_refused(root):
    a, b = MODELS
    with pytest.raises(SystemExit, match="one pool"):
        report.predicting_tables(root({a: [record(a, "A-s1", 0, 1.0)], b: [record(b, "A-s2", 0, 1.0)]}))


def test_a_few_episodes_are_enough_for_a_table(root):
    """With few episodes, some resamples hold no predictable byte at all."""
    [a, _] = MODELS
    empty = [record(a, f"A-s{k}", 0, 0.0, nbytes=0) for k in range(2, 12)]
    for n in range(1, 11):
        [row], _ = report.predicting_tables(root({a: [record(a, "A-s1", 0, 1.0), *empty[:n]]}))
        assert row["bpb_pred"] == row["bpb_pred_low"] == row["bpb_pred_high"] == round(1 / 8 / math.log(2), 4)


def test_a_run_with_only_errored_rollouts_does_not_block_the_others(tmp_path):
    a, b = MODELS
    bad = {"ok": False, "task": {"key": "A-s1"}, "traces": []}
    good = {"ok": True, "task": {"key": "A-s1"}, "traces": [{"rewards": {"solved": {"score": 1.0}}}]}
    for model, line in ((a, bad), (b, good)):
        path = files.traces(tmp_path, model, "dev")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(line) + "\n")
    rows, pairs = report.acting_tables(tmp_path, "dev")
    assert {r["model"] for r in rows} == {b} and pairs == []
