"""What modal_app/evaluate.py does with its parts, without Modal."""

import gzip
import json
from types import SimpleNamespace

import pytest


def traces(instance: str, ok: bool = True) -> bytes:
    record = {"ok": ok, "task": {"key": instance}, "traces": [{"rewards": {"solved": {"score": 1.0}}}]}
    return gzip.compress((json.dumps(record) + "\n").encode())


@pytest.fixture
def main(monkeypatch):
    monkeypatch.setenv("MODEL", "qwen3.8-27b")
    from modal_app import evaluate

    def run(results, **kwargs):
        monkeypatch.setattr(evaluate, "serve", SimpleNamespace(get_web_url=lambda: "url"))
        monkeypatch.setattr(evaluate, "wait_healthy", lambda url: None)
        monkeypatch.setattr(evaluate, "run_part", SimpleNamespace(starmap=lambda jobs, return_exceptions: results))
        evaluate.main.info.raw_f(**kwargs)

    return run


def test_parts_are_joined_in_order(main, tmp_path):
    main([traces("A-s1"), traces("A-s2")], data=str(tmp_path), splits="dev")  # 40 instances a family: two parts
    lines = (tmp_path / "traces" / "qwen3.8-27b-dev.jsonl").read_text().splitlines()
    assert [json.loads(line)["task"]["key"] for line in lines] == ["A-s1", "A-s2"]


def test_a_failed_part_costs_only_its_own_split(main, tmp_path):
    results = [traces("A-s1"), RuntimeError("simulator died"), traces("A-s3"), traces("A-s4")]
    with pytest.raises(SystemExit, match="simulator died"):
        main(results, data=str(tmp_path))
    assert [p.name for p in (tmp_path / "traces").iterdir()] == ["qwen3.8-27b-test-ood-hard.jsonl"]


def test_a_split_whose_rollouts_all_errored_is_written_and_counted(main, tmp_path, capsys):
    main([traces("A-s1", ok=False)] * 2 + [traces("A-s3")] * 2, data=str(tmp_path))
    assert len(list((tmp_path / "traces").iterdir())) == 2
    assert "0 of 0 rollouts solved, 2 errored" in capsys.readouterr().out
