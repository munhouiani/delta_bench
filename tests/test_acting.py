"""pass@1 from traces."""

import json

from delta_bench import acting


def trace(instance: str, solved: float, **record) -> str:
    return json.dumps({"ok": True, "errors": [], "task": {"key": instance}, "traces": [{"rewards": {"solved": {"score": solved}}}],
                       **record}) + "\n"


def test_errored_rollouts_are_counted_and_left_out(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(trace("A-s1", 1.0) + trace("A-s1", 0.0) + trace("B-s1", 1.0)
                    + trace("B-s1", 0.0, is_timeout=True) + trace("B-s2", 0.0, ok=False, traces=[]))
    run = acting.read(path, "m")
    assert run.counts == {"A-s1": (2, 1), "B-s1": (1, 1)} and run.errored == 2
    assert {r["group"]: r["pass@1"] for r in acting.pass1(run)} == {"all": 0.75, "A": 0.5, "B": 1.0}


def test_models_are_paired_on_the_instances_both_ran():
    a = acting.Run("a", {"A-s1": (4, 0), "A-s2": (4, 4)}, 0)
    b = acting.Run("b", {"A-s1": (4, 4), "B-s1": (4, 0)}, 0)
    [overall, family] = acting.paired(a, b)
    assert (overall["group"], overall["instances"], overall["b_minus_a"]) == ("all", 1, 1.0)
    assert family["group"] == "A"
