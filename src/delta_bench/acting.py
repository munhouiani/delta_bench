"""Acting: whether the verifier passes the final table state, as pass@1 over a split's instances.
Rollouts that failed for an infrastructure reason are left out and counted."""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from delta_world.evaluation import stats


@dataclass(frozen=True)
class Run:
    model: str
    counts: dict[str, tuple[int, int]]  # per instance: rollouts, and how many solved it
    errored: int


def read(path: Path, model: str) -> Run:
    """A model's run on a split, from its vf-eval traces."""
    counts, errored = defaultdict(lambda: [0, 0]), 0
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            failed = not record.get("ok", True) or record.get("errors") or record.get("is_timeout")
            for trace in record.get("traces") or [{}]:
                if failed or trace.get("errors"):
                    errored += 1
                    continue
                count = counts[record["task"]["key"]]
                count[0] += 1
                count[1] += bool(trace["rewards"]["solved"]["score"])
    return Run(model, {i: (n, c) for i, (n, c) in counts.items()}, errored)


def groups(ids: list[str]) -> dict[str, list[str]]:
    """All instances, then each family's. An instance id starts with its family's letter."""
    out = {"all": ids} if ids else {}
    for i in ids:
        out.setdefault(i[0], []).append(i)
    return out


def pass1(run: Run) -> list[dict]:
    """pass@1 per group, with a 95% bootstrap interval over instances."""
    rows = []
    for group, ids in groups(sorted(run.counts)).items():
        p = stats.bootstrap_interval({i: run.counts[i] for i in ids}, 1)
        rows.append({"model": run.model, "group": group, "instances": len(ids), "pass@1": round(p.estimate, 4),
                     "low": round(p.low, 4), "high": round(p.high, 4), "errored": run.errored})
    return rows


def paired(a: Run, b: Run) -> list[dict]:
    """pass@1 of b - a per group, resampling the instances both ran, so their difficulty cancels."""
    rows = []
    for group, ids in groups(sorted(set(a.counts) & set(b.counts))).items():
        d = stats.paired_bootstrap({i: a.counts[i] for i in ids}, {i: b.counts[i] for i in ids}, 1)
        rows.append({"a": a.model, "b": b.model, "group": group, "instances": len(ids), "b_minus_a": round(d.estimate, 4),
                     "low": round(d.low, 4), "high": round(d.high, 4)})
    return rows
