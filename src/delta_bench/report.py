"""The tables and the chart, from a data directory's traces and likelihoods.

    uv run python -m delta_bench.report                # data/ -> data/tables/
    uv run python -m delta_bench.report --data DIR     # any directory in that layout, such as the published dataset

    acting_<split>.csv          pass@1 per model, overall and per task family
    acting_<split>_paired.csv   the difference between every two models, on the same instances
    predicting.csv              bits per byte and top-1 accuracy per model
    predicting_paired.csv       the difference between every two models, on the same episodes
    headline.svg                predicting against acting
"""

import argparse
import csv
import itertools
import json
from pathlib import Path

from delta_bench import acting, bpb, files
from delta_bench.config import load

COLORS = ["#eda100", "#eb6834", "#1baf7a", "#2a78d6"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def acting_tables(root: Path, split: str) -> tuple[list[dict], list[dict]]:
    paths = {m: files.traces(root, m, split) for m in load().models}
    runs = [acting.read(path, m) for m, path in paths.items() if path.exists()]
    return ([row for run in runs for row in acting.pass1(run)],
            [row for a, b in itertools.combinations(runs, 2) for row in acting.paired(a, b)])


def predicting_tables(root: Path) -> tuple[list[dict], list[dict]]:
    models, resamples = load().models, load().scoring.bootstrap_resamples
    paths = {m: files.likelihoods(root, m) for m in models}
    scored = {m: [json.loads(line) for line in path.read_text().splitlines()] for m, path in paths.items() if path.exists()}
    if not scored:
        return [], []
    rows_of = bpb.shared({m: [r for r in rows if r["origin"] in models] for m, rows in scored.items()})
    if not all(rows_of.values()) or len({sum(r["obs_bytes"] for r in rows) for rows in rows_of.values()}) != 1:
        raise SystemExit("the models were scored on different observations: score them all on one pool")
    summary, pairs = [], []
    for m, rows in rows_of.items():
        row = {"model": m, "episodes": len({r["episode"] for r in rows}), "observations": len(rows)}
        for role in bpb.ROLES:
            estimate, low, high = bpb.bits_per_byte(rows, role, resamples)
            row |= {f"bpb_{role}": round(estimate, 4), f"bpb_{role}_low": round(low, 4), f"bpb_{role}_high": round(high, 4),
                    f"top1_{role}": round(bpb.ratio(rows, f"top1_{role}", f"tokens_{role}"), 4)}
        # The share of the predictable bytes that fall inside predictable tokens: close for every model, or
        # their tokenizers make them hard to compare.
        summary.append({**row, "coverage": round(bpb.ratio(rows, "covered_bytes_pred", "span_bytes_pred"), 4)})
    for (a, a_rows), (b, b_rows) in itertools.combinations(rows_of.items(), 2):
        for role in bpb.ROLES:
            estimate, low, high = bpb.paired(a_rows, b_rows, role, resamples)
            pairs.append({"a": a, "b": b, "role": role, "b_minus_a": round(estimate, 4), "low": round(low, 4),
                          "high": round(high, 4)})
    return summary, pairs


def headline(path: Path, predicting: list[dict], act: list[dict], split: str) -> None:
    """Predicting (x) against acting (y), one point per model, with 95% intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = {r["model"]: r for r in act if r["group"] == "all"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for p, color in zip([p for p in predicting if p["model"] in overall], itertools.cycle(COLORS)):
        a = overall[p["model"]]
        x, y = p["bpb_pred"], a["pass@1"]
        ax.plot([p["bpb_pred_low"], p["bpb_pred_high"]], [y, y], color=color, lw=1.5)
        ax.plot([x, x], [a["low"], a["high"]], color=color, lw=1.5)
        ax.scatter([x], [y], s=64, color=color, edgecolors="white", linewidths=2, zorder=3)
        ax.annotate(p["model"], (x, y), xytext=(7, 6), textcoords="offset points", color=INK)
    ax.set_xlim(0, 1.3 * max(p["bpb_pred_high"] for p in predicting))
    ax.set_ylim(0, 1)
    ax.set_xlabel("bits per byte on predictable tool output (lower is better)", color=MUTED)
    ax.set_ylabel(f"pass@1 on {split} (higher is better)", color=MUTED)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    root = parser.parse_args().data
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for split in load().evaluation.splits:
        act, act_paired = acting_tables(root, split)
        write(tables / f"acting_{split}.csv", act)
        write(tables / f"acting_{split}_paired.csv", act_paired)
    predicting, predicting_paired = predicting_tables(root)
    write(tables / "predicting.csv", predicting)
    write(tables / "predicting_paired.csv", predicting_paired)
    if act and predicting:  # the last split: the hardest
        headline(tables / "headline.svg", predicting, act, split)
        print(tables / "headline.svg")


if __name__ == "__main__":
    main()
