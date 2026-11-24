"""The data directory, in the published dataset's layout:

    traces/<model>-<split>.jsonl   vf-eval traces (modal_app/evaluate.py)
    likelihoods/<model>.jsonl      one record per observation (modal_app/score.py)
    tables/                        the tables and the chart (report.py)
"""

from pathlib import Path

from delta_bench.config import load


def traces(data: Path, model: str, split: str) -> Path:
    return data / "traces" / f"{model}-{split}.jsonl"


def likelihoods(data: Path, model: str) -> Path:
    return data / "likelihoods" / f"{model}.jsonl"


def pool(data: Path) -> list[dict]:
    """The episodes every model scores: each model's first clean rollout of every instance."""
    from delta_world.evaluation.obs_ce import episodes

    config = load()
    paths = {m: traces(data, m, config.scoring.split) for m in config.models}
    return [{"origin": m, **e} for m, path in paths.items() if path.exists() for e in episodes(path)]
