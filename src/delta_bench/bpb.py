"""Predicting: how likely a model finds the real tool outputs, in bits per byte.

    bits per byte = sum of -log2 p(token) / bytes the tokens cover

Per token, a tokenizer with longer tokens would pay more for the same text. Per byte, every model is
judged on the same text. Two byte sets are scored: the predictable bytes (`pred`) and every byte of
the observations (`body`)."""

import math
import random
from collections import defaultdict
from collections.abc import Callable

from delta_bench.layout import Observation

ROLES = ("pred", "body")


def records(model: str, episode: dict, observations: list[Observation], logprobs: list[float],
            ranks: list[int]) -> list[dict]:
    """One record per observation, holding sums. `logprobs` and `ranks` are per token position; rank 1
    is the model's most likely token."""
    out = []
    for o in observations:
        record = {"scorer": model, "episode": key(episode), "origin": episode["origin"],
                  "instance_id": episode["instance_id"], "obs": o.index, "obs_type": o.obs_type}
        for role in ROLES:
            tokens = [(p, n) for p, predictable, n in o.tokens if predictable or role == "body"]
            record |= {f"nll_{role}": -sum(logprobs[p] for p, _ in tokens), f"bytes_{role}": sum(n for _, n in tokens),
                       f"tokens_{role}": len(tokens), f"top1_{role}": sum(ranks[p] == 1 for p, _ in tokens)}
        out.append({**record, "obs_bytes": o.nbytes, "span_bytes_pred": o.predictable_bytes,
                    "covered_bytes_pred": o.covered_bytes})
    return out


def key(episode: dict) -> str:
    return f"{episode['origin']}/{episode['instance_id']}"


def shared(records_of: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Each model's records on the observations that every model scored."""
    keep = set.intersection(*({(r["episode"], r["obs"]) for r in rows} for rows in records_of.values()))
    return {model: [r for r in rows if (r["episode"], r["obs"]) in keep] for model, rows in records_of.items()}


def ratio(rows: list[dict], top: str, bottom: str) -> float:
    total = sum(r[bottom] for r in rows)
    return sum(r[top] for r in rows) / total if total else math.nan


def _bits_per_byte(sums: list[tuple[float, int]]) -> float:
    total = sum(nbytes for _, nbytes in sums)
    return sum(nll for nll, _ in sums) / total / math.log(2) if total else math.nan


def _by_episode(rows: list[dict], role: str) -> dict[str, tuple[float, int]]:
    sums = defaultdict(lambda: (0.0, 0))
    for r in rows:
        nll, nbytes = sums[r["episode"]]
        sums[r["episode"]] = (nll + r[f"nll_{role}"], nbytes + r[f"bytes_{role}"])
    return sums


def _interval(keys: list[str], statistic: Callable[[list[str]], float], resamples: int) -> tuple[float, float, float]:
    """The statistic and its 95% bootstrap interval over episodes. A resample without a byte to score
    has no statistic and is left out."""
    rng = random.Random(0)
    draws = [statistic([rng.choice(keys) for _ in keys]) for _ in range(resamples)]
    draws = sorted(d for d in draws if not math.isnan(d)) or [math.nan]
    return statistic(keys), draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))]


def bits_per_byte(rows: list[dict], role: str, resamples: int) -> tuple[float, float, float]:
    sums = _by_episode(rows, role)
    return _interval(sorted(sums), lambda keys: _bits_per_byte([sums[k] for k in keys]), resamples)


def paired(a: list[dict], b: list[dict], role: str, resamples: int) -> tuple[float, float, float]:
    """Bits per byte of b - a, with both on the same resampled episodes."""
    sa, sb = _by_episode(a, role), _by_episode(b, role)
    return _interval(sorted(set(sa) & set(sb)), lambda keys: (_bits_per_byte([sb[k] for k in keys])
                                                              - _bits_per_byte([sa[k] for k in keys])), resamples)
