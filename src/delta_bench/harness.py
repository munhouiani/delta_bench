"""Delta-World's evaluation (`delta_world.evaluation.run_eval`) for any model: the same `vf-eval`
command, with the model's own sampling and with timeouts. The command takes those from a JSON file,
because vf-eval passes a sampling field on its command line through as a string."""

import json
import sys
from pathlib import Path

from delta_bench.config import Model, load

FAMILIES = ["A", "B", "C", "D", "E"]


def parts(per_family: int, samples: int) -> list[tuple[int, int]]:
    """A split in parts (first instance, instances per family), each small enough for one simulator."""
    size = max(1, load().evaluation.episodes_per_simulator // (len(FAMILIES) * samples))
    return [(offset, min(size, per_family - offset)) for offset in range(0, per_family, size)]


def settings(model: Model) -> dict:
    from delta_world.config import load as delta_world

    evaluation = load().evaluation
    return {"sampling": {**model.sampling, "max_tokens": delta_world().sampling.max_new_tokens},
            "client": {"timeout": {"read": evaluation.read_timeout_s}},
            "env": {"taskset": {"families": FAMILIES}, "timeout": {"episode": evaluation.episode_timeout_s}}}


def command(model: Model, split_name: str, offset: int, per_family: int, samples: int, base_url: str, spark_url: str,
            settings_file: Path, out: Path) -> list[str]:
    """Runs `per_family` instances of each family, starting at the split's instance `offset`, and
    writes <out>/run/traces.jsonl."""
    from delta_world.config import load as delta_world
    from delta_world.tasks import splits

    split = splits.split(split_name)
    if offset + per_family > split.count:
        raise ValueError(f"{split_name} has {split.count} instances per family, not {offset + per_family}")
    return [
        str(Path(sys.executable).parent / "vf-eval"), "delta-world", "@", str(settings_file), "-m", model.name,
        "--client.base-url", base_url, "--client.api-key-var", "DW_SERVICE_TOKEN",
        "--env.agent.harness.id", "null", "--env.agent.runtime.type", "subprocess",
        "--env.taskset.seed-start", str(split.seed_start + offset), "--env.taskset.num-tasks", str(per_family),
        "--env.taskset.task.spark-url", spark_url,
        "--env.taskset.task.max-turns", str(delta_world().sampling.turn_limit),
        "--env.taskset.task.hard", json.dumps(split.hard),
        "-n", str(per_family * len(FAMILIES)), "-r", str(samples), "-c", str(load().evaluation.concurrency),
        "-o", str(out), "--run.name", "run", "--clean", "true",
        "--no-push",  # vf-eval uploads the run to Prime Intellect when it finds a login
    ]
