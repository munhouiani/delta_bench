"""Acting: runs a model on Delta-World. A vLLM server and one simulator per part of a split run in
the same Modal app, so everything stops when the run does.

    MODEL=qwen3.8-27b uv run modal run -m modal_app.evaluate                                        # 1,600 rollouts
    MODEL=qwen3.8-27b uv run modal run -m modal_app.evaluate --splits dev --per-family 2 --samples 1 --data data/smoke

Writes <data>/traces/<model>-<split>.jsonl.
"""

import gzip
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

from delta_bench import acting, files, harness, serving
from modal_app.common import HF_CACHE, MODEL, WORK, api_key, config, simulator_image, vllm_image, weights, with_source

settings = config.evaluation
app = modal.App(f"delta-bench-evaluate-{MODEL.name}")


# The server outlives the time the simulators take to start.
@app.function(image=with_source(vllm_image()), gpu=MODEL.gpu, max_containers=1, scaledown_window=600,
              timeout=config.serving.timeout_s, volumes={HF_CACHE: weights}, secrets=[api_key])
@modal.concurrent(max_inputs=config.serving.max_concurrent_requests)
@modal.web_server(config.serving.port, startup_timeout=config.serving.startup_timeout_s)
def serve() -> None:
    subprocess.Popen(serving.serve_command(MODEL, os.environ["DW_SERVICE_TOKEN"]))


@app.function(image=simulator_image(), cpu=settings.cpu, memory=settings.memory_mb, timeout=settings.timeout_s,
              secrets=[api_key])
def run_part(url: str, split: str, offset: int, per_family: int, samples: int) -> bytes:
    """One part of a split against a fresh simulator: its traces, gzipped."""
    from delta_world.config import load as delta_world
    from delta_world.simulator.server import SimulatorServer

    simulator = SimulatorServer(Path("/tmp/sim"), delta_world().simulator.port)
    simulator.start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            settings_file = Path(tmp) / "settings.json"
            settings_file.write_text(json.dumps(harness.settings(MODEL)))
            subprocess.run(harness.command(MODEL, split, offset, per_family, samples, f"{url}/v1", simulator.url,
                                           settings_file, Path(tmp)), cwd=WORK, check=True)
            return gzip.compress((Path(tmp) / "run" / "traces.jsonl").read_bytes())
    finally:
        simulator.stop()


def wait_healthy(url: str) -> None:
    deadline = time.monotonic() + config.serving.startup_timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=170) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass  # still starting
        time.sleep(10)
    raise TimeoutError(f"the server was not healthy after {config.serving.startup_timeout_s} s")


@app.local_entrypoint()
def main(data: str = "data", splits: str = ",".join(settings.splits), per_family: int = settings.per_family,
         samples: int = settings.samples) -> None:
    url = serve.get_web_url()
    print(f"waiting for {MODEL.name} on {MODEL.gpu}", flush=True)
    wait_healthy(url)
    jobs = [(split, *part) for split in splits.split(",") for part in harness.parts(per_family, samples)]
    results = list(run_part.starmap([(url, *job, samples) for job in jobs], return_exceptions=True))
    failed = [(job, r) for job, r in zip(jobs, results) if not isinstance(r, bytes)]
    for split in splits.split(","):
        if any(job[0] == split for job, _ in failed):
            continue
        path = files.traces(Path(data), MODEL.name, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(gzip.decompress(r) for job, r in zip(jobs, results) if job[0] == split))
        run = acting.read(path, MODEL.name)
        rollouts, solved = (sum(count[k] for count in run.counts.values()) for k in (0, 1))
        print(f"{path}: {solved} of {rollouts} rollouts solved, {run.errored} errored")
    if failed:
        raise SystemExit("\n".join(f"part {job} failed: {error!r}" for job, error in failed))
