"""What both Modal apps share: the model to run, and the images."""

import os

import modal

from delta_bench.config import ROOT, load

config = load()
NAME = os.environ.get("MODEL", "")
if NAME not in config.models:
    raise SystemExit(f"set MODEL to one of {sorted(config.models)}")
MODEL = config.models[NAME]

WORK, HF_CACHE = "/work", "/hf"
DELTA_WORLD = "third_party/delta_world"
weights = modal.Volume.from_name(config.modal.hf_cache_volume, create_if_missing=True)
api_key = modal.Secret.from_name(config.modal.api_key_secret, required_keys=["DW_SERVICE_TOKEN"])


def with_source(image: modal.Image) -> modal.Image:
    """Adds this package and Delta-World, in the repository's layout, and the choice of model."""
    ignore = ["**/__pycache__"]
    return (image.add_local_dir(ROOT / "src", f"{WORK}/src", copy=True, ignore=ignore)
            .add_local_file(ROOT / "config.yaml", f"{WORK}/config.yaml", copy=True)
            .add_local_dir(ROOT / DELTA_WORLD / "src", f"{WORK}/{DELTA_WORLD}/src", copy=True, ignore=ignore)
            .add_local_file(ROOT / DELTA_WORLD / "config.yaml", f"{WORK}/{DELTA_WORLD}/config.yaml", copy=True)
            .env({"PYTHONPATH": f"{WORK}/src:{WORK}/{DELTA_WORLD}/src", "HF_HOME": HF_CACHE, "MODEL": NAME}))


def vllm_image() -> modal.Image:
    if MODEL.image:
        # vLLM's image has python3 but no python, which Modal calls, and its entrypoint is the server.
        image = (modal.Image.from_registry(MODEL.image).entrypoint([])
                 .run_commands('ln -sf "$(command -v python3)" /usr/local/bin/python'))
    else:
        image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(f"vllm=={config.serving.vllm_version}")
    return image.env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})


def simulator_image() -> modal.Image:
    """Delta-World's own simulator image: its lock, Java, and Delta's jars, fetched once by starting it."""
    from delta_world.config import load as delta_world

    start_once = ("from pathlib import Path; from delta_world.config import load; "
                  "from delta_world.simulator.server import SimulatorServer as S; "
                  "s = S(Path('/tmp/warm'), load().simulator.port); s.start(); s.stop()")
    image = (modal.Image.debian_slim(python_version="3.12")
             .apt_install("openjdk-17-jre-headless", "git", "procps")
             .pip_install("uv")  # vf-eval runs each rollout as a uv script
             .env({"JAVA_HOME": delta_world().modal.java_home})
             .uv_sync(str(ROOT / DELTA_WORLD), extras=["sim"]))
    return with_source(image).run_commands(f'python -c "{start_once}"')
