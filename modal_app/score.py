"""Predicting: a model's log-probs of the real tool outputs, on the pool of every model's episodes.
Run it for each model once all of them have dev traces.

    MODEL=qwen3.8-27b uv run modal run -m modal_app.score                             # the whole pool
    MODEL=qwen3.8-27b uv run modal run -m modal_app.score --limit 5 --data data/smoke # its first 5 episodes

Writes <data>/likelihoods/<model>.jsonl.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import modal

from delta_bench import bpb, files, serving
from modal_app.common import HF_CACHE, MODEL, config, vllm_image, weights, with_source

app = modal.App(f"delta-bench-score-{MODEL.name}")
image = vllm_image()
if MODEL.scoring_engine == "transformers":
    image = image.uv_pip_install("accelerate")

LogProbs = tuple[list[float], list[int]]  # per token: its log-prob and its rank; the first token has neither


def vllm_logprobs(prompts: list[list[int]]) -> Iterator[LogProbs]:
    from vllm import LLM, SamplingParams

    outputs = LLM(**serving.engine_args(MODEL)).generate([{"prompt_token_ids": ids} for ids in prompts],
                                                         SamplingParams(max_tokens=1, prompt_logprobs=0))
    for ids, output in zip(prompts, outputs):
        entries = [entry[token] for token, entry in zip(ids[1:], output.prompt_logprobs[1:])]
        yield [0.0] + [e.logprob for e in entries], [0] + [e.rank for e in entries]


def transformers_logprobs(prompts: list[list[int]]) -> Iterator[LogProbs]:
    import torch
    from transformers import AutoModelForCausalLM

    model, loading = AutoModelForCausalLM.from_pretrained(
        MODEL.weights, revision=MODEL.revision, dtype=MODEL.vllm["dtype"], device_map="cuda", output_loading_info=True)
    if loading["missing_keys"]:  # they would be left at random values
        raise RuntimeError(f"the checkpoint lacks weights: {sorted(loading['missing_keys'])[:3]}")
    model.eval()
    for ids in prompts:
        tokens = torch.tensor(ids, device="cuda")
        logprobs, ranks = [0.0], [0]
        with torch.no_grad():
            logits = model(input_ids=tokens[None]).logits[0, :-1]  # row k predicts token k + 1
        for chunk, targets in zip(logits.split(1024), tokens[1:].split(1024)):  # bounds the float32 copy
            chunk = torch.log_softmax(chunk.float(), dim=-1)
            found = chunk.gather(1, targets[:, None])
            logprobs += found[:, 0].tolist()
            ranks += (chunk >= found).sum(dim=1).tolist()
        yield logprobs, ranks


@app.function(image=with_source(image), gpu=MODEL.gpu, timeout=config.scoring.timeout_s, volumes={HF_CACHE: weights})
def score(episodes: list[dict], tools: list[dict]) -> list[dict]:
    from transformers import AutoTokenizer

    from delta_bench.layout import layout

    tokenizer = AutoTokenizer.from_pretrained(MODEL.weights, revision=MODEL.revision, trust_remote_code=True)
    layouts = [(e, *layout(tokenizer, e, tools, MODEL.chat_template_kwargs)) for e in episodes]
    # An episode too long for one model is left out for every model (bpb.shared).
    layouts = [x for x in layouts if len(x[1]) < config.serving.max_model_len]
    engine = {"vllm": vllm_logprobs, "transformers": transformers_logprobs}[MODEL.scoring_engine]
    results = engine([ids for _, ids, _ in layouts])
    return [record for (episode, _, observations), (logprobs, ranks) in zip(layouts, results)
            for record in bpb.records(MODEL.name, episode, observations, logprobs, ranks)]


@app.local_entrypoint()
def main(data: str = "data", limit: int = 0) -> None:
    from delta_world.simulator.tools import tool_specs

    episodes = files.pool(Path(data))
    episodes = episodes[:limit] if limit else episodes
    if not episodes:
        raise SystemExit(f"no {config.scoring.split} traces in {data}: run modal_app.evaluate first")
    print(f"{MODEL.name} scores {len(episodes)} episodes on {MODEL.gpu}", flush=True)
    records = score.remote(episodes, tool_specs())
    path = files.likelihoods(Path(data), MODEL.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"{path}: {len(records)} observations")
