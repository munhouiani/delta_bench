"""How vLLM runs a model: as a server for acting, and in-process for scoring."""

import json

from delta_bench.config import Model, load


def serve_command(model: Model, api_key: str) -> list[str]:
    serving = load().serving
    command = [
        "vllm", "serve", model.weights, "--revision", model.revision, "--served-model-name", model.name,
        "--host", "0.0.0.0", "--port", str(serving.port), "--api-key", api_key,
        "--max-model-len", str(serving.max_model_len), "--trust-remote-code",
        "--enable-auto-tool-choice", "--tool-call-parser", model.tool_call_parser,
        "--reasoning-parser", model.reasoning_parser,
        "--default-chat-template-kwargs", json.dumps(model.chat_template_kwargs),
    ]
    for name, value in model.vllm.items():
        flag = name.replace("_", "-")
        if isinstance(value, bool):
            command.append(f"--{flag}" if value else f"--no-{flag}")
        else:
            command += [f"--{flag}", str(value)]
    return command


def engine_args(model: Model) -> dict:
    """Prefix caching is off: it would skip the prompt log-probs of a cached prefix."""
    return {"model": model.weights, "revision": model.revision, "max_model_len": load().serving.max_model_len,
            "trust_remote_code": True, **model.vllm, "enable_prefix_caching": False}
