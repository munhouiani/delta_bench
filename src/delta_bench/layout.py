"""Which tokens of a conversation are tool output, in any model's chat template.

Each model renders the conversation with its own template and tokenizer. An observation (the text
the environment returned) is located in the rendered text, and a token belongs to it when all of the
token's characters do. A token is predictable by Delta-World's rule (`rl/tokens.py`): its characters,
whitespace aside, are all ones the action determines, such as a row count, unlike a timestamp.

Every model must be judged on the same bytes, so observations are trimmed: some templates trim tool
output and some don't."""

import json
import os
from dataclasses import dataclass, field

from delta_world.rl.tokens import token_role
from delta_world.simulator.masks import ENV_PREDICTABLE


@dataclass
class Observation:
    index: int
    obs_type: str
    nbytes: int
    predictable_bytes: int
    covered_bytes: int = 0  # predictable bytes inside predictable tokens
    tokens: list[tuple[int, bool, int]] = field(default_factory=list)  # (position, predictable, bytes)


def template_messages(messages: list[dict]) -> list[dict]:
    """Messages as chat templates take them: tool-call arguments as a mapping, not a JSON string."""
    out = []
    for m in messages:
        calls = []
        for call in m.get("tool_calls") or []:
            arguments = call["function"]["arguments"]
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass  # a malformed call stays as the model wrote it
            calls.append({**call, "function": {**call["function"], "arguments": arguments}})
        out.append({**m, "tool_calls": calls} if calls else m)
    return out


def carriers(messages: list[dict], observations: list[dict]) -> list[int | None]:
    """The index of the tool message that showed the model each observation, matched in order. A call
    the harness rejected has a tool message but no observation. The last call of an episode that hit
    the turn limit has an observation that no message carried: None."""
    tools = [(j, m["content"].strip()) for j, m in enumerate(messages) if m["role"] == "tool"]
    out, t = [], 0
    for o in observations:
        k = next((k for k in range(t, len(tools)) if tools[k][1] == o["text"].strip()), None)
        out.append(None if k is None else tools[k][0])
        t = t if k is None else k + 1
    return out


def locate(body: str, spans: list, window: tuple[int, int], text: str, offsets: list, index: int,
           obs_type: str) -> Observation | None:
    """The observation `body` inside `window` of the rendered text, with its tokens; None when it is
    not there. `spans` are [start, end, role] over the body."""
    start = text.rfind(body, *window)
    if not body or start < 0:
        return None
    end = start + len(body)
    predictable = [(a, b) for a, b, role in spans if role == ENV_PREDICTABLE]
    out = Observation(index, obs_type, len(body.encode()), sum(len(body[a:b].encode()) for a, b in predictable))
    starts = [a for a, _, _ in spans]
    seen = start
    for position, (a, b) in enumerate(offsets):
        if not start <= a < b <= end:
            continue
        # A character split over several tokens has the same offsets in each: the first one counts its bytes.
        lo, hi = max(a, seen) - start, b - start
        seen = max(seen, b)
        is_predictable = token_role(body, spans, starts, a - start, b - start) == ENV_PREDICTABLE
        if is_predictable:
            out.covered_bytes += sum(len(body[max(lo, x):min(hi, y)].encode()) for x, y in predictable)
        out.tokens.append((position, is_predictable, len(body[lo:hi].encode())))
    return out


def trim(text: str, spans: list) -> tuple[str, list]:
    """The text without surrounding whitespace, and its spans moved and clipped to it."""
    body = text.strip()
    lead = len(text) - len(text.lstrip())
    moved = [[max(a - lead, 0), min(b - lead, len(body)), role] for a, b, role in spans]
    return body, [s for s in moved if s[0] < s[1]]


def layout(tokenizer, episode: dict, tools: list[dict], template_kwargs: dict) -> tuple[list[int], list[Observation]]:
    """An episode's token ids in a model's chat template, and its observations.

    An observation is looked for only in the stretch its own tool message occupies: from where the
    conversation up to that message stops agreeing with the whole, to where the conversation through
    it does. So an output that the model's own code contains is never mistaken for the output."""
    messages = template_messages(episode["messages"])

    def render(conversation: list[dict]) -> str:
        return tokenizer.apply_chat_template(conversation, tools=tools, tokenize=False, add_generation_prompt=False,
                                             **template_kwargs)

    text = render(messages)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    observed = [o for o in episode["observations"] if o["text"]]
    out = []
    for index, (o, j) in enumerate(zip(observed, carriers(messages, observed))):
        if j is None:
            continue
        window = [len(os.path.commonprefix([render(messages[:n]), text])) for n in (j, j + 1)]
        body, spans = trim(o["text"], o["spans"])
        found = locate(body, spans, window, text, encoded["offset_mapping"], index, o["obs_type"])
        if found and found.tokens:
            out.append(found)
    return list(encoded["input_ids"]), out
