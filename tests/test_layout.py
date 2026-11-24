"""Finding tool outputs in a rendered conversation, with each token's role and bytes."""

import itertools
import json

from hypothesis import assume, given
from hypothesis import strategies as st

from delta_bench.layout import carriers, layout, locate, trim

P, T, U = "env_predictable", "env_trivial", "env_unpredictable"


class PlainTemplate:
    """Writes each message as <role>...</role>, and has one token per character."""

    def apply_chat_template(self, messages, **kwargs):
        out = ""
        for m in messages:
            calls = "".join(f"[{c['function']['name']}:{c['function']['arguments']}]" for c in m.get("tool_calls") or [])
            out += f"<{m['role']}>{m['content']}{calls}</{m['role']}>"
        return out

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def call(i: str, **arguments) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": i, "type": "function", "function": {"name": "run_sql", "arguments": json.dumps(arguments)}}]}


def tool(text: str) -> dict:
    return {"role": "tool", "content": text}


def observed(text: str, role: str = P) -> dict:
    return {"text": text, "spans": [[0, len(text), role]], "obs_type": "query_result"}


def run(messages: list[dict], observations: list[dict]):
    episode = {"messages": [{"role": "user", "content": "go"}, *messages], "observations": observations}
    return layout(PlainTemplate(), episode, [], {})


def at(out) -> dict[int, list[int]]:
    """Per observation index, its token positions."""
    return {o.index: [p for p, _, _ in o.tokens] for o in out}


def test_an_output_the_models_code_contains_is_found_in_the_tool_message():
    ids, out = run([call("1", code="print('done')"), tool("done")], [observed("done")])
    start = "".join(map(chr, ids)).index("<tool>") + len("<tool>")
    assert at(out) == {0: list(range(start, start + 4))}


def test_the_same_output_twice_is_found_twice():
    ids, out = run([call("1", query="SELECT 7"), tool("7"), call("2", query="SELECT 7"), tool("7")], [observed("7")] * 2)
    text = "".join(map(chr, ids))
    assert at(out) == {0: [text.index("<tool>7") + 6], 1: [text.rindex("<tool>7") + 6]}


def test_an_output_that_never_reached_the_model_is_not_scored():
    """The last call of an episode that hit the turn limit. Its text is in the model's own code."""
    _, out = run([call("1", code="1"), tool("1"), call("2", code="print('done')")], [observed("1"), observed("done")])
    assert list(at(out)) == [0]


def test_a_call_the_harness_rejected_does_not_shift_the_pairing():
    messages = [call("1", query="SELECT 1"), tool("1"), call("2", code="x"), tool("validation error"),
                call("3", query="SELECT 22"), tool("22")]
    ids, out = run(messages, [observed("1"), observed("22")])
    start = "".join(map(chr, ids)).rindex("<tool>22") + 6
    assert at(out)[1] == [start, start + 1]


def test_the_whitespace_around_an_output_is_not_scored():
    """Some templates trim tool output and some don't, and every model is judged on the same bytes."""
    ids, out = run([call("1", query="SELECT 1"), tool(" 42\n")], [observed(" 42\n")])
    start = "".join(map(chr, ids)).index("<tool> 42") + len("<tool> ")
    assert at(out) == {0: [start, start + 1]} and out[0].nbytes == 2


def test_carriers_ignore_surrounding_whitespace_and_skip_what_was_never_shown():
    messages = [tool(" x\n"), tool("y")]
    assert carriers(messages, [{"text": "x"}, {"text": "never shown"}, {"text": "y"}]) == [0, None, 1]


def test_tokens_carry_roles_and_bytes():
    body = "rows: 3 at <TS>"
    text = f"<tool>{body}</tool>"
    cuts = [0, 6, 11, 12, 13, 14, 16, 17, 21, len(text)]
    o = locate(body, [[0, 7, P], [7, 11, T], [11, 15, U]], (0, len(text)), text, list(itertools.pairwise(cuts)), 0, "dml")
    assert o.tokens == [(1, True, 5), (2, False, 1), (3, True, 1), (4, False, 1), (5, False, 2), (6, False, 1), (7, False, 4)]
    assert (o.nbytes, o.predictable_bytes, o.covered_bytes) == (15, 7, 6)  # the space after "rows:" is not covered


@st.composite
def observation(draw):
    """A body whose spans partition it, inside template text."""
    body = draw(st.text(st.characters(codec="utf-8", exclude_categories=("Cs",)), min_size=1, max_size=60))
    cuts = sorted(draw(st.sets(st.integers(1, max(1, len(body) - 1)), max_size=8)) | {0, len(body)})
    spans = [[a, b, draw(st.sampled_from([P, T, U]))] for a, b in itertools.pairwise(cuts) if a < b]
    text = f"<tool>{draw(st.text(max_size=10))}\n{body}\n</tool>"
    token_cuts = sorted(draw(st.sets(st.integers(1, len(text) - 1), max_size=40)) | {0, len(text)})
    return body, spans, text, list(itertools.pairwise(token_cuts))


def located(case, offsets):
    body, spans, text, _ = case
    return locate(*trim(body, spans), (0, len(text)), text, offsets, 0, "x")


@given(observation())
def test_scored_bytes_never_exceed_the_body(case):
    assume(case[0].strip())
    o = located(case, case[3])
    positions = [p for p, _, _ in o.tokens]
    assert positions == sorted(set(positions))
    assert sum(n for _, _, n in o.tokens) <= o.nbytes == len(case[0].strip().encode())
    assert o.covered_bytes <= o.predictable_bytes


@given(observation())
def test_tokens_that_tile_the_body_score_every_byte(case):
    assume(case[0].strip())
    o = located(case, [(i, i + 1) for i in range(len(case[2]))])
    assert sum(n for _, _, n in o.tokens) == o.nbytes


@given(observation())
def test_a_character_split_over_tokens_is_counted_once(case):
    """Byte-level BPE gives each half of a character the same offsets."""
    assume(case[0].strip())
    once, twice = located(case, case[3]), located(case, [o for o in case[3] for _ in (0, 1)])
    assert sum(n for _, _, n in once.tokens) == sum(n for _, _, n in twice.tokens)
    assert once.covered_bytes == twice.covered_bytes


@given(observation())
def test_trimming_keeps_every_characters_role(case):
    body, spans, _, _ = case
    trimmed, moved = trim(body, spans)
    lead = len(body) - len(body.lstrip())

    def role(of, i):
        return next(r for a, b, r in of if a <= i < b)

    assert all(role(moved, i) == role(spans, i + lead) for i in range(len(trimmed)))
