"""Bits per byte: records, the shared observations, and what must hold for any records."""

import math

from hypothesis import given
from hypothesis import strategies as st

from delta_bench import bpb
from delta_bench.layout import Observation

EPISODE = {"origin": "x", "instance_id": "E-s1"}


def test_a_record_sums_its_observations_tokens():
    o = Observation(2, "error", nbytes=10, predictable_bytes=6, covered_bytes=5, tokens=[(1, True, 5), (3, False, 4)])
    [r] = bpb.records("m", EPISODE, [o], logprobs=[0.0, -2.0, -9.0, -0.5], ranks=[0, 1, 7, 2])
    assert (r["episode"], r["obs"], r["obs_bytes"], r["covered_bytes_pred"]) == ("x/E-s1", 2, 10, 5)
    assert (r["nll_pred"], r["bytes_pred"], r["tokens_pred"], r["top1_pred"]) == (2.0, 5, 1, 1)
    assert (r["nll_body"], r["bytes_body"], r["tokens_body"], r["top1_body"]) == (2.5, 9, 2, 1)


def test_a_model_twice_as_sure_pays_half_the_bits():
    def rows(nll):
        return [{"episode": "e", "nll_body": nll, "bytes_body": 8}]

    assert bpb.bits_per_byte(rows(8 * math.log(2)), "body", 10) == (1.0, 1.0, 1.0)
    assert bpb.paired(rows(8 * math.log(2)), rows(4 * math.log(2)), "body", 10) == (-0.5, -0.5, -0.5)


def test_the_interval_holds_95_percent_of_the_resamples():
    """One episode in five costs 1 bit per byte and the others cost none, so the resampled score is the
    binomial share of the costly one, over 100 episodes: 0.2, with 95% between about 0.12 and 0.28."""
    rows = [{"episode": f"e{i}", "nll_body": math.log(2) * 8 * (i % 5 == 0), "bytes_body": 8} for i in range(100)]
    estimate, low, high = bpb.bits_per_byte(rows, "body", 2000)
    assert math.isclose(estimate, 0.2) and 0.10 < low < 0.15 and 0.25 < high < 0.30


@st.composite
def records(draw):
    return [{"episode": f"x/E-s{draw(st.integers(0, 6))}", "obs": k, "nll_body": draw(st.floats(0, 1e4)),
             "bytes_body": draw(st.integers(1, 5000))} for k in range(draw(st.integers(1, 30)))]


@given(records(), st.randoms())
def test_the_order_of_the_records_does_not_matter(rows, rng):
    shuffled = list(rows)
    rng.shuffle(shuffled)
    assert math.isclose(bpb.bits_per_byte(rows, "body", 1)[0], bpb.bits_per_byte(shuffled, "body", 1)[0], rel_tol=1e-9)


@given(records())
def test_the_interval_lies_within_the_episodes(rows):
    each = [bpb.bits_per_byte([r for r in rows if r["episode"] == e], "body", 1)[0] for e in {r["episode"] for r in rows}]
    estimate, low, high = bpb.bits_per_byte(rows, "body", 50)
    assert min(each) - 1e-9 <= low <= high <= max(each) + 1e-9
    assert min(each) - 1e-9 <= estimate <= max(each) + 1e-9


@given(records())
def test_a_model_differs_from_itself_by_nothing(rows):
    assert bpb.paired(rows, rows, "body", 20) == (0.0, 0.0, 0.0)


@given(records(), records())
def test_every_model_is_judged_on_the_same_observations(a, b):
    kept = bpb.shared({"a": a, "b": b})
    assert {(r["episode"], r["obs"]) for r in kept["a"]} == {(r["episode"], r["obs"]) for r in kept["b"]}
