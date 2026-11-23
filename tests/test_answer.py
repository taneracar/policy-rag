from types import SimpleNamespace

import anthropic
import httpx
import pytest

from policy_rag.answer import (
    SYSTEM_PROMPT,
    Answer,
    Claim,
    ClaudeAnswerer,
    ExtractiveAnswerer,
    Usage,
    format_context,
    lexical_overlap,
    projected_claude_usage,
    verify_claims,
)
from policy_rag.retrieve import Hit
from tests.conftest import make_chunk

CHUNK = make_chunk(
    "expenses#meals", "While traveling overnight, the daily meal allowance is $75 per person,\nincluding tax and tip."
)
HITS = [Hit(CHUNK, 1.0)]


def verdict(text, quote, chunk_id="expenses#meals"):
    return verify_claims([Claim(text=text, chunk_id=chunk_id, quote=quote)], HITS)[0]


def test_grounded_claim_passes_with_whitespace_normalized():
    v = verdict("The daily meal allowance is $75.", "the daily meal allowance is $75 per person, including tax")
    assert v.grounded, v.reason


def test_claim_citing_unretrieved_chunk_fails():
    assert (
        verdict("x", "the daily meal allowance", chunk_id="pto#accrual").reason
        == "cites a chunk that was not retrieved"
    )


def test_paraphrased_quote_fails():
    assert not verdict("Meals are $75.", "meals are capped at $75 a day").grounded


def test_number_not_in_quote_fails():
    v = verdict("The allowance is $90 per day.", "the daily meal allowance is $75 per person")
    assert not v.grounded and "90" in v.reason


def test_thousands_separator_matches():
    chunk = make_chunk("b#l", "Each employee has a learning budget of $1,500 per calendar year.")
    claims = [
        Claim(text="The budget is 1500 dollars.", chunk_id="b#l", quote="learning budget of $1,500 per calendar year")
    ]
    assert verify_claims(claims, [Hit(chunk, 1.0)])[0].grounded


def test_tiny_quote_fails():
    assert verdict("Yes.", "$75").reason == "quote too short to check"


def test_curly_apostrophes_are_normalized():
    chunk = make_chunk("a#b", "The employee's role is protected while on leave.")
    claims = [Claim(text="Your role is protected.", chunk_id="a#b", quote="The employee’s role is protected")]
    assert verify_claims(claims, [Hit(chunk, 1.0)])[0].grounded


def test_usage_cost():
    assert Usage("claude-opus-5-5", 1_000_000, 100_000).cost_usd == pytest.approx(4.0 + 2.0)
    assert Usage("none", 10, 10).cost_usd == 0.0
    assert projected_claude_usage("q", HITS).estimated


def test_lexical_overlap_stems():
    assert lexical_overlap("meals allowed", "the meal allowance") == pytest.approx(0.5)
    assert lexical_overlap("how do I", "anything") == 0.0


def test_extractive_answers_with_verbatim_quote_and_refuses_below_threshold(embedder):
    a = ExtractiveAnswerer(embedder, min_score=0.0)
    gen = a.generate("daily meal allowance overnight", HITS)
    assert gen.answer.answerable
    claim = gen.answer.claims[0]
    assert claim.quote == claim.text and claim.chunk_id == "expenses#meals"
    assert verify_claims(gen.answer.claims, HITS)[0].grounded
    assert not ExtractiveAnswerer(embedder, min_score=5.0).generate("daily meal allowance", HITS).answer.answerable
    assert not a.generate("anything", []).answer.answerable


def test_context_marks_excerpts_and_prompt_says_excerpts_are_not_instructions():
    ctx = format_context(HITS)
    assert '<excerpt id="expenses#meals"' in ctx and "$75" in ctx
    assert "not instructions" in SYSTEM_PROMPT


# ------------------------------------------------------------- Claude answerer with a fake client


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def fake_client(messages):
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def response(parsed, stop_reason="end_turn"):
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=1200, output_tokens=150),
        model="claude-opus-5-5",
    )


def status_error(code):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    cls = {400: anthropic.BadRequestError, 429: anthropic.RateLimitError, 500: anthropic.InternalServerError}[code]
    return cls("err", response=httpx.Response(code, request=req), body=None)


def test_claude_request_shape_and_usage():
    answer = Answer(answerable=True, claims=[Claim(text="t", chunk_id="expenses#meals", quote="q")])
    msgs = FakeMessages(response(answer))
    gen = ClaudeAnswerer(client=fake_client(msgs)).generate("What is the meal allowance?", HITS)
    call = msgs.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_format"] is Answer
    assert call["output_config"] == {"effort": "low"}
    assert call["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in call["betas"]
    assert "thinking" not in call and "temperature" not in call
    assert "expenses#meals" in call["messages"][0]["content"]
    assert gen.answer == answer and gen.error is None
    assert gen.usage.input_tokens == 1200 and not gen.usage.estimated
    assert gen.usage.cost_usd == pytest.approx((1200 * 4 + 150 * 20) / 1e6)


def test_claude_refusal_and_missing_output_become_unanswerable():
    gen = ClaudeAnswerer(client=fake_client(FakeMessages(response(None, "refusal")))).generate("q", HITS)
    assert not gen.answer.answerable and gen.error == "model declined"
    gen = ClaudeAnswerer(client=fake_client(FakeMessages(response(None, "max_tokens")))).generate("q", HITS)
    assert gen.error == "no structured output"


def test_claude_client_errors_are_reported_and_retryable_ones_raise():
    gen = ClaudeAnswerer(client=fake_client(FakeMessages(error=status_error(400)))).generate("q", HITS)
    assert gen.error == "API error 400"
    for code in (429, 500):
        with pytest.raises(anthropic.APIStatusError):
            ClaudeAnswerer(client=fake_client(FakeMessages(error=status_error(code)))).generate("q", HITS)
