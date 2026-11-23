import json

import pytest

from policy_rag.answer import Claim, Usage
from policy_rag.cli import GOLDEN, gate
from policy_rag.evaluate import Case, calibrate, load_golden, retrieval_metrics, score_case, split, summarize
from policy_rag.graph import Result
from policy_rag.retrieve import Hit, Retrieval
from tests.conftest import make_chunk


def case(**kw):
    base = dict(id="a1", question="q", answerable=True, expected=("pto#accrual",), must_contain=("15",), split="dev")
    return Case(**{**base, **kw})


def result(status="answered", text="You accrue 15 days.", cid="pto#accrual", usage=None, path=None):
    hits = [Hit(make_chunk(cid, "You accrue 15 days per year."), 1.0)]
    claims = [Claim(text=text, chunk_id=cid, quote="You accrue 15 days")] if status == "answered" else []
    return Result(
        "q",
        status,
        claims=claims,
        retrieval=Retrieval("q", "hybrid", hits),
        usage=usage,
        path=path or ["retrieve", "generate", "verify", status if status == "answered" else "refuse"],
    )


def test_golden_file_is_consistent(corpus_chunks):
    cases = load_golden(GOLDEN)
    sections = {c.section_id for c in corpus_chunks}
    assert len({c.id for c in cases}) == len(cases)
    assert {c.split for c in cases} == {"dev", "test"}
    for c in cases:
        assert set(c.expected) <= sections, c.id
        assert c.answerable == bool(c.expected) == bool(c.must_contain), c.id
        texts = " ".join(ch.text for ch in corpus_chunks if ch.section_id in c.expected).lower()
        assert all(p.lower() in texts for p in c.must_contain), f"{c.id}: required phrase not in its section"
    assert split(cases, "dev") and split(cases, "all") == cases


def test_score_answerable():
    assert score_case(case(), result()).correct
    assert not score_case(case(must_contain=("20",)), result()).correct
    assert not score_case(case(must_not_contain=("15",)), result()).correct  # e.g. superseded value
    assert not score_case(case(), result("refused")).correct


def test_score_unanswerable_and_citations():
    assert score_case(case(answerable=False, expected=(), must_contain=()), result("refused")).correct
    assert not score_case(case(answerable=False, expected=(), must_contain=()), result()).correct
    assert score_case(case(), result()).citation_ok is True
    assert score_case(case(expected=("pto#carryover",)), result()).citation_ok is False


def test_cost_real_vs_projected_vs_gate_refusal():
    real = score_case(case(), result(usage=Usage("claude-opus-5-5", 1000, 100)))
    assert not real.cost_is_projection and real.cost_usd == pytest.approx(0.006)
    projected = score_case(case(), result(usage=Usage("none", estimated=True)))
    assert projected.cost_is_projection and projected.cost_usd > 0
    gated = score_case(case(), result("refused", path=["retrieve", "refuse"]))
    assert gated.cost_usd == 0.0


def test_summarize_rates():
    rows = [
        score_case(case(), result()),
        score_case(case(id="a2", must_contain=("99",)), result()),
        score_case(case(id="u1", answerable=False, expected=(), must_contain=()), result("refused")),
    ]
    s = summarize(rows)
    assert s["answer_accuracy"] == 0.5 and s["refusal_accuracy"] == 1.0
    assert s["groundedness"] == 1.0 and s["false_refusal_rate"] == 0.0


class StubRetriever:
    def __init__(self, ranked):
        self.ranked = ranked

    def retrieve(self, question, k):
        return Retrieval(question, "dense", [Hit(make_chunk(cid, "t"), 1.0) for cid in self.ranked[:k]])


def test_retrieval_metrics():
    cases = [case(expected=("x#a",)), case(id="a2", expected=("x#b", "x#c")), case(id="u", answerable=False)]
    m = retrieval_metrics(StubRetriever(["x#z", "x#a", "x#b"]), cases, ks=(1, 3))
    assert m["n"] == 2
    assert m["recall@1"] == 0.0
    assert m["recall@3"] == pytest.approx((1 + 0.5) / 2)
    assert m["mrr"] == pytest.approx((1 / 2 + 1 / 3) / 2)


def test_calibrate_prefers_stricter_on_ties():
    class G:
        def __init__(self, mc, ms):
            self.mc, self.ms = mc, ms

    import policy_rag.evaluate as ev

    original = ev.end_to_end
    ev.end_to_end = lambda g, cases: [
        score_case(case(answerable=False, expected=(), must_contain=()), result("refused"))
    ]
    try:
        c = calibrate(G, [], cosines=[0.0, 0.3], scores=[0.4, 0.5])
    finally:
        ev.end_to_end = original
    assert (c.min_cosine, c.min_score) == (0.3, 0.5)
    assert len(c.grid) == 4


def test_gate_thresholds():
    report = {"retrieval": {"test": {"hybrid": {"recall@5": 0.9}}}, "x": {"cost": 0.02}}
    assert gate(report, {"retrieval.test.hybrid.recall@5": 0.9}) == []
    assert gate(report, {"retrieval.test.hybrid.recall@5": 0.95})
    assert gate(report, {"max:x.cost": 0.01})
    assert gate(report, {"max:x.cost": 0.05}) == []


def test_gate_file_parses():
    from policy_rag.cli import ROOT

    thresholds = json.loads((ROOT / "eval" / "gate.json").read_text())
    assert thresholds and all(isinstance(v, int | float) for v in thresholds.values())
