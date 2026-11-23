"""Evaluation: retrieval quality per mode, then end-to-end answer quality.

Metrics
- recall@k: share of a question's expected sections found in the top k chunks.
- MRR: mean of 1 / rank of the first expected section (0 if not in the top 20).
- answer accuracy: answerable question answered, every required phrase is in the answer, and no
  forbidden phrase is (e.g. the old value from a superseded policy).
- refusal accuracy: unanswerable question refused.
- false refusal rate: answerable question refused.
- groundedness: grounded claims / all claims the answerer produced (before dropping).
- citation accuracy: answered questions whose citations include an expected section.
- cost per query: real API cost when Claude ran; otherwise a chars/4 projection, labelled.

Thresholds (gate cosine, extractive min score) are tuned on the `dev` split and reported on
`test`, so the headline numbers are not scored on the questions used for tuning.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .answer import projected_claude_usage, verify_claims
from .graph import Result, ask
from .retrieve import MODES, Retriever


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    answerable: bool
    expected: tuple[str, ...]
    must_contain: tuple[str, ...]
    split: str
    must_not_contain: tuple[str, ...] = ()
    kind: str = ""


def load_golden(path: Path) -> list[Case]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            cases.append(
                Case(
                    d["id"],
                    d["question"],
                    d["answerable"],
                    tuple(d["expected"]),
                    tuple(d["must_contain"]),
                    d["split"],
                    tuple(d.get("must_not_contain", ())),
                    d.get("kind", ""),
                )
            )
    return cases


def split(cases: Iterable[Case], name: str | None) -> list[Case]:
    return [c for c in cases if name in (None, "all") or c.split == name]


# ---------------------------------------------------------------- retrieval


def retrieval_metrics(retriever: Retriever, cases: list[Case], ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    cases = [c for c in cases if c.answerable]
    recall = {k: [] for k in ks}
    rr: list[float] = []
    for c in cases:
        sections = [h.chunk.section_id for h in retriever.retrieve(c.question, k=20).hits]
        for k in ks:
            recall[k].append(len(set(c.expected) & set(sections[:k])) / len(c.expected))
        first = next((i for i, s in enumerate(sections, start=1) if s in c.expected), None)
        rr.append(1.0 / first if first else 0.0)
    out = {f"recall@{k}": _mean(v) for k, v in recall.items()}
    out["mrr"] = _mean(rr)
    out["n"] = len(cases)
    return out


def compare_modes(make_retriever: Callable[[str], Retriever], cases: list[Case]) -> dict[str, dict]:
    return {mode: retrieval_metrics(make_retriever(mode), cases) for mode in MODES}


# ---------------------------------------------------------------- end to end


@dataclass
class CaseResult:
    case: Case
    result: Result
    correct: bool
    claims_total: int
    claims_grounded: int
    citation_ok: bool | None
    cost_usd: float
    cost_is_projection: bool
    latency_ms: int


def score_case(case: Case, result: Result) -> CaseResult:
    answered = result.status == "answered"
    if case.answerable:
        text = result.answer_text.lower()
        correct = (
            answered
            and all(p.lower() in text for p in case.must_contain)
            and not any(p.lower() in text for p in case.must_not_contain)
        )
    else:
        correct = not answered
    total = len(result.claims) + len(result.dropped)
    citation_ok = None
    if answered and case.answerable:
        section_of = {h.chunk.id: h.chunk.section_id for h in result.retrieval.hits}
        cited_sections = {section_of.get(cid, cid) for cid in result.citations}
        citation_ok = bool(cited_sections & set(case.expected))

    usage = result.usage
    if usage is not None and not usage.estimated:
        cost, projected = usage.cost_usd, False
    elif result.retrieval and result.retrieval.hits and "generate" in result.path:
        cost, projected = projected_claude_usage(case.question, result.retrieval.hits).cost_usd, True
    else:
        cost, projected = 0.0, usage is None or usage.estimated  # refused at the gate: no model call
    return CaseResult(
        case,
        result,
        correct,
        total,
        len(result.claims),
        citation_ok,
        cost,
        projected,
        usage.latency_ms if usage else 0,
    )


def end_to_end(graph, cases: list[Case]) -> list[CaseResult]:
    return [score_case(c, ask(graph, c.question)) for c in cases]


def summarize(rows: list[CaseResult]) -> dict:
    ans = [r for r in rows if r.case.answerable]
    una = [r for r in rows if not r.case.answerable]
    claims = sum(r.claims_total for r in rows)
    cites = [r.citation_ok for r in rows if r.citation_ok is not None]
    return {
        "n": len(rows),
        "answer_accuracy": _mean([r.correct for r in ans]),
        "refusal_accuracy": _mean([r.correct for r in una]),
        "false_refusal_rate": _mean([r.result.status == "refused" for r in ans]),
        "groundedness": (sum(r.claims_grounded for r in rows) / claims) if claims else None,
        "citation_accuracy": _mean(cites),
        "model_calls": sum("generate" in r.result.path for r in rows),
        "cost_per_query_usd": _mean([r.cost_usd for r in rows]),
        "cost_is_projection": any(r.cost_is_projection for r in rows),
        "p50_latency_ms": statistics.median([r.latency_ms for r in rows]) if rows else 0,
        "accuracy_by_kind": {
            kind: _mean([r.correct for r in rows if r.case.kind == kind])
            for kind in sorted({r.case.kind for r in rows})
        },
    }


def balanced(summary: dict) -> float:
    return ((summary["answer_accuracy"] or 0) + (summary["refusal_accuracy"] or 0)) / 2


@dataclass
class Calibration:
    min_cosine: float
    min_score: float
    dev_balanced_accuracy: float
    grid: list[dict] = field(default_factory=list)


def calibrate(make_graph: Callable[[float, float], object], dev: list[Case], cosines, scores) -> Calibration:
    """Grid search on dev. Ties go to the stricter setting (more refusals), which is the safer error."""
    best: Calibration | None = None
    grid = []
    for mc in cosines:
        for ms in scores:
            s = summarize(end_to_end(make_graph(mc, ms), dev))
            b = balanced(s)
            grid.append({"min_cosine": mc, "min_score": ms, "balanced": b})
            if (
                best is None
                or b > best.dev_balanced_accuracy + 1e-9
                or (abs(b - best.dev_balanced_accuracy) < 1e-9 and (mc, ms) > (best.min_cosine, best.min_score))
            ):
                best = Calibration(mc, ms, b)
    assert best is not None
    best.grid = grid
    return best


def check_grounding_rejects_fabrication(rows: list[CaseResult]) -> int:
    """Re-verify every answered claim with its number changed; return how many still passed (should be 0)."""
    leaked = 0
    for r in rows:
        for c in r.result.claims:
            if any(ch.isdigit() for ch in c.text):
                forged = c.model_copy(update={"text": c.text + " (or 99 in some cases)"})
                leaked += verify_claims([forged], r.result.retrieval.hits)[0].grounded
    return leaked


def _mean(values) -> float | None:
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else None
