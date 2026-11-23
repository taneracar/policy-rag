"""The question-answering flow as a LangGraph state graph.

    START -> retrieve -> gate --(nothing relevant)--> refuse -> END
                           \\--> generate -> verify --(no grounded claims)--> refuse -> END
                                                   \\--> answer -> END

The gate refuses before any model call when retrieval found nothing close enough, which saves
a paid call on questions the corpus cannot answer. The verify step is code, not a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from .answer import Claim, Generation, Usage, Verdict, verify_claims
from .retrieve import Hit, Retrieval, Retriever


class Answerer(Protocol):
    name: str

    def generate(self, question: str, hits: list[Hit]) -> Generation: ...


@dataclass
class Result:
    question: str
    status: Literal["answered", "refused"]
    claims: list[Claim] = field(default_factory=list)
    dropped: list[Verdict] = field(default_factory=list)
    refusal_reason: str | None = None
    retrieval: Retrieval | None = None
    usage: Usage | None = None
    path: list[str] = field(default_factory=list)

    @property
    def answer_text(self) -> str:
        return " ".join(c.text for c in self.claims)

    @property
    def citations(self) -> list[str]:
        return list(dict.fromkeys(c.chunk_id for c in self.claims))

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "status": self.status,
            "answer": self.answer_text,
            "claims": [c.model_dump() for c in self.claims],
            "citations": self.citations,
            "dropped_claims": [{"claim": v.claim.model_dump(), "reason": v.reason} for v in self.dropped],
            "refusal_reason": self.refusal_reason,
            "retrieved": [h.chunk.id for h in self.retrieval.hits] if self.retrieval else [],
            "path": self.path,
        }


class State(TypedDict, total=False):
    question: str
    retrieval: Retrieval
    generation: Generation
    verdicts: list[Verdict]
    result: Result
    path: list[str]


def build_graph(retriever: Retriever, answerer: Answerer, k: int = 5, min_cosine: float = 0.0):
    def retrieve(state: State) -> State:
        return {"retrieval": retriever.retrieve(state["question"], k), "path": [*state.get("path", []), "retrieve"]}

    def gate(state: State) -> str:
        r = state["retrieval"]
        if not r.hits:
            return "refuse"
        # Keyword-only retrieval has no cosine; then any hit passes the gate.
        if any(h.cosine is not None for h in r.hits) and r.top_cosine < min_cosine:
            return "refuse"
        return "generate"

    def generate(state: State) -> State:
        gen = answerer.generate(state["question"], state["retrieval"].hits)
        return {"generation": gen, "path": [*state["path"], "generate"]}

    def verify(state: State) -> State:
        gen = state["generation"]
        verdicts = verify_claims(gen.answer.claims, state["retrieval"].hits) if gen.answer.answerable else []
        return {"verdicts": verdicts, "path": [*state["path"], "verify"]}

    def after_verify(state: State) -> str:
        return "answer" if any(v.grounded for v in state["verdicts"]) else "refuse"

    def answer(state: State) -> State:
        verdicts = state["verdicts"]
        path = [*state["path"], "answer"]
        return {
            "path": path,
            "result": Result(
                question=state["question"],
                status="answered",
                claims=[v.claim for v in verdicts if v.grounded],
                dropped=[v for v in verdicts if not v.grounded],
                retrieval=state["retrieval"],
                usage=state["generation"].usage,
                path=path,
            ),
        }

    def refuse(state: State) -> State:
        gen = state.get("generation")
        verdicts = state.get("verdicts", [])
        if gen is None:
            reason = "no relevant policy text was retrieved"
        elif gen.error:
            reason = gen.error
        elif not gen.answer.answerable:
            reason = "the retrieved policy text does not answer this"
        else:
            reason = "no claim passed the grounding check"
        path = [*state["path"], "refuse"]
        return {
            "path": path,
            "result": Result(
                question=state["question"],
                status="refused",
                dropped=[v for v in verdicts if not v.grounded],
                refusal_reason=reason,
                retrieval=state["retrieval"],
                usage=gen.usage if gen else None,
                path=path,
            ),
        }

    g = StateGraph(State)
    for name, fn in [
        ("retrieve", retrieve),
        ("generate", generate),
        ("verify", verify),
        ("answer", answer),
        ("refuse", refuse),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", gate, {"generate": "generate", "refuse": "refuse"})
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", after_verify, {"answer": "answer", "refuse": "refuse"})
    g.add_edge("answer", END)
    g.add_edge("refuse", END)
    return g.compile()


def ask(graph, question: str) -> Result:
    return graph.invoke({"question": question, "path": []})["result"]
