"""Dense, keyword and hybrid retrieval.

Hybrid uses reciprocal rank fusion (RRF): score = sum over rankers of 1 / (rrf_k + rank).
RRF only uses ranks, so it needs no calibration between cosine similarity and BM25, whose
scales are unrelated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .chunking import Chunk
from .embed import Embedder
from .store import DEFAULT_FILTER, SourceFilter, Store

Mode = Literal["dense", "keyword", "hybrid"]
MODES: tuple[Mode, ...] = ("dense", "keyword", "hybrid")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None  # 1-based
    keyword_rank: int | None = None
    cosine: float | None = None  # similarity of question and chunk, when dense ran


@dataclass
class Retrieval:
    question: str
    mode: Mode
    hits: list[Hit] = field(default_factory=list)

    @property
    def top_cosine(self) -> float:
        values = [h.cosine for h in self.hits if h.cosine is not None]
        return max(values) if values else 0.0


def rrf(rankings: list[list[str]], rrf_k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Retriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        mode: Mode = "hybrid",
        candidates: int = 20,
        rrf_k: int = 60,
        source_filter: SourceFilter = DEFAULT_FILTER,
    ):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.store = store
        self.embedder = embedder
        self.mode = mode
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.source_filter = source_filter

    def retrieve(self, question: str, k: int = 5) -> Retrieval:
        dense = keyword = []
        if self.mode in ("dense", "hybrid"):
            vec = self.embedder.embed([question])[0]
            dense = self.store.dense(vec, max(k, self.candidates), self.source_filter)
        if self.mode in ("keyword", "hybrid"):
            keyword = self.store.keyword(question, max(k, self.candidates), self.source_filter)

        cosine = {s.chunk.id: s.score for s in dense}
        d_rank = {s.chunk.id: i for i, s in enumerate(dense, start=1)}
        k_rank = {s.chunk.id: i for i, s in enumerate(keyword, start=1)}
        chunks = {s.chunk.id: s.chunk for s in [*dense, *keyword]}

        if self.mode == "dense":
            ranked = [(s.chunk.id, s.score) for s in dense]
        elif self.mode == "keyword":
            ranked = [(s.chunk.id, s.score) for s in keyword]
        else:
            ranked = rrf([[s.chunk.id for s in dense], [s.chunk.id for s in keyword]], self.rrf_k)

        hits = [Hit(chunks[cid], score, d_rank.get(cid), k_rank.get(cid), cosine.get(cid)) for cid, score in ranked[:k]]
        return Retrieval(question, self.mode, hits)
