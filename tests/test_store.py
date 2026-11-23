"""Store contract tests. Each runs on SQLite, and on Postgres + pgvector when POLICY_RAG_PG_DSN is set."""

import numpy as np
import pytest

from policy_rag.store import SourceFilter, keyword_terms
from tests.conftest import make_chunk

ALL = SourceFilter(include_superseded=True, include_unreviewed=True)


def fill(store, embedder, chunks):
    store.add(chunks, embedder.embed([c.embedding_text() for c in chunks]))


def test_add_get_count_and_upsert(empty_store, embedder):
    c = make_chunk("pto#accrual", "Employees accrue 15 days per year.")
    fill(empty_store, embedder, [c])
    assert empty_store.get("pto#accrual") == c
    assert empty_store.get("missing#x") is None
    changed = make_chunk("pto#accrual", "Employees accrue 20 days per year.")
    fill(empty_store, embedder, [changed])
    assert empty_store.count() == 1
    assert empty_store.get("pto#accrual").text.startswith("Employees accrue 20")
    # the keyword index follows the upsert too
    assert [s.chunk.id for s in empty_store.keyword("20 days", 5)] == ["pto#accrual"]
    assert empty_store.keyword("15", 5) == []


def test_add_rejects_mismatched_vectors(empty_store, embedder):
    with pytest.raises(ValueError):
        empty_store.add([make_chunk("a#b", "text")], np.zeros((2, embedder.dim), np.float32))


def test_dense_is_sorted_by_cosine_and_exact_match_wins(corpus_store, embedder, corpus_chunks):
    target = next(c for c in corpus_chunks if c.id == "pto#sick-leave")
    hits = corpus_store.dense(embedder.embed([target.embedding_text()])[0], 5)
    assert hits[0].chunk.id == "pto#sick-leave"
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_keyword_finds_rare_terms(corpus_store):
    hits = corpus_store.keyword("SEV1 page acknowledge", 3)
    assert hits[0].chunk.doc_id == "incident-response"


@pytest.mark.parametrize(
    "hostile",
    ['" OR 1=1 --', "NEAR(a b)", "pto*", "a & b | !c", "'; DROP TABLE chunks; --", "", "?!"],
)
def test_keyword_query_syntax_cannot_be_injected(corpus_store, hostile):
    corpus_store.keyword(hostile, 5)  # must not raise
    assert corpus_store.count() > 0


def test_keyword_terms_drop_stopwords_and_symbols():
    assert keyword_terms("How many SICK days do I get?!") == ["sick", "days"]
    assert keyword_terms('"quoted" OR x') == ["quoted"]


def test_default_filter_excludes_superseded_and_unreviewed(corpus_store, embedder):
    q = "daily meal allowance while traveling overnight"
    vec = embedder.embed([q])[0]
    default_docs = {s.chunk.doc_id for s in corpus_store.dense(vec, 60)} | {
        s.chunk.doc_id for s in corpus_store.keyword(q, 60)
    }
    assert "expenses-2024" not in default_docs and "staff-faq" not in default_docs
    all_docs = {s.chunk.doc_id for s in corpus_store.dense(vec, 60, ALL)}
    assert {"expenses-2024", "staff-faq"} <= all_docs


def test_filter_is_applied_before_top_k(empty_store, embedder):
    """An excluded chunk that is the best match must not leave the top-k empty."""
    old = make_chunk("old#meals", "Meal allowance is 60 dollars.", status="superseded")
    new = make_chunk("new#meals", "Meal allowance is 75 dollars per day while traveling.")
    fill(empty_store, embedder, [old, new])
    vec = embedder.embed([old.embedding_text()])[0]
    assert [s.chunk.id for s in empty_store.dense(vec, 1)] == ["new#meals"]
    assert [s.chunk.id for s in empty_store.dense(vec, 1, ALL)] == ["old#meals"]
    assert [s.chunk.id for s in empty_store.keyword("meal allowance 60", 1)] == ["new#meals"]


def test_only_superseded_toggle(empty_store, embedder):
    fill(
        empty_store,
        embedder,
        [
            make_chunk("old#x", "meal allowance", status="superseded"),
            make_chunk("wiki#x", "meal allowance", reviewed=False),
            make_chunk("cur#x", "meal allowance"),
        ],
    )
    got = {s.chunk.id for s in empty_store.keyword("meal allowance", 10, SourceFilter(include_superseded=True))}
    assert got == {"old#x", "cur#x"}
    got = {s.chunk.id for s in empty_store.keyword("meal allowance", 10, SourceFilter(include_unreviewed=True))}
    assert got == {"wiki#x", "cur#x"}
