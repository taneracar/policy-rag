import pytest

from policy_rag.retrieve import Retriever, rrf


def test_rrf_math_and_tie_order():
    fused = dict(rrf([["a", "b"], ["b", "c"]], rrf_k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert [cid for cid, _ in rrf([["b"], ["a"]])] == ["a", "b"]  # equal scores sort by id


def test_rrf_rewards_agreement():
    order = [cid for cid, _ in rrf([["x", "both"], ["y", "both"]])]
    assert order[0] == "both"


def test_unknown_mode_rejected(corpus_store, embedder):
    with pytest.raises(ValueError):
        Retriever(corpus_store, embedder, "semantic")


@pytest.mark.parametrize("mode", ["dense", "keyword", "hybrid"])
def test_modes_return_k_hits_with_ranks(corpus_store, embedder, mode):
    r = Retriever(corpus_store, embedder, mode).retrieve("How many sick days per year?", k=4)
    assert len(r.hits) == 4
    assert len({h.chunk.id for h in r.hits}) == 4
    if mode == "keyword":
        assert all(h.cosine is None and h.keyword_rank for h in r.hits)
        assert r.top_cosine == 0.0
    if mode == "dense":
        assert all(h.dense_rank and h.cosine is not None for h in r.hits)


def test_hybrid_recovers_what_one_ranker_misses(corpus_store, embedder):
    q = "SEV1 acknowledge page minutes"
    hybrid = Retriever(corpus_store, embedder, "hybrid").retrieve(q, k=3)
    assert hybrid.hits[0].chunk.id == "incident-response#response-times"
    assert hybrid.hits[0].dense_rank is not None and hybrid.hits[0].keyword_rank is not None


def test_question_with_no_content_words(corpus_store, embedder):
    r = Retriever(corpus_store, embedder, "keyword").retrieve("how do I?", k=3)
    assert r.hits == []
