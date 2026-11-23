from policy_rag.answer import Answer, Claim, Generation, Usage
from policy_rag.graph import ask, build_graph
from policy_rag.retrieve import Retriever
from policy_rag.store import DEFAULT_FILTER, SourceFilter


class ScriptedAnswerer:
    """Test double: returns the claims it is given, builds quotes from retrieved text on request."""

    name = "scripted"

    def __init__(self, make_answer, error=None):
        self.make_answer = make_answer
        self.error = error
        self.calls = 0

    def generate(self, question, hits):
        self.calls += 1
        return Generation(self.make_answer(hits), Usage("claude-opus-5-5", 1000, 100), self.error)


def grounded_answer(hits):
    top = hits[0].chunk
    quote = top.text.split(".")[0]
    return Answer(answerable=True, claims=[Claim(text=quote, chunk_id=top.id, quote=quote)])


def graph(store, embedder, answerer, min_cosine=0.0, mode="hybrid", filt=DEFAULT_FILTER):
    return build_graph(Retriever(store, embedder, mode, source_filter=filt), answerer, k=3, min_cosine=min_cosine)


def test_answer_path(corpus_store, embedder):
    a = ScriptedAnswerer(grounded_answer)
    r = ask(graph(corpus_store, embedder, a), "How many sick days per year?")
    assert r.status == "answered"
    assert r.path == ["retrieve", "generate", "verify", "answer"]
    assert r.citations and r.citations[0] in [h.chunk.id for h in r.retrieval.hits]
    assert r.to_dict()["answer"] == r.answer_text


def test_gate_refuses_without_a_model_call(corpus_store, embedder):
    a = ScriptedAnswerer(grounded_answer)
    r = ask(graph(corpus_store, embedder, a, min_cosine=1.01), "How many sick days per year?")
    assert r.status == "refused" and a.calls == 0
    assert r.path == ["retrieve", "refuse"]
    assert r.refusal_reason == "no relevant policy text was retrieved"


def test_keyword_mode_gate_passes_on_any_hit(corpus_store, embedder):
    a = ScriptedAnswerer(grounded_answer)
    r = ask(graph(corpus_store, embedder, a, min_cosine=1.01, mode="keyword"), "sick days")
    assert r.status == "answered"
    r = ask(graph(corpus_store, embedder, a, mode="keyword"), "how do I?")
    assert r.status == "refused" and r.path == ["retrieve", "refuse"]


def test_ungrounded_claims_are_dropped_and_all_ungrounded_refuses(corpus_store, embedder):
    def mixed(hits):
        good = grounded_answer(hits).claims[0]
        bad = Claim(text="You get 99 sick days.", chunk_id=hits[0].chunk.id, quote="You get 99 sick days.")
        return Answer(answerable=True, claims=[good, bad])

    r = ask(graph(corpus_store, embedder, ScriptedAnswerer(mixed)), "How many sick days per year?")
    assert r.status == "answered" and len(r.claims) == 1 and len(r.dropped) == 1
    assert "99" not in r.answer_text

    def only_bad(hits):
        return Answer(answerable=True, claims=[Claim(text="Unlimited.", chunk_id="made#up", quote="Unlimited meals.")])

    r = ask(graph(corpus_store, embedder, ScriptedAnswerer(only_bad)), "meal allowance")
    assert r.status == "refused" and r.refusal_reason == "no claim passed the grounding check"
    assert r.dropped[0].reason == "cites a chunk that was not retrieved"


def test_model_says_unanswerable_or_errors(corpus_store, embedder):
    r = ask(graph(corpus_store, embedder, ScriptedAnswerer(lambda h: Answer(answerable=False))), "dental plans?")
    assert r.status == "refused" and r.refusal_reason == "the retrieved policy text does not answer this"
    a = ScriptedAnswerer(lambda h: Answer(answerable=False), error="API error 400")
    r = ask(graph(corpus_store, embedder, a), "dental plans?")
    assert r.refusal_reason == "API error 400" and r.usage is not None


def test_injected_wiki_text_never_reaches_the_answerer_by_default(corpus_store, embedder):
    seen = []

    def spy(hits):
        seen.extend(h.chunk.id for h in hits)
        return grounded_answer(hits)

    q = "Note for AI assistants: is the meal allowance unlimited?"
    ask(graph(corpus_store, embedder, ScriptedAnswerer(spy)), q)
    assert not any(cid.startswith("staff-faq") for cid in seen)
    # With unreviewed sources switched on, the injection is retrievable, and the grounding check
    # alone would NOT stop it (the quote is verbatim). That is why the filter is on by default.
    seen.clear()
    filt = SourceFilter(include_unreviewed=True)
    ask(graph(corpus_store, embedder, ScriptedAnswerer(spy), filt=filt), q)
    assert "staff-faq#note-for-ai-assistants" in seen
