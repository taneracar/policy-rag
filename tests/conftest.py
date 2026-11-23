import os
from pathlib import Path

import pytest

from policy_rag.chunking import Chunk, load_corpus
from policy_rag.embed import HashEmbedder
from policy_rag.store import PgVectorStore, SqliteStore

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
PG_DSN = os.environ.get("POLICY_RAG_PG_DSN")


def make_chunk(cid: str, text: str, section: str | None = None, status: str = "active", reviewed: bool = True) -> Chunk:
    doc, frag = cid.split("#")
    section = section or frag  # so that section_id == id, as for unsplit corpus chunks
    return Chunk(cid, doc, doc.title(), section, text, status, reviewed, "2026-01-01")


@pytest.fixture(scope="session")
def embedder():
    return HashEmbedder()


@pytest.fixture(scope="session")
def corpus_chunks():
    return load_corpus(CORPUS)


STORE_PARAMS = ["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)]


@pytest.fixture(params=STORE_PARAMS)
def empty_store(request, embedder):
    if request.param == "sqlite":
        yield SqliteStore()
    else:
        if not PG_DSN:
            pytest.skip("POLICY_RAG_PG_DSN not set")
        store = PgVectorStore(PG_DSN, embedder.dim, table="test_chunks", reset=True)
        yield store
        store.conn.execute("DROP TABLE IF EXISTS test_chunks")
        store.conn.close()


@pytest.fixture
def corpus_store(empty_store, embedder, corpus_chunks):
    empty_store.add(corpus_chunks, embedder.embed([c.embedding_text() for c in corpus_chunks]))
    return empty_store
