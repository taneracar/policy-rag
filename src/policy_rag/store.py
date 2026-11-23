"""Chunk stores with dense (vector) and keyword (full-text) search, filtered by source metadata.

Two backends behind one interface:
- SqliteStore: vectors as float32 blobs scored with numpy (exact cosine), keyword search with
  FTS5 BM25. Zero setup; used locally and in most tests.
- PgVectorStore: Postgres + pgvector, HNSW cosine index, and tsvector/GIN full-text search
  ranked with ts_rank_cd. Runs in CI against a pgvector service container.

Both apply the source filter inside the query (superseded documents and unreviewed sources are
excluded by default), so a filtered-out chunk can never take a top-k slot.

The two keyword rankers are different functions (BM25 vs cover density), so keyword rankings
can differ between backends. Tests assert behavior both share, not identical scores.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .chunking import Chunk

STOPWORDS = frozenset(
    """a an and are as at be before by can could do does for from get go has have how i if in is it
    its me my of on or our should so than that the their them there they this to was we what when
    where which who whom why will with would you your am been being did done many much any some
    long often need must may per each""".split()
)

COLUMNS = "id, doc_id, doc_title, section, text, status, reviewed, effective"


def keyword_terms(question: str) -> list[str]:
    """Lowercased content words, deduplicated in order. Quotes and operators are stripped,
    so user text can never inject FTS5 or tsquery syntax."""
    seen: list[str] = []
    for w in re.findall(r"[a-z0-9]+", question.lower()):
        if w not in STOPWORDS and len(w) > 1 and w not in seen:
            seen.append(w)
    return seen


@dataclass(frozen=True)
class SourceFilter:
    include_superseded: bool = False
    include_unreviewed: bool = False

    def allows(self, chunk: Chunk) -> bool:
        return (self.include_superseded or chunk.status == "active") and (self.include_unreviewed or chunk.reviewed)

    def sql(self, true_literal: str) -> str:
        parts = []
        if not self.include_superseded:
            parts.append("status = 'active'")
        if not self.include_unreviewed:
            parts.append(f"reviewed = {true_literal}")
        return " AND ".join(parts) or "1 = 1"


DEFAULT_FILTER = SourceFilter()


@dataclass(frozen=True)
class Scored:
    chunk: Chunk
    score: float


class Store(Protocol):
    name: str

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def dense(self, vector: np.ndarray, k: int, filt: SourceFilter = DEFAULT_FILTER) -> list[Scored]: ...
    def keyword(self, question: str, k: int, filt: SourceFilter = DEFAULT_FILTER) -> list[Scored]: ...
    def get(self, chunk_id: str) -> Chunk | None: ...
    def count(self) -> int: ...


def _chunk(row) -> Chunk:
    return Chunk(*row[:6], bool(row[6]), row[7])


class SqliteStore:
    name = "sqlite"

    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, doc_title TEXT NOT NULL,
              section TEXT NOT NULL, text TEXT NOT NULL, status TEXT NOT NULL,
              reviewed INTEGER NOT NULL, effective TEXT NOT NULL, vector BLOB NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
              id UNINDEXED, section, text, tokenize = 'porter unicode61');
            """
        )
        self._cache: tuple[list[Chunk], np.ndarray] | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("one vector per chunk")
        with self.db:
            for c, v in zip(chunks, vectors, strict=True):
                self.db.execute("DELETE FROM chunks_fts WHERE id = ?", (c.id,))
                self.db.execute(
                    f"INSERT OR REPLACE INTO chunks ({COLUMNS}, vector) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        c.id,
                        c.doc_id,
                        c.doc_title,
                        c.section,
                        c.text,
                        c.status,
                        int(c.reviewed),
                        c.effective,
                        np.asarray(v, np.float32).tobytes(),
                    ),
                )
                self.db.execute(
                    "INSERT INTO chunks_fts (id, section, text) VALUES (?, ?, ?)", (c.id, c.section, c.text)
                )
        self._cache = None

    def _all(self) -> tuple[list[Chunk], np.ndarray]:
        if self._cache is None:
            rows = self.db.execute(f"SELECT {COLUMNS}, vector FROM chunks ORDER BY id").fetchall()
            chunks = [_chunk(r) for r in rows]
            matrix = (
                np.vstack([np.frombuffer(r[8], np.float32) for r in rows]) if rows else np.zeros((0, 0), np.float32)
            )
            self._cache = (chunks, matrix)
        return self._cache

    def dense(self, vector: np.ndarray, k: int, filt: SourceFilter = DEFAULT_FILTER) -> list[Scored]:
        chunks, matrix = self._all()
        keep = [i for i, c in enumerate(chunks) if filt.allows(c)]
        if not keep:
            return []
        sims = matrix[keep] @ np.asarray(vector, np.float32)
        order = np.argsort(-sims, kind="stable")[:k]
        return [Scored(chunks[keep[i]], float(sims[i])) for i in order]

    def keyword(self, question: str, k: int, filt: SourceFilter = DEFAULT_FILTER) -> list[Scored]:
        terms = keyword_terms(question)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        rows = self.db.execute(
            f"""SELECT {", ".join("c." + col.strip() for col in COLUMNS.split(","))}, bm25(chunks_fts, 2.0, 1.0) AS s
                FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.id
                WHERE chunks_fts MATCH ? AND {filt.sql("1")}
                ORDER BY s, c.id LIMIT ?""",
            (match, k),
        ).fetchall()
        # FTS5 bm25() is lower-is-better; flip the sign so higher is better everywhere.
        return [Scored(_chunk(r), -float(r[8])) for r in rows]

    def get(self, chunk_id: str) -> Chunk | None:
        r = self.db.execute(f"SELECT {COLUMNS} FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return _chunk(r) if r else None

    def count(self) -> int:
        return self.db.execute("SELECT count(*) FROM chunks").fetchone()[0]


class PgVectorStore:
    name = "postgres"

    def __init__(self, dsn: str, dim: int, table: str = "policy_chunks", reset: bool = False):
        import psycopg
        from pgvector.psycopg import register_vector

        if not re.fullmatch(r"[a-z_][a-z0-9_]*", table):
            raise ValueError("table name must be a plain identifier")
        self.table = table
        self.dim = dim
        # No server-side prepared statements: they break behind transaction poolers (PgBouncer) and
        # on PGlite's single shared session, which the local test run uses.
        self.conn = psycopg.connect(dsn, autocommit=True, prepare_threshold=None)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        if reset:
            self.conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
              id text PRIMARY KEY, doc_id text NOT NULL, doc_title text NOT NULL,
              section text NOT NULL, text text NOT NULL,
              status text NOT NULL CHECK (status IN ('active', 'superseded')),
              reviewed boolean NOT NULL, effective text NOT NULL,
              embedding vector({dim}) NOT NULL,
              tsv tsvector GENERATED ALWAYS AS (
                setweight(to_tsvector('english', section), 'A') || setweight(to_tsvector('english', text), 'B')
              ) STORED)
            """
        )
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_hnsw ON {table} USING hnsw (embedding vector_cosine_ops)"
        )
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_tsv ON {table} USING gin (tsv)")
        # pgvector's HNSW scan applies WHERE after the index walk; with a small filtered-out share
        # this is fine, and iterative scans keep top-k full when the filter removes candidates.
        self.conn.execute("SET hnsw.iterative_scan = relaxed_order")

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("one vector per chunk")
        with self.conn.cursor() as cur:
            for c, v in zip(chunks, vectors, strict=True):
                cur.execute(
                    f"""INSERT INTO {self.table} ({COLUMNS}, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET doc_id = EXCLUDED.doc_id, doc_title = EXCLUDED.doc_title,
                          section = EXCLUDED.section, text = EXCLUDED.text, status = EXCLUDED.status,
                          reviewed = EXCLUDED.reviewed, effective = EXCLUDED.effective, embedding = EXCLUDED.embedding""",
                    (
                        c.id,
                        c.doc_id,
                        c.doc_title,
                        c.section,
                        c.text,
                        c.status,
                        c.reviewed,
                        c.effective,
                        np.asarray(v, np.float32),
                    ),
                )

    def dense(self, vector: np.ndarray, k: int, filt: SourceFilter = DEFAULT_FILTER) -> list[Scored]:
        v = np.asarray(vector, np.float32)
        rows = self.conn.execute(
            f"""SELECT {COLUMNS}, 1 - (embedding <=> %s) AS sim FROM {self.table}
                WHERE {filt.sql("true")} ORDER BY embedding <=> %s, id LIMIT %s""",
            (v, v, k),
        ).fetchall()
        return [Scored(_chunk(r), float(r[8])) for r in rows]

    def keyword(self, question: str, k: int, filt: SourceFilter = DEFAULT_FILTER) -> list[Scored]:
        terms = keyword_terms(question)
        if not terms:
            return []
        # Terms are [a-z0-9]+ only, so joining them with | is a safe OR query.
        query = " | ".join(terms)
        rows = self.conn.execute(
            f"""SELECT {COLUMNS}, ts_rank_cd(tsv, q) AS s
                FROM {self.table}, to_tsquery('english', %s) q
                WHERE tsv @@ q AND {filt.sql("true")} ORDER BY s DESC, id LIMIT %s""",
            (query, k),
        ).fetchall()
        return [Scored(_chunk(r), float(r[8])) for r in rows]

    def get(self, chunk_id: str) -> Chunk | None:
        r = self.conn.execute(f"SELECT {COLUMNS} FROM {self.table} WHERE id = %s", (chunk_id,)).fetchone()
        return _chunk(r) if r else None

    def count(self) -> int:
        return self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]
