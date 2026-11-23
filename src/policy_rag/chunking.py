"""Split markdown policy documents into section chunks.

One chunk per `##` section keeps a citation pointing at a heading a person can find.
Long sections are split on paragraph boundaries with one paragraph of overlap.

Each document starts with a small frontmatter block (id, title, effective, status, owner,
supersedes / superseded_by). Every chunk carries that metadata so retrieval can filter out
superseded versions and sources no policy owner has reviewed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc_id>#<section-slug>" or "<doc_id>#<section-slug>-<n>" for split sections
    doc_id: str
    doc_title: str
    section: str
    text: str
    status: str = "active"  # "active" or "superseded"
    reviewed: bool = True  # False for sources no policy owner reviewed (e.g. an employee wiki)
    effective: str = ""

    @property
    def section_id(self) -> str:
        """The section this chunk belongs to, without the split suffix."""
        return f"{self.doc_id}#{slugify(self.section)}"

    def embedding_text(self) -> str:
        # Title and heading carry most of the topic signal for short sections.
        return f"{self.doc_title}. {self.section}. {self.text}"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def parse_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    """Return (fields, body). Only flat `key: value` lines are supported, which is all the corpus uses."""
    if not markdown.startswith("---\n"):
        return {}, markdown
    end = markdown.find("\n---", 4)
    if end == -1:
        raise ValueError("frontmatter is not closed with ---")
    fields = {}
    for line in markdown[4:end].splitlines():
        if line.strip():
            key, sep, value = line.partition(":")
            if not sep:
                raise ValueError(f"bad frontmatter line: {line!r}")
            fields[key.strip()] = value.strip()
    return fields, markdown[end + 4 :].lstrip("\n")


def is_reviewed(owner: str) -> bool:
    return bool(owner) and "not reviewed" not in owner.lower()


def chunk_markdown(doc_id: str, markdown: str, max_chars: int = 1200) -> list[Chunk]:
    meta, markdown = parse_frontmatter(markdown)
    doc_id = meta.get("id", doc_id)
    status = meta.get("status", "active")
    if status not in ("active", "superseded"):
        raise ValueError(f"{doc_id}: unknown status {status!r}")
    reviewed = is_reviewed(meta.get("owner", "unknown"))
    effective = meta.get("effective", "")
    title = meta.get("title", doc_id)
    sections: list[tuple[str, list[str]]] = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif sections:
            sections[-1][1].append(line)
    # Text before the first `##` (the preamble) is not a policy statement and is not indexed.

    chunks: list[Chunk] = []
    for heading, lines in sections:
        paragraphs = [p.strip() for p in "\n".join(lines).split("\n\n") if p.strip()]
        if not paragraphs:
            continue
        parts = _pack(paragraphs, max_chars)
        for n, part in enumerate(parts):
            suffix = "" if len(parts) == 1 else f"-{n + 1}"
            chunks.append(
                Chunk(
                    id=f"{doc_id}#{slugify(heading)}{suffix}",
                    doc_id=doc_id,
                    doc_title=title,
                    section=heading,
                    text=part,
                    status=status,
                    reviewed=reviewed,
                    effective=effective,
                )
            )
    return chunks


def _pack(paragraphs: list[str], max_chars: int) -> list[str]:
    """Greedy packing; each new part starts with the previous part's last paragraph."""
    parts: list[list[str]] = [[]]
    for p in paragraphs:
        current = parts[-1]
        if current and len("\n\n".join([*current, p])) > max_chars:
            parts.append([current[-1], p] if len(current) > 1 else [p])
        else:
            current.append(p)
    return ["\n\n".join(p) for p in parts]


def load_corpus(directory: Path, max_chars: int = 1200) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_markdown(path.stem, path.read_text(encoding="utf-8"), max_chars))
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chunk ids; two sections in one document share a heading")
    return chunks
