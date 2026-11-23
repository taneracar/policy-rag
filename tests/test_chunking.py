import pytest

from policy_rag.chunking import chunk_markdown, load_corpus, parse_frontmatter, slugify

DOC = """---
id: travel
title: Travel Policy
effective: 2026-03-01
status: active
owner: Finance
---
# Travel Policy

Intro text that is not a policy statement.

## Air travel
Economy class is required.

## Lodging
The cap is $250.
"""


def test_frontmatter_and_sections():
    chunks = chunk_markdown("file-name", DOC)
    assert [c.id for c in chunks] == ["travel#air-travel", "travel#lodging"]
    assert chunks[0].doc_title == "Travel Policy"
    assert chunks[0].effective == "2026-03-01"
    assert chunks[0].status == "active" and chunks[0].reviewed
    assert "Intro text" not in " ".join(c.text for c in chunks)


def test_superseded_and_unreviewed_metadata():
    old = chunk_markdown("x", DOC.replace("status: active", "status: superseded"))
    assert {c.status for c in old} == {"superseded"}
    wiki = chunk_markdown("x", DOC.replace("owner: Finance", "owner: Employee wiki (not reviewed by policy owners)"))
    assert not any(c.reviewed for c in wiki)


def test_unknown_status_is_rejected():
    with pytest.raises(ValueError, match="unknown status"):
        chunk_markdown("x", DOC.replace("status: active", "status: draft"))


def test_unclosed_frontmatter_is_rejected():
    with pytest.raises(ValueError):
        parse_frontmatter("---\nid: x\n# Title\n")


def test_document_without_frontmatter_defaults_to_active_reviewed_unknown_owner():
    chunks = chunk_markdown("plain", "# Plain\n\n## One\nText.\n")
    assert chunks[0].id == "plain#one"
    assert chunks[0].status == "active"


def test_long_section_splits_with_paragraph_overlap():
    paragraphs = [f"Paragraph {i} " + "word " * 40 for i in range(6)]
    md = "# T\n\n## Long section\n" + "\n\n".join(paragraphs)
    chunks = chunk_markdown("doc", md, max_chars=500)
    assert len(chunks) > 1
    assert [c.id for c in chunks] == [f"doc#long-section-{i + 1}" for i in range(len(chunks))]
    assert all(c.section_id == "doc#long-section" for c in chunks)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.text.split("\n\n")[-1] == b.text.split("\n\n")[0]  # one paragraph of overlap
    assert all(f"Paragraph {i} " in "".join(c.text for c in chunks) for i in range(6))


def test_slugify():
    assert slugify("401(k) plan") == "401-k-plan"
    assert slugify("  Ground transportation and mileage ") == "ground-transportation-and-mileage"


def test_real_corpus_loads_with_unique_ids(corpus_chunks):
    ids = [c.id for c in corpus_chunks]
    assert len(ids) == len(set(ids)) > 40
    assert {c.doc_id for c in corpus_chunks if c.status == "superseded"} == {"expenses-2024"}
    assert {c.doc_id for c in corpus_chunks if not c.reviewed} == {"staff-faq"}


def test_duplicate_headings_in_one_document_fail(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\n## Same\nOne.\n\n## Same\nTwo.\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(tmp_path)
