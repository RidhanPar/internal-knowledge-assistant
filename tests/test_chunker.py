"""Chunking is the retrieval quality lever, so its invariants are tested directly.

These tests need no database or Bedrock — chunking is pure text processing.
"""

from __future__ import annotations

from app.ingestion import chunker
from app.ingestion.chunker import TARGET_TOKENS, chunk_markdown, count_tokens


def test_heading_path_is_tracked_hierarchically():
    text = (
        "# Policy\n\n"
        "Intro paragraph.\n\n"
        "## Access Control\n\n"
        "Least privilege applies.\n\n"
        "### MFA\n\n"
        "MFA is mandatory for all accounts.\n"
    )
    chunks = chunk_markdown("policy", text)
    paths = {c.heading_path for c in chunks}
    assert "Policy > Access Control > MFA" in paths
    # A sibling deeper heading must not inherit an unrelated branch.
    assert "Policy > Access Control" in paths


def test_preamble_before_first_heading_is_not_dropped():
    text = "Some preamble with no heading at all.\n\n# Later Heading\n\nBody."
    chunks = chunk_markdown("doc", text)
    joined = "\n".join(c.content for c in chunks)
    assert "preamble" in joined


def test_chunks_respect_token_budget_with_tolerance():
    # Build a long single section that must be split into multiple chunks.
    paragraph = ("This is a sentence about company policy. " * 20).strip()
    body = "\n\n".join([paragraph] * 10)
    text = f"# Big Section\n\n{body}\n"
    chunks = chunk_markdown("big", text)
    assert len(chunks) > 1
    # Allow headroom for the prepended heading path and overlap seed.
    for c in chunks:
        assert count_tokens(c.content) <= TARGET_TOKENS * 1.6


def test_chunk_indices_are_contiguous_and_zero_based():
    text = "# A\n\nalpha body here.\n\n# B\n\nbeta body here.\n"
    chunks = chunk_markdown("doc", text)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_overlap_carries_context_between_adjacent_chunks():
    # Distinct, sentence-like units so we can detect shared tail/head text.
    units = [f"Fact number {i} about the policy is important." for i in range(60)]
    body = "\n\n".join(units)
    text = f"# Section\n\n{body}\n"
    chunks = chunk_markdown("doc", text)
    assert len(chunks) >= 2
    # The end of chunk 0 should reappear at the start of chunk 1 (overlap).
    first_tail = chunks[0].content.split("\n\n")[-1]
    assert first_tail in chunks[1].content


def test_tiny_trailing_fragment_is_merged():
    # A section whose final unit is tiny should not become its own chunk.
    big = "\n\n".join([("word " * 200).strip()] * 3)
    text = f"# S\n\n{big}\n\ntiny.\n"
    chunks = chunk_markdown("doc", text)
    assert all(count_tokens(c.content) >= chunker.MIN_CHUNK_TOKENS for c in chunks)
