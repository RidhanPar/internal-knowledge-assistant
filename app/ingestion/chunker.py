"""Structure-aware Markdown chunking.

Chunking strategy (defensible design choices)
---------------------------------------------
1. **Split on Markdown headings first.** Internal docs are written in sections,
   and a heading marks a topic boundary. Cutting there keeps each chunk about a
   single subject, which is exactly what a dense retriever needs: one coherent
   idea per vector. We keep the full heading path (H1 > H2 > H3) as metadata so
   citations can say *where* in the document an answer came from, and so the
   heading text itself is embedded as useful context.

2. **Pack to a token budget, not a character count.** Embedding models and the
   generator both think in tokens, and retrieval granularity matters: chunks
   that are too large blur multiple facts into one vector (poor precision);
   chunks that are too small lose the context needed to answer (poor recall).
   ~450 tokens is a good middle ground for policy/handbook prose — roughly a
   few paragraphs.

3. **Respect natural boundaries.** We pack whole paragraphs, and only fall back
   to sentence splitting when a single paragraph exceeds the budget. We never
   cut mid-sentence, which would strand half a thought in each neighbour.

4. **Overlap between chunks.** A fixed token overlap (~60) carries the tail of
   one chunk into the head of the next so a fact spanning a boundary is still
   fully present in at least one chunk. This trades a little storage for
   materially better recall on boundary-straddling questions.

Tokenisation uses tiktoken's `cl100k_base` as a fast, dependency-light
*approximation*. Titan/Claude tokenise slightly differently, but for sizing
chunks the exact tokenizer doesn't matter — we only need consistent, roughly
model-scale counts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Chunking parameters. Exposed as module constants so tests and callers can
# reason about them; tunable without touching the algorithm.
TARGET_TOKENS = 450
OVERLAP_TOKENS = 60
# A section shorter than this is emitted whole rather than being packed with a
# budget loop — avoids pointless single-paragraph splitting.
MIN_CHUNK_TOKENS = 16

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Sentence splitter: break after ., !, ? followed by whitespace. Good enough for
# prose; deliberately not a full NLP sentence tokenizer.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


@dataclass
class Chunk:
    document_id: str
    chunk_index: int
    heading_path: str
    content: str
    token_count: int = field(default=0)

    def __post_init__(self) -> None:
        if not self.token_count:
            self.token_count = count_tokens(self.content)


@dataclass
class _Section:
    heading_path: str
    body: str


def _iter_sections(text: str) -> list[_Section]:
    """Split raw markdown into sections keyed by their heading path.

    Text appearing before the first heading (e.g. a preamble) is attached to a
    synthetic empty heading path so it is never dropped.
    """
    sections: list[_Section] = []
    # Stack of (level, title) tracking the current heading hierarchy.
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            path = " > ".join(title for _, title in heading_stack)
            sections.append(_Section(heading_path=path, body=body))

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            # New heading — close out the section accumulated so far.
            flush()
            current_lines = []
            level = len(m.group(1))
            title = m.group(2).strip()
            # Pop deeper-or-equal headings, then push this one.
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        else:
            current_lines.append(line)

    flush()
    return sections


def _split_paragraph(paragraph: str) -> list[str]:
    """Split an over-long paragraph into sentence-bounded pieces under budget."""
    sentences = _SENTENCE_RE.split(paragraph)
    pieces: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for sentence in sentences:
        st = count_tokens(sentence)
        if buf and buf_tokens + st > TARGET_TOKENS:
            pieces.append(" ".join(buf))
            buf, buf_tokens = [], 0
        buf.append(sentence)
        buf_tokens += st
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def _pack_section(body: str) -> list[str]:
    """Pack a section body into token-bounded chunks with overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

    # Normalise to a list of units no larger than the budget.
    units: list[str] = []
    for para in paragraphs:
        if count_tokens(para) > TARGET_TOKENS:
            units.extend(_split_paragraph(para))
        else:
            units.append(para)

    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for unit in units:
        ut = count_tokens(unit)
        if buf and buf_tokens + ut > TARGET_TOKENS:
            chunks.append("\n\n".join(buf))
            # Seed the next buffer with a token-bounded overlap taken from the
            # tail of the one we just emitted.
            overlap = _tail_overlap(buf)
            buf = overlap
            buf_tokens = sum(count_tokens(u) for u in buf)
        buf.append(unit)
        buf_tokens += ut

    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _tail_overlap(units: list[str]) -> list[str]:
    """Take whole trailing units up to OVERLAP_TOKENS to carry into next chunk."""
    overlap: list[str] = []
    total = 0
    for unit in reversed(units):
        t = count_tokens(unit)
        if total + t > OVERLAP_TOKENS and overlap:
            break
        overlap.insert(0, unit)
        total += t
    return overlap


def chunk_markdown(document_id: str, text: str) -> list[Chunk]:
    """Chunk a markdown document into retrievable, heading-tagged chunks."""
    chunks: list[Chunk] = []
    index = 0
    for section in _iter_sections(text):
        for piece in _pack_section(section.body):
            if (
                count_tokens(piece) < MIN_CHUNK_TOKENS
                and chunks
                and chunks[-1].heading_path == section.heading_path
            ):
                # Tiny trailing fragment *within the same section* — fold into the
                # previous chunk instead of emitting a near-empty vector. We never
                # merge across sections: that would file text under the wrong
                # heading and corrupt citations.
                prev = chunks[-1]
                merged = f"{prev.content}\n\n{piece}"
                chunks[-1] = Chunk(
                    document_id=prev.document_id,
                    chunk_index=prev.chunk_index,
                    heading_path=prev.heading_path,
                    content=merged,
                )
                continue
            # Prepend the heading path so the embedded text carries its context.
            embedded = f"{section.heading_path}\n\n{piece}" if section.heading_path else piece
            chunks.append(
                Chunk(
                    document_id=document_id,
                    chunk_index=index,
                    heading_path=section.heading_path,
                    content=embedded,
                )
            )
            index += 1
    return chunks
