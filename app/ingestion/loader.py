"""Load raw documents from the corpus directory."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_H1_RE = re.compile(r"^#\s+(.*\S)\s*$", re.MULTILINE)


@dataclass
class RawDocument:
    id: str
    title: str
    source_path: str
    content: str
    content_hash: str


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_corpus(corpus_dir: str | Path) -> list[RawDocument]:
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_path.resolve()}")

    docs: list[RawDocument] = []
    for path in sorted(corpus_path.glob("**/*.md")):
        content = path.read_text(encoding="utf-8")
        m = _H1_RE.search(content)
        title = m.group(1).strip() if m else path.stem.replace("-", " ").title()
        docs.append(
            RawDocument(
                id=_slugify(path.stem),
                title=title,
                source_path=str(path.as_posix()),
                content=content,
                content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
        )
    if not docs:
        raise ValueError(f"No .md documents found under {corpus_path.resolve()}")
    return docs
