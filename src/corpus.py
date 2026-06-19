"""Loading the 10-text corpus and the larger sentence pool used for PCA fitting.

Document order is fixed and shared everywhere: for each source the 5 texts are in
TOPIC order (photosynthesis, cellular_respiration, jazz, blues, mount_everest),
originals first then generated -> 10 labelled documents total.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.config import (
    GENERATED_DIR,
    ORIGINAL_DIR,
    TOPIC_SLUGS,
    TOPICS,
)
from src.text_utils import split_sentences


@dataclass(frozen=True)
class Doc:
    idx: int          # 0..9 global index
    source: str       # "original" | "generated"
    slug: str         # topic slug
    domain: str       # biology | music | geography
    topic_idx: int    # 0..4 within a source
    text: str

    @property
    def label(self) -> str:
        return f"{self.source[:4]}:{self.slug}"


def load_documents() -> List[Doc]:
    """Return all 10 documents in fixed order: 5 original, then 5 generated."""
    docs: List[Doc] = []
    idx = 0
    for source, folder, prefix in (
        ("original", ORIGINAL_DIR, "wiki"),
        ("generated", GENERATED_DIR, "wiki_gen"),
    ):
        for t_idx, (slug, _title, domain) in enumerate(TOPICS):
            n = t_idx + 1
            path = folder / f"{prefix}_{n:02d}_{slug}.txt"
            docs.append(
                Doc(idx, source, slug, domain, t_idx, path.read_text("utf-8").strip())
            )
            idx += 1
    return docs


def load_sentence_pool() -> List[str]:
    """A large pool of sentences for fitting well-conditioned PCA bases.

    = sentences from the FULL original Wikipedia articles (thousands) plus the
    sentences of the 5 generated texts. Far more samples than the 10 documents,
    so PCA can yield genuine 30- and 300-component bases (sections 4/7).
    """
    pool: List[str] = []
    full_dir = ORIGINAL_DIR / "_full"
    for slug in TOPIC_SLUGS:
        fp = full_dir / f"{slug}.full.txt"
        if fp.exists():
            pool.extend(split_sentences(fp.read_text("utf-8")))
    for doc in load_documents():
        if doc.source == "generated":
            pool.extend(split_sentences(doc.text))
    # Keep only reasonably contentful sentences.
    return [s for s in pool if len(s.split()) >= 4]
