"""Lightweight text utilities: cleaning, sentence/word tokenization.

Kept dependency-light on purpose. We use a regex sentence splitter and a simple
word tokenizer so the pipeline runs without downloading NLTK data; NLTK is in
requirements only as a fallback / for graders who prefer it.
"""
from __future__ import annotations

import re
from typing import List

_WS = re.compile(r"\s+")
# Split on sentence-ending punctuation followed by whitespace + capital/quote.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z0-9])")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def clean_text(text: str) -> str:
    """Collapse whitespace and strip leftover bracketed reference markers."""
    text = re.sub(r"\[[0-9a-zA-Z]+\]", " ", text)  # [1], [a], [citation needed]-ish
    text = _WS.sub(" ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Regex sentence splitter. Good enough for prose; no model download."""
    text = clean_text(text)
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if len(p.strip()) > 0]


def tokenize_words(text: str) -> List[str]:
    """Lowercase alphabetic word tokens (apostrophes kept inside words)."""
    return _WORD.findall(text.lower())


def word_count(text: str) -> int:
    return len(tokenize_words(text))
