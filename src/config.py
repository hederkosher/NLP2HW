"""Shared configuration for the Homework 2 (Embeddings) pipeline.

Central place for paths, the 5 chosen topics, embedding methods/sizes, and the
section-9 probe sentences, so every script agrees on the same conventions.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ORIGINAL_DIR = DATA / "original"
GENERATED_DIR = DATA / "generated"
EMB_DIR = DATA / "embeddings"
RESULTS_DIR = ROOT / "results"
GEN_PROMPT_FILE = DATA / "generation_prompt.txt"

for _d in (ORIGINAL_DIR, GENERATED_DIR, EMB_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- The 5 Wikipedia topics (confirmed with user) --------------------------
# Two same-domain "close pairs" + one outlier, so both "identify similar" and
# "separate different" analyses are meaningful.
#   slug, Wikipedia page title, short domain tag
TOPICS = [
    ("photosynthesis", "Photosynthesis", "biology"),
    ("cellular_respiration", "Cellular respiration", "biology"),
    ("jazz", "Jazz", "music"),
    ("blues", "Blues", "music"),
    ("mount_everest", "Mount Everest", "geography"),
]
TOPIC_SLUGS = [t[0] for t in TOPICS]
# Index pairs expected to be SIMILAR (same domain) under a good method.
CLOSE_PAIRS = [(0, 1), (2, 3)]  # photosynthesis~respiration, jazz~blues
OUTLIER_IDX = 4  # mount_everest

MIN_WORDS = 300

# --- Embedding configuration -----------------------------------------------
METHODS = ["bow", "tfidf", "fasttext", "minilm"]
DIMS = [30, 300]
SOURCES = ["original", "generated"]  # "text types": human vs AI-generated
ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
ST_NATIVE_DIM = 384

# --- Section-9 probe sentences (hand-crafted to expose method differences) --
# Group A: similar MEANING, different WORDS (low lexical overlap, high semantics)
PROBE_SIMILAR_MEANING = [
    "The doctor treated the patient.",
    "The physician helped the sick person.",
    "Medical care was provided to the injured man.",
]
# Group B: similar WORDS, different MEANING ("bank" polysemy)
PROBE_SIMILAR_WORDS = [
    "The bank approved the loan.",
    "The fisherman sat on the bank.",
    "The river bank was full of stones.",
]
