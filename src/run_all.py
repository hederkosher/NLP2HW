"""Run the whole Homework-2 pipeline end to end, in order.

    python -m src.run_all

Steps (each is also runnable on its own):
  1. data_collection     (סעיף א)  -> data/original/
  2. generate_texts      (סעיף ב)  -> validates data/generated/
  3. embeddings          (ג + ד)   -> data/embeddings/ (16 sets + artifacts)
  4. analysis            (ה, ו, ז) -> results/*.csv, results/heatmaps_*.png
  5. manual_walkthrough  (ח)       -> results/section_het_manual.md
  6. probes              (ט)       -> results/section_tet_probes.{md,png}

Re-running is idempotent. Network is needed only for step 1 (Wikipedia) and the
first run of MiniLM (model download).
"""
from __future__ import annotations

import sys


def main() -> int:
    from src import (analysis, data_collection, embeddings, generate_texts,
                     manual_walkthrough, probes)

    steps = [
        ("1/6 data_collection (א)", data_collection.main),
        ("2/6 generate_texts (ב)", lambda: generate_texts.main([])),
        ("3/6 embeddings (ג+ד)", embeddings.main),
        ("4/6 analysis (ה,ו,ז)", analysis.main),
        ("5/6 manual_walkthrough (ח)", manual_walkthrough.main),
        ("6/6 probes (ט)", probes.main),
    ]
    for name, fn in steps:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        rc = fn()
        if rc:
            print(f"!! step '{name}' returned {rc}; stopping.")
            return rc
    print("\nAll steps complete. See results/ and REPORT.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
