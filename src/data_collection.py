"""Phase 1 (סעיף א): fetch 5 English Wikipedia articles.

Saves two things per topic:
  * data/original/wiki_0N_<slug>.txt  -> a >=300-word cleaned EXCERPT (the
    graded deliverable: 5 natural-prose texts).
  * data/original/_full/<slug>.full.txt -> the FULL cleaned article, used later
    only as a larger sentence pool for fitting PCA bases (sections ד/ז need more
    samples than 5 documents can provide). Not part of the 5 graded texts.

Run:  python -m src.data_collection
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

from src.config import MIN_WORDS, ORIGINAL_DIR, TOPICS
from src.text_utils import clean_text, split_sentences, word_count

FULL_DIR = ORIGINAL_DIR / "_full"

API_URL = "https://en.wikipedia.org/w/api.php"
# Wikipedia requires a descriptive User-Agent; requests without one are blocked
# (the legacy `wikipedia` pip package omits it and now gets empty responses).
USER_AGENT = "NLP2HW-Embeddings/1.0 (educational coursework; contact: student)"


def fetch_article(title: str, retries: int = 4) -> str:
    """Return the full plain-text content of a Wikipedia page via the action API."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            pages = data["query"]["pages"]
            page = next(iter(pages.values()))
            extract = page.get("extract", "")
            if extract:
                return extract
            last_err = RuntimeError(f"empty extract for '{title}'")
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"could not fetch '{title}': {last_err}")


def make_excerpt(full_text: str, min_words: int = MIN_WORDS) -> str:
    """Take leading sentences until we exceed `min_words` (natural prose)."""
    sentences = split_sentences(full_text)
    out, count = [], 0
    for s in sentences:
        # Skip Wikipedia section headers like "== History =="
        if s.startswith("=="):
            continue
        out.append(s)
        count += word_count(s)
        if count >= min_words:
            break
    return " ".join(out)


def main() -> int:
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(TOPICS)} English Wikipedia articles...\n")
    for i, (slug, title, domain) in enumerate(TOPICS, start=1):
        try:
            raw = fetch_article(title)
        except Exception as e:  # noqa: BLE001 - report, do not fake
            print(f"  !! FAILED to fetch '{title}': {type(e).__name__}: {e}")
            return 1

        # Strip section markers from the full text for the PCA pool.
        full_clean = clean_text(
            "\n".join(
                line for line in raw.splitlines() if not line.strip().startswith("==")
            )
        )
        (FULL_DIR / f"{slug}.full.txt").write_text(full_clean, encoding="utf-8")

        excerpt = make_excerpt(raw)
        wc = word_count(excerpt)
        out_path = ORIGINAL_DIR / f"wiki_{i:02d}_{slug}.txt"
        out_path.write_text(excerpt, encoding="utf-8")
        flag = "OK " if wc >= MIN_WORDS else "!! "
        print(f"  {flag}wiki_{i:02d}_{slug}.txt  ({wc} words, domain={domain})")

    print("\nDone. Originals in data/original/, full articles in data/original/_full/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
