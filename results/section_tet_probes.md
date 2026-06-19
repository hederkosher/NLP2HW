# Section 9 - Probe Sentences (method-difference stress test)

**Group A - similar meaning, different words:**
- A1: "The doctor treated the patient."
- A2: "The physician helped the sick person."
- A3: "Medical care was provided to the injured man."

**Group B - similar words, different meaning:**
- B1: "The bank approved the loan."
- B2: "The fisherman sat on the bank."
- B3: "The river bank was full of stones."

## Within-group mean cosine by method

| method | within-A (meaning) | within-B (shared words) | A − B |
| --- | --- | --- | --- |
| bow | 0.351 | 0.499 | -0.149 |
| tfidf | 0.11 | 0.244 | -0.133 |
| fasttext | 0.852 | 0.899 | -0.047 |
| minilm | 0.694 | 0.376 | 0.318 |

## Answers

- **Best at semantic similarity** (high A, and the only positive A−B): **minilm** (within-A=0.694, A−B=0.318). It is the only method that rates the different-words/same-meaning Group A as MORE similar than the shared-word Group B.
- **Most confused by lexical overlap**: **bow** (A−B=-0.149, within-B=0.499 > within-A=0.351). The shared token "bank" makes unrelated Group B sentences look MORE similar than the genuinely-related Group A sentences.
- **Caveat - FastText**: its cosines are uniformly high (within-A=0.852, within-B=0.899); mean-pooled subword vectors push every short sentence into a similar direction, so it barely discriminates either way.
- **Largest semantic-vs-lexical gap** is for pair **A1~A3** (MiniLM − lexical = +0.52): bow=0.27, tfidf=0.08, fasttext=0.84, minilm=0.70.
  - Why: for A1~A3, MiniLM sees meaning the lexical methods miss. This is exactly where word overlap and meaning point in opposite directions, so lexical and semantic encoders disagree the most.

## Full 6×6 cosine matrices

### bow

| | A1 | A2 | A3 | B1 | B2 | B3 |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | 1.00 | 0.53 | 0.27 | 0.57 | 0.53 | 0.29 |
| A2 | 0.53 | 1.00 | 0.25 | 0.53 | 0.50 | 0.27 |
| A3 | 0.27 | 0.25 | 1.00 | 0.27 | 0.25 | 0.27 |
| B1 | 0.57 | 0.53 | 0.27 | 1.00 | 0.67 | 0.43 |
| B2 | 0.53 | 0.50 | 0.25 | 0.67 | 1.00 | 0.40 |
| B3 | 0.29 | 0.27 | 0.27 | 0.43 | 0.40 | 1.00 |

### tfidf

| | A1 | A2 | A3 | B1 | B2 | B3 |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | 1.00 | 0.19 | 0.08 | 0.22 | 0.20 | 0.09 |
| A2 | 0.19 | 1.00 | 0.07 | 0.20 | 0.17 | 0.08 |
| A3 | 0.08 | 0.07 | 1.00 | 0.08 | 0.07 | 0.14 |
| B1 | 0.22 | 0.20 | 0.08 | 1.00 | 0.34 | 0.21 |
| B2 | 0.20 | 0.17 | 0.07 | 0.34 | 1.00 | 0.18 |
| B3 | 0.09 | 0.08 | 0.14 | 0.21 | 0.18 | 1.00 |

### fasttext

| | A1 | A2 | A3 | B1 | B2 | B3 |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | 1.00 | 0.83 | 0.84 | 0.89 | 0.79 | 0.81 |
| A2 | 0.83 | 1.00 | 0.88 | 0.92 | 0.90 | 0.91 |
| A3 | 0.84 | 0.88 | 1.00 | 0.85 | 0.82 | 0.88 |
| B1 | 0.89 | 0.92 | 0.85 | 1.00 | 0.91 | 0.91 |
| B2 | 0.79 | 0.90 | 0.82 | 0.91 | 1.00 | 0.88 |
| B3 | 0.81 | 0.91 | 0.88 | 0.91 | 0.88 | 1.00 |

### minilm

| | A1 | A2 | A3 | B1 | B2 | B3 |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | 1.00 | 0.72 | 0.70 | 0.10 | 0.22 | 0.05 |
| A2 | 0.72 | 1.00 | 0.66 | 0.13 | 0.19 | 0.07 |
| A3 | 0.70 | 0.66 | 1.00 | 0.12 | 0.16 | 0.09 |
| B1 | 0.10 | 0.13 | 0.12 | 1.00 | 0.35 | 0.36 |
| B2 | 0.22 | 0.19 | 0.16 | 0.35 | 1.00 | 0.42 |
| B3 | 0.05 | 0.07 | 0.09 | 0.36 | 0.42 | 1.00 |
