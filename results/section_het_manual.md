# Section 8 - Manual BoW / TF-IDF / Cosine Walkthrough

Three sentences taken verbatim from `data/generated/wiki_gen_01_photosynthesis.txt`:

- **S1:** "Plants make their own food."
- **S2:** "They make this food from sunlight, water, and air."
- **S3:** "The green parts of a plant, like the leaves, do most of this work."

**Tokenization** (lowercase, alphabetic tokens only):

- S1 → ['plants', 'make', 'their', 'own', 'food']
- S2 → ['they', 'make', 'this', 'food', 'from', 'sunlight', 'water', 'and', 'air']
- S3 → ['the', 'green', 'parts', 'of', 'a', 'plant', 'like', 'the', 'leaves', 'do', 'most', 'of', 'this', 'work']

**Vocabulary** (sorted, 23 terms): `['a', 'air', 'and', 'do', 'food', 'from', 'green', 'leaves', 'like', 'make', 'most', 'of', 'own', 'parts', 'plant', 'plants', 'sunlight', 'the', 'their', 'they', 'this', 'water', 'work']`

## 1. Bag-of-Words (counting by hand)

Each cell = number of times the term occurs in the sentence.

| doc | a | air | and | do | food | from | green | leaves | like | make | most | of | own | parts | plant | plants | sunlight | the | their | they | this | water | work |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| S2 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 1 | 0 |
| S3 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 1 | 2 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 |

## 2. TF-IDF (step by step)

**TF** = raw count above. **DF** = #sentences containing the term. **IDF** = ln((1+N)/(1+DF)) + 1, with N = 3 (smoothed, as in scikit-learn).

| term | DF | IDF |
| --- | --- | --- |
| a | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| air | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| and | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| do | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| food | 2 | ln((1+3)/(1+2))+1 = 1.2877 |
| from | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| green | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| leaves | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| like | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| make | 2 | ln((1+3)/(1+2))+1 = 1.2877 |
| most | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| of | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| own | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| parts | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| plant | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| plants | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| sunlight | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| the | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| their | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| they | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| this | 2 | ln((1+3)/(1+2))+1 = 1.2877 |
| water | 1 | ln((1+3)/(1+1))+1 = 1.6931 |
| work | 1 | ln((1+3)/(1+1))+1 = 1.6931 |

**TF·IDF** (raw), then each row is L2-normalized (divide the row by its Euclidean norm - scikit-learn default):

| doc | a | air | and | do | food | from | green | leaves | like | make | most | of | own | parts | plant | plants | sunlight | the | their | they | this | water | work |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 raw | 0.000 | 0.000 | 0.000 | 0.000 | 1.288 | 0.000 | 0.000 | 0.000 | 0.000 | 1.288 | 0.000 | 0.000 | 1.693 | 0.000 | 0.000 | 1.693 | 0.000 | 0.000 | 1.693 | 0.000 | 0.000 | 0.000 | 0.000 |
| S1 norm | 0.000 | 0.000 | 0.000 | 0.000 | 0.373 | 0.000 | 0.000 | 0.000 | 0.000 | 0.373 | 0.000 | 0.000 | 0.490 | 0.000 | 0.000 | 0.490 | 0.000 | 0.000 | 0.490 | 0.000 | 0.000 | 0.000 | 0.000 |
| S2 raw | 0.000 | 1.693 | 1.693 | 0.000 | 1.288 | 1.693 | 0.000 | 0.000 | 0.000 | 1.288 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.693 | 0.000 | 0.000 | 1.693 | 1.288 | 1.693 | 0.000 |
| S2 norm | 0.000 | 0.360 | 0.360 | 0.000 | 0.273 | 0.360 | 0.000 | 0.000 | 0.000 | 0.273 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.360 | 0.000 | 0.000 | 0.360 | 0.273 | 0.360 | 0.000 |
| S3 raw | 1.693 | 0.000 | 0.000 | 1.693 | 0.000 | 0.000 | 1.693 | 1.693 | 1.693 | 0.000 | 1.693 | 3.386 | 0.000 | 1.693 | 1.693 | 0.000 | 0.000 | 3.386 | 0.000 | 0.000 | 1.288 | 0.000 | 1.693 |
| S3 norm | 0.239 | 0.000 | 0.000 | 0.239 | 0.000 | 0.000 | 0.239 | 0.239 | 0.239 | 0.000 | 0.239 | 0.477 | 0.000 | 0.239 | 0.239 | 0.000 | 0.000 | 0.477 | 0.000 | 0.000 | 0.181 | 0.000 | 0.239 |

Row norms used for normalization: ‖S1‖=3.4520, ‖S2‖=4.7090, ‖S3‖=7.0988

## 3. Cosine similarity between word pairs

Per the chosen definition, a **word's vector is its column** across the 3 sentences. We compute cosine on (a) the BoW count columns and (b) the normalized TF-IDF columns.

### BoW

**cos('make', 'food')**

- column('make') = [1.0000, 1.0000, 0.0000]
- column('food') = [1.0000, 1.0000, 0.0000]
- dot = (1.0000)(1.0000) + (1.0000)(1.0000) + (0.0000)(0.0000) = **2.0000**
- ‖make‖ = √2.0000 = 1.4142,  ‖food‖ = √2.0000 = 1.4142
- cosine = 2.0000 / (1.4142 × 1.4142) = **1.0000**

**cos('food', 'this')**

- column('food') = [1.0000, 1.0000, 0.0000]
- column('this') = [0.0000, 1.0000, 1.0000]
- dot = (1.0000)(0.0000) + (1.0000)(1.0000) + (0.0000)(1.0000) = **1.0000**
- ‖food‖ = √2.0000 = 1.4142,  ‖this‖ = √2.0000 = 1.4142
- cosine = 1.0000 / (1.4142 × 1.4142) = **0.5000**

**cos('plant', 'the')**

- column('plant') = [0.0000, 0.0000, 1.0000]
- column('the') = [0.0000, 0.0000, 2.0000]
- dot = (0.0000)(0.0000) + (0.0000)(0.0000) + (1.0000)(2.0000) = **2.0000**
- ‖plant‖ = √1.0000 = 1.0000,  ‖the‖ = √4.0000 = 2.0000
- cosine = 2.0000 / (1.0000 × 2.0000) = **1.0000**

### TF-IDF (L2-normalized)

**cos('make', 'food')**

- column('make') = [0.3730, 0.2735, 0.0000]
- column('food') = [0.3730, 0.2735, 0.0000]
- dot = (0.3730)(0.3730) + (0.2735)(0.2735) + (0.0000)(0.0000) = **0.2139**
- ‖make‖ = √0.2139 = 0.4625,  ‖food‖ = √0.2139 = 0.4625
- cosine = 0.2139 / (0.4625 × 0.4625) = **1.0000**

**cos('food', 'this')**

- column('food') = [0.3730, 0.2735, 0.0000]
- column('this') = [0.0000, 0.2735, 0.1814]
- dot = (0.3730)(0.0000) + (0.2735)(0.2735) + (0.0000)(0.1814) = **0.0748**
- ‖food‖ = √0.2139 = 0.4625,  ‖this‖ = √0.1077 = 0.3281
- cosine = 0.0748 / (0.4625 × 0.3281) = **0.4927**

**cos('plant', 'the')**

- column('plant') = [0.0000, 0.0000, 0.2385]
- column('the') = [0.0000, 0.0000, 0.4770]
- dot = (0.0000)(0.0000) + (0.0000)(0.0000) + (0.2385)(0.4770) = **0.1138**
- ‖plant‖ = √0.0569 = 0.2385,  ‖the‖ = √0.2276 = 0.4770
- cosine = 0.1138 / (0.2385 × 0.4770) = **1.0000**

## Summary of word-pair cosines

| word pair | BoW cosine | TF-IDF cosine |
| --- | --- | --- |
| make ~ food | 1.0000 | 1.0000 |
| food ~ this | 0.5000 | 0.4927 |
| plant ~ the | 1.0000 | 1.0000 |

> Note: BoW weights every shared sentence equally, so columns with the same occurrence pattern have cosine 1.0. After TF-IDF's IDF weighting and per-sentence L2 normalization, the same columns are re-scaled differently per sentence, which changes the cosines - showing how TF-IDF down-weights terms that appear in many sentences.

---
_Cross-check: the by-hand BoW matches scikit-learn CountVectorizer = **True**; the by-hand normalized TF-IDF matches scikit-learn TfidfVectorizer = **True**._
