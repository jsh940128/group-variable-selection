# Unstructured Variable Selection (UVS) — Minimal Example

This repository provides a **minimal, shareable example** of running
**Unstructured Variable Selection (UVS)** on review text data, with an **optional OpenAI API step** for vocabulary cleaning.

Focus:

* simple UVS pipeline
* explicit aggregation logic
* optional, auditable OpenAI usage

---

## 1. High-Level Pipeline

```text
Raw Review CSV
   ↓
review_text = title + text
   ↓
Aggregate by some column such as id / this can be adjusted
   ↓
UnstructuredVariableSelection (UVS)
   ↓
(Optional) Vocabulary Cleaning (OpenAI API)
```

## 2. Input Data Assumptions

```text
Required columns in raw CSV:

locationId   → aggregation unit (hotel / venue / property)
title        → review title
text         → review body
rating       → numeric outcome (or equivalent)

Construction rules:
- review_text = title + " " + text
- aggregation level = locationId
```

## 3. Core Model: UnstructureVariableSelection

```text
UnstructureVariableSelection (UVS):

- extracts word-level features from unstructured text
- models word co-occurrence vs. numeric outcomes
- selects informative lexical variables via Bayesian selection

Example constraints:
- small vocabulary (top_n kept low)
- short MCMC runs

Purpose: teaching / reference, not production tuning.
```

## 4. Optional OpenAI API Usage

```text
OpenAI is used ONLY for vocabulary cleaning.

Rationale:
- remove noise tokens
- normalize variants
- optionally assign lemmas

Not used for:
- text generation
- labeling
- prediction
```

## 5. Vocabulary Cleaner

```text
File: word_corpus_cleaner.py

What it does:
- drops obvious noise (URLs, IDs, fragments)
- normalizes tokens
- optionally assigns lemmas (e.g., cleaner → clean)

What it does NOT do:
- does not modify review text
- does not touch ratings
- does not affect model structure

All decisions are written to Excel for inspection.
```

## 6. OpenAI API Principles

```text
Design rules:

1) Batch-based calls (default: 150 tokens)
2) Deterministic fallback on API failure
3) Explicit keep / drop / lemma logging
4) No dependency between UVS and API

Result: reproducible and publication-safe.
```

## 7. Minimal API Setup

```text
Dependencies:
pip install -U openai python-dotenv pandas openpyxl

.env file:
OPENAI_API_KEY=your_api_key_here
```

## 8. When to Use the API Step

```text
Use if:
- vocabulary is large or noisy
- interpretability matters

Skip if:
- quick exploration
- curated vocabulary
- zero external dependencies needed
```

## 9. Guiding Principle

```text
If a step is heuristic:
- keep it optional
- keep it inspectable
- keep it replaceable
```