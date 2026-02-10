# pip install -U openai python-dotenv pandas openpyxl
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# 0) Config / CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean a vocabulary Excel file: drop noise, normalize tokens, optional lemma."
    )
    p.add_argument(
        "--word-corpus-xlsx",
        dest="word_corpus_xlsx",
        default=None,
        help='Path to the input XLSX file. If omitted, uses env var WORD_CORPUS_XLSX.',
    )
    p.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=150,
        help="Number of unique tokens per API batch (default: 150).",
    )
    return p.parse_args()


def resolve_input_path(cli_value: str | None) -> Path:
    raw = cli_value or os.getenv("WORD_CORPUS_XLSX")
    if not raw:
        print("ERROR: Provide --word-corpus-xlsx PATH or set env var WORD_CORPUS_XLSX.")
        sys.exit(1)
    p = Path(raw).expanduser().resolve()
    if not p.exists():
        print(f"ERROR: File not found: {p}")
        sys.exit(1)
    if p.suffix.lower() not in {".xlsx", ".xls"}:
        print(f"ERROR: Expected an Excel file (.xlsx/.xls). Got: {p.name}")
        sys.exit(1)
    return p


# ============================================================
# 1) Column detection
# ============================================================
def detect_word_col(columns) -> str:
    cols = list(columns)
    low = {str(c).strip().lower(): c for c in cols}
    for k in ("word", "token", "vocab", "term", "original", "valid_corpus"):
        if k in low:
            return low[k]
    return cols[0]


# ============================================================
# 2) Deterministic cleaning helpers
# ============================================================
def norm_cell(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none"):
        return ""
    return s


def is_obviously_noise(w: str) -> bool:
    if not w or len(w) <= 1:
        return True
    if w.isdigit():
        return True
    if not any(ch.isalpha() for ch in w):
        return True
    wl = w.lower()
    if "http" in wl or "www." in wl or "@" in wl:
        return True
    if re.fullmatch(r"[\W_]+", w):
        return True
    return False


# ============================================================
# 3) OpenAI review
# ============================================================
SYSTEM_INSTRUCTIONS = """
You are cleaning a vocabulary list for a text-based statistical model
that relies on word frequency, co-occurrence, and Bayesian variable selection.

For each input token, output ONE JSON object with EXACT keys:
- word: original string (exactly as given)
- keep: true/false
- corrected: a SINGLE TOKEN correction (lowercase, no spaces). If no change, return the original token.
- lemma: "" OR a SINGLE TOKEN lemma (no spaces). Leave "" if unnecessary.

Hard constraints:
- NEVER output multiword strings in corrected/lemma (no spaces).
- NEVER introduce new tokens not grounded in the input word.
- If unsure, keep=true and leave lemma="".

Lemma guidance (conservative):
- Use lemma only when it reduces trivial morphological variants (plural/tense/derivation).
- Do NOT over-collapse meanings (e.g., staff ≠ service; room ≠ suite).

Keep criteria:
- Content words carrying sentiment, quality, amenities, service, location, or experience signal
  (e.g., clean, noisy, breakfast, spa, concierge, view, location, checkout).

Remove criteria:
- Stopwords, filler words, artifacts, punctuation fragments, encoding noise,
  URLs/emails/IDs, and non-lexical tokens.

Return STRICT JSONL (one JSON per line).
No markdown. No commentary.
""".strip()


def call_review_batch(client: OpenAI, model: str, batch_words: list[str]) -> list[dict]:
    """
    Sends a batch of words to OpenAI and parses the JSONL response.
    Includes error handling to prevent the script from crashing on bad API responses.
    """
    payload = {"words": batch_words}
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            temperature=0.0
        )
        text = (resp.choices[0].message.content or "").strip()

        rows: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Handle potential markdown code blocks if the model ignores instructions
            if line.startswith("```"): continue 
            
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  [Warning] Skipping malformed JSON line: {line[:40]}...")
                continue
        return rows

    except Exception as e:
        print(f"  [Error] API Batch Failed: {e}")
        return [] # Return empty list; logic will fallback to keeping original words


# ============================================================
# 4) Main Logic: clean_word_corpus(...)
# ============================================================
def clean_word_corpus(
    word_corpus_xlsx: str | Path,
    *,
    model: str = "gpt-4o-mini",
    batch_size: int = 150,
    load_env: bool = True,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """
    Clean a vocabulary XLSX in-place (token column) and write a *_cleaned.xlsx copy.
    """
    if load_env:
        load_dotenv()

    if client is None:
        client = OpenAI()

    in_path = Path(word_corpus_xlsx).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"File not found: {in_path}")
    
    out_path = in_path.with_name(in_path.stem + "_cleaned.xlsx")

    print(f"[word_corpus_cleaner] Input file : {in_path}")
    print(f"[word_corpus_cleaner] Output file: {out_path}")

    # --- Load workbook
    df = pd.read_excel(in_path)

    # --- Detect token column
    word_col = detect_word_col(df.columns)

    # --- Ensure lemma column exists
    if "lemma" not in df.columns:
        # Place lemma immediately to the right of the token column
        cols = list(df.columns)
        cols.insert(cols.index(word_col) + 1, "lemma")
        df = df.reindex(columns=cols)

    # --- Normalize word cells + drop blank rows
    raw_rows = len(df)
    df[word_col] = df[word_col].apply(norm_cell)
    blank_dropped = int((df[word_col] == "").sum())
    df = df[df[word_col].ne("")].copy()

    # --- Unique tokens list (pre-correction)
    original_tokens = df[word_col].unique().tolist()
    original_unique = len(original_tokens)

    # --- Deterministic noise filter (drop without API)
    obvious_noise_set = {w for w in original_tokens if is_obviously_noise(w)}
    obvious_noise_unique = len(obvious_noise_set)
    tokens_for_api = [w for w in original_tokens if w not in obvious_noise_set]

    # --- OpenAI review
    batch_size = max(1, int(batch_size))
    num_batches = math.ceil(len(tokens_for_api) / batch_size) if tokens_for_api else 0

    review_map: dict[str, dict] = {}  # word -> {keep, corrected, lemma}
    
    print(f"[word_corpus_cleaner] Processing {len(tokens_for_api)} unique tokens in {num_batches} batches...")

    for i in range(num_batches):
        batch = tokens_for_api[i * batch_size : (i + 1) * batch_size]
        result_rows = call_review_batch(client=client, model=model, batch_words=batch)

        for r in result_rows:
            w = str(r.get("word", "")).strip()
            if not w: 
                continue
            review_map[w] = {
                "keep": bool(r.get("keep", True)),
                "corrected": (r.get("corrected") or w).strip(),
                "lemma": (r.get("lemma") or "").strip(),
            }
        
        # Simple progress indicator
        if (i + 1) % 5 == 0:
            print(f"  Batch {i+1}/{num_batches} complete.")

    # --- Apply decisions (Optimized: Single Pass) ---
    def get_decision_row(w: str) -> Tuple[bool, str, Any, str]:
        # 1. Obvious noise check
        if w in obvious_noise_set:
            return False, w, pd.NA, "obvious_noise"

        # 2. API Result check
        ann = review_map.get(w)
        if ann is None:
            # Fallback: keep as-is if model missed it
            return True, w, pd.NA, "fallback_keep"

        keep = ann["keep"]
        corrected = ann["corrected"] or w
        # Use pd.NA for empty lemmas to ensure clean Excel output
        lemma = ann["lemma"] if ann["lemma"] else pd.NA
        reason = "model_keep" if keep else "model_drop"
        
        return keep, corrected, lemma, reason

    # Generate all columns in one list comprehension (fastest method)
    results = [get_decision_row(w) for w in df[word_col]]
    
    # Unpack results into separate tuples
    keep_flags, corrected_vals, lemma_vals, reasons = zip(*results)

    # Assign back to DataFrame
    df["__keep"] = keep_flags
    df[word_col] = corrected_vals
    df["lemma"] = lemma_vals
    
    # Calculate stats
    before_filter_rows = len(df)
    dropped_obvious_rows = reasons.count("obvious_noise")
    dropped_model_rows = reasons.count("model_drop")
    
    # Correction stats (approximate, row-based)
    corrections_rows = sum(1 for orig, new in zip(df[word_col], corrected_vals) if orig != new)
    lemma_rows_filled = sum(1 for x in lemma_vals if not pd.isna(x))

    # --- Final Filtering ---
    df = df[df["__keep"]].drop(columns=["__keep"]).copy()
    final_rows = len(df)
    final_unique = df[word_col].nunique(dropna=True)

    # --- Save
    df.to_excel(out_path, index=False)

    summary = {
        "input_path": str(in_path),
        "output_path": str(out_path),
        "token_column": str(word_col),
        "rows": {
            "raw_rows": raw_rows,
            "blank_dropped_rows": blank_dropped,
            "rows_before_keep_filter": before_filter_rows,
            "dropped_obvious_noise_rows": dropped_obvious_rows,
            "dropped_model_rows": dropped_model_rows,
            "final_rows": final_rows,
        },
        "tokens": {
            "unique_before": original_unique,
            "unique_obvious_noise": obvious_noise_unique,
            "unique_sent_to_api": len(tokens_for_api),
            "unique_after": int(final_unique),
            "corrected_rows": corrections_rows,
            "lemma_filled_rows": lemma_rows_filled,
        },
        "model": {"name": model, "batch_size": batch_size},
    }

    print("\n[word_corpus_cleaner] Summary")
    print(f"  Token column detected        : {summary['token_column']}")
    print(f"  Rows (raw → final)           : {raw_rows:,} → {final_rows:,}")
    if blank_dropped:
        print(f"  Blank-token rows dropped     : {blank_dropped:,}")
    print(f"  Obvious-noise rows dropped   : {dropped_obvious_rows:,}")
    print(f"  Model-based rows dropped     : {dropped_model_rows:,}")
    print(f"  Unique tokens (before → after): {original_unique:,} → {final_unique:,}")
    print(f"  Unique tokens sent to API    : {len(tokens_for_api):,}")
    print(f"  Row-level token corrections  : {corrections_rows:,}")
    print(f"  Rows with lemma filled       : {lemma_rows_filled:,}")
    print(f"  Saved cleaned workbook       : {out_path}")

    return summary


# ============================================================
# 5) CLI entry point
# ============================================================
def main() -> None:
    args = parse_args()
    in_path = resolve_input_path(args.word_corpus_xlsx)
    clean_word_corpus(
        in_path,
        model=args.model,
        batch_size=args.batch_size,
        load_env=True,
        client=None,
    )


if __name__ == "__main__":
    main()