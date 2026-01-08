# pip install -U openai python-dotenv pandas openpyxl
import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

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
        raise SystemExit(
            "ERROR: Provide --word-corpus-xlsx PATH or set env var WORD_CORPUS_XLSX."
        )
    p = Path(raw).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"ERROR: File not found: {p}")
    if p.suffix.lower() not in {".xlsx", ".xls"}:
        raise SystemExit(f"ERROR: Expected an Excel file (.xlsx/.xls). Got: {p.name}")
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

For each input token, output a JSON object with:
- word: original string (exactly)
- keep: true/false (remove if inappropriate for modeling)
- corrected: corrected/normalized token (only if needed; else return the original)
- lemma: "" OR a base/root form WHEN morphological variants would otherwise
         split signal across related tokens.

IMPORTANT LEMMA RULE:
- If an adjective/adverb/verb is clearly derived from a common noun or verb root,
  provide the root as lemma.
  Examples:
    comfortable → comfort
    comforting → comfort
    cleanliness → clean
    cleaner → clean
    cleanest → clean
- Do NOT invent lemmas for proper nouns, hotel names, locations, or brands.
- If the word is already the root form, leave lemma empty.

Keep criteria:
- Content words carrying sentiment, quality, or topic signal.
- Domain-relevant terms in hospitality reviews.

Remove criteria:
- Stopwords, filler words, artifacts, punctuation fragments,
  encoding noise, URLs/emails/IDs, meaningless tokens.

Return STRICT JSONL (one JSON object per line).
No markdown. No commentary.
""".strip()


def call_review_batch(client: OpenAI, model: str, batch_words: list[str]) -> list[dict]:
    payload = {"words": batch_words}
    resp = client.responses.create(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=json.dumps(payload, ensure_ascii=False),
    )
    text = (resp.output_text or "").strip()

    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            # Ignore malformed lines, keep going
            continue
    return rows


# ============================================================
# 4) Public method: clean_word_corpus(...)
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

    Prints:
      - input path
      - output path
      - summary of changes

    Returns a summary dict for programmatic use.
    """
    if load_env:
        load_dotenv()

    if client is None:
        client = OpenAI()

    in_path = Path(word_corpus_xlsx).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"File not found: {in_path}")
    if in_path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(f"Expected an Excel file (.xlsx/.xls). Got: {in_path.name}")

    out_path = in_path.with_name(in_path.stem + "_cleaned.xlsx")

    print(f"[word_corpus_cleaner] Input file : {in_path}")
    print(f"[word_corpus_cleaner] Output file: {out_path}")

    # --- Load workbook
    df = pd.read_excel(in_path)

    # --- Detect token column
    word_col = detect_word_col(df.columns)

    # --- Ensure lemma column exists (NO other new columns)
    if "lemma" not in df.columns:
        df["lemma"] = pd.NA

    # place lemma immediately to the right of the token column
    cols = [c for c in df.columns if c != "lemma"]
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
    malformed_jsonl_lines = 0

    for i in range(num_batches):
        batch = tokens_for_api[i * batch_size : (i + 1) * batch_size]
        result_rows = call_review_batch(client=client, model=model, batch_words=batch)

        # Track coverage: if model returns fewer rows than inputs, that's a signal of formatting loss
        # (not fatal; we fallback to keep-as-is)
        returned_words = set()
        for r in result_rows:
            w = str(r.get("word", "")).strip()
            if not w:
                malformed_jsonl_lines += 1
                continue
            returned_words.add(w)
            review_map[w] = {
                "keep": bool(r.get("keep", True)),
                "corrected": (r.get("corrected") or w).strip(),
                "lemma": (r.get("lemma") or "").strip(),
            }

    # --- Apply decisions + compute summary stats
    def apply_decision(w: str):
        # drop obvious noise
        if w in obvious_noise_set:
            return False, w, "", "obvious_noise"

        ann = review_map.get(w)
        if ann is None:
            # fallback: keep as-is, no lemma
            return True, w, "", "fallback_keep"

        keep = ann["keep"]
        corrected = ann["corrected"] or w
        lemma = ann["lemma"] or ""
        reason = "model_keep" if keep else "model_drop"
        return keep, corrected, lemma, reason

    decisions = df[word_col].apply(apply_decision)

    df["__keep"] = decisions.map(lambda t: t[0])
    corrected_series = decisions.map(lambda t: t[1])
    lemma_series = decisions.map(lambda t: t[2])
    reason_series = decisions.map(lambda t: t[3])

    # token-level stats (unique)
    # Count how many tokens changed after correction
    # (based on per-row mapping; approximate but useful)
    corrections_rows = int((df[word_col] != corrected_series).sum())
    lemma_rows_filled = int((lemma_series != "").sum())

    # Apply corrections
    df[word_col] = corrected_series
    df["lemma"] = lemma_series.map(lambda x: x if x else pd.NA)

    before_filter_rows = len(df)
    dropped_obvious_rows = int((reason_series == "obvious_noise").sum())
    dropped_model_rows = int((reason_series == "model_drop").sum())

    df = df[df["__keep"]].drop(columns=["__keep"]).copy()
    final_rows = len(df)

    # Unique tokens after correction (post-drop)
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
        "notes": {
            "malformed_jsonl_lines_seen": malformed_jsonl_lines,
            "fallback_behavior": "If a token is missing from model output, it is kept as-is with empty lemma.",
        },
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
    if malformed_jsonl_lines:
        print(f"  Warning: malformed JSONL lines ignored: {malformed_jsonl_lines:,}")
    print(f"  Saved cleaned workbook       : {out_path}")

    return summary


# ============================================================
# 5) CLI entry point (still works)
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
