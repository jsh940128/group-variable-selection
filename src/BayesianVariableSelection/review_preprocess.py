import re
import pandas as pd

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import pandas as pd

_NUM_RE = re.compile(r"\d+")

def _strip_numbers(s: pd.Series) -> pd.Series:
    """
    Remove digits from text (e.g., '5-star' -> '-star').
    Keeps punctuation; you can optionally normalize whitespace downstream.
    """
    return s.fillna("").astype(str).str.replace(_NUM_RE, "", regex=True)

"""Utilities for turning raw review rows into (aggregate_id, review_text) for UVS.

Design goals:
- Minimal dependencies (pandas only)
- Deterministic, auditable text construction
- Easy to extend with new tags (seasonality, platform, expert events, etc.)
"""

@dataclass(frozen=True)
class TimeBinningConfig:
    """Configuration for converting a date column into a time bin string."""
    date_col: str = "createdDate"
    freq: str = "M"  # "M"=month, "Q"=quarter
    tz: Optional[str] = None  # e.g., "UTC" if your timestamps include timezones


def _to_datetime(s: pd.Series, tz: Optional[str] = None) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if tz is not None:
        # If dt is tz-naive, localize; if tz-aware, convert.
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(tz)
        else:
            dt = dt.dt.tz_convert(tz)
    return dt


def add_time_bin(df: pd.DataFrame, config: TimeBinningConfig, out_col: str = "time_bin") -> pd.DataFrame:
    dt = _to_datetime(df[config.date_col], tz=config.tz)
    if config.freq.upper() == "M":
        df[out_col] = dt.dt.to_period("M").astype(str)
    elif config.freq.upper() == "Q":
        df[out_col] = dt.dt.to_period("Q").astype(str)
    else:
        raise ValueError("Unsupported freq. Use 'M' or 'Q'.")
    return df


def add_aggregate_id(
    df: pd.DataFrame,
    location_col: str = "locationId",
    time_bin_col: str = "time_bin",
    out_col: str = "agg_id",
    sep: str = "__",
) -> pd.DataFrame:
    df[out_col] = df[location_col].astype(str) + sep + df[time_bin_col].astype(str)
    return df



def build_review_text(
    df: pd.DataFrame,
    title_col: str = "title",
    text_col: str = "text",
    *,
    tag_cols: Optional[Mapping[str, str]] = None,
    include_tags: bool = True,
    out_col: str = "review_text",
) -> pd.DataFrame:
    """
    Fast, vectorized construction:
      review_text = "[tag=val] [tag=val] " + title + ". " + text
    """
    out = df  # IMPORTANT: avoid copying; caller decides if it needs a copy

    title = out[title_col].fillna("").astype(str).str.strip()
    body  = out[text_col].fillna("").astype(str).str.strip()
    base  = (title + ". " + body).str.strip()

    if include_tags and tag_cols:
        # Build tags using vectorized ops and DataFrame aggregation (no Python zip/list)
        tag_parts = []
        for tag_name, col in tag_cols.items():
            if col not in out.columns:
                continue
            val = out[col].fillna("").astype(str).str.strip()
            tag_parts.append("[" + tag_name + "=" + val + "]")

        if tag_parts:
            # Combine tags per row efficiently
            tag_df = pd.concat(tag_parts, axis=1)
            tag_prefix = tag_df.agg(" ".join, axis=1).str.strip()
            base = (tag_prefix + " " + base).str.strip()

    out[out_col] = base
    return out


def minimal_uves_input(
    df: pd.DataFrame,
    *,
    location_col: str = "locationId",
    rating_col: str = "rating",
    title_col: str = "title",
    text_col: str = "text",
    time_config: TimeBinningConfig = TimeBinningConfig(),
    include_tags: bool = True,
) -> pd.DataFrame:
    """Return a minimally clean dataframe for UVS: agg_id, rating, review_text.

    - Drops rows with missing locationId, rating, or review_text after construction
    - Creates month bin from config.date_col
    """
    out = df.copy()  # one copy only

    out = add_time_bin(out, time_config, out_col="time_bin")
    out = add_aggregate_id(out, location_col=location_col, time_bin_col="time_bin", out_col="agg_id")

    tag_cols = {
        "platform": "publishPlatform",
        "language": "language",
        "tripType": "tripType",
        "stayMonth": "stayMonth",
    }

    out = build_review_text(
        out,
        title_col=title_col,
        text_col=text_col,
        tag_cols=tag_cols,
        include_tags=include_tags,
        out_col="review_text",
    )

    keep = ["agg_id", rating_col, "review_text", location_col, "time_bin"]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()

    out = out.dropna(subset=["agg_id", rating_col, "review_text"])
    out = out[out["review_text"].str.len() > 0]
    return out

def minimal_uves_input_quarterly(
    df: pd.DataFrame,
    *,
    location_col: str = "locationId",
    rating_col: str = "rating",
    title_col: str = "title",
    text_col: str = "text",
    date_col: str = "createdDate",
    tz: Optional[str] = None,
    include_tags: bool = True,
) -> pd.DataFrame:
    """
    Quarterly version of minimal_uves_input().

    - time_bin will look like '2025Q2'
    - agg_id will look like '23688737__2025Q2'

    This is a thin wrapper that sets TimeBinningConfig(freq='Q').
    """
    q_cfg = TimeBinningConfig(date_col=date_col, freq="Q", tz=tz)
    return minimal_uves_input(
        df,
        location_col=location_col,
        rating_col=rating_col,
        title_col=title_col,
        text_col=text_col,
        time_config=q_cfg,
        include_tags=include_tags,
    )
