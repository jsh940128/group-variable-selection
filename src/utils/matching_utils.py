import numpy as np
import pandas as pd

def calculate_business_distance(
    row_treated: pd.Series, 
    row_candidate: pd.Series, 
    w_rating: float = 1.0, 
    w_reviews: float = 1.0
) -> float:
    """
    Calculates the 'Business Distance' between a treated hotel and a candidate control.
    LOWER distance = BETTER match.
    """
    # 1. Rating Difference (0-5 scale)
    r_t = float(row_treated.get("rating_mean", 0) or 0)
    r_c = float(row_candidate.get("rating_mean", 0) or 0)
    diff_rating = abs(r_t - r_c)

    # 2. Review Count Magnitude Difference (Log Scale)
    # We use log1p to handle 0 reviews and compress the scale (10 vs 100 >> 10000 vs 10100)
    # NOTE: Using 'rating_count' as per your data schema
    rev_t = float(row_treated.get("rating_count", 0) or 0)
    rev_c = float(row_candidate.get("rating_count", 0) or 0)
    
    log_rev_t = np.log1p(rev_t)
    log_rev_c = np.log1p(rev_c)
    diff_reviews = abs(log_rev_t - log_rev_c)

    # 3. Euclidean Distance
    distance = np.sqrt(
        (w_rating * diff_rating) ** 2 + 
        (w_reviews * diff_reviews) ** 2
    )
    
    return distance

def score_all_candidates(
    treated_row: pd.Series, 
    candidates_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates match distance for ALL candidates.
    Does NOT filter or select; just scores.
    
    Returns:
        pd.DataFrame: The candidates DataFrame with a new 'match_distance' column.
    """
    if candidates_df.empty:
        return candidates_df

    # Calculate distance for every candidate
    distances = []
    for idx, candidate in candidates_df.iterrows():
        dist = calculate_business_distance(treated_row, candidate)
        distances.append(dist)
    
    # Return copy with scores
    scored = candidates_df.copy()
    scored["match_distance"] = distances
    
    return scored