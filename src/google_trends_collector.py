"""
Google Trends Collector for ElderDemand AI
Fetches demand signals for elder care search terms across India
"""

import pandas as pd
import time
import os

# ── Compatibility patch: urllib3 2.x removed `method_whitelist` ──────────────
# pytrends 4.9.2 still uses the old kwarg; patch Retry to accept it as alias.
import urllib3.util.retry as _retry_mod
_OrigRetry = _retry_mod.Retry
class _PatchedRetry(_OrigRetry):
    def __init__(self, *args, method_whitelist=None, **kwargs):
        if method_whitelist is not None and "allowed_methods" not in kwargs:
            kwargs["allowed_methods"] = method_whitelist
        super().__init__(*args, **kwargs)
_retry_mod.Retry = _PatchedRetry
# ─────────────────────────────────────────────────────────────────────────────

from pytrends.request import TrendReq

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

SEARCH_TERMS = [
    "home nurse india",
    "elder care india",
    "old age care bangalore",
    "caregiver india",
    "senior care service india",
]

# Indian states/regions with their geo codes (ISO 3166-2)
INDIAN_STATES = {
    "Maharashtra":      "IN-MH",
    "Karnataka":        "IN-KA",
    "Tamil Nadu":       "IN-TN",
    "Delhi":            "IN-DL",
    "West Bengal":      "IN-WB",
    "Gujarat":          "IN-GJ",
    "Rajasthan":        "IN-RJ",
    "Uttar Pradesh":    "IN-UP",
    "Kerala":           "IN-KL",
    "Andhra Pradesh":   "IN-AP",
    "Telangana":        "IN-TG",
    "Madhya Pradesh":   "IN-MP",
    "Punjab":           "IN-PB",
    "Haryana":          "IN-HR",
    "Bihar":            "IN-BR",
}


def build_pytrends():
    """Initialize pytrends with retries."""
    return TrendReq(hl='en-US', tz=330, timeout=(10, 25), retries=3, backoff_factor=0.5)


def fetch_interest_over_time(terms: list, timeframe: str = "2019-01-01 2024-01-01") -> pd.DataFrame:
    """Fetch interest over time for given search terms (India-wide)."""
    pytrends = build_pytrends()
    # pytrends takes max 5 terms per request
    all_dfs = []
    for i in range(0, len(terms), 5):
        batch = terms[i:i+5]
        print(f"  Fetching interest over time for: {batch}")
        try:
            pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo='IN', gprop='')
            df = pytrends.interest_over_time()
            if df.empty:
                print("    [WARN] Empty result for this batch")
                continue
            if 'isPartial' in df.columns:
                df = df.drop(columns=['isPartial'])
            all_dfs.append(df)
        except Exception as e:
            print(f"    [ERROR] {e}")
        time.sleep(5)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, axis=1)


def fetch_interest_by_region(term: str) -> pd.DataFrame:
    """Fetch regional interest (by state) for a single search term."""
    pytrends = build_pytrends()
    print(f"  Fetching regional interest for: '{term}'")
    try:
        pytrends.build_payload([term], cat=0, timeframe='today 5-y', geo='IN', gprop='')
        df = pytrends.interest_by_region(resolution='REGION', inc_low_vol=True, inc_geo_code=True)
        df = df.reset_index()
        df['search_term'] = term
        return df
    except Exception as e:
        print(f"    [ERROR] {e}")
        return pd.DataFrame()


def fetch_related_queries(term: str) -> dict:
    """Fetch related queries for a search term."""
    pytrends = build_pytrends()
    print(f"  Fetching related queries for: '{term}'")
    try:
        pytrends.build_payload([term], cat=0, timeframe='today 5-y', geo='IN', gprop='')
        related = pytrends.related_queries()
        return related.get(term, {})
    except Exception as e:
        print(f"    [ERROR] {e}")
        return {}


def collect_google_trends() -> dict:
    """Main function: collect all Google Trends data."""
    results = {}
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # 1. Interest over time (all terms, India-wide)
    print("\n[1/3] Fetching interest over time (India-wide)...")
    iot_df = fetch_interest_over_time(SEARCH_TERMS)
    if not iot_df.empty:
        path = os.path.join(RAW_DATA_DIR, "trends_interest_over_time.csv")
        iot_df.to_csv(path)
        print(f"  Saved: {path} — shape {iot_df.shape}")
        results['interest_over_time'] = iot_df
    else:
        print("  [WARN] No interest-over-time data collected")

    time.sleep(5)

    # 2. Interest by region (each term separately)
    print("\n[2/3] Fetching regional breakdown by state...")
    regional_frames = []
    for term in SEARCH_TERMS:
        df = fetch_interest_by_region(term)
        if not df.empty:
            regional_frames.append(df)
        time.sleep(5)

    if regional_frames:
        regional_df = pd.concat(regional_frames, ignore_index=True)
        path = os.path.join(RAW_DATA_DIR, "trends_interest_by_region.csv")
        regional_df.to_csv(path, index=False)
        print(f"  Saved: {path} — shape {regional_df.shape}")
        results['interest_by_region'] = regional_df
    else:
        print("  [WARN] No regional data collected")

    time.sleep(5)

    # 3. Related queries (top 2 terms to save API calls)
    print("\n[3/3] Fetching related queries...")
    related_rows = []
    for term in SEARCH_TERMS[:3]:
        queries = fetch_related_queries(term)
        for query_type in ['top', 'rising']:
            qdf = queries.get(query_type)
            if qdf is not None and not qdf.empty:
                qdf['search_term'] = term
                qdf['query_type'] = query_type
                related_rows.append(qdf)
        time.sleep(5)

    if related_rows:
        related_df = pd.concat(related_rows, ignore_index=True)
        path = os.path.join(RAW_DATA_DIR, "trends_related_queries.csv")
        related_df.to_csv(path, index=False)
        print(f"  Saved: {path} — shape {related_df.shape}")
        results['related_queries'] = related_df

    return results


if __name__ == "__main__":
    print("Google Trends Collector — ElderDemand AI")
    print("=" * 50)
    results = collect_google_trends()
    print(f"\nCollected {len(results)} datasets from Google Trends.")
    print("Done.")
