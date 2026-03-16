"""
World Bank Data Collector for ElderDemand AI
Fetches India demographics from World Bank API
"""

import requests
import pandas as pd
import json
import time
import os

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')


def fetch_indicator(indicator: str, country: str = "IN", start: int = 2000, end: int = 2023) -> list:
    """Fetch a single indicator from World Bank API."""
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    params = {
        "format": "json",
        "date": f"{start}:{end}",
        "per_page": 100,
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if len(data) < 2:
            print(f"  [WARN] No data returned for {indicator}")
            return []
        records = data[1]
        return records or []
    except Exception as e:
        print(f"  [ERROR] Failed to fetch {indicator}: {e}")
        return []


def parse_records(records: list, value_col: str) -> pd.DataFrame:
    """Parse World Bank API records into a DataFrame."""
    rows = []
    for rec in records:
        if rec.get("value") is not None:
            rows.append({
                "year": int(rec["date"]),
                "country": rec["country"]["value"],
                "country_code": rec["countryiso3code"],
                value_col: float(rec["value"]),
            })
    return pd.DataFrame(rows).sort_values("year")


INDICATORS = {
    "SP.POP.65UP.TO.ZS": "pct_pop_65plus",           # Population 65+ (% of total)
    "SP.POP.65UP.TO":    "count_pop_65plus",           # Population 65+ (absolute)
    "SP.URB.TOTL.IN.ZS": "pct_urban_population",      # Urban population (%)
    "NY.GDP.PCAP.CD":    "gdp_per_capita_usd",         # GDP per capita (current USD)
    "SP.DYN.LE00.IN":    "life_expectancy_at_birth",   # Life expectancy (years)
    "NY.GNP.PCAP.CD":    "gni_per_capita_usd",         # GNI per capita (Atlas method)
    "SP.POP.TOTL":       "total_population",            # Total population
    "SH.XPD.CHEX.PC.CD": "health_expenditure_per_cap", # Health expenditure per capita (USD)
    "SP.DYN.CDRT.IN":   "crude_death_rate",            # Crude death rate
}


def collect_world_bank_data() -> pd.DataFrame:
    """Fetch all indicators and merge into a single DataFrame."""
    print("Fetching World Bank data for India...")
    merged = None

    for indicator_code, col_name in INDICATORS.items():
        print(f"  Fetching: {indicator_code} → {col_name}")
        records = fetch_indicator(indicator_code)
        df = parse_records(records, col_name)

        if df.empty:
            print(f"    [SKIP] Empty result")
            continue

        if merged is None:
            merged = df[["year", "country", "country_code", col_name]]
        else:
            merged = merged.merge(
                df[["year", col_name]], on="year", how="outer"
            )

        time.sleep(0.3)  # polite rate limiting

    if merged is None:
        print("[ERROR] No data fetched from World Bank.")
        return pd.DataFrame()

    merged = merged.sort_values("year").reset_index(drop=True)
    print(f"\nWorld Bank data shape: {merged.shape}")
    print(merged.tail(5).to_string(index=False))
    return merged


def save_world_bank_data(df: pd.DataFrame):
    """Save World Bank data to CSV and JSON."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    csv_path = os.path.join(RAW_DATA_DIR, "world_bank_india_demographics.csv")
    json_path = os.path.join(RAW_DATA_DIR, "world_bank_india_demographics.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    df = collect_world_bank_data()
    if not df.empty:
        save_world_bank_data(df)
        print("\nWorld Bank collection complete.")
    else:
        print("\n[FAILED] No World Bank data collected.")
