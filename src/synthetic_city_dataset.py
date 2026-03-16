"""
Synthetic City Dataset Generator for ElderDemand AI
Creates realistic data for 50 major Indian cities using census-anchored estimates
"""

import pandas as pd
import numpy as np
import os
import json

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

# Seed for reproducibility
np.random.seed(42)

# ─────────────────────────────────────────────────────────
# TIER 1: Anchor cities with real/well-documented estimates
# Sources: Census 2011, NSSO, RBI Household Finance, NITI Aayog
# ─────────────────────────────────────────────────────────
ANCHOR_CITIES = [
    # name, state, lat, lon, total_pop_2023_M, pct_65plus, pct_nuclear, median_hh_income_INR, tier
    ("Mumbai",          "Maharashtra",   19.076,  72.877, 21.0,  7.2,  68, 540000,  1),
    ("Delhi",           "Delhi",         28.704,  77.102, 32.0,  6.5,  72, 480000,  1),
    ("Bangalore",       "Karnataka",     12.972,  77.594, 13.0,  6.1,  74, 620000,  1),
    ("Hyderabad",       "Telangana",     17.385,  78.486, 10.5,  6.8,  67, 510000,  1),
    ("Chennai",         "Tamil Nadu",    13.083,  80.270, 11.0,  8.1,  65, 490000,  1),
    ("Kolkata",         "West Bengal",   22.572,  88.364, 15.0,  9.2,  58, 360000,  1),
    ("Pune",            "Maharashtra",   18.520,  73.856,  7.4,  6.9,  71, 550000,  1),
    ("Ahmedabad",       "Gujarat",       23.023,  72.572,  8.0,  7.5,  62, 420000,  1),
    ("Surat",           "Gujarat",       21.170,  72.831,  7.0,  5.8,  64, 390000,  2),
    ("Jaipur",          "Rajasthan",     26.912,  75.787,  4.0,  7.8,  60, 310000,  2),
    ("Lucknow",         "Uttar Pradesh", 26.847,  80.947,  3.8,  7.4,  59, 280000,  2),
    ("Kanpur",          "Uttar Pradesh", 26.449,  80.331,  3.2,  7.1,  57, 250000,  2),
    ("Nagpur",          "Maharashtra",   21.145,  79.089,  2.9,  7.7,  63, 340000,  2),
    ("Indore",          "Madhya Pradesh",22.718,  75.857,  2.8,  6.9,  65, 330000,  2),
    ("Thane",           "Maharashtra",   19.218,  72.978,  2.6,  6.3,  75, 480000,  2),
    ("Bhopal",          "Madhya Pradesh",23.259,  77.413,  2.5,  7.2,  61, 300000,  2),
    ("Visakhapatnam",   "Andhra Pradesh",17.687,  83.218,  2.3,  8.0,  62, 320000,  2),
    ("Patna",           "Bihar",         25.594,  85.138,  2.1,  7.6,  54, 220000,  2),
    ("Vadodara",        "Gujarat",       22.307,  73.181,  2.2,  7.3,  63, 380000,  2),
    ("Ghaziabad",       "Uttar Pradesh", 28.669,  77.438,  2.3,  5.9,  73, 400000,  2),
    ("Ludhiana",        "Punjab",        30.901,  75.857,  1.9,  8.5,  60, 430000,  2),
    ("Agra",            "Uttar Pradesh", 27.176,  78.008,  1.8,  7.9,  58, 260000,  3),
    ("Nashik",          "Maharashtra",   19.998,  73.790,  1.7,  7.0,  66, 350000,  3),
    ("Faridabad",       "Haryana",       28.408,  77.313,  1.6,  6.2,  70, 420000,  3),
    ("Meerut",          "Uttar Pradesh", 28.984,  77.706,  1.6,  7.1,  57, 280000,  3),
    ("Rajkot",          "Gujarat",       22.308,  70.800,  1.5,  7.8,  62, 340000,  3),
    ("Varanasi",        "Uttar Pradesh", 25.316,  82.974,  1.5,  8.7,  53, 240000,  3),
    ("Srinagar",        "J&K",           34.084,  74.797,  1.2,  6.5,  55, 250000,  3),
    ("Aurangabad",      "Maharashtra",   19.877,  75.343,  1.4,  6.8,  62, 310000,  3),
    ("Dhanbad",         "Jharkhand",     23.795,  86.429,  1.3,  7.4,  53, 230000,  3),
    ("Amritsar",        "Punjab",        31.634,  74.872,  1.2,  8.9,  58, 370000,  3),
    ("Allahabad",       "Uttar Pradesh", 25.436,  81.846,  1.3,  7.8,  55, 245000,  3),
    ("Ranchi",          "Jharkhand",     23.344,  85.309,  1.2,  6.9,  58, 260000,  3),
    ("Howrah",          "West Bengal",   22.588,  88.310,  1.1,  9.4,  56, 310000,  3),
    ("Coimbatore",      "Tamil Nadu",    11.017,  76.956,  1.1,  9.0,  64, 390000,  3),
    ("Vijayawada",      "Andhra Pradesh",16.519,  80.619,  1.1,  8.2,  62, 300000,  3),
    ("Jodhpur",         "Rajasthan",     26.294,  73.040,  1.0,  7.6,  59, 280000,  3),
    ("Madurai",         "Tamil Nadu",     9.919,  78.119,  1.0,  9.5,  62, 330000,  3),
    ("Raipur",          "Chhattisgarh",  21.250,  81.630,  1.0,  6.7,  60, 270000,  3),
    ("Kota",            "Rajasthan",     25.147,  75.849,  0.9,  7.3,  61, 290000,  3),
    ("Guwahati",        "Assam",         26.145,  91.736,  0.9,  7.1,  60, 260000,  3),
    ("Chandigarh",      "Chandigarh",    30.734,  76.779,  1.1,  8.1,  68, 560000,  2),
    ("Thiruvananthapuram","Kerala",       8.524,  76.936,  0.9, 11.2,  68, 400000,  3),
    ("Kochi",           "Kerala",        10.009,  76.261,  0.8, 10.8,  69, 450000,  3),
    ("Bhubaneswar",     "Odisha",        20.296,  85.824,  0.9,  7.4,  61, 270000,  3),
    ("Salem",           "Tamil Nadu",    11.665,  78.146,  0.8,  9.1,  61, 310000,  3),
    ("Mysuru",          "Karnataka",     12.296,  76.639,  1.0,  9.8,  65, 350000,  3),
    ("Jabalpur",        "Madhya Pradesh",23.182,  79.987,  1.1,  7.3,  58, 265000,  3),
    ("Tiruchirappalli", "Tamil Nadu",    10.795,  78.687,  0.9,  9.3,  62, 300000,  3),
    ("Bareilly",        "Uttar Pradesh", 28.367,  79.415,  1.0,  7.0,  55, 235000,  3),
]


def compute_healthcare_density(tier: int, income: float) -> float:
    """
    Healthcare facility density per 10,000 seniors.
    Higher in tier-1 cities and higher-income cities.
    """
    base = {1: 12.0, 2: 7.5, 3: 4.2}[tier]
    income_factor = 1 + (income - 300000) / 1_000_000
    noise = np.random.normal(0, 0.5)
    return round(max(1.5, base * income_factor + noise), 2)


def compute_caregiver_index(tier: int, pct_nuclear: float) -> float:
    """
    Caregiver availability index (0–10).
    Lower when nuclear family % is high (fewer informal caregivers available).
    """
    base = {1: 5.5, 2: 6.2, 3: 7.0}[tier]
    nuclear_penalty = (pct_nuclear - 60) * 0.04
    noise = np.random.normal(0, 0.3)
    return round(max(1.0, min(10.0, base - nuclear_penalty + noise)), 2)


def compute_willingness_to_pay(income: float, pct_nuclear: float, tier: int) -> dict:
    """
    Estimated monthly WTP (INR) for elder care services.
    Based on: income level, nuclear family structure, urban tier.
    """
    # Affordability ceiling: ~8–12% of household income
    base_wtp = income * np.random.uniform(0.08, 0.12)
    # Nuclear families pay more (no informal support)
    nuclear_premium = (pct_nuclear - 60) * 150
    # Tier premium (tier 1 cities expect premium services)
    tier_premium = {1: 3000, 2: 1000, 3: 0}[tier]
    total = base_wtp + nuclear_premium + tier_premium
    noise = np.random.normal(0, 500)
    monthly_wtp = max(2000, round(total + noise, -2))
    return {
        "wtp_monthly_basic_care_INR":    int(monthly_wtp * 0.6),   # basic in-home help
        "wtp_monthly_nurse_visit_INR":   int(monthly_wtp * 1.0),   # skilled nurse visits
        "wtp_monthly_full_time_INR":     int(monthly_wtp * 2.2),   # full-time live-in
        "wtp_monthly_daycare_INR":       int(monthly_wtp * 0.75),  # adult daycare
    }


def compute_market_size(senior_pop: int, wtp: dict, pct_nuclear: float) -> dict:
    """
    Estimated addressable market per city.
    Assumes 15–25% of high-nuclear families are potential customers.
    """
    addressable_fraction = np.random.uniform(0.12, 0.22)
    potential_customers = int(senior_pop * (pct_nuclear / 100) * addressable_fraction)
    annual_revenue_potential_CR = round(
        potential_customers * wtp["wtp_monthly_nurse_visit_INR"] * 12 / 1e7, 2
    )
    return {
        "potential_customer_households": potential_customers,
        "annual_revenue_potential_CR_INR": annual_revenue_potential_CR,
    }


def generate_city_dataset() -> pd.DataFrame:
    """Generate the full 50-city dataset."""
    rows = []
    for record in ANCHOR_CITIES:
        name, state, lat, lon, total_pop_M, pct_65, pct_nuclear, hh_income, tier = record

        total_pop = int(total_pop_M * 1_000_000)
        senior_pop = int(total_pop * pct_65 / 100)

        # Add realistic noise to anchored values
        pct_65_final     = round(pct_65 + np.random.normal(0, 0.2), 2)
        pct_nuclear_final = round(min(85, max(45, pct_nuclear + np.random.normal(0, 2))), 1)
        hh_income_final  = int(hh_income * np.random.uniform(0.95, 1.05))

        hc_density  = compute_healthcare_density(tier, hh_income_final)
        cg_index    = compute_caregiver_index(tier, pct_nuclear_final)
        wtp         = compute_willingness_to_pay(hh_income_final, pct_nuclear_final, tier)
        market      = compute_market_size(senior_pop, wtp, pct_nuclear_final)

        # Demand score (composite 0–100)
        demand_score = round(
            (pct_nuclear_final / 85 * 30) +
            (hh_income_final / 620000 * 30) +
            ((10 - cg_index) / 9 * 20) +
            (pct_65_final / 12 * 20),
            1
        )

        row = {
            "city":                         name,
            "state":                        state,
            "city_tier":                    tier,
            "latitude":                     lat,
            "longitude":                    lon,
            "total_population_2023":        total_pop,
            "senior_population_65plus":     senior_pop,
            "pct_population_65plus":        pct_65_final,
            "pct_nuclear_families":         pct_nuclear_final,
            "median_hh_income_INR":         hh_income_final,
            "healthcare_facility_density":  hc_density,
            "caregiver_availability_index": cg_index,
            "demand_score_0_100":           demand_score,
            **wtp,
            **market,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("annual_revenue_potential_CR_INR", ascending=False).reset_index(drop=True)
    df.insert(0, "rank_by_market_size", range(1, len(df) + 1))
    return df


def save_city_dataset(df: pd.DataFrame):
    """Save to CSV and JSON."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    csv_path = os.path.join(RAW_DATA_DIR, "india_50cities_eldercare.csv")
    json_path = os.path.join(RAW_DATA_DIR, "india_50cities_eldercare.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")

    # Print summary
    print(f"\nTop 10 Cities by Market Potential:")
    cols = ["rank_by_market_size", "city", "state", "senior_population_65plus",
            "demand_score_0_100", "annual_revenue_potential_CR_INR"]
    print(df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    print("Synthetic City Dataset Generator — ElderDemand AI")
    print("=" * 55)
    df = generate_city_dataset()
    save_city_dataset(df)
    print(f"\nGenerated {len(df)} cities. Done.")
