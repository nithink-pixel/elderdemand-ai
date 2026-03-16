# ElderDemand AI
### Senior Care Market Intelligence Platform — India

> **A data science portfolio project and startup feasibility study.**
> Uses real demographic data, live search trends, and ML-driven demand scoring to identify the highest-opportunity cities for launching elder care services in India.

---

## The Problem

India is entering a demographic inflection point. With **99.5 million seniors** (65+) today — growing at **3.8% CAGR** — and a simultaneous collapse of joint family structures (urban nuclear family rate: 68%), the supply of informal home-based care is evaporating. Professional elder care services are a ₹24.8 lakh crore opportunity that is structurally underserved.

---

## Key Findings

| Metric | Value |
|---|---|
| Senior Population (2023) | **99.5 Million** |
| Senior Population (2028 forecast) | **119 Million** |
| TAM — All seniors needing care | **₹24,84,541 Crore** |
| SAM — Urban nuclear-family seniors | **₹14,15,816 Crore** |
| SOM — 5-year target (3% SAM) | **₹42,474 Crore** |
| Cities analysed | **50 major Indian cities** |
| GO cities (launch-ready) | **11** |
| Conditional cities | 32 |
| #1 Launch City | **Delhi** (demand score 70.6/100) |
| Top 1 city revenue potential | **₹12,886 Crore/year** |
| Top 5 cities combined | **₹43,282 Crore/year** |

### Top 10 Launch Cities by Revenue Potential

| Rank | City | State | Senior Pop | Revenue Potential |
|---|---|---|---|---|
| 1 | Delhi | Delhi | 20,80,000 | ₹12,886 Cr |
| 2 | Bangalore | Karnataka | 7,93,000 | ₹9,276 Cr |
| 3 | Mumbai | Maharashtra | 15,12,000 | ₹7,886 Cr |
| 4 | Chennai | Tamil Nadu | 8,91,000 | ₹6,261 Cr |
| 5 | Kolkata | West Bengal | 13,80,000 | ₹5,478 Cr |
| 6 | Hyderabad | Telangana | 7,14,000 | ₹5,373 Cr |
| 7 | Ahmedabad | Gujarat | 6,00,000 | ₹4,356 Cr |
| 8 | Pune | Maharashtra | 5,10,600 | ₹3,504 Cr |
| 9 | Surat | Gujarat | 4,06,000 | ₹1,713 Cr |
| 10 | Jaipur | Rajasthan | 3,12,000 | ₹1,264 Cr |

---

## Data Sources

| Source | Data | Method |
|---|---|---|
| **World Bank API** | India demographics 2000–2023 (9 indicators) | Live REST API |
| **Google Trends** | Search demand for 5 elder care queries across all Indian states | pytrends |
| **Synthetic Dataset** | 50 cities × 20 features anchored to Census 2011, NSSO, RBI data | Stochastic simulation |

**World Bank indicators fetched:** Senior population (%), senior population count, urban %, GDP per capita, life expectancy, GNI per capita, total population, health expenditure, crude death rate.

**Google Trends queries:** `home nurse india`, `elder care india`, `old age care bangalore`, `caregiver india`, `senior care service india`.

---

## Technical Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION                             │
│  World Bank API → demographics  │  Google Trends → demand signals│
│  Synthetic generator → 50 cities with census-anchored estimates │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                   ANALYSIS PIPELINE                             │
│  Composite demand scoring (6-factor weighted z-score)           │
│  TAM / SAM / SOM per city and India-wide                        │
│  8-factor Go/No-Go scorecard per city                          │
│  5-year revenue forecast (3.8% senior CAGR, 6.5% income CAGR)  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                   ML DEMAND MODEL                               │
│  Ridge Regression in log-log space (captures multiplicative     │
│  demand: population × income × nuclear rate)                    │
│  Feature importances via Random Forest (n=300 trees, depth=4)  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│               INVESTOR DASHBOARD (19MB HTML)                    │
│  10 interactive Plotly charts · Dark theme · Self-contained     │
│  India state choropleth · City bubble map · 5-yr forecast       │
│  Go/No-Go scorecard table · Radar charts · Feature importances  │
└─────────────────────────────────────────────────────────────────┘
```

### Demand Score Formula

```
Demand Score (0–100) =
  Nuclear Family %  × 0.22   (proxy for informal care gap)
+ Household Income  × 0.22   (affordability & WTP)
+ Senior Population × 0.20   (absolute market size)
+ % Population 65+  × 0.16   (ageing intensity)
+ Caregiver Gap     × 0.12   (supply-demand mismatch)
+ Google Trends     × 0.08   (real-time search demand signal)
```

### Go/No-Go Scorecard Criteria (8 factors, each 0–10)

| Factor | Weight Rationale |
|---|---|
| Market Size (senior pop) | Absolute opportunity ceiling |
| Willingness to Pay | Household income as affordability proxy |
| Nuclear Family % | Primary driver — no informal caregivers |
| Caregiver Gap | Supply shortage = structural demand |
| Affordability | Income level → premium service viability |
| Search Demand | Google Trends → latent demand signal |
| Healthcare Density | Infrastructure readiness |
| Ageing Intensity | % seniors of total population |

**Thresholds:** ≥65% → GO · 48–64% → Conditional · <48% → No-Go

---

## Project Structure

```
elderdemand-ai/
├── data/
│   ├── raw/                          # Collected data
│   │   ├── world_bank_india_demographics.csv
│   │   ├── trends_interest_over_time.csv
│   │   ├── trends_interest_by_region.csv
│   │   ├── india_50cities_eldercare.csv
│   │   └── _collection_report.json
│   └── processed/                    # Analysis outputs
│       ├── cities_scored.csv
│       ├── go_nogo_scorecard.csv
│       ├── city_forecast.csv
│       └── india_forecast.csv
├── src/
│   ├── world_bank_collector.py       # World Bank API client
│   ├── google_trends_collector.py    # pytrends collector + urllib3 patch
│   ├── synthetic_city_dataset.py     # 50-city realistic dataset generator
│   ├── build_dashboard.py            # Full pipeline + dashboard builder
│   └── run_data_collection.py        # Master runner for data collection
├── reports/
│   └── elderdemand_dashboard.html    # Self-contained investor dashboard
├── models/                           # Saved model artifacts (future)
├── notebooks/                        # Exploratory analysis (future)
├── visualizations/                   # Exported charts (future)
├── requirements.txt
└── README.md
```

---

## Technologies

| Category | Libraries |
|---|---|
| Data collection | `requests`, `pytrends` |
| Data processing | `pandas`, `numpy`, `scipy` |
| Machine learning | `scikit-learn` (Ridge, RandomForest, Pipeline) |
| Visualisation | `plotly` (choropleth, scatter-geo, subplots) |
| Geospatial | `geopandas`, `folium` |
| Statistics | `statsmodels` |
| Output | Self-contained HTML (Plotly.js CDN) |

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect live data (optional — raw data already included)

```bash
cd src
python run_data_collection.py
```

This fetches fresh data from World Bank API and Google Trends (~3–5 minutes). Google Trends has rate limits — the collector includes backoff and retry logic.

> **Note:** pytrends requires a compatibility patch for urllib3 ≥2.0 — already applied inside `google_trends_collector.py`.

### 3. Build the full analysis + dashboard

```bash
cd src
python build_dashboard.py
```

Opens `reports/elderdemand_dashboard.html` — a fully self-contained file (~19MB) with all 10 interactive charts. No server needed; open directly in any browser.

### 4. Run individual collectors

```bash
cd src
python world_bank_collector.py          # World Bank demographics
python synthetic_city_dataset.py        # Regenerate 50-city dataset
python google_trends_collector.py       # Google Trends signals
```

---

## Business Value & Investor Relevance

### Why Now
- India's senior population crosses **100M in 2024** — a demographic milestone
- Life expectancy grew +5.2 years since 2000 (more care-years per senior)
- Nuclear family rate rising ~0.8 ppt/year — informal care network shrinking
- GDP per capita up 33% since 2019 — households can now afford premium services
- Google Trends shows **accelerating search demand** for home care terms post-2021

### Market Entry Strategy (data-driven)
Based on this analysis, a staged launch is recommended:

**Phase 1 (Year 1–2):** Delhi, Bangalore, Mumbai — highest WTP + nuclear rate + trends signal
**Phase 2 (Year 2–3):** Chennai, Hyderabad, Pune — strong demand scores, lower competition
**Phase 3 (Year 3–5):** Ahmedabad, Kolkata, Chandigarh, Kochi — conditional markets maturing

### Revenue Assumptions
| Year | Penetration | Projected Revenue |
|---|---|---|
| 2024 | 0.8% | ₹487 Cr |
| 2025 | 1.8% | ₹1,174 Cr |
| 2026 | 3.0% | ₹2,095 Cr |
| 2027 | 4.5% | ₹3,392 Cr |
| 2028 | 6.2% | ₹5,064 Cr |

*Conservative scenario: 3.8% senior CAGR, 6.5% income CAGR, avg WTP ₹52,000/mo for skilled nursing visits*

### Comparable Global Precedents
- **Honor (USA):** $1.7B valuation in home care matching
- **Hometeam (USA):** $60M raised, $400M revenue
- **Portea Medical (India):** ₹500 Cr+ revenue in home healthcare
- **Care24 (India):** Serving 50+ cities with verified caregivers

India is 5–8 years behind the US home care market maturity curve — the window for category leadership is open now.

---

## Limitations & Methodology Notes

- **Synthetic city data** is anchored to Census 2011, NSSO surveys, and RBI Household Finance reports but includes stochastic noise. Validate against Census 2021 (pending full release) before investment decisions.
- **ML model** (Ridge, log-log space) demonstrates demand driver direction on n=50 cities. Production-grade prediction requires n≥500 real survey data points.
- **WTP estimates** are modelled at 8–12% of household income for professional care — consistent with NSSO health expenditure data but should be validated with primary consumer research.
- **Google Trends** indices are relative (0–100), not absolute volumes. Use for directional state-level comparison only.

---

## Author

Built as a data science portfolio project demonstrating end-to-end market intelligence: real API data collection, statistical modelling, ML demand prediction, geospatial analysis, and investor-grade reporting.

---

*Data sources: World Bank Open Data API · Google Trends (via pytrends) · Census of India 2011 · NSSO surveys · RBI Household Finance Committee Report*
