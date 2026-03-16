"""
Master Data Collection Runner — ElderDemand AI
Runs all collectors in sequence and produces a collection report
"""

import time
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')


def run_world_bank():
    print("\n" + "="*60)
    print("STEP 1: World Bank API — India Demographics")
    print("="*60)
    from world_bank_collector import collect_world_bank_data, save_world_bank_data
    df = collect_world_bank_data()
    if not df.empty:
        save_world_bank_data(df)
        return {"status": "success", "rows": len(df), "cols": list(df.columns)}
    return {"status": "failed", "rows": 0}


def run_google_trends():
    print("\n" + "="*60)
    print("STEP 2: Google Trends — Demand Signals")
    print("="*60)
    from google_trends_collector import collect_google_trends
    results = collect_google_trends()
    summary = {}
    for key, df in results.items():
        summary[key] = {"rows": len(df), "cols": list(df.columns)}
    return {"status": "success" if results else "failed", "datasets": summary}


def run_synthetic_cities():
    print("\n" + "="*60)
    print("STEP 3: Synthetic City Dataset — 50 Indian Cities")
    print("="*60)
    from synthetic_city_dataset import generate_city_dataset, save_city_dataset
    df = generate_city_dataset()
    save_city_dataset(df)
    return {"status": "success", "rows": len(df), "cols": list(df.columns)}


def write_collection_report(results: dict):
    report = {
        "project": "ElderDemand AI — Senior Care Market Intelligence",
        "collection_timestamp": datetime.now().isoformat(),
        "data_sources": results,
        "raw_data_directory": RAW_DATA_DIR,
        "files_collected": sorted(os.listdir(RAW_DATA_DIR)) if os.path.exists(RAW_DATA_DIR) else [],
    }
    report_path = os.path.join(RAW_DATA_DIR, "_collection_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nCollection report saved: {report_path}")
    return report


if __name__ == "__main__":
    import sys
    # Run from src/ directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("ElderDemand AI — Data Collection Pipeline")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    start = time.time()

    try:
        results["world_bank"] = run_world_bank()
    except Exception as e:
        print(f"[ERROR] World Bank: {e}")
        results["world_bank"] = {"status": "error", "error": str(e)}

    try:
        results["google_trends"] = run_google_trends()
    except Exception as e:
        print(f"[ERROR] Google Trends: {e}")
        results["google_trends"] = {"status": "error", "error": str(e)}

    try:
        results["synthetic_cities"] = run_synthetic_cities()
    except Exception as e:
        print(f"[ERROR] Synthetic Cities: {e}")
        results["synthetic_cities"] = {"status": "error", "error": str(e)}

    elapsed = round(time.time() - start, 1)
    report = write_collection_report(results)

    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    for source, result in results.items():
        status = result.get("status", "unknown")
        icon = "✓" if status == "success" else "✗"
        print(f"  {icon} {source}: {status}")
    print(f"\nTotal time: {elapsed}s")
    print(f"Files in data/raw: {len(report['files_collected'])}")
    print("\nAll done.")
