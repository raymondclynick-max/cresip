"""
02_gee_fill_extraction.py
Extract monthly fill levels for coastal reservoirs using Google Earth Engine.
Source: JRC Global Surface Water monthly history (band: water, 0=nodata 1=land 2=water)

Usage:
    python pipeline/02_gee_fill_extraction.py \
        --reservoirs data/coastal_reservoirs.csv \
        --start-date 2015-01-01 \
        --end-date 2024-12-31 \
        --output data/
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    import ee
except ImportError:
    print("ERROR: earthengine-api not installed. Run: pip install earthengine-api")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────
GEE_PROJECT     = "cresip-gee"
PRICE_PER_M3    = 0.50
BUFFER_RADIUS_M = 2000
HYPSOMETRIC_EXP = 1.5
MAX_RETRIES     = 3
RETRY_DELAY     = 10


def init_gee():
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"✓ GEE initialised (project: {GEE_PROJECT})")
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project=GEE_PROJECT)
            print(f"✓ GEE authenticated (project: {GEE_PROJECT})")
        except Exception as e:
            print(f"ERROR: GEE initialisation failed: {e}")
            sys.exit(1)


def get_jrc_water_fraction_batch(lon: float, lat: float, months: list) -> dict:
    """
    Fetch water fractions for all months in a single GEE call.
    Returns dict: {(year, month): fraction or None}
    """
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(BUFFER_RADIUS_M)

    results = {}
    try:
        jrc = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")

        # Build a multi-band image: one band per month
        def month_fraction(ym):
            year, month = ym
            img = (jrc.filter(ee.Filter.calendarRange(year, year, "year"))
                      .filter(ee.Filter.calendarRange(month, month, "month"))
                      .first())
            band = img.select("water")
            water = band.eq(2).rename(f"w_{year}_{month:02d}")
            total = band.gte(1).rename(f"t_{year}_{month:02d}")
            return water.addBands(total)

        # Process in chunks of 24 months to avoid GEE payload limits
        chunk_size = 24
        for i in range(0, len(months), chunk_size):
            chunk = months[i:i+chunk_size]
            images = [month_fraction(ym) for ym in chunk]
            stacked = ee.Image(images[0])
            for img in images[1:]:
                stacked = stacked.addBands(img)

            vals = stacked.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=region,
                scale=30,
                maxPixels=1e6
            ).getInfo()

            for year, month in chunk:
                w = vals.get(f"w_{year}_{month:02d}", 0) or 0
                t = vals.get(f"t_{year}_{month:02d}", 0) or 0
                results[(year, month)] = float(w) / float(t) if t > 0 else None

    except Exception as e:
        # Fall back to None for all months in this batch
        for ym in months:
            if ym not in results:
                results[ym] = None

    return results


def get_jrc_water_fraction(lon: float, lat: float, year: int, month: int) -> float | None:
    """Single month fallback."""
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(BUFFER_RADIUS_M)
    jrc = (ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")
             .filter(ee.Filter.calendarRange(year, year, "year"))
             .filter(ee.Filter.calendarRange(month, month, "month")))
    try:
        img = jrc.first()
        band = img.select("water")
        water_px = band.eq(2).reduceRegion(ee.Reducer.sum(), region, 30, maxPixels=1e6).getInfo()
        total_px = band.gte(1).reduceRegion(ee.Reducer.sum(), region, 30, maxPixels=1e6).getInfo()
        w = water_px.get("water", 0) or 0
        t = total_px.get("water", 0) or 0
        return float(w) / float(t) if t > 0 else None
    except Exception:
        return None


def fill_metrics(water_fraction, cap_m3: float) -> dict:
    if water_fraction is None or (isinstance(water_fraction, float) and np.isnan(water_fraction)):
        return {"fill_fraction": None, "fill_pct": None,
                "current_volume_m3": None, "void_volume_m3": None}
    frac = max(0.0, min(1.0, water_fraction))
    vol  = cap_m3 * (frac ** HYPSOMETRIC_EXP)
    return {
        "fill_fraction":     round(frac, 4),
        "fill_pct":          round(frac * 100, 2),
        "current_volume_m3": round(vol, 0),
        "void_volume_m3":    round(cap_m3 - vol, 0),
    }


def extract_reservoir(row: pd.Series, months: list) -> list:
    records = []
    res_id = row.get("id", row.name)
    lon, lat, cap = float(row["lon"]), float(row["lat"]), float(row["cap_m3"])

    # Batch fetch all months in ~5 GEE calls instead of 120
    for attempt in range(MAX_RETRIES):
        try:
            fractions = get_jrc_water_fraction_batch(lon, lat, months)
            break
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                fractions = {ym: None for ym in months}

    for year, month in months:
        wf = fractions.get((year, month))
        records.append({
            "reservoir_id": res_id,
            "year": year, "month": month,
            "date": f"{year}-{month:02d}-01",
            "water_fraction": wf,
            **fill_metrics(wf, cap),
            "source": "jrc",
        })
    return records


def build_months(start_date: str, end_date: str) -> list:
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(day=1)
    end   = datetime.strptime(end_date,   "%Y-%m-%d")
    months, cur = [], start
    while cur <= end:
        months.append((cur.year, cur.month))
        cur = cur.replace(month=cur.month % 12 + 1,
                          year=cur.year + (1 if cur.month == 12 else 0))
    return months


def seasonal_profile(ts: pd.DataFrame) -> pd.DataFrame:
    return (ts.dropna(subset=["fill_pct"])
              .groupby(["reservoir_id", "month"])["fill_pct"]
              .mean().round(2).reset_index()
              .rename(columns={"fill_pct": "mean_fill_pct"}))


def revenue_potential(ts: pd.DataFrame, reservoirs: pd.DataFrame) -> pd.DataFrame:
    mean_void = (ts.dropna(subset=["void_volume_m3"])
                   .groupby(["reservoir_id", "year"])["void_volume_m3"].mean()
                   .reset_index()
                   .groupby("reservoir_id")["void_volume_m3"].mean()
                   .round(0).reset_index()
                   .rename(columns={"void_volume_m3": "mean_annual_void_m3"}))
    rev = mean_void.merge(
        reservoirs[["id","cap_m3","lat","lon","country"]].rename(columns={"id":"reservoir_id"}),
        on="reservoir_id", how="left")
    rev["mean_fill_pct"]            = ((1 - rev["mean_annual_void_m3"] / rev["cap_m3"]) * 100).round(2)
    rev["revenue_conservative_usd"] = (rev["mean_annual_void_m3"] * 0.5 * PRICE_PER_M3).round(0)
    rev["revenue_base_usd"]         = (rev["mean_annual_void_m3"] * PRICE_PER_M3).round(0)
    rev["revenue_optimistic_usd"]   = (rev["cap_m3"] * PRICE_PER_M3).round(0)
    return rev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reservoirs",     required=True)
    parser.add_argument("--start-date",     default="2015-01-01")
    parser.add_argument("--end-date",       default="2024-12-31")
    parser.add_argument("--output",         default="data/")
    parser.add_argument("--max-reservoirs", type=int, default=None)
    parser.add_argument("--resume",         action="store_true")
    args = parser.parse_args()

    if not Path(args.reservoirs).exists():
        print(f"ERROR: {args.reservoirs} not found.")
        sys.exit(1)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    ts_path = Path(args.output) / "fill_timeseries.csv"

    init_gee()
    print("Loading reservoirs...")

    reservoirs = pd.read_csv(args.reservoirs)
    reservoirs = reservoirs.dropna(subset=["lat","lon","cap_m3"])
    reservoirs = reservoirs[reservoirs["cap_m3"] > 0]
    print(f"Loaded {len(reservoirs):,} reservoirs")

    if args.max_reservoirs:
        reservoirs = reservoirs.head(args.max_reservoirs)
        print(f"  Limiting to {args.max_reservoirs} for testing")

    months = build_months(args.start_date, args.end_date)
    print(f"  {len(months)} months: {args.start_date} → {args.end_date}")

    done_ids, existing = set(), []
    if args.resume and ts_path.exists():
        df_ex = pd.read_csv(ts_path)
        done_ids = set(df_ex["reservoir_id"].unique())
        existing = df_ex.to_dict("records")
        print(f"  Resuming: {len(done_ids)} done")

    todo = reservoirs[~reservoirs["id"].isin(done_ids)] if done_ids else reservoirs
    print(f"\nExtracting {len(todo):,} reservoirs × {len(months)} months...")
    print(f"Estimated time: {len(todo) * len(months) * 0.3 / 60:.0f} minutes\n")

    all_records = list(existing)
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc="Reservoirs"):
        all_records.extend(extract_reservoir(row, months))
        if len(all_records) % (50 * len(months)) == 0:
            pd.DataFrame(all_records).to_csv(ts_path, index=False)

    ts = pd.DataFrame(all_records)
    ts.to_csv(ts_path, index=False)
    print(f"\n✓ {len(ts):,} records → {ts_path}")

    sp = seasonal_profile(ts)
    sp.to_csv(Path(args.output) / "seasonal_profile.csv", index=False)

    ms = (ts.dropna(subset=["fill_pct"])
            .groupby(["reservoir_id","year","month"])
            .agg(fill_pct=("fill_pct","mean"), void_m3=("void_volume_m3","mean"))
            .round(2).reset_index())
    ms.to_csv(Path(args.output) / "monthly_stats.csv", index=False)

    rev = revenue_potential(ts, reservoirs)
    rev.to_csv(Path(args.output) / "revenue_potential.csv", index=False)

    print(f"✓ Seasonal profile, monthly stats, revenue saved")
    print(f"\nRevenue summary:")
    print(f"  Base case total: ${rev['revenue_base_usd'].sum()/1e9:.2f}B/year")
    print(f"  Mean fill:       {rev['mean_fill_pct'].mean():.1f}%")


if __name__ == "__main__":
    main()
