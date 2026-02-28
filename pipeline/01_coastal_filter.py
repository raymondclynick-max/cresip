"""
01_coastal_filter.py
Filter GDW v1.0 reservoirs within 20km of the GSHHG coastline.
Outputs: data/coastal_reservoirs.csv + data/coastal_reservoirs.gpkg

Usage:
    python pipeline/01_coastal_filter.py \
        --gdw "E:/OneDrive/uuwater/GDW_v1_0_shp/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp" \
        --gshhg "E:/OneDrive/uuwater/gshhg-shp-2.3.7/GSHHS_shp/l/GSHHS_l_L1.shp" \
        --buffer-km 20 \
        --output "data/coastal_reservoirs"
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union  # kept for potential future use
from tqdm import tqdm

# ── Column mappings: GDW v1.0 field names ────────────────────────────────────
# Inspect with: gdf.columns.tolist()
GDW_COLS = {
    "id":       "GDW_ID",       # unique reservoir ID
    "name":     "DAM_NAME",     # dam/reservoir name
    "country":  "COUNTRY",      # country
    "lat":      "LAT_DD",       # latitude decimal degrees
    "lon":      "LONG_DD",      # longitude decimal degrees
    "cap_mcm":  "CAP_MCM",      # capacity million cubic metres
    "area_skm": "AREA_SKM",     # surface area km²
    "depth_m":  "DEPTH_M",      # mean depth metres (may be null)
    "purpose":  "MAIN_USE",     # primary use
    "year":     "YEAR",         # year completed
}

MOLLWEIDE = "ESRI:54009"   # equal-area projection for buffering


def load_gdw(path: str) -> gpd.GeoDataFrame:
    print(f"Loading GDW reservoirs from: {path}")
    gdf = gpd.read_file(path)
    print(f"  Loaded {len(gdf):,} reservoirs")
    print(f"  Columns: {gdf.columns.tolist()}")

    # Remap columns to standard names where they exist
    rename = {}
    for std, gdw in GDW_COLS.items():
        if gdw in gdf.columns:
            rename[gdw] = std
        else:
            # Try case-insensitive match
            match = [c for c in gdf.columns if c.upper() == gdw.upper()]
            if match:
                rename[match[0]] = std

    gdf = gdf.rename(columns=rename)
    print(f"  Mapped columns: {list(rename.values())}")
    return gdf


def load_coastline(path: str) -> gpd.GeoDataFrame:
    print(f"Loading GSHHG coastline from: {path}")
    coast = gpd.read_file(path)
    print(f"  Loaded {len(coast):,} coastline polygons")
    return coast


def filter_coastal(reservoirs: gpd.GeoDataFrame,
                   coastline: gpd.GeoDataFrame,
                   buffer_km: float) -> gpd.GeoDataFrame:
    print(f"\nProjecting to Mollweide equal-area for {buffer_km}km buffer...")

    # Use geometry centroid for coordinates (LONG_DAM/LAT_DAM are empty in GDW v1.0)
    if "LONG_DAM" in reservoirs.columns and "LAT_DAM" in reservoirs.columns:
        non_zero = (reservoirs["LONG_DAM"] != 0) & (reservoirs["LAT_DAM"] != 0)
        use_dam_coords = non_zero.sum() > len(reservoirs) * 0.5
    else:
        use_dam_coords = False

    if use_dam_coords:
        print("  Building point geometry from DAM coordinates...")
        from shapely.geometry import Point
        reservoirs = reservoirs.copy()
        reservoirs["geometry"] = reservoirs.apply(
            lambda r: Point(r["LONG_DAM"], r["LAT_DAM"])
            if pd.notna(r["LONG_DAM"]) and r["LONG_DAM"] != 0 else None,
            axis=1
        )
        reservoirs = reservoirs[reservoirs["geometry"].notna()]
        reservoirs = gpd.GeoDataFrame(reservoirs, geometry="geometry", crs="EPSG:4326")
    else:
        print("  Using shapefile geometry (DAM coords empty)...")
        # Ensure WGS84
        if reservoirs.crs is None:
            reservoirs = reservoirs.set_crs("EPSG:4326")
        else:
            reservoirs = reservoirs.to_crs("EPSG:4326")
        # Convert polygon geometries to centroids if needed
        geom_types = reservoirs.geometry.geom_type.unique()
        print(f"  Geometry types: {geom_types}")
        if any(t in ['Polygon', 'MultiPolygon'] for t in geom_types):
            print("  Converting polygons to centroids...")
            reservoirs = reservoirs.copy()
            reservoirs["geometry"] = reservoirs.geometry.centroid

    if coastline.crs is None:
        coastline = coastline.set_crs("EPSG:4326")
    else:
        coastline = coastline.to_crs("EPSG:4326")

    # Project to equal-area
    res_proj = reservoirs.to_crs(MOLLWEIDE).copy()
    coast_proj = coastline.to_crs(MOLLWEIDE).copy()

    buffer_m = buffer_km * 1000

    # Convert coastline polygons to linestrings (the actual shoreline)
    # This is what we want distance to — the land/sea boundary
    print("  Extracting coastline boundaries (shorelines)...")
    coast_proj["geometry"] = coast_proj.geometry.buffer(0)  # fix invalids
    coast_lines = coast_proj.copy()
    coast_lines["geometry"] = coast_proj.geometry.boundary
    coast_lines = coast_lines[~coast_lines.geometry.is_empty].reset_index(drop=True)
    print(f"  {len(coast_lines):,} shoreline segments")

    # Use sjoin_nearest to find distance from each reservoir to nearest shoreline
    print("  Finding nearest shoreline to each reservoir (this takes ~5 mins)...")
    res_proj = res_proj.reset_index(drop=True)

    result = gpd.sjoin_nearest(
        res_proj[["geometry"]],
        coast_lines[["geometry"]],
        how="left",
        distance_col="dist_coast_m"
    )

    # sjoin_nearest may return duplicates if equidistant — keep minimum
    result = result.groupby(result.index)["dist_coast_m"].min()

    within_mask = result <= buffer_m
    coastal = reservoirs[within_mask].copy()
    coastal["dist_coast_km"] = (result[within_mask] / 1000).round(3).values

    print(f"  {within_mask.sum():,} of {len(reservoirs):,} reservoirs within {buffer_km}km of coast")
    return coastal


def derive_metrics(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Derive standardised capacity metrics from GDW columns."""
    df = gdf.copy()

    # Capacity: convert million cubic metres → cubic metres
    if "cap_mcm" in df.columns:
        df["cap_m3"] = pd.to_numeric(df["cap_mcm"], errors="coerce") * 1e6
    else:
        df["cap_m3"] = np.nan

    # Area: convert km² → m²
    if "area_skm" in df.columns:
        df["area_m2"] = pd.to_numeric(df["area_skm"], errors="coerce") * 1e6
    else:
        df["area_m2"] = np.nan

    # Depth: use directly if available, else estimate from cap/area
    if "depth_m" in df.columns:
        df["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce")
    else:
        df["depth_m"] = np.nan

    # Estimate missing depth from volume / area
    mask = df["depth_m"].isna() & df["cap_m3"].notna() & df["area_m2"].notna() & (df["area_m2"] > 0)
    df.loc[mask, "depth_m"] = (df.loc[mask, "cap_m3"] / df.loc[mask, "area_m2"]).round(2)

    # Drop reservoirs with no capacity data
    before = len(df)
    df = df[df["cap_m3"].notna() & (df["cap_m3"] > 0)]
    print(f"  Dropped {before - len(df)} reservoirs with missing/zero capacity")

    return df


def main():
    parser = argparse.ArgumentParser(description="Filter GDW reservoirs within coastal buffer")
    parser.add_argument("--gdw", required=True, help="Path to GDW_reservoirs_v1_0.shp")
    parser.add_argument("--gshhg", required=True, help="Path to GSHHS_l_L1.shp")
    parser.add_argument("--buffer-km", type=float, default=20.0, help="Coastal buffer distance in km")
    parser.add_argument("--output", default="data/coastal_reservoirs", help="Output path prefix (no extension)")
    args = parser.parse_args()

    # Validate inputs
    for p in [args.gdw, args.gshhg]:
        if not Path(p).exists():
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load
    reservoirs = load_gdw(args.gdw)
    coastline = load_coastline(args.gshhg)

    # Filter
    coastal = filter_coastal(reservoirs, coastline, args.buffer_km)

    # Derive metrics
    print("\nDeriving capacity metrics...")
    coastal = derive_metrics(coastal)

    # Ensure lat/lon columns exist - derive from geometry in WGS84
    if "lat" not in coastal.columns or coastal["lat"].isna().all():
        wgs84 = coastal.to_crs("EPSG:4326") if coastal.crs != "EPSG:4326" else coastal
        coastal["lon"] = wgs84.geometry.centroid.x
        coastal["lat"] = wgs84.geometry.centroid.y

    # Select output columns
    keep = ["id", "name", "country", "lat", "lon",
            "cap_m3", "cap_mcm", "area_m2", "depth_m",
            "dist_coast_km", "purpose", "year", "geometry"]
    keep = [c for c in keep if c in coastal.columns]
    coastal = coastal[keep]

    # Save
    csv_path = f"{args.output}.csv"
    gpkg_path = f"{args.output}.gpkg"

    coastal.drop(columns=["geometry"]).to_csv(csv_path, index=False)
    coastal.to_file(gpkg_path, driver="GPKG")

    print(f"\n✓ Saved {len(coastal):,} coastal reservoirs")
    print(f"  CSV:  {csv_path}")
    print(f"  GPKG: {gpkg_path}")
    print(f"\nCapacity summary:")
    print(f"  Total capacity: {coastal['cap_m3'].sum()/1e9:.2f} Bm³")
    print(f"  Mean capacity:  {coastal['cap_m3'].mean()/1e6:.1f} Mm³")
    print(f"  Countries:      {coastal['country'].nunique() if 'country' in coastal.columns else 'N/A'}")


if __name__ == "__main__":
    main()
