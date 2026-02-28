"""
03_cluster_analysis.py
DBSCAN spatial clustering of coastal reservoirs + depot placement optimisation.

Usage:
    python pipeline/03_cluster_analysis.py \
        --reservoirs data/coastal_reservoirs.csv \
        --revenue data/revenue_potential.csv \
        --seasonal data/seasonal_profile.csv \
        --output data/
"""

import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Cost model constants ──────────────────────────────────────────────────────
TRANSPORT_COST_PER_KM_PER_MM3 = 5000   # USD per km per million m³
DEPOT_CAPEX_USD = 10_000_000            # USD fixed cost per depot
PAYBACK_YEARS = 10                      # capex amortisation period
MAX_VIABLE_PAYBACK = 20                 # years — above this: not viable
MIN_CLUSTER_REVENUE = 500_000           # USD/yr minimum to consider

# ── DBSCAN parameters ─────────────────────────────────────────────────────────
DBSCAN_EPS_KM = 150        # max distance between cluster members
DBSCAN_MIN_SAMPLES = 3     # minimum reservoirs per cluster


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def haversine_matrix(coords: np.ndarray) -> np.ndarray:
    """Pairwise haversine distance matrix for (lat, lon) array."""
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = haversine_km(coords[i,0], coords[i,1], coords[j,0], coords[j,1])
            D[i,j] = d
            D[j,i] = d
    return D


def optimise_depot(lats: np.ndarray, lons: np.ndarray,
                   weights: np.ndarray) -> tuple[float, float, float]:
    """
    Find optimal depot location minimising weighted sum of distances.
    Returns: (depot_lat, depot_lon, weighted_mean_distance_km)
    """
    weights = weights / weights.sum()

    def objective(x):
        depot_lat, depot_lon = x
        dists = np.array([
            haversine_km(depot_lat, depot_lon, lat, lon)
            for lat, lon in zip(lats, lons)
        ])
        return np.sum(weights * dists)

    # Initial guess: weighted centroid
    x0 = [np.average(lats, weights=weights), np.average(lons, weights=weights)]
    bounds = [(-90, 90), (-180, 180)]

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    depot_lat, depot_lon = result.x

    # Mean weighted distance
    dists = np.array([
        haversine_km(depot_lat, depot_lon, lat, lon)
        for lat, lon in zip(lats, lons)
    ])
    mean_dist = float(np.average(dists, weights=weights))

    return float(depot_lat), float(depot_lon), mean_dist


def fill_correlation(seasonal: pd.DataFrame, reservoir_ids: list) -> float:
    """
    Mean pairwise Pearson correlation of seasonal fill profiles.
    High correlation → reservoirs dry at same time → depot useful.
    """
    profiles = []
    for rid in reservoir_ids:
        s = seasonal[seasonal["reservoir_id"] == rid].sort_values("month")
        if len(s) == 12:
            profiles.append(s["mean_fill_pct"].values)

    if len(profiles) < 2:
        return 0.0

    corrs = []
    for i in range(len(profiles)):
        for j in range(i+1, len(profiles)):
            c = np.corrcoef(profiles[i], profiles[j])[0,1]
            if not np.isnan(c):
                corrs.append(c)

    return round(float(np.mean(corrs)), 4) if corrs else 0.0


def compute_cluster_economics(cluster_df: pd.DataFrame,
                               revenue_df: pd.DataFrame,
                               seasonal_df: pd.DataFrame,
                               cluster_id: int,
                               region_name: str) -> dict:
    """Compute full economic profile for a cluster."""
    res_ids = cluster_df["reservoir_id"].tolist()

    # Revenue
    rev_data = revenue_df[revenue_df["reservoir_id"].isin(res_ids)]
    annual_void_m3 = rev_data["mean_annual_void_m3"].sum()
    annual_revenue = rev_data["revenue_base_usd"].sum()

    if annual_revenue < MIN_CLUSTER_REVENUE:
        viable = False
    else:
        viable = None  # determined later

    # Depot placement
    lats = cluster_df["lat"].values
    lons = cluster_df["lon"].values
    weights = rev_data.set_index("reservoir_id")["mean_annual_void_m3"].reindex(res_ids).fillna(1).values

    depot_lat, depot_lon, mean_dist_km = optimise_depot(lats, lons, weights)

    # Transport cost
    void_mm3 = annual_void_m3 / 1e6  # million m³
    transport_cost = mean_dist_km * void_mm3 * TRANSPORT_COST_PER_KM_PER_MM3

    # Net value
    annual_capex = DEPOT_CAPEX_USD / PAYBACK_YEARS
    net_annual_value = annual_revenue - transport_cost - annual_capex

    # Payback
    if annual_revenue > transport_cost:
        payback = DEPOT_CAPEX_USD / (annual_revenue - transport_cost)
    else:
        payback = float("inf")

    viable = (net_annual_value > 0) and (payback < MAX_VIABLE_PAYBACK)

    # Fill correlation
    corr = fill_correlation(seasonal_df, res_ids)

    # Country breakdown
    countries = cluster_df["country"].dropna().unique().tolist() if "country" in cluster_df.columns else []

    return {
        "cluster_id": cluster_id,
        "region": region_name,
        "n_reservoirs": len(res_ids),
        "reservoir_ids": ",".join(str(r) for r in res_ids),
        "depot_lat": round(depot_lat, 4),
        "depot_lon": round(depot_lon, 4),
        "countries": ",".join(countries),
        "annual_void_m3": round(annual_void_m3, 0),
        "annual_void_mm3": round(void_mm3, 3),
        "annual_revenue_usd": round(annual_revenue, 0),
        "transport_cost_usd": round(transport_cost, 0),
        "annual_capex_usd": round(annual_capex, 0),
        "net_annual_value_usd": round(net_annual_value, 0),
        "payback_years": round(payback, 2) if payback != float("inf") else 999,
        "mean_depot_distance_km": round(mean_dist_km, 2),
        "fill_correlation": corr,
        "viable": viable,
    }


def infer_region_name(lats: np.ndarray, lons: np.ndarray) -> str:
    """Very rough region name from centroid coordinates."""
    lat = np.mean(lats)
    lon = np.mean(lons)

    if lat > 55:
        if lon < 30: return "N Europe"
        return "N Asia"
    elif lat > 35:
        if lon < -30: return "E USA"
        if lon < 0: return "W Europe"
        if lon < 60: return "Med / Middle East"
        if lon < 100: return "C Asia"
        return "E Asia"
    elif lat > 10:
        if lon < -30: return "Caribbean"
        if lon < 20: return "W Africa"
        if lon < 80: return "S Asia"
        if lon < 130: return "SE Asia"
        return "Pacific"
    elif lat > -10:
        if lon < -30: return "N Brazil"
        return "Equatorial Africa"
    elif lat > -35:
        if lon < -30: return "S America"
        if lon < 30: return "S Africa"
        return "Australia"
    else:
        return "S Ocean"


def main():
    parser = argparse.ArgumentParser(description="Cluster analysis and depot optimisation")
    parser.add_argument("--reservoirs", required=True)
    parser.add_argument("--revenue", required=True)
    parser.add_argument("--seasonal", required=True)
    parser.add_argument("--output", default="data/")
    args = parser.parse_args()

    for p in [args.reservoirs, args.revenue, args.seasonal]:
        if not Path(p).exists():
            print(f"ERROR: {p} not found")
            sys.exit(1)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    reservoirs = pd.read_csv(args.reservoirs)
    revenue = pd.read_csv(args.revenue)
    seasonal = pd.read_csv(args.seasonal)

    # Standardise reservoir_id in revenue
    if "reservoir_id" not in revenue.columns and "id" in revenue.columns:
        revenue = revenue.rename(columns={"id": "reservoir_id"})

    # Revenue already has lat/lon from the extraction step; merge only if missing
    if "lat" not in revenue.columns or revenue["lat"].isna().all():
        res_coords = reservoirs[["id", "lat", "lon", "country"]].rename(columns={"id": "reservoir_id"})
        revenue = revenue.merge(res_coords, on="reservoir_id", how="left")
    revenue = revenue.dropna(subset=["lat", "lon"])

    print(f"  {len(reservoirs):,} reservoirs")
    print(f"  {len(revenue):,} with revenue data")
    print(f"  {len(seasonal):,} seasonal profile records")

    # DBSCAN clustering
    print(f"\nRunning DBSCAN (eps={DBSCAN_EPS_KM}km, min_samples={DBSCAN_MIN_SAMPLES})...")
    coords = revenue[["lat", "lon"]].values
    eps_rad = DBSCAN_EPS_KM / 6371.0  # convert km to radians

    db = DBSCAN(eps=eps_rad, min_samples=DBSCAN_MIN_SAMPLES, algorithm="ball_tree", metric="haversine")
    labels = db.fit_predict(np.radians(coords))

    revenue["cluster_label"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  {n_clusters} clusters found, {n_noise} noise points")

    # Compute economics for each cluster
    print("\nComputing cluster economics...")
    cluster_results = []
    unique_labels = sorted(set(labels) - {-1})

    for label in tqdm(unique_labels, desc="Clusters"):
        mask = revenue["cluster_label"] == label
        cluster_df = revenue[mask].copy()

        region = infer_region_name(cluster_df["lat"].values, cluster_df["lon"].values)
        result = compute_cluster_economics(cluster_df, revenue, seasonal, label, region)
        cluster_results.append(result)

    clusters_df = pd.DataFrame(cluster_results)

    # Sort by net annual value descending
    clusters_df = clusters_df.sort_values("net_annual_value_usd", ascending=False).reset_index(drop=True)
    clusters_df["rank"] = clusters_df.index + 1

    # Save clusters
    clusters_path = Path(args.output) / "clusters.csv"
    clusters_df.to_csv(clusters_path, index=False)

    # Save reservoirs with cluster assignments
    res_clustered = reservoirs.merge(
        revenue[["reservoir_id", "cluster_label"]].rename(columns={"reservoir_id": "id"}),
        on="id", how="left"
    )
    res_clustered_path = Path(args.output) / "reservoirs_clustered.csv"
    res_clustered.to_csv(res_clustered_path, index=False)

    # Summary
    viable = clusters_df[clusters_df["viable"] == True]
    print(f"\n✓ Saved {len(clusters_df)} clusters → {clusters_path}")
    print(f"\nResults:")
    print(f"  Total clusters:   {len(clusters_df)}")
    print(f"  Viable clusters:  {len(viable)}")
    print(f"  Total viable revenue: ${viable['annual_revenue_usd'].sum()/1e9:.2f}B/year")
    print(f"\nTop 5 clusters:")
    cols = ["rank", "region", "n_reservoirs", "annual_void_mm3",
            "annual_revenue_usd", "payback_years", "viable"]
    print(clusters_df[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
