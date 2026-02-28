"""
04_seed_supabase.py
Create tables and seed Supabase with processed CRESIP data.

Usage:
    python pipeline/04_seed_supabase.py \
        --reservoirs data/coastal_reservoirs.csv \
        --timeseries data/fill_timeseries.csv \
        --seasonal data/seasonal_profile.csv \
        --revenue data/revenue_potential.csv \
        --clusters data/clusters.csv \
        --url https://wbuoekaaurcqcfrepsdu.supabase.co \
        --key YOUR_SERVICE_ROLE_KEY
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from supabase import create_client, Client
except ImportError:
    print("ERROR: supabase not installed.")
    sys.exit(1)

BATCH_SIZE = 500   # rows per upsert batch


def connect(url: str, key: str) -> Client:
    client = create_client(url, key)
    print(f"✓ Connected to Supabase: {url}")
    return client


def create_schema(client: Client):
    """
    Execute SQL to create tables via Supabase REST (postgrest RPC).
    Tables: reservoirs, fill_data, seasonal_profiles, clusters
    """
    sql_statements = [
        # Enable PostGIS (should already be enabled, idempotent)
        "CREATE EXTENSION IF NOT EXISTS postgis;",

        # Reservoirs table
        """
        CREATE TABLE IF NOT EXISTS reservoirs (
            id              TEXT PRIMARY KEY,
            name            TEXT,
            country         TEXT,
            lat             DOUBLE PRECISION NOT NULL,
            lon             DOUBLE PRECISION NOT NULL,
            cap_m3          DOUBLE PRECISION,
            cap_mcm         DOUBLE PRECISION,
            area_m2         DOUBLE PRECISION,
            depth_m         DOUBLE PRECISION,
            dist_coast_km   DOUBLE PRECISION,
            purpose         TEXT,
            year            INTEGER,
            cluster_label   INTEGER,
            geom            GEOMETRY(POINT, 4326),
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );
        """,

        # Spatial index
        "CREATE INDEX IF NOT EXISTS idx_reservoirs_geom ON reservoirs USING GIST (geom);",
        "CREATE INDEX IF NOT EXISTS idx_reservoirs_country ON reservoirs (country);",
        "CREATE INDEX IF NOT EXISTS idx_reservoirs_cluster ON reservoirs (cluster_label);",

        # Fill timeseries
        """
        CREATE TABLE IF NOT EXISTS fill_data (
            id              BIGSERIAL PRIMARY KEY,
            reservoir_id    TEXT NOT NULL REFERENCES reservoirs(id) ON DELETE CASCADE,
            date            DATE NOT NULL,
            year            INTEGER,
            month           INTEGER,
            fill_fraction   DOUBLE PRECISION,
            fill_pct        DOUBLE PRECISION,
            current_volume_m3 DOUBLE PRECISION,
            void_volume_m3  DOUBLE PRECISION,
            water_fraction  DOUBLE PRECISION,
            source          TEXT DEFAULT 'jrc',
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(reservoir_id, date)
        );
        """,

        "CREATE INDEX IF NOT EXISTS idx_fill_data_reservoir ON fill_data (reservoir_id);",
        "CREATE INDEX IF NOT EXISTS idx_fill_data_date ON fill_data (date);",

        # Seasonal profiles
        """
        CREATE TABLE IF NOT EXISTS seasonal_profiles (
            id              BIGSERIAL PRIMARY KEY,
            reservoir_id    TEXT NOT NULL REFERENCES reservoirs(id) ON DELETE CASCADE,
            month           INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
            mean_fill_pct   DOUBLE PRECISION,
            UNIQUE(reservoir_id, month)
        );
        """,

        # Clusters / depot candidates
        """
        CREATE TABLE IF NOT EXISTS clusters (
            id              BIGSERIAL PRIMARY KEY,
            cluster_id      INTEGER NOT NULL,
            rank            INTEGER,
            region          TEXT,
            n_reservoirs    INTEGER,
            reservoir_ids   TEXT,
            depot_lat       DOUBLE PRECISION,
            depot_lon       DOUBLE PRECISION,
            depot_geom      GEOMETRY(POINT, 4326),
            countries       TEXT,
            annual_void_m3  DOUBLE PRECISION,
            annual_void_mm3 DOUBLE PRECISION,
            annual_revenue_usd DOUBLE PRECISION,
            transport_cost_usd DOUBLE PRECISION,
            net_annual_value_usd DOUBLE PRECISION,
            payback_years   DOUBLE PRECISION,
            mean_depot_distance_km DOUBLE PRECISION,
            fill_correlation DOUBLE PRECISION,
            viable          BOOLEAN,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );
        """,

        "CREATE INDEX IF NOT EXISTS idx_clusters_viable ON clusters (viable);",
        "CREATE INDEX IF NOT EXISTS idx_clusters_depot_geom ON clusters USING GIST (depot_geom);",
    ]

    print("Creating schema...")
    for sql in sql_statements:
        sql = sql.strip()
        if not sql:
            continue
        try:
            client.rpc("exec_sql", {"query": sql}).execute()
        except Exception as e:
            # Try direct postgrest approach for DDL
            print(f"  Note: {str(e)[:80]}")

    print("  Schema creation attempted (verify in Supabase dashboard if needed)")


def create_tables_via_dashboard_sql(client: Client):
    """
    Alternative: print SQL for manual execution in Supabase SQL editor.
    Use this if RPC exec_sql is not available.
    """
    sql = """
-- Run this in Supabase SQL Editor: https://supabase.com/dashboard/project/wbuoekaaurcqcfrepsdu/sql

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS reservoirs (
    id              TEXT PRIMARY KEY,
    name            TEXT,
    country         TEXT,
    lat             DOUBLE PRECISION NOT NULL,
    lon             DOUBLE PRECISION NOT NULL,
    cap_m3          DOUBLE PRECISION,
    cap_mcm         DOUBLE PRECISION,
    area_m2         DOUBLE PRECISION,
    depth_m         DOUBLE PRECISION,
    dist_coast_km   DOUBLE PRECISION,
    purpose         TEXT,
    year            INTEGER,
    cluster_label   INTEGER,
    geom            GEOMETRY(POINT, 4326),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reservoirs_geom ON reservoirs USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_reservoirs_country ON reservoirs (country);
CREATE INDEX IF NOT EXISTS idx_reservoirs_cluster ON reservoirs (cluster_label);

CREATE TABLE IF NOT EXISTS fill_data (
    id              BIGSERIAL PRIMARY KEY,
    reservoir_id    TEXT NOT NULL REFERENCES reservoirs(id) ON DELETE CASCADE,
    date            DATE NOT NULL,
    year            INTEGER,
    month           INTEGER,
    fill_fraction   DOUBLE PRECISION,
    fill_pct        DOUBLE PRECISION,
    current_volume_m3 DOUBLE PRECISION,
    void_volume_m3  DOUBLE PRECISION,
    water_fraction  DOUBLE PRECISION,
    source          TEXT DEFAULT 'jrc',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(reservoir_id, date)
);

CREATE INDEX IF NOT EXISTS idx_fill_data_reservoir ON fill_data (reservoir_id);
CREATE INDEX IF NOT EXISTS idx_fill_data_date ON fill_data (date);

CREATE TABLE IF NOT EXISTS seasonal_profiles (
    id              BIGSERIAL PRIMARY KEY,
    reservoir_id    TEXT NOT NULL REFERENCES reservoirs(id) ON DELETE CASCADE,
    month           INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
    mean_fill_pct   DOUBLE PRECISION,
    UNIQUE(reservoir_id, month)
);

CREATE TABLE IF NOT EXISTS clusters (
    id              BIGSERIAL PRIMARY KEY,
    cluster_id      INTEGER NOT NULL,
    rank            INTEGER,
    region          TEXT,
    n_reservoirs    INTEGER,
    reservoir_ids   TEXT,
    depot_lat       DOUBLE PRECISION,
    depot_lon       DOUBLE PRECISION,
    depot_geom      GEOMETRY(POINT, 4326),
    countries       TEXT,
    annual_void_m3  DOUBLE PRECISION,
    annual_void_mm3 DOUBLE PRECISION,
    annual_revenue_usd DOUBLE PRECISION,
    transport_cost_usd DOUBLE PRECISION,
    net_annual_value_usd DOUBLE PRECISION,
    payback_years   DOUBLE PRECISION,
    mean_depot_distance_km DOUBLE PRECISION,
    fill_correlation DOUBLE PRECISION,
    viable          BOOLEAN,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_clusters_viable ON clusters (viable);
CREATE INDEX IF NOT EXISTS idx_clusters_depot_geom ON clusters USING GIST (depot_geom);
    """
    sql_path = Path("data/schema.sql")
    sql_path.write_text(sql)
    print(f"\n⚠  Schema SQL saved to {sql_path}")
    print("   Please run it manually in the Supabase SQL Editor:")
    print("   https://supabase.com/dashboard/project/wbuoekaaurcqcfrepsdu/sql/new")
    return sql


def clean_row(row: dict) -> dict:
    """Convert NaN/inf to None for JSON serialisation."""
    cleaned = {}
    for k, v in row.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            cleaned[k] = None
        elif isinstance(v, np.integer):
            cleaned[k] = int(v)
        elif isinstance(v, np.floating):
            cleaned[k] = float(v) if not (np.isnan(v) or np.isinf(v)) else None
        else:
            cleaned[k] = v
    return cleaned


def batch_upsert(client: Client, table: str, records: list[dict],
                  conflict_col: str = None):
    """Upsert records in batches."""
    total = len(records)
    inserted = 0
    for i in range(0, total, BATCH_SIZE):
        batch = records[i:i+BATCH_SIZE]
        batch = [clean_row(r) for r in batch]
        try:
            q = client.table(table).upsert(batch)
            if conflict_col:
                pass  # supabase-py handles upsert via primary key
            q.execute()
            inserted += len(batch)
            print(f"  {table}: {inserted}/{total}", end="\r")
        except Exception as e:
            print(f"\n  ERROR batch {i}: {e}")
    print(f"  ✓ {table}: {inserted} records")


def seed_reservoirs(client: Client, path: str):
    df = pd.read_csv(path)
    print(f"\nSeeding reservoirs ({len(df):,} rows)...")

    # Add PostGIS WKT geometry
    records = []
    for _, row in df.iterrows():
        r = row.to_dict()
        r["geom"] = f"POINT({row['lon']} {row['lat']})"
        # Ensure id is string
        r["id"] = str(r["id"])
        # Convert year to int if possible
        if "year" in r and pd.notna(r["year"]):
            r["year"] = int(r["year"])
        records.append(r)

    batch_upsert(client, "reservoirs", records)


def seed_fill_data(client: Client, path: str):
    df = pd.read_csv(path)
    # Only insert rows with valid fill data
    df = df.dropna(subset=["reservoir_id", "date"])
    print(f"\nSeeding fill_data ({len(df):,} rows)...")

    records = df.to_dict("records")
    for r in records:
        r["reservoir_id"] = str(r["reservoir_id"])

    batch_upsert(client, "fill_data", records)


def seed_seasonal(client: Client, path: str):
    df = pd.read_csv(path)
    print(f"\nSeeding seasonal_profiles ({len(df):,} rows)...")
    records = df.to_dict("records")
    for r in records:
        r["reservoir_id"] = str(r["reservoir_id"])
    batch_upsert(client, "seasonal_profiles", records)


def seed_clusters(client: Client, path: str):
    df = pd.read_csv(path)
    print(f"\nSeeding clusters ({len(df):,} rows)...")

    records = []
    for _, row in df.iterrows():
        r = row.to_dict()
        r["depot_geom"] = f"POINT({row['depot_lon']} {row['depot_lat']})"
        records.append(r)

    batch_upsert(client, "clusters", records)


def main():
    parser = argparse.ArgumentParser(description="Seed Supabase with CRESIP data")
    parser.add_argument("--reservoirs", required=True)
    parser.add_argument("--timeseries", required=True)
    parser.add_argument("--seasonal", required=True)
    parser.add_argument("--revenue", default=None)
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--url", required=True, help="Supabase project URL")
    parser.add_argument("--key", required=True, help="Supabase service_role key")
    parser.add_argument("--schema-only", action="store_true",
                        help="Only output schema SQL, don't insert data")
    args = parser.parse_args()

    # Validate files
    for p in [args.reservoirs, args.timeseries, args.seasonal, args.clusters]:
        if not Path(p).exists():
            print(f"ERROR: {p} not found")
            sys.exit(1)

    client = connect(args.url, args.key)

    # Always output schema SQL for manual execution
    create_tables_via_dashboard_sql(client)

    if args.schema_only:
        print("Schema SQL written. Run it in Supabase SQL editor then re-run without --schema-only")
        return

    print("\nStarting data seeding...")
    seed_reservoirs(client, args.reservoirs)
    seed_seasonal(client, args.seasonal)
    seed_clusters(client, args.clusters)
    seed_fill_data(client, args.timeseries)  # largest table, last

    print("\n✓ Seeding complete")


if __name__ == "__main__":
    main()
