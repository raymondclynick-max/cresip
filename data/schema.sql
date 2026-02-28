
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
    