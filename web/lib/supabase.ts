import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseKey)

export type Reservoir = {
  id: string
  name: string | null
  country: string
  lat: number
  lon: number
  cap_m3: number
  dist_coast_km: number
  cluster_label: number | null
}

export type Cluster = {
  id: number
  cluster_id: number
  rank: number
  region: string
  n_reservoirs: number
  depot_lat: number
  depot_lon: number
  countries: string
  annual_void_m3: number
  annual_revenue_usd: number
  payback_years: number
  viable: boolean
  fill_correlation: number
  mean_depot_distance_km: number
}

export type SeasonalProfile = {
  reservoir_id: string
  month: number
  mean_fill_pct: number
}
