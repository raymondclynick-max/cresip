'use client'

import { useEffect, useRef, useState } from 'react'
import type { Reservoir, Cluster } from '@/lib/supabase'

interface GlobeProps {
  reservoirs: Reservoir[]
  clusters: Cluster[]
  selectedCluster: Cluster | null
  onClusterClick: (cl: Cluster) => void
}

export default function GlobeComponent({ reservoirs, clusters, selectedCluster, onClusterClick }: GlobeProps) {
  const globeEl = useRef<any>(null)
  const [GlobeLib, setGlobeLib] = useState<any>(null)

  // Dynamically import Globe on client only
  useEffect(() => {
    import('react-globe.gl').then(mod => setGlobeLib(() => mod.default))
  }, [])

  // Auto-rotate and initial POV once ready
  const handleGlobeReady = () => {
    if (!globeEl.current) return
    globeEl.current.controls().autoRotate = true
    globeEl.current.controls().autoRotateSpeed = 0.3
    globeEl.current.pointOfView({ lat: 20, lng: 0, altitude: 2.2 })
  }

  // Fly to selected cluster
  useEffect(() => {
    if (!globeEl.current || !selectedCluster) return
    globeEl.current.controls().autoRotate = false
    globeEl.current.pointOfView(
      { lat: selectedCluster.depot_lat, lng: selectedCluster.depot_lon, altitude: 1.4 },
      1000
    )
  }, [selectedCluster])

  const resPoints = reservoirs.map(r => ({
    lat: r.lat,
    lng: r.lon,
    size: Math.max(0.15, Math.min(0.8, Math.log10((r.cap_m3 || 1e6) / 1e6) * 0.25)),
    color: r.cluster_label !== null && r.cluster_label >= 0 ? '#00c7ffaa' : '#ffffff22',
    label: r.name || r.country,
    type: 'reservoir',
    data: r,
  }))

  const depotPoints = clusters.map(cl => ({
    lat: cl.depot_lat,
    lng: cl.depot_lon,
    size: selectedCluster?.id === cl.id ? 1.0 : 0.6,
    color: selectedCluster?.id === cl.id ? '#ffffff' : cl.viable ? '#00ffa3' : '#ff7b4e',
    label: `${cl.region} — $${(cl.annual_revenue_usd / 1e6).toFixed(1)}M/yr`,
    type: 'depot',
    data: cl,
  }))

  if (!GlobeLib) return null

  const Globe = GlobeLib

  return (
    <Globe
      ref={globeEl}
      width={typeof window !== 'undefined' ? window.innerWidth : 1920}
      height={typeof window !== 'undefined' ? window.innerHeight : 1080}
      backgroundColor="#050810"
      globeImageUrl="//unpkg.com/three-globe/example/img/earth-dark.jpg"
      atmosphereColor="#1a4a6b"
      atmosphereAltitude={0.15}
      pointsData={[...resPoints, ...depotPoints]}
      pointLat="lat"
      pointLng="lng"
      pointAltitude={0.001}
      pointRadius="size"
      pointColor="color"
      pointLabel="label"
      onPointClick={(point: any) => {
        if (point.type === 'depot') onClusterClick(point.data)
      }}
      onGlobeReady={handleGlobeReady}
    />
  )
}
