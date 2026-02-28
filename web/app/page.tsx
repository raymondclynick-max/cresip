'use client'

import { useEffect, useState, useRef } from 'react'
import { supabase, type Reservoir, type Cluster } from '@/lib/supabase'
import dynamic from 'next/dynamic'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts'

const Globe = dynamic(() => import('@/components/Globe'), { ssr: false })

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

function fmt(n: number, decimals = 1) {
  if (n >= 1e9) return `$${(n/1e9).toFixed(decimals)}B`
  if (n >= 1e6) return `$${(n/1e6).toFixed(decimals)}M`
  return `$${n.toLocaleString()}`
}

function fmtVoid(m3: number) {
  if (m3 >= 1e9) return `${(m3/1e9).toFixed(2)} Bm³`
  if (m3 >= 1e6) return `${(m3/1e6).toFixed(1)} Mm³`
  return `${m3.toLocaleString()} m³`
}

export default function Dashboard() {
  const [reservoirs, setReservoirs] = useState<Reservoir[]>([])
  const [clusters, setClusters] = useState<Cluster[]>([])
  const [selectedCluster, setSelectedCluster] = useState<Cluster | null>(null)
  const [seasonal, setSeasonal] = useState<{month:number,mean_fill_pct:number}[]>([])
  const [loading, setLoading] = useState(true)
  const [globeReady, setGlobeReady] = useState(false)

  useEffect(() => {
    async function load() {
      const [resResult, clResult] = await Promise.all([
        supabase.from('reservoirs').select('id,name,country,lat,lon,cap_m3,dist_coast_km,cluster_label').limit(3000),
        supabase.from('clusters').select('*').order('rank')
      ])
      if (resResult.data) setReservoirs(resResult.data)
      if (clResult.data) setClusters(clResult.data)
      setLoading(false)
      setTimeout(() => setGlobeReady(true), 300)
    }
    load()
  }, [])

  async function selectCluster(cl: Cluster) {
    setSelectedCluster(cl)
    const ids = cl.reservoir_ids ? cl.reservoir_ids.split(',') : []
    if (ids.length === 0) return
    const { data } = await supabase
      .from('seasonal_profiles')
      .select('month,mean_fill_pct')
      .in('reservoir_id', ids.slice(0, 20))
    if (data) {
      const byMonth: Record<number, number[]> = {}
      data.forEach(r => {
        if (!byMonth[r.month]) byMonth[r.month] = []
        byMonth[r.month].push(r.mean_fill_pct)
      })
      setSeasonal(Array.from({length:12}, (_,i) => ({
        month: i+1,
        mean_fill_pct: byMonth[i+1] ? byMonth[i+1].reduce((a,b)=>a+b,0)/byMonth[i+1].length : 0
      })))
    }
  }

  const totalRevenue = clusters.filter(c=>c.viable).reduce((s,c)=>s+c.annual_revenue_usd,0)
  const viableCount = clusters.filter(c=>c.viable).length
  const totalVoid = clusters.reduce((s,c)=>s+(c.annual_void_m3||0),0)

  return (
    <div className="min-h-screen bg-[#050810] text-white font-mono overflow-hidden">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-3 border-b border-white/5 bg-[#050810]/90 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="text-cyan-400 text-xs tracking-[0.3em] uppercase">CRESIP</span>
          <span className="text-white/20 text-xs">|</span>
          <span className="text-white/40 text-xs tracking-widest">Coastal Reservoir Intelligence Platform</span>
        </div>
        <div className="flex items-center gap-6 text-xs text-white/30">
          <span>{reservoirs.length.toLocaleString()} RESERVOIRS</span>
          <span>{clusters.length} CLUSTERS</span>
          <span className="text-cyan-400/70">2015–2024 · JRC GSW</span>
        </div>
      </header>

      {/* Globe */}
      <div className="fixed inset-0 top-10">
        {globeReady && (
          <Globe
            reservoirs={reservoirs}
            clusters={clusters}
            selectedCluster={selectedCluster}
            onClusterClick={selectCluster}
          />
        )}
      </div>

      {/* Left panel — KPIs */}
      <div className="fixed left-4 top-16 bottom-4 w-64 flex flex-col gap-3 z-10 pointer-events-none">
        <div className="bg-[#0a0f1a]/80 backdrop-blur border border-white/5 p-4 rounded-sm pointer-events-auto">
          <div className="text-white/30 text-[10px] tracking-[0.25em] uppercase mb-1">Total Viable Revenue</div>
          <div className="text-2xl text-cyan-400 font-light">{fmt(totalRevenue)}<span className="text-sm text-white/30">/yr</span></div>
          <div className="text-white/20 text-[10px] mt-1">{viableCount} viable clusters</div>
        </div>

        <div className="bg-[#0a0f1a]/80 backdrop-blur border border-white/5 p-4 rounded-sm pointer-events-auto">
          <div className="text-white/30 text-[10px] tracking-[0.25em] uppercase mb-1">Total Void Capacity</div>
          <div className="text-2xl text-emerald-400 font-light">{fmtVoid(totalVoid)}<span className="text-sm text-white/30">/yr</span></div>
          <div className="text-white/20 text-[10px] mt-1">mean fill 4.1% — significant opportunity</div>
        </div>

        <div className="bg-[#0a0f1a]/80 backdrop-blur border border-white/5 p-4 rounded-sm pointer-events-auto">
          <div className="text-white/30 text-[10px] tracking-[0.25em] uppercase mb-2">Price per m³</div>
          <div className="text-lg text-white/70">$0.50 <span className="text-xs text-white/30">base case</span></div>
          <div className="text-white/20 text-[10px] mt-1">279,960 satellite observations · 2015–2024</div>
        </div>

        {/* Top clusters list */}
        <div className="bg-[#0a0f1a]/80 backdrop-blur border border-white/5 p-3 rounded-sm pointer-events-auto flex-1 overflow-auto">
          <div className="text-white/30 text-[10px] tracking-[0.25em] uppercase mb-2">Top Clusters</div>
          <div className="flex flex-col gap-1">
            {clusters.slice(0, 12).map(cl => (
              <button
                key={cl.id}
                onClick={() => selectCluster(cl)}
                className={`text-left px-2 py-1.5 rounded-sm text-xs transition-all border ${
                  selectedCluster?.id === cl.id
                    ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-300'
                    : 'border-transparent text-white/50 hover:bg-white/5 hover:text-white/80'
                }`}
              >
                <div className="flex justify-between items-center">
                  <span className="font-medium">{cl.region}</span>
                  <span className={cl.viable ? 'text-emerald-400' : 'text-orange-400'}>
                    {fmt(cl.annual_revenue_usd)}
                  </span>
                </div>
                <div className="text-white/30 text-[10px]">{cl.n_reservoirs} res · {cl.payback_years?.toFixed(1)}yr payback</div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Right panel — selected cluster detail */}
      {selectedCluster && (
        <div className="fixed right-4 top-16 bottom-4 w-72 flex flex-col gap-3 z-10">
          <div className="bg-[#0a0f1a]/90 backdrop-blur border border-cyan-500/20 p-4 rounded-sm">
            <div className="flex justify-between items-start mb-3">
              <div>
                <div className="text-cyan-400 text-sm font-medium">{selectedCluster.region}</div>
                <div className="text-white/30 text-[10px]">Cluster #{selectedCluster.rank} · {selectedCluster.countries}</div>
              </div>
              <button onClick={() => setSelectedCluster(null)} className="text-white/20 hover:text-white/60 text-xs">✕</button>
            </div>

            <div className="grid grid-cols-2 gap-2 mb-3">
              {[
                ['Revenue', fmt(selectedCluster.annual_revenue_usd) + '/yr', 'text-cyan-400'],
                ['Void Cap', fmtVoid(selectedCluster.annual_void_m3||0), 'text-white/70'],
                ['Payback', `${selectedCluster.payback_years?.toFixed(1)}yr`, 'text-white/70'],
                ['Reservoirs', selectedCluster.n_reservoirs, 'text-white/70'],
                ['Depot dist', `${selectedCluster.mean_depot_distance_km?.toFixed(0)}km`, 'text-white/70'],
                ['Fill corr', `${((selectedCluster.fill_correlation||0)*100).toFixed(0)}%`, 'text-white/70'],
              ].map(([label, val, color]) => (
                <div key={label as string} className="bg-white/3 p-2 rounded-sm">
                  <div className="text-white/30 text-[10px] uppercase tracking-wider">{label}</div>
                  <div className={`text-sm font-medium ${color}`}>{val}</div>
                </div>
              ))}
            </div>

            <div className={`text-[10px] px-2 py-1 rounded-sm text-center tracking-wider ${
              selectedCluster.viable
                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                : 'bg-orange-500/10 text-orange-400 border border-orange-500/20'
            }`}>
              {selectedCluster.viable ? '● COMMERCIALLY VIABLE' : '○ REVIEW REQUIRED'}
            </div>
          </div>

          {seasonal.length > 0 && (
            <div className="bg-[#0a0f1a]/90 backdrop-blur border border-white/5 p-4 rounded-sm flex-1">
              <div className="text-white/30 text-[10px] tracking-[0.25em] uppercase mb-3">Seasonal Fill Profile</div>
              <ResponsiveContainer width="100%" height={140}>
                <AreaChart data={seasonal.map(s => ({ month: MONTHS[s.month-1], fill: +s.mean_fill_pct.toFixed(1) }))}>
                  <defs>
                    <linearGradient id="fillGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="month" tick={{fill:'#ffffff30', fontSize:9}} axisLine={false} tickLine={false}/>
                  <YAxis tick={{fill:'#ffffff30', fontSize:9}} axisLine={false} tickLine={false} width={28}/>
                  <Tooltip
                    contentStyle={{background:'#0a0f1a',border:'1px solid #ffffff10',borderRadius:'2px',fontSize:'11px'}}
                    labelStyle={{color:'#ffffff60'}}
                    itemStyle={{color:'#06b6d4'}}
                  />
                  <Area type="monotone" dataKey="fill" stroke="#06b6d4" strokeWidth={1.5} fill="url(#fillGrad)"/>
                </AreaChart>
              </ResponsiveContainer>

              <div className="text-white/30 text-[10px] tracking-[0.25em] uppercase mt-3 mb-2">Revenue by Cluster</div>
              <ResponsiveContainer width="100%" height={100}>
                <BarChart data={clusters.slice(0,8).map(c => ({ name: c.region.slice(0,8), rev: +(c.annual_revenue_usd/1e6).toFixed(1) }))}>
                  <XAxis dataKey="name" tick={{fill:'#ffffff30',fontSize:8}} axisLine={false} tickLine={false}/>
                  <YAxis hide />
                  <Tooltip
                    contentStyle={{background:'#0a0f1a',border:'1px solid #ffffff10',borderRadius:'2px',fontSize:'10px'}}
                    labelStyle={{color:'#ffffff60'}}
                    itemStyle={{color:'#10b981'}}
                    formatter={(v:number|undefined) => [`$${v??0}M`, 'Revenue' as const]}
                  />
                  <Bar dataKey="rev" radius={[2,2,0,0]}>
                    {clusters.slice(0,8).map((c,i) => (
                      <Cell key={i} fill={c.viable ? '#10b981' : '#f97316'} opacity={0.7}/>
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {loading && (
        <div className="fixed inset-0 bg-[#050810] flex items-center justify-center z-50">
          <div className="text-center">
            <div className="text-cyan-400 text-xs tracking-[0.4em] uppercase animate-pulse mb-2">CRESIP</div>
            <div className="text-white/20 text-xs">Loading satellite data...</div>
          </div>
        </div>
      )}
    </div>
  )
}
