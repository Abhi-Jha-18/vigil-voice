import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import {
  Siren,
  AlertTriangle,
  ShieldAlert,
  Gauge,
  Radio,
  UploadCloud,
  Cpu,
  BarChart3,
  PieChart as PieIcon,
} from 'lucide-react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts'
import PageContainer, { PageHeader } from '../components/layout/PageContainer'
import StatCard from '../components/dashboard/StatCard'
import SystemStatus from '../components/dashboard/SystemStatus'
import RecentIncidents from '../components/dashboard/RecentIncidents'
import { useIncidents } from '../hooks/useIncidents'
import { riskMeta } from '../utils/risk'

export default function Dashboard({ model }) {
  const { incidents, loading } = useIncidents()

  const stats = useMemo(() => {
    const total = incidents.length
    const high = incidents.filter((i) => i.risk_level === 'HIGH_SPOOF_RISK').length
    const susp = incidents.filter((i) => i.risk_level === 'SUSPICIOUS').length
    const avgPeak = total
      ? Math.round((incidents.reduce((a, i) => a + (i.peak_spoof_score || 0), 0) / total) * 100)
      : 0
    return { total, high, susp, avgPeak }
  }, [incidents])

  const distribution = useMemo(() => {
    const counts = { LOW_RISK: 0, SUSPICIOUS: 0, HIGH_SPOOF_RISK: 0 }
    incidents.forEach((i) => {
      counts[i.risk_level] = (counts[i.risk_level] || 0) + 1
    })
    return [
      { name: 'Low Risk', value: counts.LOW_RISK, color: riskMeta('LOW_RISK').color },
      { name: 'Suspicious', value: counts.SUSPICIOUS, color: riskMeta('SUSPICIOUS').color },
      { name: 'High Spoof', value: counts.HIGH_SPOOF_RISK, color: riskMeta('HIGH_SPOOF_RISK').color },
    ]
  }, [incidents])

  const activity = useMemo(() => {
    // Group saved incidents by day for a real (not fabricated) activity chart.
    const byDay = {}
    incidents.forEach((i) => {
      const d = new Date(i.created_at * 1000)
      const key = d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
      byDay[key] = (byDay[key] || 0) + 1
    })
    return Object.entries(byDay)
      .map(([day, count]) => ({ day, count }))
      .slice(-7)
  }, [incidents])

  return (
    <PageContainer>
      <PageHeader
        title="Security Command Center"
        description="Real-time overview of the VigilVoice detection engine, model health, and saved voice-spoofing incidents."
        actions={
          <>
            <Link to="/live" className="btn-primary">
              <Radio size={15} /> Live Monitor
            </Link>
            <Link to="/analyze" className="btn-ghost">
              <UploadCloud size={15} /> Analyze Audio
            </Link>
          </>
        }
      />

      {/* Stat cards — derived only from real saved incidents */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard icon={Siren} label="Saved incidents" value={stats.total} sub="From evidence exports" accent="#22d3ee" loading={loading} />
        <StatCard icon={AlertTriangle} label="Suspicious" value={stats.susp} accent="#fbbf24" loading={loading} />
        <StatCard icon={ShieldAlert} label="High-risk events" value={stats.high} accent="#fb7185" loading={loading} />
        <StatCard icon={Gauge} label="Avg peak spoof" value={`${stats.avgPeak}%`} accent="#a78bfa" loading={loading} />
      </div>

      <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-3">
        {/* Charts */}
        <div className="space-y-5 lg:col-span-2">
          <div className="panel p-5">
            <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-200">
              <BarChart3 size={16} className="text-cyan-300" /> Incident activity
              <span className="text-[11px] font-normal text-slate-500">· saved records by day</span>
            </h3>
            {activity.length > 0 ? (
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={activity} margin={{ top: 4, right: 8, left: -22, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" vertical={false} />
                    <XAxis dataKey="day" stroke="#475569" fontSize={11} tickLine={false} axisLine={false} />
                    <YAxis allowDecimals={false} stroke="#475569" fontSize={11} tickLine={false} axisLine={false} />
                    <Tooltip
                      cursor={{ fill: 'rgba(148,163,184,0.06)' }}
                      contentStyle={{ background: '#0e121b', border: '1px solid rgba(148,163,184,0.2)', borderRadius: 10, fontSize: 12 }}
                      formatter={(v) => [v, 'Incidents']}
                    />
                    <Bar dataKey="count" fill="#22d3ee" radius={[6, 6, 0, 0]} maxBarSize={48} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <ChartEmpty label="No saved incidents yet — run live monitoring and export evidence to see activity." />
            )}
          </div>

          <div className="panel p-5">
            <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-200">
              <PieIcon size={16} className="text-cyan-300" /> Risk distribution
            </h3>
            {stats.total > 0 ? (
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={distribution.filter((d) => d.value > 0)} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80} paddingAngle={3}>
                      {distribution.map((d) => (
                        <Cell key={d.name} fill={d.color} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ background: '#0e121b', border: '1px solid rgba(148,163,184,0.2)', borderRadius: 10, fontSize: 12 }} />
                    <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <ChartEmpty label="Risk distribution appears once incidents are recorded." />
            )}
          </div>
        </div>

        {/* Right column */}
        <div className="space-y-5">
          <SystemStatus model={model} />
          <RecentIncidents incidents={incidents} loading={loading} />
          <Link to="/model" className="panel-soft flex items-center gap-3 p-4 transition hover:bg-white/[0.04]">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-500/10 text-indigo-300">
              <Cpu size={18} />
            </span>
            <div className="flex-1">
              <p className="text-sm font-semibold text-slate-200">Model &amp; AI Center</p>
              <p className="text-[11px] text-slate-500">Architecture, metrics &amp; integrity</p>
            </div>
          </Link>
        </div>
      </div>
    </PageContainer>
  )
}

function ChartEmpty({ label }) {
  return (
    <div className="flex h-56 flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-line text-center">
      <BarChart3 size={22} className="text-slate-600" />
      <p className="max-w-xs text-xs text-slate-500">{label}</p>
    </div>
  )
}
