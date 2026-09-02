import { Link, useParams } from 'react-router-dom'
import {
  ArrowLeft,
  Download,
  Fingerprint,
  Clock,
  Activity,
  Cpu,
  FileText,
  Siren,
} from 'lucide-react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from 'recharts'
import PageContainer from '../components/layout/PageContainer'
import ErrorState from '../components/common/ErrorState'
import { PageLoader } from '../components/common/Loading'
import TakeActionPanel from '../components/incidents/TakeActionPanel'
import { useIncident } from '../hooks/useIncidents'
import { riskMeta, modelStatusMeta } from '../utils/risk'
import { formatClockFromEpoch, formatDuration, downloadJson } from '../utils/format'

export default function IncidentDetails() {
  const { id } = useParams()
  const { incident, loading, error, refresh } = useIncident(id)

  if (loading) return <PageLoader label="Loading incident…" />
  if (error || !incident)
    return (
      <PageContainer>
        <BackLink />
        <ErrorState
          title="Incident not found"
          message={error || 'This incident may have expired or been removed.'}
          onRetry={refresh}
        />
      </PageContainer>
    )

  const meta = riskMeta(incident.risk_level)
  const modelMeta = modelStatusMeta(incident.model_status)
  const timeline = (incident.timeline || []).map((w) => ({
    t: w.start_time,
    score: Math.round((w.fake_probability ?? 0) * 100),
    level: w.risk_level,
  }))

  const fields = [
    { label: 'Incident ID', value: incident.incident_id, mono: true },
    { label: 'Session ID', value: incident.session_id, mono: true },
    { label: 'Created', value: formatClockFromEpoch(incident.created_at) },
    { label: 'Duration', value: formatDuration(incident.duration_seconds) },
    { label: 'Risk level', value: meta.label, color: meta.color },
    { label: 'Model status', value: modelMeta.label, color: modelMeta.color },
  ]

  return (
    <PageContainer>
      <BackLink />

      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <span
            className="flex h-12 w-12 items-center justify-center rounded-xl"
            style={{ background: `${meta.color}1a`, color: meta.color }}
          >
            <Siren size={24} />
          </span>
          <div>
            <h2 className="text-xl font-bold text-white">{incident.incident_id}</h2>
            <p className="text-sm text-slate-400">{formatClockFromEpoch(incident.created_at)}</p>
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => downloadJson(`${incident.incident_id}.json`, incident)}
            className="btn-ghost"
          >
            <Download size={15} /> Download evidence JSON
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        <div className="space-y-5 lg:col-span-2">
          {/* Overview */}
          <section className="panel p-5">
            <SectionTitle icon={Fingerprint} title="Incident overview" />
            <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {fields.map((f) => (
                <div key={f.label} className="rounded-lg bg-ink-950/50 px-3 py-2.5">
                  <dt className="text-[10px] uppercase tracking-wider text-slate-500">{f.label}</dt>
                  <dd
                    className={`mt-0.5 truncate text-sm font-semibold ${f.mono ? 'font-mono' : ''} ${
                      f.color || 'text-slate-100'
                    }`}
                  >
                    {f.value}
                  </dd>
                </div>
              ))}
            </dl>
          </section>

          {/* Detection summary */}
          <section className="panel p-5">
            <SectionTitle icon={Activity} title="Detection summary" />
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <SummaryStat label="Avg spoof" value={`${Math.round((incident.average_spoof_score || 0) * 100)}%`} />
              <SummaryStat label="Peak spoof" value={`${Math.round((incident.peak_spoof_score || 0) * 100)}%`} tone="bad" />
              <SummaryStat label="Suspicious" value={incident.suspicious_windows ?? 0} tone="warn" />
              <SummaryStat label="Total windows" value={incident.analyzed_windows ?? 0} />
            </div>

            <div className="mt-5 h-56">
              {timeline.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={timeline} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" vertical={false} />
                    <XAxis dataKey="t" stroke="#475569" fontSize={10} tickLine={false} axisLine={false} tickFormatter={(t) => `${Math.round(t)}s`} />
                    <YAxis domain={[0, 100]} stroke="#475569" fontSize={10} tickLine={false} axisLine={false} unit="%" />
                    <Tooltip
                      contentStyle={{ background: '#0e121b', border: '1px solid rgba(148,163,184,0.2)', borderRadius: 10, fontSize: 12 }}
                      labelFormatter={(t) => `${Number(t).toFixed(1)}s`}
                      formatter={(v) => [`${v}%`, 'Spoof probability']}
                    />
                    <ReferenceLine y={60} stroke="#fbbf24" strokeDasharray="4 4" strokeOpacity={0.5} />
                    <ReferenceLine y={80} stroke="#fb7185" strokeDasharray="4 4" strokeOpacity={0.6} />
                    <Line type="monotone" dataKey="score" stroke="#22d3ee" strokeWidth={2} dot={{ r: 2, fill: '#22d3ee' }} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <p className="flex h-full items-center justify-center text-sm text-slate-600">
                  No window timeline available for this record.
                </p>
              )}
            </div>
          </section>

          {/* Model info */}
          <section className="panel p-5">
            <SectionTitle icon={Cpu} title="Model information" />
            <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <InfoRow label="Model status" value={modelMeta.label} />
              <InfoRow label="Model SHA-256" value={incident.model_sha256 || 'Not available'} mono={!!incident.model_sha256} />
              <InfoRow label="Analyzed windows" value={incident.analyzed_windows ?? '—'} />
              <InfoRow label="Suspicious windows" value={incident.suspicious_windows ?? '—'} />
            </dl>
            {incident.social_engineering_indicators?.length > 0 && (
              <div className="mt-4">
                <p className="eyebrow mb-2">Social engineering indicators</p>
                <div className="flex flex-wrap gap-2">
                  {incident.social_engineering_indicators.map((ind) => (
                    <span key={ind} className="rounded-md bg-amber-500/10 px-2.5 py-1 text-xs font-medium text-amber-200 ring-1 ring-inset ring-amber-500/30">
                      {ind.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </section>

          {/* Disclaimer / evidence */}
          <section className="panel p-5">
            <SectionTitle icon={FileText} title="Evidence & disclaimer" />
            {incident.disclaimer && (
              <p className="rounded-lg border border-line bg-ink-950/40 p-3 text-xs leading-relaxed text-slate-400">
                {incident.disclaimer}
              </p>
            )}
          </section>
        </div>

        {/* Right: take action */}
        <div className="space-y-5">
          <div className="panel p-5">
            <div className="mb-3 flex items-center gap-2" style={{ color: meta.color }}>
              <Clock size={16} />
              <h3 className="text-sm font-semibold">Response guidance</h3>
            </div>
            <p className="text-xs leading-relaxed text-slate-400">{meta.description}</p>
          </div>
          <TakeActionPanel
            onExportEvidence={() => downloadJson(`${incident.incident_id}.json`, incident)}
            evidenceBusy={false}
          />
        </div>
      </div>
    </PageContainer>
  )
}

function BackLink() {
  return (
    <Link to="/incidents" className="mb-4 inline-flex items-center gap-1.5 text-sm text-slate-400 transition hover:text-slate-200">
      <ArrowLeft size={15} /> Back to Incident Center
    </Link>
  )
}

function SectionTitle({ icon: Icon, title }) {
  return (
    <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-200">
      <Icon size={16} className="text-cyan-300" /> {title}
    </h3>
  )
}

function SummaryStat({ label, value, tone }) {
  const color = tone === 'bad' ? 'text-rose-400' : tone === 'warn' ? 'text-amber-400' : 'text-slate-100'
  return (
    <div className="rounded-lg bg-ink-950/50 py-3 text-center">
      <p className={`font-mono text-xl font-bold ${color}`}>{value}</p>
      <p className="mt-0.5 text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
    </div>
  )
}

function InfoRow({ label, value, mono }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg bg-ink-950/50 px-3 py-2.5">
      <span className="text-xs text-slate-500">{label}</span>
      <span className={`truncate text-sm font-medium text-slate-200 ${mono ? 'font-mono text-xs' : ''}`} title={value}>
        {value}
      </span>
    </div>
  )
}
