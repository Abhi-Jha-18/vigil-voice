import { Link } from 'react-router-dom'
import { ChevronRight, Siren } from 'lucide-react'
import { riskMeta } from '../../utils/risk'
import { formatClockFromEpoch } from '../../utils/format'

export default function IncidentCard({ incident }) {
  const meta = riskMeta(incident.risk_level)
  return (
    <Link
      to={`/incidents/${incident.incident_id}`}
      className="group panel-soft block p-4 transition hover:border-line-strong hover:bg-white/[0.04]"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <span
            className="flex h-10 w-10 items-center justify-center rounded-lg"
            style={{ background: `${meta.color}1a`, color: meta.color }}
          >
            <Siren size={18} />
          </span>
          <div>
            <p className="font-mono text-sm font-semibold text-slate-100">{incident.incident_id}</p>
            <p className="text-xs text-slate-500">{formatClockFromEpoch(incident.created_at)}</p>
          </div>
        </div>
        <ChevronRight size={16} className="mt-2 text-slate-600 transition group-hover:translate-x-0.5 group-hover:text-slate-300" />
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2 text-center">
        <MiniStat label="Peak spoof" value={`${Math.round((incident.peak_spoof_score || 0) * 100)}%`} color={meta.color} />
        <MiniStat label="Suspicious" value={incident.suspicious_windows ?? 0} />
        <MiniStat label="Windows" value={incident.analyzed_windows ?? 0} />
      </div>

      <div className="mt-3 flex items-center justify-between">
        <span
          className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold"
          style={{ color: meta.color, background: `${meta.color}1a` }}
        >
          <span className="h-1.5 w-1.5 rounded-full" style={{ background: meta.color }} />
          {meta.label}
        </span>
        <span className="font-mono text-[11px] text-slate-500">{incident.model_status}</span>
      </div>
    </Link>
  )
}

function MiniStat({ label, value, color }) {
  return (
    <div className="rounded-md bg-ink-950/50 py-2">
      <p className="font-mono text-sm font-bold" style={{ color: color || '#e2e8f0' }}>
        {value}
      </p>
      <p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
    </div>
  )
}
