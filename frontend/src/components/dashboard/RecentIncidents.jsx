import { Link } from 'react-router-dom'
import { Siren, ArrowRight } from 'lucide-react'
import { riskMeta } from '../../utils/risk'
import { formatClockFromEpoch } from '../../utils/format'
import EmptyState from '../common/EmptyState'

export default function RecentIncidents({ incidents, loading }) {
  const recent = [...(incidents || [])].sort((a, b) => b.created_at - a.created_at).slice(0, 5)

  return (
    <div className="panel p-5">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-200">
          <Siren size={16} className="text-rose-300" /> Recent incidents
        </h3>
        <Link to="/incidents" className="flex items-center gap-1 text-xs font-medium text-cyan-300 hover:text-cyan-200">
          View all <ArrowRight size={13} />
        </Link>
      </div>

      {loading ? (
        <div className="space-y-2">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="h-12 animate-pulse rounded-lg bg-white/5" />
          ))}
        </div>
      ) : recent.length === 0 ? (
        <EmptyState
          icon={Siren}
          title="No incidents yet"
          description="Saved live-detection incidents will appear here."
        />
      ) : (
        <div className="space-y-2">
          {recent.map((inc) => {
            const meta = riskMeta(inc.risk_level)
            return (
              <Link
                key={inc.incident_id}
                to={`/incidents/${inc.incident_id}`}
                className="flex items-center justify-between rounded-lg bg-ink-950/50 px-3 py-2.5 transition hover:bg-white/5"
              >
                <div className="flex items-center gap-3">
                  <span className="h-2 w-2 rounded-full" style={{ background: meta.color }} />
                  <div>
                    <p className="font-mono text-xs font-semibold text-slate-200">{inc.incident_id}</p>
                    <p className="text-[11px] text-slate-500">{formatClockFromEpoch(inc.created_at)}</p>
                  </div>
                </div>
                <span className="font-mono text-xs font-semibold" style={{ color: meta.color }}>
                  {Math.round((inc.peak_spoof_score || 0) * 100)}%
                </span>
              </Link>
            )
          })}
        </div>
      )}
    </div>
  )
}
