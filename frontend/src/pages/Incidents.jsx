import { useMemo, useState } from 'react'
import { Siren, Search } from 'lucide-react'
import PageContainer, { PageHeader } from '../components/layout/PageContainer'
import IncidentCard from '../components/incidents/IncidentCard'
import EmptyState from '../components/common/EmptyState'
import ErrorState from '../components/common/ErrorState'
import { Skeleton } from '../components/common/Loading'
import { useIncidents } from '../hooks/useIncidents'

const FILTERS = [
  { key: 'ALL', label: 'All' },
  { key: 'LOW_RISK', label: 'Low' },
  { key: 'SUSPICIOUS', label: 'Suspicious' },
  { key: 'HIGH_SPOOF_RISK', label: 'High Risk' },
]

export default function Incidents() {
  const { incidents, loading, error, refresh } = useIncidents()
  const [filter, setFilter] = useState('ALL')
  const [query, setQuery] = useState('')

  const filtered = useMemo(() => {
    return incidents
      .filter((i) => (filter === 'ALL' ? true : i.risk_level === filter))
      .filter((i) =>
        query ? i.incident_id.toLowerCase().includes(query.toLowerCase()) : true,
      )
      .sort((a, b) => b.created_at - a.created_at)
  }, [incidents, filter, query])

  return (
    <PageContainer>
      <PageHeader
        title="Incident Center"
        description="Saved detection incidents from live monitoring. These are probabilistic risk records, not legal findings."
      />

      <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap gap-1.5">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                filter === f.key
                  ? 'bg-cyan-500/15 text-cyan-200 ring-1 ring-inset ring-cyan-500/30'
                  : 'bg-white/5 text-slate-400 hover:bg-white/10'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
        <div className="relative sm:w-64">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search incident ID…"
            className="input-base pl-9"
          />
        </div>
      </div>

      {error && <ErrorState title="Could not load incidents" message={error} onRetry={refresh} />}

      {!error && loading && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-44" />
          ))}
        </div>
      )}

      {!error && !loading && filtered.length === 0 && (
        <EmptyState
          icon={Siren}
          title="No incidents yet"
          description="Live-detection incidents are saved here when suspicious activity is detected and evidence is exported."
        />
      )}

      {!error && !loading && filtered.length > 0 && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {filtered.map((inc) => (
            <IncidentCard key={inc.incident_id} incident={inc} />
          ))}
        </div>
      )}
    </PageContainer>
  )
}
