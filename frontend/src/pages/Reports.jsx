import { Link } from 'react-router-dom'
import { FileText, Download, FileJson, ShieldCheck, FlaskConical, Siren } from 'lucide-react'
import PageContainer, { PageHeader } from '../components/layout/PageContainer'
import EmptyState from '../components/common/EmptyState'
import ErrorState from '../components/common/ErrorState'
import { Skeleton } from '../components/common/Loading'
import { useIncidents } from '../hooks/useIncidents'
import { riskMeta } from '../utils/risk'
import { formatClockFromEpoch, downloadJson } from '../utils/format'

export default function Reports({ model }) {
  const { incidents, loading, error, refresh } = useIncidents()
  const evalData = model?.modelInfo?.evaluation
  const robustness = model?.modelInfo?.robustness

  return (
    <PageContainer>
      <PageHeader
        title="Reports & Evidence"
        description="Incident evidence packages, model evaluation, and robustness reports. Only data exposed by the backend is shown."
      />

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        {/* Report type cards */}
        <ReportCard
          icon={ShieldCheck}
          title="Model evaluation report"
          available={!!evalData && Object.keys(evalData).length > 0}
          detail={
            evalData?.dataset_type
              ? `Dataset: ${evalData.dataset_type}${evalData.eer != null ? ` · EER ${(evalData.eer * 100).toFixed(2)}%` : ''}`
              : 'No evaluation_report.json found on the server.'
          }
          onDownload={evalData ? () => downloadJson('evaluation_report.json', evalData) : null}
        />
        <ReportCard
          icon={FlaskConical}
          title="Robustness report"
          available={robustness?.available}
          detail={
            robustness?.available
              ? 'Noise and resampling generalization benchmark completed.'
              : 'No robustness_report.json found on the server.'
          }
          onDownload={robustness?.available ? () => downloadJson('robustness_report.json', robustness.data) : null}
        />
        <ReportCard
          icon={Siren}
          title="Incident evidence packages"
          available={incidents.length > 0}
          detail={`${incidents.length} saved incident record(s) in reports/incidents.`}
        />
      </div>

      {/* Incident evidence table */}
      <div className="panel mt-5 p-5">
        <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-200">
          <FileJson size={16} className="text-cyan-300" /> Evidence packages
        </h3>

        {error && <ErrorState title="Could not load reports" message={error} onRetry={refresh} />}

        {!error && loading && (
          <div className="space-y-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-12" />
            ))}
          </div>
        )}

        {!error && !loading && incidents.length === 0 && (
          <EmptyState
            icon={FileText}
            title="No evidence packages yet"
            description="Export evidence from a live session to generate a downloadable incident JSON package."
          />
        )}

        {!error && !loading && incidents.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-[11px] uppercase tracking-wider text-slate-500">
                <tr className="border-b border-line">
                  <th className="px-3 py-2.5">Incident</th>
                  <th className="px-3 py-2.5">Created</th>
                  <th className="px-3 py-2.5">Risk</th>
                  <th className="px-3 py-2.5">Peak</th>
                  <th className="px-3 py-2.5 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {[...incidents]
                  .sort((a, b) => b.created_at - a.created_at)
                  .map((inc) => {
                    const meta = riskMeta(inc.risk_level)
                    return (
                      <tr key={inc.incident_id} className="border-b border-line/60 last:border-0 hover:bg-white/[0.02]">
                        <td className="px-3 py-2.5">
                          <Link to={`/incidents/${inc.incident_id}`} className="font-mono text-xs font-semibold text-cyan-300 hover:underline">
                            {inc.incident_id}
                          </Link>
                        </td>
                        <td className="px-3 py-2.5 text-xs text-slate-400">{formatClockFromEpoch(inc.created_at)}</td>
                        <td className="px-3 py-2.5">
                          <span className="rounded-full px-2 py-0.5 text-[11px] font-semibold" style={{ color: meta.color, background: `${meta.color}1a` }}>
                            {meta.label}
                          </span>
                        </td>
                        <td className="px-3 py-2.5 font-mono text-xs text-slate-200">
                          {Math.round((inc.peak_spoof_score || 0) * 100)}%
                        </td>
                        <td className="px-3 py-2.5 text-right">
                          <button
                            onClick={() => downloadJson(`${inc.incident_id}.json`, inc)}
                            className="inline-flex items-center gap-1.5 rounded-md border border-line bg-white/5 px-2.5 py-1.5 text-xs text-slate-300 transition hover:border-line-strong hover:text-white"
                          >
                            <Download size={13} /> JSON
                          </button>
                        </td>
                      </tr>
                    )
                  })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </PageContainer>
  )
}

function ReportCard({ icon: Icon, title, detail, available, onDownload }) {
  return (
    <div className="panel-soft flex flex-col p-5">
      <div className="flex items-center justify-between">
        <span className={`flex h-10 w-10 items-center justify-center rounded-lg ${available ? 'bg-cyan-500/10 text-cyan-300' : 'bg-white/5 text-slate-500'}`}>
          <Icon size={20} />
        </span>
        <span
          className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
            available ? 'bg-emerald-500/10 text-emerald-300' : 'bg-white/5 text-slate-500'
          }`}
        >
          {available ? 'Available' : 'Not available'}
        </span>
      </div>
      <h4 className="mt-3 text-sm font-semibold text-slate-100">{title}</h4>
      <p className="mt-1 flex-1 text-xs leading-relaxed text-slate-500">{detail}</p>
      {onDownload && (
        <button onClick={onDownload} className="btn-ghost mt-3 w-full">
          <Download size={14} /> Download JSON
        </button>
      )}
    </div>
  )
}
