import { useEffect, useState } from 'react'
import { Globe, Phone, Wifi, FileDown, ShieldAlert, ExternalLink } from 'lucide-react'
import { getReportingResources } from '../../services/api'

const LINK_ICONS = {
  national_cyber_crime_portal: Globe,
  i4c_ncrp_suspect_report: Phone,
  sanchar_saathi_chakshu: Wifi,
}

/**
 * Official Government of India cybercrime reporting panel.
 * Never auto-submits anything — it only links users to official resources.
 */
export default function TakeActionPanel({ onExportEvidence, evidenceBusy }) {
  const [resources, setResources] = useState(null)

  useEffect(() => {
    getReportingResources()
      .then((r) => setResources(r?.resources || null))
      .catch(() => setResources(null))
  }, [])

  const links = resources
    ? ['national_cyber_crime_portal', 'i4c_ncrp_suspect_report', 'sanchar_saathi_chakshu']
        .map((k) => ({ key: k, ...resources[k] }))
        .filter((l) => l?.url)
    : []
  const helpline = resources?.cyber_crime_helpline

  return (
    <div className="rounded-xl border border-amber-500/25 bg-amber-500/[0.04] p-4">
      <div className="mb-3 flex items-center gap-2">
        <ShieldAlert size={16} className="text-amber-400" />
        <h4 className="text-sm font-semibold text-amber-200">Take action — official reporting</h4>
      </div>
      <p className="mb-3 text-xs leading-relaxed text-slate-400">
        If you suspect financial fraud, impersonation, or a voice scam, preserve the evidence and
        report through the official Government of India portals below. VigilVoice does not file
        reports automatically.
      </p>

      <div className="space-y-2">
        {links.length === 0 && (
          <p className="text-xs text-slate-500">
            Official portals: cybercrime.gov.in · sancharsaathi.gov.in
          </p>
        )}
        {links.map((l) => {
          const Icon = LINK_ICONS[l.key] || Globe
          return (
            <a
              key={l.key}
              href={l.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex items-center gap-3 rounded-lg border border-line bg-ink-950/50 px-3 py-2.5 transition hover:border-cyan-500/40 hover:bg-cyan-500/5"
            >
              <span className="flex h-8 w-8 items-center justify-center rounded-md bg-cyan-500/10 text-cyan-300">
                <Icon size={15} />
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium text-slate-200">{l.name}</span>
                <span className="block truncate text-[11px] text-slate-500">{l.purpose}</span>
              </span>
              <ExternalLink size={14} className="text-slate-600 transition group-hover:text-cyan-300" />
            </a>
          )
        })}

        {helpline && (
          <div className="flex items-center gap-3 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-md bg-rose-500/20 text-rose-300">
              <Phone size={15} />
            </span>
            <span className="flex-1 text-sm font-medium text-rose-200">
              Cyber crime emergency helpline
            </span>
            <span className="font-mono text-lg font-bold text-rose-100">{helpline.number}</span>
          </div>
        )}
      </div>

      {onExportEvidence && (
        <button onClick={onExportEvidence} disabled={evidenceBusy} className="btn-ghost mt-3 w-full">
          <FileDown size={15} />
          {evidenceBusy ? 'Preparing evidence…' : 'Save incident evidence (JSON)'}
        </button>
      )}

      <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
        Detection is probabilistic and indicates acoustic anomalies associated with synthetic
        speech. VigilVoice does not make legal accusations.
      </p>
    </div>
  )
}
