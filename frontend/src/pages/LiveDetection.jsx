import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Play, Square, Siren, Download, CheckCircle2, ArrowRight } from 'lucide-react'
import PageContainer, { PageHeader } from '../components/layout/PageContainer'
import RiskMeter from '../components/live/RiskMeter'
import AudioVisualizer from '../components/live/AudioVisualizer'
import ConnectionStatus from '../components/live/ConnectionStatus'
import DetectionTimeline from '../components/live/DetectionTimeline'
import TakeActionPanel from '../components/incidents/TakeActionPanel'
import ErrorState from '../components/common/ErrorState'
import Badge from '../components/common/Badge'
import { useLiveDetection } from '../hooks/useLiveDetection'
import { riskMeta } from '../utils/risk'
import { formatDuration, downloadJson } from '../utils/format'

const SE_FLAGS = [
  { value: 'CREDENTIAL_REQUEST', label: 'Password / OTP' },
  { value: 'FINANCIAL_REQUEST', label: 'Money / UPI' },
  { value: 'IMPERSONATION', label: 'Authority / Bank' },
  { value: 'URGENCY_PRESSURE', label: 'Urgent threat' },
]

export default function LiveDetection() {
  const live = useLiveDetection()
  const [evidenceBusy, setEvidenceBusy] = useState(false)
  const [exportedId, setExportedId] = useState(null)
  const [activeFlags, setActiveFlags] = useState([])

  const isActive = live.connectionState === 'connected'
  const meta = riskMeta(live.riskLevel)

  const handleExport = async () => {
    setEvidenceBusy(true)
    const res = await live.exportEvidence()
    setEvidenceBusy(false)
    if (res.ok) {
      downloadJson(`${res.incidentId}_evidence.json`, res.incident)
      setExportedId(res.incidentId)
    } else {
      alert(res.error)
    }
  }

  const toggleFlag = (value) => {
    setActiveFlags((prev) => {
      const has = prev.includes(value)
      if (!has) live.addIndicator(value)
      return has ? prev.filter((v) => v !== value) : [...prev, value]
    })
  }

  return (
    <PageContainer>
      <PageHeader
        title="Live Voice Monitor"
        description="Stream a live call or microphone feed. VigilVoice analyzes rolling 2.5s windows in real time and raises a risk alert without making accusations."
        actions={<ConnectionStatus state={live.connectionState} />}
      />

      {live.error && (
        <div className="mb-5">
          <ErrorState
            title="Could not start monitoring"
            message={live.error}
            onRetry={live.start}
          />
        </div>
      )}

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
        {/* Left: risk + controls */}
        <div className="space-y-5">
          <RiskMeter score={live.riskScore} level={live.riskLevel} sublabel={meta.description} />

          <div className="panel-soft p-4">
            <div className="mb-3 grid grid-cols-3 gap-2 text-center">
              <Stat label="Windows" value={live.analyzedWindows} />
              <Stat label="Peak" value={`${live.peakScore}%`} tone={live.peakScore >= 80 ? 'bad' : live.peakScore >= 60 ? 'warn' : 'ok'} />
              <Stat label="Avg" value={`${live.avgScore}%`} />
            </div>

            <div className="mb-3 flex items-center justify-between rounded-lg bg-ink-950/60 px-3 py-2 font-mono text-sm">
              <span className="text-slate-500">Elapsed</span>
              <span className="text-slate-200">{formatDuration(live.elapsed)}</span>
            </div>

            <div className="flex gap-2">
              {!isActive ? (
                <button onClick={live.start} disabled={live.connectionState === 'connecting'} className="btn-ok flex-1">
                  <Play size={16} />
                  {live.connectionState === 'connecting' ? 'Connecting…' : 'Start Monitoring'}
                </button>
              ) : (
                <button onClick={live.stop} className="btn-bad flex-1">
                  <Square size={15} /> Stop Monitoring
                </button>
              )}
            </div>

            <div className="mt-4">
              <p className="eyebrow mb-2">Social engineering indicators</p>
              <div className="flex flex-wrap gap-2">
                {SE_FLAGS.map((f) => (
                  <button
                    key={f.value}
                    onClick={() => toggleFlag(f.value)}
                    disabled={!isActive}
                    className={`rounded-lg border px-2.5 py-1.5 text-xs font-medium transition disabled:opacity-40 ${
                      activeFlags.includes(f.value)
                        ? 'border-amber-400/50 bg-amber-500/15 text-amber-200'
                        : 'border-line bg-white/5 text-slate-400 hover:border-line-strong'
                    }`}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
              <p className="mt-2 text-[11px] text-slate-500">
                Flags are attached to the session evidence; they are manual observations, not
                automated verdicts.
              </p>
            </div>
          </div>

          {live.incident && (
            <div className="rounded-xl border border-rose-500/30 bg-rose-500/[0.06] p-4">
              <div className="mb-2 flex items-center gap-2">
                <Siren size={16} className="text-rose-400" />
                <h4 className="text-sm font-semibold text-rose-200">Potential voice spoofing detected</h4>
              </div>
              <p className="mb-3 text-xs leading-relaxed text-slate-400">
                Repeated windows showed elevated indicators associated with synthetic or manipulated
                speech. Treat requests made on this call with caution and verify identity through an
                official channel.
              </p>
              <dl className="grid grid-cols-2 gap-2 text-xs">
                <IncidentField label="Incident ID" value={live.incident.incident_id} />
                <IncidentField label="Peak score" value={`${Math.round(live.incident.peak_spoof_score * 100)}%`} danger />
                <IncidentField label="Suspicious windows" value={live.incident.suspicious_windows} />
                <IncidentField label="Model" value={live.incident.model_status} />
              </dl>
            </div>
          )}
        </div>

        {/* Center: visualizer + timeline */}
        <div className="space-y-5 xl:col-span-2">
          <AudioVisualizer analyserRef={live.analyserRef} active={isActive} />
          <DetectionTimeline points={live.levelHistory} windows={live.timeline} />
          <div className="panel-soft p-4">
            <div className="flex items-center justify-between">
              <span className="eyebrow">Evidence & reporting</span>
              {exportedId && (
                <Badge className="bg-emerald-500/10 text-emerald-300 ring-emerald-500/30">
                  <CheckCircle2 size={12} /> Evidence saved
                </Badge>
              )}
            </div>
            <div className="mt-3">
              <TakeActionPanel onExportEvidence={handleExport} evidenceBusy={evidenceBusy} />
            </div>
            {exportedId ? (
              <Link
                to={`/incidents/${exportedId}`}
                className="btn-ghost mt-3 w-full"
              >
                <Siren size={15} /> View incident {exportedId}
                <ArrowRight size={14} />
              </Link>
            ) : (
              <button
                onClick={handleExport}
                disabled={evidenceBusy || !live.sessionId}
                className="btn-ghost mt-3 w-full"
              >
                <Download size={15} /> Save evidence JSON
              </button>
            )}
          </div>
        </div>
      </div>
    </PageContainer>
  )
}

function Stat({ label, value, tone }) {
  const toneClass =
    tone === 'bad' ? 'text-rose-400' : tone === 'warn' ? 'text-amber-400' : 'text-slate-100'
  return (
    <div className="rounded-lg bg-ink-950/60 py-2.5">
      <div className={`font-mono text-lg font-bold ${toneClass}`}>{value}</div>
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
    </div>
  )
}

function IncidentField({ label, value, danger }) {
  return (
    <div className="rounded-md bg-ink-950/50 px-2.5 py-2">
      <dt className="text-[10px] uppercase tracking-wider text-slate-500">{label}</dt>
      <dd className={`mt-0.5 font-mono font-semibold ${danger ? 'text-rose-300' : 'text-slate-200'}`}>
        {value}
      </dd>
    </div>
  )
}
