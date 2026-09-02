import { useRef, useState } from 'react'
import {
  UploadCloud,
  FileAudio,
  X,
  Loader2,
  ShieldCheck,
  ShieldAlert,
  HelpCircle,
  ScanLine,
  Image as ImageIcon,
  Hash,
} from 'lucide-react'
import PageContainer, { PageHeader } from '../components/layout/PageContainer'
import ErrorState from '../components/common/ErrorState'
import EmptyState from '../components/common/EmptyState'
import { detectAudio } from '../services/api'
import { PHASE_OPTIONS, modelStatusMeta } from '../utils/risk'
import { formatBytes, pct100 } from '../utils/format'

const ACCEPT = '.wav,.mp3,.m4a,.mp4,.ogg,.flac,.webm,.aac,audio/*,video/mp4'
const STEPS = ['VAD & Prep', 'Feature Extraction', 'AI Inference', 'Decision']

export default function UploadAnalysis({ model }) {
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [phase, setPhase] = useState('phase2')
  const [forceVerdict, setForceVerdict] = useState('')
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)
  const [step, setStep] = useState(-1)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const maxMb = model?.status?.configuration?.max_upload_mb || 50

  const validate = (f) => {
    if (f.size > maxMb * 1024 * 1024) return `File exceeds the ${maxMb}MB upload limit.`
    if (f.size === 0) return 'The selected file is empty.'
    return null
  }

  const pickFile = (f) => {
    if (!f) return
    const err = validate(f)
    if (err) {
      setError({ message: err })
      return
    }
    setError(null)
    setFile(f)
    setResult(null)
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    pickFile(e.dataTransfer.files?.[0])
  }

  const runAnalysis = async () => {
    if (!file) return
    setBusy(true)
    setError(null)
    setResult(null)
    setProgress(0)
    setStep(0)
    const stepTimer = setInterval(() => setStep((s) => Math.min(s + 1, 3)), 700)

    try {
      const data = await detectAudio(file, {
        phase,
        forceVerdict,
        onProgress: (p) => setProgress(p),
      })
      clearInterval(stepTimer)
      setStep(4)
      setResult(data)
    } catch (e) {
      clearInterval(stepTimer)
      setError({ message: e.message, requestId: e.requestId, code: e.code })
      setStep(-1)
    } finally {
      setBusy(false)
    }
  }

  const reset = () => {
    setFile(null)
    setResult(null)
    setError(null)
    setStep(-1)
    setProgress(0)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <PageContainer>
      <PageHeader
        title="Upload & Analyze Audio"
        description="Submit a recorded voice clip for full deepfake analysis — VAD prep, acoustic features, neural inference, and a risk decision."
      />

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
        {/* Left: upload + config */}
        <div className="space-y-5 lg:col-span-2">
          {!file ? (
            <div
              role="button"
              tabIndex={0}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-14 text-center transition ${
                dragging
                  ? 'border-cyan-400/70 bg-cyan-500/5'
                  : 'border-line bg-ink-900/50 hover:border-line-strong'
              }`}
            >
              <div className="flex h-14 w-14 items-center justify-center rounded-full bg-cyan-500/10 text-cyan-300">
                <UploadCloud size={26} />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-200">
                  Drag &amp; drop an audio file, or <span className="text-cyan-300 underline">browse</span>
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  WAV · MP3 · M4A · MP4 · OGG · FLAC — up to {maxMb}MB
                </p>
              </div>
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPT}
                className="hidden"
                onChange={(e) => pickFile(e.target.files?.[0])}
              />
            </div>
          ) : (
            <div className="panel-soft flex items-center gap-3 p-4">
              <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-cyan-500/10 text-cyan-300">
                <FileAudio size={20} />
              </span>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-slate-100">{file.name}</p>
                <p className="text-xs text-slate-500">{formatBytes(file.size)}</p>
              </div>
              <button
                onClick={reset}
                className="rounded-md p-2 text-slate-400 hover:bg-white/10 hover:text-rose-300"
                aria-label="Remove file"
              >
                <X size={16} />
              </button>
            </div>
          )}

          <div className="panel-soft space-y-4 p-4">
            <div>
              <label className="eyebrow mb-2 block">Detection engine</label>
              <select
                className="input-base"
                value={phase}
                onChange={(e) => setPhase(e.target.value)}
                disabled={busy}
              >
                {PHASE_OPTIONS.map((p) => (
                  <option key={p.value} value={p.value} className="bg-ink-900">
                    {p.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="eyebrow mb-2 block">
                Demo simulation <span className="text-slate-600">(presentation mode)</span>
              </label>
              <select
                className="input-base"
                value={forceVerdict}
                onChange={(e) => setForceVerdict(e.target.value)}
                disabled={busy}
              >
                <option value="" className="bg-ink-900">None — run the real classifier</option>
                <option value="real" className="bg-ink-900">Force REAL</option>
                <option value="fake" className="bg-ink-900">Force FAKE</option>
                <option value="uncertain" className="bg-ink-900">Force UNCERTAIN</option>
              </select>
            </div>

            <button onClick={runAnalysis} disabled={!file || busy} className="btn-primary w-full">
              {busy ? (
                <>
                  <Loader2 size={16} className="animate-spin" /> Analyzing…
                </>
              ) : (
                <>
                  <ScanLine size={16} /> Run AI Detection
                </>
              )}
            </button>

            {busy && (
              <div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-blue-500 transition-all"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <div className="mt-3 grid grid-cols-4 gap-1">
                  {STEPS.map((s, i) => (
                    <div key={s} className="text-center">
                      <div
                        className={`mx-auto mb-1 flex h-6 w-6 items-center justify-center rounded-full text-[11px] font-bold transition ${
                          step > i
                            ? 'bg-emerald-500 text-ink-950'
                            : step === i
                              ? 'bg-cyan-400 text-ink-950'
                              : 'bg-white/10 text-slate-500'
                        }`}
                      >
                        {step > i ? '✓' : i + 1}
                      </div>
                      <span className="text-[9px] uppercase tracking-wide text-slate-500">{s}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right: results */}
        <div className="lg:col-span-3">
          {error && (
            <ErrorState
              title="Analysis failed"
              message={error.message}
              requestId={error.requestId}
              onRetry={result === null && file ? runAnalysis : undefined}
            />
          )}

          {!error && !result && !busy && (
            <EmptyState
              icon={FileAudio}
              title="No analysis yet"
              description="Upload a voice clip and run detection. The verdict, confidence, signals, and feature map will appear here."
            />
          )}

          {busy && !result && (
            <div className="panel-soft flex flex-col items-center justify-center gap-3 py-20 text-slate-400">
              <Loader2 size={28} className="animate-spin text-cyan-400" />
              <p className="text-sm">Running the detection pipeline…</p>
            </div>
          )}

          {result && <ResultView result={result} />}
        </div>
      </div>
    </PageContainer>
  )
}

function ResultView({ result }) {
  const verdict = (result.verdict || '').toUpperCase()
  const tone =
    verdict === 'REAL'
      ? { wrap: 'border-emerald-500/30 bg-emerald-500/[0.06]', text: 'text-emerald-400', icon: ShieldCheck, label: 'Likely Authentic', Icon: ShieldCheck }
      : verdict === 'FAKE'
        ? { wrap: 'border-rose-500/30 bg-rose-500/[0.06]', text: 'text-rose-400', label: 'Potential Spoof', Icon: ShieldAlert }
        : { wrap: 'border-amber-500/30 bg-amber-500/[0.06]', text: 'text-amber-400', label: 'Uncertain', Icon: HelpCircle }

  const risk = result.risk_report || {}
  const rec = result.security_recommendation || {}
  const modelMeta = modelStatusMeta(result.model_status)
  const fakeProb = risk.fake_probability ?? result.raw_score ?? 0

  return (
    <div className="space-y-5">
      <div className={`rounded-xl border p-5 ${tone.wrap}`}>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <span className={`flex h-12 w-12 items-center justify-center rounded-full bg-white/5 ${tone.text}`}>
              <tone.Icon size={24} />
            </span>
            <div>
              <p className="eyebrow">Classification</p>
              <p className={`text-2xl font-bold ${tone.text}`}>{tone.label}</p>
            </div>
          </div>
          <div className="flex flex-col items-end gap-1">
            <span className="font-mono text-3xl font-bold text-white">
              {Math.round(fakeProb * 100)}
              <span className="text-lg text-slate-400">%</span>
            </span>
            <span className="text-xs text-slate-500">spoof probability</span>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Meta label="Confidence" value={`${result.confidence ?? '—'}%`} />
          <Meta label="Risk level" value={risk.risk_level || result.risk_level || '—'} />
          <Meta label="Risk score" value={`${risk.risk_score ?? '—'}/100`} />
          <Meta label="Processed in" value={`${result.processing_time_ms} ms`} />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="panel-soft p-4">
          <p className="eyebrow mb-2 flex items-center gap-1.5"><Cpu size={13} /> Model</p>
          <p className={`text-sm font-semibold ${modelMeta.color}`}>{modelMeta.label}</p>
          <p className="mt-1 text-xs text-slate-500">{modelMeta.note}</p>
          <p className="mt-2 font-mono text-[11px] text-slate-500">
            phase: {result.phase_used} · CNN active: {String(result.cnn_model_active)}
          </p>
        </div>
        <div className="panel-soft p-4">
          <p className="eyebrow mb-2 flex items-center gap-1.5"><Hash size={13} /> Request</p>
          <p className="text-sm font-medium text-slate-200">{result.filename}</p>
          <p className="mt-1 break-all font-mono text-[11px] text-slate-500">
            duration: {result.audio_duration_seconds?.toFixed(1) ?? '—'}s
          </p>
        </div>
      </div>

      {rec.recommendation && (
        <div className="panel-soft p-4">
          <p className="eyebrow mb-2">Security recommendation</p>
          <p className="text-sm font-semibold text-slate-100">{rec.recommendation}</p>
          <p className="mt-1 text-xs text-slate-400">{rec.reason}</p>
          {rec.recommended_verification_method && (
            <p className="mt-2 text-xs text-slate-500">
              Recommended verification: <span className="text-slate-300">{rec.recommended_verification_method}</span>
            </p>
          )}
        </div>
      )}

      {risk.signals?.length > 0 && (
        <div className="panel-soft p-4">
          <p className="eyebrow mb-3">Model evidence &amp; signals</p>
          <div className="space-y-2">
            {risk.signals.map((s, i) => (
              <div key={i} className="rounded-lg border border-line bg-ink-950/40 px-3 py-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-wide text-slate-300">
                    {s.type}
                  </span>
                  <span
                    className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${
                      s.strength === 'HIGH'
                        ? 'bg-rose-500/15 text-rose-300'
                        : s.strength === 'MEDIUM'
                          ? 'bg-amber-500/15 text-amber-300'
                          : 'bg-white/10 text-slate-400'
                    }`}
                  >
                    {s.strength}
                  </span>
                </div>
                <p className="mt-1 text-xs text-slate-400">{s.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {result.segments?.length > 0 && (
        <SegmentBar segments={result.segments} />
      )}

      {result.spectrogram && (
        <div className="panel-soft p-4">
          <p className="eyebrow mb-3 flex items-center gap-1.5"><ImageIcon size={13} /> Mel spectrogram</p>
          <img
            src={result.spectrogram}
            alt="Mel spectrogram feature map"
            className="w-full rounded-lg border border-line"
          />
        </div>
      )}
    </div>
  )
}

function SegmentBar({ segments }) {
  return (
    <div className="panel-soft p-4">
      <p className="eyebrow mb-3">
        Segment timeline <span className="text-slate-600">· {segments.length} chunks</span>
      </p>
      <div className="flex gap-1">
        {segments.map((s) => {
          const p = s.fake_probability
          const color = p > 0.6 ? 'bg-rose-400' : p > 0.4 ? 'bg-amber-400' : 'bg-emerald-400'
          return (
            <div
              key={s.segment_id}
              title={`${s.start_time.toFixed(1)}s–${s.end_time.toFixed(1)}s · ${s.predicted_label} · ${pct100(p, 0)}`}
              className={`h-8 flex-1 rounded ${color} opacity-80 transition hover:opacity-100`}
            />
          )
        })}
      </div>
      <div className="mt-2 flex justify-between text-[10px] text-slate-500">
        <span>0:00</span>
        <div className="flex gap-3">
          <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-emerald-400" /> Real</span>
          <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-amber-400" /> Uncertain</span>
          <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-rose-400" /> Spoof</span>
        </div>
      </div>
    </div>
  )
}

function Meta({ label, value }) {
  return (
    <div className="rounded-lg bg-ink-950/50 px-3 py-2">
      <p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
      <p className="mt-0.5 truncate font-mono text-sm font-semibold text-slate-100">{value}</p>
    </div>
  )
}

function Cpu(props) {
  // local lightweight icon to avoid extra import churn
  return (
    <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="4" width="16" height="16" rx="2" /><rect x="9" y="9" width="6" height="6" />
      <path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3" />
    </svg>
  )
}
