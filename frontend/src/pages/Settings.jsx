import { useState } from 'react'
import { Cpu, ShieldCheck, Sliders, Wifi, Moon, Mic, Database } from 'lucide-react'
import PageContainer, { PageHeader } from '../components/layout/PageContainer'
import { modelStatusMeta } from '../utils/risk'

export default function Settings({ model }) {
  const [reducedMotion, setReducedMotion] = useState(false)
  const cfg = model?.status?.configuration
  const status = model?.health?.model_status || model?.modelInfo?.model_status
  const meta = status ? modelStatusMeta(status) : null

  return (
    <PageContainer>
      <PageHeader
        title="Settings"
        description="Interface preferences, detection configuration exposed by the backend, privacy posture, and system information."
      />

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        {/* General */}
        <section className="panel p-5">
          <SectionTitle icon={Sliders} title="General" />
          <Toggle
            icon={Moon}
            label="Dark theme"
            description="VigilVoice uses a dedicated dark cybersecurity theme."
            checked
            readOnly
          />
          <Toggle
            icon={Sliders}
            label="Reduce motion"
            description="Minimize animations and transitions."
            checked={reducedMotion}
            onChange={() => setReducedMotion((v) => !v)}
          />
        </section>

        {/* Privacy */}
        <section className="panel p-5">
          <SectionTitle icon={ShieldCheck} title="Privacy" />
          <div className="space-y-3">
            <PrivacyRow
              icon={Mic}
              title="Live audio is not stored by default"
              body="Microphone audio is streamed transiently to the backend for windowed analysis and is not persisted. Microphone permission is requested only when you start monitoring."
            />
            <PrivacyRow
              icon={Database}
              title="Evidence is saved only when you export"
              body="An incident evidence JSON is written to reports/incidents only after you explicitly choose to save it. Nothing is reported to authorities automatically."
            />
          </div>
        </section>

        {/* Detection config */}
        <section className="panel p-5">
          <SectionTitle icon={Cpu} title="Detection configuration" />
          <div className="grid grid-cols-2 gap-3">
            <Cfg label="Sample rate" value={cfg ? `${cfg.sample_rate_hz} Hz` : '—'} />
            <Cfg label="Max upload" value={cfg ? `${cfg.max_upload_mb} MB` : '—'} />
            <Cfg label="Max audio length" value={cfg ? `${cfg.max_audio_duration_seconds}s` : '—'} />
            <Cfg label="Inference timeout" value={cfg ? `${cfg.inference_timeout_seconds}s` : '—'} />
            <Cfg label="Max concurrency" value={cfg?.max_concurrent_inferences ?? '—'} />
            <Cfg label="Live window" value={cfg ? `${cfg.live_window_seconds}s / ${cfg.live_hop_seconds}s hop` : '—'} />
            <Cfg label="High-risk threshold" value={cfg ? `${(cfg.live_high_risk_threshold * 100).toFixed(0)}%` : '—'} />
            <Cfg label="Suspicious threshold" value={cfg ? `${(cfg.live_suspicious_threshold * 100).toFixed(0)}%` : '—'} />
          </div>
          <p className="mt-3 text-[11px] text-slate-500">
            These values are read from the backend and cannot be changed from the client.
          </p>
        </section>

        {/* System */}
        <section className="panel p-5">
          <SectionTitle icon={Wifi} title="System" />
          <dl className="space-y-2">
            <SysRow label="API status" value={model?.online ? 'Online' : 'Offline'} ok={model?.online} />
            <SysRow label="Backend" value={model?.status?.server || 'VigilVoice API'} />
            <SysRow label="Version" value={model?.status?.version || '—'} mono />
            <SysRow label="Environment" value={model?.status?.configuration?.environment || '—'} />
            <SysRow label="Model status" value={meta?.label || '—'} valueClass={meta?.color} />
            <SysRow
              label="Demo overrides"
              value={cfg?.demo_mode ? 'Enabled (demo mode)' : 'Disabled'}
              ok={!cfg?.demo_mode}
            />
          </dl>
        </section>
      </div>
    </PageContainer>
  )
}

function SectionTitle({ icon: Icon, title }) {
  return (
    <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-200">
      <Icon size={16} className="text-cyan-300" /> {title}
    </h3>
  )
}

function Toggle({ icon: Icon, label, description, checked, onChange, readOnly }) {
  return (
    <div className="mb-3 flex items-center justify-between gap-3 rounded-lg bg-ink-950/50 px-3 py-3">
      <div className="flex items-start gap-3">
        <Icon size={16} className="mt-0.5 text-slate-500" />
        <div>
          <p className="text-sm font-medium text-slate-200">{label}</p>
          <p className="text-[11px] text-slate-500">{description}</p>
        </div>
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={readOnly}
        onClick={onChange}
        className={`relative h-6 w-11 shrink-0 rounded-full transition ${
          checked ? 'bg-cyan-500/80' : 'bg-white/10'
        } ${readOnly ? 'opacity-70' : ''}`}
      >
        <span
          className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-all ${
            checked ? 'left-[22px]' : 'left-0.5'
          }`}
        />
      </button>
    </div>
  )
}

function PrivacyRow({ icon: Icon, title, body }) {
  return (
    <div className="flex items-start gap-3 rounded-lg bg-ink-950/50 px-3 py-3">
      <Icon size={16} className="mt-0.5 text-emerald-300" />
      <div>
        <p className="text-sm font-medium text-slate-200">{title}</p>
        <p className="mt-0.5 text-[11px] leading-relaxed text-slate-500">{body}</p>
      </div>
    </div>
  )
}

function Cfg({ label, value }) {
  return (
    <div className="rounded-lg bg-ink-950/50 px-3 py-2.5">
      <p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
      <p className="mt-0.5 font-mono text-sm font-semibold text-slate-100">{value}</p>
    </div>
  )
}

function SysRow({ label, value, ok, valueClass, mono }) {
  return (
    <div className="flex items-center justify-between rounded-lg bg-ink-950/50 px-3 py-2.5">
      <dt className="text-xs text-slate-400">{label}</dt>
      <dd className={`flex items-center gap-2 text-sm font-semibold ${valueClass || 'text-slate-200'} ${mono ? 'font-mono text-xs' : ''}`}>
        {ok !== undefined && (
          <span className={`h-1.5 w-1.5 rounded-full ${ok ? 'bg-emerald-400' : 'bg-rose-400'}`} />
        )}
        {value}
      </dd>
    </div>
  )
}
