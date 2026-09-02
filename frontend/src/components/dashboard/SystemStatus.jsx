import { Server, Cpu, Activity, Wifi } from 'lucide-react'
import { modelStatusMeta } from '../../utils/risk'

function Row({ icon: Icon, label, ok, value, valueClass }) {
  return (
    <div className="flex items-center justify-between rounded-lg bg-ink-950/50 px-3 py-2.5">
      <span className="flex items-center gap-2 text-xs text-slate-400">
        <Icon size={14} className="text-slate-500" /> {label}
      </span>
      <span className={`flex items-center gap-2 text-xs font-semibold ${valueClass || 'text-slate-200'}`}>
        <span className={`h-1.5 w-1.5 rounded-full ${ok ? 'bg-emerald-400' : ok === false ? 'bg-rose-400' : 'bg-slate-500'}`} />
        {value}
      </span>
    </div>
  )
}

export default function SystemStatus({ model }) {
  const online = model?.online
  const status = model?.health?.model_status || model?.modelInfo?.model_status
  const meta = status ? modelStatusMeta(status) : null
  const ready = model?.ready?.ready
  const uptime = model?.status?.uptime_seconds

  return (
    <div className="panel p-5">
      <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-200">
        <Activity size={16} className="text-cyan-300" /> System status
      </h3>
      <div className="space-y-2">
        <Row icon={Wifi} label="API connection" ok={online} value={online ? 'Online' : 'Offline'} valueClass={online ? 'text-emerald-300' : 'text-rose-300'} />
        <Row icon={Server} label="Readiness" ok={ready} value={ready ? 'Ready' : 'Not ready'} valueClass={ready ? 'text-emerald-300' : 'text-amber-300'} />
        <Row
          icon={Cpu}
          label="Detection model"
          ok={status === 'REAL_MODEL' || status === 'DEMO_MODEL'}
          value={meta?.label || 'Unknown'}
          valueClass={meta?.color}
        />
        <Row icon={Activity} label="Backend uptime" ok={online} value={uptime != null ? `${Math.floor(uptime / 60)} min` : '—'} />
      </div>
      {meta && <p className="mt-3 text-[11px] leading-relaxed text-slate-500">{meta.note}</p>}
    </div>
  )
}
