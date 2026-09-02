import { Wifi, WifiOff, Loader2 } from 'lucide-react'

const MAP = {
  idle: { label: 'Disconnected', className: 'bg-white/5 text-slate-400 ring-line', dot: 'bg-slate-500', icon: WifiOff },
  connecting: { label: 'Connecting…', className: 'bg-amber-500/10 text-amber-300 ring-amber-500/30', dot: 'bg-amber-400', icon: Loader2, spin: true },
  connected: { label: 'Connected', className: 'bg-emerald-500/10 text-emerald-300 ring-emerald-500/30', dot: 'bg-emerald-400', icon: Wifi },
  error: { label: 'Connection error', className: 'bg-rose-500/10 text-rose-300 ring-rose-500/30', dot: 'bg-rose-400', icon: WifiOff },
}

export default function ConnectionStatus({ state }) {
  const m = MAP[state] || MAP.idle
  const Icon = m.icon
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold ring-1 ring-inset ${m.className}`}
    >
      <span className={`h-2 w-2 rounded-full ${m.dot} ${state === 'connected' ? 'animate-pulse' : ''}`} />
      <Icon size={14} className={m.spin ? 'animate-spin' : ''} />
      {m.label}
    </span>
  )
}
