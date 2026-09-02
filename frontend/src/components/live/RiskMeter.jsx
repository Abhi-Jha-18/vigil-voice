import { riskMeta } from '../../utils/risk'

/**
 * Premium radial risk gauge. `score` is 0..100 fake probability.
 */
export default function RiskMeter({ score = 0, level = 'LOW_RISK', label, sublabel }) {
  const meta = riskMeta(level)
  const radius = 74
  const circumference = 2 * Math.PI * radius
  const pct = Math.max(0, Math.min(100, score))
  const offset = circumference - (pct / 100) * circumference

  return (
    <div
      className="panel-soft relative flex flex-col items-center justify-center gap-3 overflow-hidden px-6 py-8 text-center transition-colors duration-500"
      style={{ boxShadow: `inset 0 0 60px -30px ${meta.color}` }}
    >
      <div
        className="pointer-events-none absolute inset-0 opacity-60 transition-opacity duration-700"
        style={{
          background: `radial-gradient(22rem 12rem at 50% 0%, ${meta.color}1f, transparent 70%)`,
        }}
      />
      <span className="eyebrow relative">{label || 'Call Risk Status'}</span>

      <div className="relative h-44 w-44">
        <svg className="h-full w-full -rotate-90" viewBox="0 0 180 180">
          <circle
            cx="90"
            cy="90"
            r={radius}
            fill="none"
            stroke="rgba(148,163,184,0.12)"
            strokeWidth="12"
          />
          <circle
            cx="90"
            cy="90"
            r={radius}
            fill="none"
            stroke={meta.color}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{
              transition: 'stroke-dashoffset 0.6s ease, stroke 0.6s ease',
              filter: `drop-shadow(0 0 8px ${meta.color}80)`,
            }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-mono text-4xl font-bold tracking-tight text-white">
            {pct.toFixed(1)}
            <span className="text-xl text-slate-400">%</span>
          </span>
          <span className="mt-1 text-[11px] font-medium uppercase tracking-widest text-slate-500">
            spoof probability
          </span>
        </div>
      </div>

      <div className="relative">
        <span
          className="inline-flex items-center gap-2 rounded-full px-3.5 py-1.5 text-sm font-bold ring-1 ring-inset"
          style={{ color: meta.color, background: `${meta.color}1a`, boxShadow: `inset 0 0 0 1px ${meta.color}40` }}
        >
          <span className="h-2 w-2 rounded-full" style={{ background: meta.color }} />
          {meta.label}
        </span>
        {sublabel && <p className="mt-2 max-w-xs text-xs text-slate-400">{sublabel}</p>}
      </div>
    </div>
  )
}
