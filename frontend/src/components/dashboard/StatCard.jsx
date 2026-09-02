export default function StatCard({ icon: Icon, label, value, sub, accent = '#22d3ee', loading }) {
  return (
    <div className="panel-soft relative overflow-hidden p-4">
      <div
        className="pointer-events-none absolute -right-6 -top-6 h-24 w-24 rounded-full opacity-20 blur-2xl"
        style={{ background: accent }}
      />
      <div className="flex items-center justify-between">
        <span className="flex h-9 w-9 items-center justify-center rounded-lg" style={{ background: `${accent}1a`, color: accent }}>
          {Icon ? <Icon size={18} /> : null}
        </span>
      </div>
      <p className="mt-3 font-mono text-2xl font-bold text-white">
        {loading ? <span className="inline-block h-7 w-16 animate-pulse rounded bg-white/10" /> : value}
      </p>
      <p className="text-xs font-medium text-slate-400">{label}</p>
      {sub && <p className="mt-0.5 text-[11px] text-slate-600">{sub}</p>}
    </div>
  )
}
