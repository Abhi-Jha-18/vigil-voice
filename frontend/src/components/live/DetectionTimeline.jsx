import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  CartesianGrid,
} from 'recharts'
import { riskMeta } from '../../utils/risk'
import { timeLabelFromOffset } from '../../utils/format'

/**
 * Rolling spoof-probability chart + per-window event log.
 * `points`: [{ t, score }] (0..100)  `windows`: full timeline entries.
 */
export default function DetectionTimeline({ points = [], windows = [], highThreshold = 80, suspiciousThreshold = 60 }) {
  return (
    <div className="panel-soft flex h-full flex-col p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="eyebrow">Spoof probability over time</span>
        <div className="flex items-center gap-3 text-[11px] text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="h-1.5 w-3 rounded-full bg-amber-400" /> Suspicious
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-1.5 w-3 rounded-full bg-rose-400" /> High risk
          </span>
        </div>
      </div>

      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={points} margin={{ top: 4, right: 4, left: -18, bottom: 0 }}>
            <defs>
              <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.45} />
                <stop offset="100%" stopColor="#22d3ee" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" vertical={false} />
            <XAxis dataKey="t" tickFormatter={timeLabelFromOffset} stroke="#475569" fontSize={10} tickLine={false} axisLine={false} minTickGap={30} />
            <YAxis domain={[0, 100]} stroke="#475569" fontSize={10} tickLine={false} axisLine={false} unit="%" />
            <Tooltip
              contentStyle={{ background: '#0e121b', border: '1px solid rgba(148,163,184,0.2)', borderRadius: 10, fontSize: 12 }}
              labelStyle={{ color: '#94a3b8' }}
              labelFormatter={(t) => `Time ${timeLabelFromOffset(t)}`}
              formatter={(v) => [`${v}%`, 'Spoof probability']}
            />
            <ReferenceLine y={suspiciousThreshold} stroke="#fbbf24" strokeDasharray="4 4" strokeOpacity={0.5} />
            <ReferenceLine y={highThreshold} stroke="#fb7185" strokeDasharray="4 4" strokeOpacity={0.6} />
            <Area type="monotone" dataKey="score" stroke="#22d3ee" strokeWidth={2} fill="url(#riskFill)" isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 border-t border-line pt-3">
        <div className="mb-2 flex items-center justify-between">
          <span className="eyebrow">Detection windows</span>
          <span className="text-[11px] text-slate-500">{windows.length} analyzed</span>
        </div>
        <div className="max-h-44 space-y-1 overflow-y-auto pr-1">
          {windows.length === 0 && (
            <p className="py-6 text-center text-xs text-slate-600">
              Windows will appear here once speech is detected.
            </p>
          )}
          {[...windows].reverse().map((w) => {
            const meta = riskMeta(w.riskLevel)
            return (
              <div
                key={w.id}
                className="flex items-center justify-between rounded-md bg-white/[0.03] px-3 py-1.5 text-xs"
              >
                <span className="flex items-center gap-2 font-mono text-slate-400">
                  <span className="h-1.5 w-1.5 rounded-full" style={{ background: meta.color }} />
                  {timeLabelFromOffset(w.t)}
                </span>
                <span className="font-medium" style={{ color: meta.color }}>
                  {meta.short}
                </span>
                <span className="font-mono text-slate-300">{Math.round(w.fakeProb * 100)}%</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
