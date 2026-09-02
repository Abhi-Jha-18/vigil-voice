import { Menu, Activity } from 'lucide-react'
import { modelStatusMeta } from '../../utils/risk'
import Badge from '../common/Badge'

export default function Topbar({ onMenu, title, model }) {
  const online = model?.online
  const status = model?.health?.model_status || model?.modelInfo?.model_status || null
  const meta = status ? modelStatusMeta(status) : null

  return (
    <header className="sticky top-0 z-20 flex h-16 items-center justify-between gap-3 border-b border-line bg-ink-950/80 px-4 backdrop-blur-xl sm:px-6">
      <div className="flex items-center gap-3">
        <button
          className="rounded-md p-2 text-slate-400 hover:bg-white/10 lg:hidden"
          onClick={onMenu}
          aria-label="Open menu"
        >
          <Menu size={20} />
        </button>
        <div>
          <h1 className="text-[15px] font-semibold text-white sm:text-base">{title}</h1>
          <p className="hidden text-xs text-slate-500 sm:block">
            AI voice deepfake &amp; spoofing detection
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Badge
          className={
            online
              ? 'bg-emerald-500/10 text-emerald-300 ring-emerald-500/30'
              : 'bg-rose-500/10 text-rose-300 ring-rose-500/30'
          }
          dot
          dotClass={online ? 'bg-emerald-400' : 'bg-rose-400'}
        >
          <span className="hidden sm:inline">{online ? 'API Online' : 'API Offline'}</span>
          <span className="sm:hidden">{online ? 'Online' : 'Offline'}</span>
        </Badge>

        {meta && (
          <Badge className={`hidden md:inline-flex ${meta.bg} ${meta.color}`} dot dotClass={meta.dot}>
            <span className="inline-flex items-center gap-1.5">
              <Activity size={12} />
              {meta.label}
            </span>
          </Badge>
        )}
      </div>
    </header>
  )
}
