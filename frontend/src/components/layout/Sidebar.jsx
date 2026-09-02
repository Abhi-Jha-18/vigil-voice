import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Radio,
  UploadCloud,
  Siren,
  Cpu,
  FileText,
  Settings,
  ShieldCheck,
  X,
} from 'lucide-react'

const NAV = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/live', label: 'Live Detection', icon: Radio },
  { to: '/analyze', label: 'Upload & Analyze', icon: UploadCloud },
  { to: '/incidents', label: 'Incident Center', icon: Siren },
  { to: '/model', label: 'Model Center', icon: Cpu },
  { to: '/reports', label: 'Reports', icon: FileText },
  { to: '/settings', label: 'Settings', icon: Settings },
]

export default function Sidebar({ open, onClose }) {
  return (
    <>
      {open && <div className="fixed inset-0 z-30 bg-black/60 lg:hidden" onClick={onClose} />}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-line bg-ink-900/95 backdrop-blur-xl transition-transform duration-300 lg:translate-x-0 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex items-center justify-between px-5 py-5">
          <div className="flex items-center gap-2.5">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 text-ink-950 shadow-glow">
              <ShieldCheck size={20} strokeWidth={2.4} />
            </span>
            <div className="leading-tight">
              <div className="text-[15px] font-bold tracking-tight text-white">VigilVoice</div>
              <div className="text-[10px] font-medium uppercase tracking-[0.16em] text-slate-500">
                Deepfake Defense
              </div>
            </div>
          </div>
          <button
            className="rounded-md p-1 text-slate-400 hover:bg-white/10 lg:hidden"
            onClick={onClose}
            aria-label="Close menu"
          >
            <X size={18} />
          </button>
        </div>

        <nav className="flex-1 space-y-1 px-3 py-2">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={onClose}
              className={({ isActive }) => `nav-link ${isActive ? 'nav-link-active' : ''}`}
            >
              <Icon size={17} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="m-3 rounded-lg border border-line bg-ink-850/70 p-3">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            Privacy-first
          </p>
          <p className="mt-1 text-[11px] leading-relaxed text-slate-400">
            Live audio is processed transiently and is not stored by default.
          </p>
        </div>
      </aside>
    </>
  )
}
