import { ShieldCheck } from 'lucide-react'

export default function EmptyState({ icon: Icon = ShieldCheck, title, description, action }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-line px-6 py-14 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-white/5 text-slate-500">
        <Icon size={22} />
      </div>
      <div>
        <p className="text-sm font-semibold text-slate-300">{title}</p>
        {description && <p className="mx-auto mt-1 max-w-sm text-sm text-slate-500">{description}</p>}
      </div>
      {action}
    </div>
  )
}
