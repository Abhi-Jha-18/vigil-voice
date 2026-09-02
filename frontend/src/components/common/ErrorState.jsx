import { AlertTriangle, RefreshCw } from 'lucide-react'

export default function ErrorState({
  title = 'Something went wrong',
  message,
  requestId,
  onRetry,
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-rose-500/20 bg-rose-500/5 px-6 py-12 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-rose-500/10 text-rose-400">
        <AlertTriangle size={22} />
      </div>
      <div>
        <p className="text-sm font-semibold text-rose-200">{title}</p>
        {message && <p className="mx-auto mt-1 max-w-md text-sm text-slate-400">{message}</p>}
        {requestId && (
          <p className="mt-2 font-mono text-[11px] text-slate-600">Request ID: {requestId}</p>
        )}
      </div>
      {onRetry && (
        <button onClick={onRetry} className="btn-ghost mt-1">
          <RefreshCw size={14} /> Try again
        </button>
      )}
    </div>
  )
}
