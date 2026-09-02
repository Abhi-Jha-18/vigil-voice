import { Cpu, CheckCircle2, XCircle, ShieldCheck, Database, Gauge } from 'lucide-react'
import PageContainer, { PageHeader } from '../components/layout/PageContainer'
import { Skeleton } from '../components/common/Loading'
import { modelStatusMeta } from '../utils/risk'

function Metric({ label, value, available = true }) {
  return (
    <div className="rounded-lg bg-ink-950/50 px-4 py-3">
      <p className="text-[11px] uppercase tracking-wider text-slate-500">{label}</p>
      <p className="mt-1 font-mono text-lg font-bold text-slate-100">
        {available ? value : <span className="text-sm font-normal text-slate-600">Not available</span>}
      </p>
    </div>
  )
}

export default function ModelCenter({ model }) {
  const info = model?.modelInfo
  const status = info?.model_status || model?.health?.model_status
  const meta = status ? modelStatusMeta(status) : null
  const evalData = info?.evaluation
  const isRealDataset = evalData?.dataset_type === 'REAL'
  const robustness = info?.robustness

  const eer = isRealDataset ? evalData?.eer : null
  const auc = isRealDataset ? evalData?.roc_auc : null
  const far = evalData?.far ?? evalData?.false_accept_rate ?? null
  const frr = evalData?.frr ?? evalData?.false_reject_rate ?? null
  const acc = evalData?.accuracy ?? evalData?.accuracy_pct ?? null

  return (
    <PageContainer>
      <PageHeader
        title="Model & AI Center"
        description="Detection engine capabilities, verified evaluation metrics, and checkpoint integrity. Only metrics exposed by the backend are shown."
      />

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        {/* Status hero */}
        <div className="panel p-5">
          <div className="flex items-center gap-3">
            <span className={`flex h-12 w-12 items-center justify-center rounded-xl ${meta?.bg || 'bg-white/5'} ${meta?.color || 'text-slate-400'}`}>
              <Cpu size={24} />
            </span>
            <div>
              <p className="eyebrow">Model status</p>
              <p className={`text-lg font-bold ${meta?.color || 'text-slate-200'}`}>
                {model?.loading ? 'Loading…' : meta?.label || 'Unknown'}
              </p>
            </div>
          </div>
          {meta && <p className="mt-3 text-xs leading-relaxed text-slate-400">{meta.note}</p>}

          <div className="mt-4 space-y-2">
            <CapRow label="CNN spectrogram classifier" available={!!model?.status?.models?.phase2_cnn?.available} />
            <CapRow label="CNN-LSTM temporal" available={!!model?.status?.models?.phase3_cnn_lstm?.available} />
            <CapRow label="Wav2Vec transformer" available={!!model?.status?.models?.phase4_wav2vec?.available} />
            <CapRow label="Acoustic heuristics (fallback)" available />
          </div>
        </div>

        {/* Engine info */}
        <div className="panel p-5 lg:col-span-2">
          <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-200">
            <ShieldCheck size={16} className="text-cyan-300" /> Engine information
          </h3>
          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <Info label="Architecture" value="SimpleCNN (MFCC spectrogram)" />
            <Info label="Framework" value="PyTorch" />
            <Info label="Model version" value={model?.health?.model?.version || model?.status?.models?.phase2_cnn?.version || '—'} mono />
            <Info label="Sample rate" value={`${model?.status?.configuration?.sample_rate_hz ?? 16000} Hz`} mono />
            <Info label="Dataset" value={isRealDataset ? 'ASVspoof / real' : evalData?.dataset_type || 'Not available'} />
            <Info label="Environment" value={model?.status?.configuration?.environment || '—'} />
          </dl>

          <h3 className="mb-3 mt-6 flex items-center gap-2 text-sm font-semibold text-slate-200">
            <Gauge size={16} className="text-cyan-300" /> Evaluation metrics
            {!isRealDataset && <span className="text-[11px] font-normal text-slate-500">· requires a REAL evaluation report</span>}
          </h3>
          {model?.loading ? (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-16" />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Metric label="EER" value={`${(eer * 100).toFixed(2)}%`} available={eer != null} />
              <Metric label="ROC-AUC" value={auc?.toFixed(3)} available={auc != null} />
              <Metric label="FAR" value={far != null ? `${(far * 100).toFixed(1)}%` : '—'} available={far != null} />
              <Metric label="FRR" value={frr != null ? `${(frr * 100).toFixed(1)}%` : '—'} available={frr != null} />
            </div>
          )}
          {acc != null && (
            <p className="mt-3 text-xs text-slate-500">Reported accuracy: <span className="font-mono text-slate-300">{(acc * 100).toFixed(1)}%</span></p>
          )}
        </div>
      </div>

      {/* Robustness */}
      <div className="panel mt-5 p-5">
        <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-200">
          <Database size={16} className="text-cyan-300" /> Robustness &amp; noise generalization
        </h3>
        {robustness?.available ? (
          <RobustnessTable data={robustness.data} />
        ) : (
          <p className="rounded-lg border border-dashed border-line px-4 py-8 text-center text-sm text-slate-500">
            Robustness evaluation not available. Run the robustness benchmark to populate noise and
            resampling metrics.
          </p>
        )}
      </div>

      <div className="panel mt-5 p-5">
        <h3 className="mb-2 text-sm font-semibold text-slate-200">Integrity & provenance</h3>
        <p className="text-xs leading-relaxed text-slate-400">
          The backend verifies the model checkpoint against a recorded SHA-256 checksum. A mismatch
          sets <span className="font-mono text-slate-300">MODEL_INTEGRITY_FAILURE</span> and disables
          neural inference. When no trained checkpoint is present, the system safely falls back to
          acoustic heuristics (<span className="font-mono text-slate-300">HEURISTIC_ONLY</span>).
        </p>
      </div>
    </PageContainer>
  )
}

function CapRow({ label, available }) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-slate-400">{label}</span>
      {available ? (
        <span className="flex items-center gap-1 text-emerald-300">
          <CheckCircle2 size={13} /> Ready
        </span>
      ) : (
        <span className="flex items-center gap-1 text-slate-600">
          <XCircle size={13} /> Not available
        </span>
      )}
    </div>
  )
}

function Info({ label, value, mono }) {
  return (
    <div className="rounded-lg bg-ink-950/50 px-3 py-2.5">
      <p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
      <p className={`mt-0.5 text-sm font-semibold text-slate-200 ${mono ? 'font-mono text-xs' : ''}`}>{value}</p>
    </div>
  )
}

function RobustnessTable({ data }) {
  const rows = [
    { label: 'Clean audio baseline', acc: data?.baseline?.accuracy, eer: data?.baseline?.eer },
    { label: 'Background noise (10 dB)', acc: data?.conditions?.noise_10db?.accuracy },
    { label: 'Resampled (8 kHz)', acc: data?.conditions?.resample_8khz?.accuracy },
  ]
  return (
    <div className="overflow-hidden rounded-lg border border-line">
      <table className="w-full text-sm">
        <thead className="bg-ink-950/60 text-left text-[11px] uppercase tracking-wider text-slate-500">
          <tr>
            <th className="px-4 py-2.5">Condition</th>
            <th className="px-4 py-2.5">Accuracy</th>
            <th className="px-4 py-2.5">EER</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className="border-t border-line">
              <td className="px-4 py-2.5 text-slate-300">{r.label}</td>
              <td className="px-4 py-2.5 font-mono text-slate-100">
                {r.acc != null ? `${(r.acc * 100).toFixed(1)}%` : 'Not available'}
              </td>
              <td className="px-4 py-2.5 font-mono text-slate-100">
                {r.eer != null ? `${(r.eer * 100).toFixed(2)}%` : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
