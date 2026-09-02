/**
 * Risk / verdict presentation mappings shared across the app.
 * Language is deliberately cautious and probabilistic — VigilVoice never makes
 * legal accusations (per product requirements).
 */

export const RISK_LEVELS = {
  LOW_RISK: {
    label: 'Low Risk',
    short: 'LOW',
    color: '#34d399',
    text: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    ring: 'ring-emerald-500/30',
    dot: 'bg-emerald-400',
    description: 'No significant synthetic-speech indicators detected.',
  },
  SUSPICIOUS: {
    label: 'Suspicious',
    short: 'SUSPICIOUS',
    color: '#fbbf24',
    text: 'text-amber-400',
    bg: 'bg-amber-500/10',
    ring: 'ring-amber-500/30',
    dot: 'bg-amber-400',
    description: 'Some voice characteristics look unusual. Stay cautious and verify identity.',
  },
  HIGH_SPOOF_RISK: {
    label: 'High Spoof Risk',
    short: 'HIGH',
    color: '#fb7185',
    text: 'text-rose-400',
    bg: 'bg-rose-500/10',
    ring: 'ring-rose-500/30',
    dot: 'bg-rose-400',
    description:
      'Elevated indicators associated with potential voice spoofing. Do not trust requests made on this call.',
  },
}

export function riskMeta(level) {
  return RISK_LEVELS[level] || RISK_LEVELS.LOW_RISK
}

/** Maps a fake probability (0..1) to a live risk level using the same buckets as the UI. */
export function riskLevelFromScore(fakeProb, highThreshold = 0.8, suspiciousThreshold = 0.6) {
  if (fakeProb >= highThreshold) return 'HIGH_SPOOF_RISK'
  if (fakeProb >= suspiciousThreshold) return 'SUSPICIOUS'
  return 'LOW_RISK'
}

export const MODEL_STATUS = {
  REAL_MODEL: {
    label: 'Real Model',
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10 ring-emerald-500/30',
    dot: 'bg-emerald-400',
    note: 'Trained CNN checkpoint verified and active.',
  },
  DEMO_MODEL: {
    label: 'Demo Model',
    color: 'text-cyan-300',
    bg: 'bg-cyan-500/10 ring-cyan-500/30',
    dot: 'bg-cyan-400',
    note: 'Demo / synthetic-data checkpoint. Results are illustrative.',
  },
  HEURISTIC_ONLY: {
    label: 'Heuristic Only',
    color: 'text-amber-400',
    bg: 'bg-amber-500/10 ring-amber-500/30',
    dot: 'bg-amber-400',
    note: 'No neural model loaded — acoustic heuristics only.',
  },
  MODEL_UNAVAILABLE: {
    label: 'Model Unavailable',
    color: 'text-amber-400',
    bg: 'bg-amber-500/10 ring-amber-500/30',
    dot: 'bg-amber-400',
    note: 'The detection model is currently unavailable.',
  },
  MODEL_INTEGRITY_FAILURE: {
    label: 'Integrity Failure',
    color: 'text-rose-400',
    bg: 'bg-rose-500/10 ring-rose-500/30',
    dot: 'bg-rose-400',
    note: 'Model checksum verification failed — inference disabled.',
  },
}

export function modelStatusMeta(status) {
  return MODEL_STATUS[status] || MODEL_STATUS.HEURISTIC_ONLY
}

export const PHASE_OPTIONS = [
  { value: 'phase1', label: 'Phase 1 — Stub (Heuristics)' },
  { value: 'phase2', label: 'Phase 2 — CNN Spectrogram' },
  { value: 'phase3', label: 'Phase 3 — CNN-LSTM (falls back)' },
  { value: 'phase4', label: 'Phase 4 — Wav2Vec (falls back)' },
]
