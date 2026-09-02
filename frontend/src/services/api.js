/**
 * Centralized API service layer.
 * All REST calls to the FastAPI backend go through here so components never
 * hardcode URLs or duplicate error handling.
 *
 * VITE_API_BASE_URL is optional: leave it empty for same-origin requests
 * (production build is served by FastAPI; dev uses the Vite proxy).
 */

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

/** Convert a backend structured error into a normalized Error instance. */
export class ApiError extends Error {
  constructor(message, { code, status, requestId, details } = {}) {
    super(message)
    this.name = 'ApiError'
    this.code = code || 'ERROR'
    this.status = status
    this.requestId = requestId
    this.details = details
  }
}

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`
  let res
  try {
    res = await fetch(url, {
      ...options,
      headers: {
        'X-Request-ID': crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
        ...(options.headers || {}),
      },
    })
  } catch (networkErr) {
    throw new ApiError(
      'Unable to reach the VigilVoice backend. Check that the FastAPI server is running.',
      { code: 'NETWORK_ERROR', status: 0 },
    )
  }

  const requestId = res.headers.get('X-Request-ID')

  let data = null
  const contentType = res.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    try {
      data = await res.json()
    } catch {
      data = null
    }
  }

  if (!res.ok) {
    const code = data?.error?.code || data?.detail?.code || 'ERROR'
    const message =
      data?.error?.message ||
      (typeof data?.detail === 'string' ? data.detail : null) ||
      defaultMessageForStatus(res.status)
    throw new ApiError(message, { code, status: res.status, requestId, details: data })
  }

  return data
}

function defaultMessageForStatus(status) {
  switch (status) {
    case 413:
      return 'File too large. It exceeds the configured upload limit.'
    case 415:
      return 'Unsupported file format.'
    case 429:
      return 'Server busy — too many concurrent analyses. Please retry.'
    case 504:
      return 'Analysis timed out. The server could not complete analysis within the time limit.'
    case 503:
      return 'Service temporarily unavailable.'
    default:
      return `Request failed with status ${status}.`
  }
}

// ── System ────────────────────────────────────────────────────────────────────

export const getHealth = () => request('/api/health')
export const getReady = () => request('/api/ready')
export const getStatus = () => request('/api/status')
export const getModelInfo = () => request('/api/model/info')

// ── Detection ─────────────────────────────────────────────────────────────────

/**
 * Upload an audio file for analysis.
 * @param {File} file
 * @param {{ phase?: string, forceVerdict?: string, onProgress?: (pct:number)=>void }} opts
 */
export function detectAudio(file, { phase = 'phase2', forceVerdict = '', onProgress } = {}) {
  const form = new FormData()
  form.append('file', file)
  form.append('phase', phase)
  if (forceVerdict) form.append('force_verdict', forceVerdict)

  // XHR so we can surface real upload progress (fetch lacks upload progress).
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/api/detect`)
    xhr.setRequestHeader(
      'X-Request-ID',
      crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
    )

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100))
    }

    xhr.onload = () => {
      let data = null
      try {
        data = JSON.parse(xhr.responseText)
      } catch {
        /* non-JSON */
      }
      const requestId = xhr.getResponseHeader('X-Request-ID')
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(data)
      } else {
        const code = data?.error?.code || 'ERROR'
        const message =
          data?.error?.message ||
          (typeof data?.detail === 'string' ? data.detail : null) ||
          defaultMessageForStatus(xhr.status)
        reject(new ApiError(message, { code, status: xhr.status, requestId, details: data }))
      }
    }

    xhr.onerror = () =>
      reject(
        new ApiError(
          'Unable to reach the VigilVoice backend. Check that the FastAPI server is running.',
          { code: 'NETWORK_ERROR', status: 0 },
        ),
      )

    xhr.send(form)
  })
}

// ── Live sessions ─────────────────────────────────────────────────────────────

export const createLiveSession = () => request('/api/live/session', { method: 'POST' })
export const getLiveStatus = (sessionId) => request(`/api/live/status/${sessionId}`)
export const getReportingResources = () => request('/api/live/reporting-resources')

export async function generateIncidentReport(sessionId) {
  return request(`/api/live/incident/${sessionId}/report`, { method: 'POST' })
}

// ── Incidents (read-only, backed by reports/incidents/*.json) ─────────────────

export const listIncidents = () => request('/api/incidents')
export const getIncident = (id) => request(`/api/incidents/${id}`)
