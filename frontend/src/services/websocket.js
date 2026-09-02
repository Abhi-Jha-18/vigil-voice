/**
 * Builds a WebSocket URL for a live detection session.
 * Uses VITE_API_BASE_URL when set (absolute), otherwise derives a same-origin
 * URL from the current page (works through the Vite dev proxy and in prod).
 */
export function buildLiveSocketUrl(sessionId) {
  const base = import.meta.env.VITE_API_BASE_URL
  if (base) {
    const httpBase = base.replace(/^http/, 'ws').replace(/\/$/, '')
    return `${httpBase}/api/live/audio/${sessionId}`
  }
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}/api/live/audio/${sessionId}`
}
