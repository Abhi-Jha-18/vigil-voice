import { useCallback, useEffect, useState } from 'react'
import { getIncident, listIncidents } from '../services/api'

/**
 * Lists saved incidents. The first load sets `loading`; subsequent automatic
 * refetches (poll + window refocus) run silently so the UI never flashes
 * skeletons or wipes already-rendered data when a transient error occurs.
 */
export function useIncidents({ pollMs = 20000 } = {}) {
  const [incidents, setIncidents] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const refresh = useCallback(async (silent = false) => {
    if (!silent) setLoading(true)
    try {
      const res = await listIncidents()
      setIncidents(res?.incidents || [])
      setError(null)
    } catch (e) {
      // Keep any previously loaded data; only set error on the initial load.
      setError((prev) => (silent ? prev : e.message))
      if (!silent) setIncidents([])
    } finally {
      if (!silent) setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh(false)
    const poll = setInterval(() => refresh(true), pollMs)
    const onFocus = () => {
      if (document.visibilityState === 'visible') refresh(true)
    }
    document.addEventListener('visibilitychange', onFocus)
    return () => {
      clearInterval(poll)
      document.removeEventListener('visibilitychange', onFocus)
    }
  }, [refresh, pollMs])

  return { incidents, loading, error, refresh: () => refresh(false) }
}

export function useIncident(id) {
  const [incident, setIncident] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const refresh = useCallback(
    async (silent = false) => {
      if (!id) return
      if (!silent) setLoading(true)
      try {
        const res = await getIncident(id)
        setIncident(res?.incident || null)
        setError(null)
      } catch (e) {
        setError((prev) => (silent ? prev : e.message))
        if (!silent) setIncident(null)
      } finally {
        if (!silent) setLoading(false)
      }
    },
    [id],
  )

  useEffect(() => {
    refresh(false)
    const onFocus = () => {
      if (document.visibilityState === 'visible') refresh(true)
    }
    document.addEventListener('visibilitychange', onFocus)
    return () => document.removeEventListener('visibilitychange', onFocus)
  }, [refresh])

  return { incident, loading, error, refresh: () => refresh(false) }
}
