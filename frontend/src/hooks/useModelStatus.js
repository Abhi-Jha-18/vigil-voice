import { useCallback, useEffect, useState } from 'react'
import { getHealth, getModelInfo, getReady, getStatus } from '../services/api'

/**
 * Aggregates /api/health, /api/ready, /api/status and /api/model/info into a
 * single model/system status object. Polls on a light interval while mounted.
 */
export function useModelStatus(pollMs = 15000) {
  const [data, setData] = useState({
    online: false,
    loading: true,
    health: null,
    ready: null,
    status: null,
    modelInfo: null,
    error: null,
  })

  const refresh = useCallback(async () => {
    const [healthRes, readyRes, statusRes, infoRes] = await Promise.allSettled([
      getHealth(),
      getReady(),
      getStatus(),
      getModelInfo(),
    ])

    const health = healthRes.status === 'fulfilled' ? healthRes.value : null
    const ready = readyRes.status === 'fulfilled' ? readyRes.value : null
    const status = statusRes.status === 'fulfilled' ? statusRes.value : null
    const modelInfo = infoRes.status === 'fulfilled' ? infoRes.value : null
    const online = !!health

    setData({
      online,
      loading: false,
      health,
      ready,
      status,
      modelInfo,
      error: online ? null : 'Backend unreachable',
    })
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, pollMs)
    return () => clearInterval(id)
  }, [refresh, pollMs])

  return { ...data, refresh }
}
