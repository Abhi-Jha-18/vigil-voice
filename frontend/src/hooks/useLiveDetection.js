import { useCallback, useEffect, useRef, useState } from 'react'
import { createLiveSession, generateIncidentReport } from '../services/api'
import { buildLiveSocketUrl } from '../services/websocket'

const SAMPLE_RATE = 16000
const TIMELINE_CAP = 40
const WINDOW_SECONDS = 2.5

function downsampleTo16k(buffer, inSampleRate) {
  if (!buffer || !buffer.length) return buffer
  if (inSampleRate === SAMPLE_RATE) return buffer
  const ratio = inSampleRate / SAMPLE_RATE
  const newLength = Math.round(buffer.length / ratio)
  const result = new Float32Array(newLength)
  let offsetResult = 0
  let offsetBuffer = 0
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio)
    let accum = 0
    let count = 0
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i]
      count++
    }
    result[offsetResult] = count > 0 ? accum / count : buffer[offsetBuffer] || 0
    offsetResult++
    offsetBuffer = nextOffsetBuffer
  }
  return result
}

/**
 * Manages a full live-call monitoring session:
 * REST session creation -> microphone capture -> WebSocket float32 streaming ->
 * rolling risk state / timeline / incident detection, with guaranteed cleanup.
 */
export function useLiveDetection() {
  const [connectionState, setConnectionState] = useState('idle') // idle|connecting|connected|error
  const [sessionId, setSessionId] = useState(null)
  const [elapsed, setElapsed] = useState(0)
  const [riskLevel, setRiskLevel] = useState('LOW_RISK')
  const [riskScore, setRiskScore] = useState(0) // call-level rolling fake score (0..100)
  const [confidence, setConfidence] = useState(0)
  const [windowCount, setWindowCount] = useState(0)
  const [analyzedWindows, setAnalyzedWindows] = useState(0)
  const [peakScore, setPeakScore] = useState(0)
  const [timeline, setTimeline] = useState([])
  const [incident, setIncident] = useState(null)
  const [indicators, setIndicators] = useState([])
  const [error, setError] = useState(null)
  const [levelHistory, setLevelHistory] = useState([]) // { t, score } for chart

  const refs = useRef({
    sessionId: null,
    ws: null,
    stream: null,
    audioCtx: null,
    source: null,
    processor: null,
    mutedGain: null,
    analyser: null,
    raf: null,
    timer: null,
    startedAt: 0,
    peak: 0,
    scores: [],
  })

  const resetState = useCallback(() => {
    setSessionId(null)
    setElapsed(0)
    setRiskLevel('LOW_RISK')
    setRiskScore(0)
    setConfidence(0)
    setWindowCount(0)
    setAnalyzedWindows(0)
    setPeakScore(0)
    setTimeline([])
    setIncident(null)
    setIndicators([])
    setError(null)
    setLevelHistory([])
    const r = refs.current
    r.sessionId = null
    r.peak = 0
    r.scores = []
  }, [])

  const teardownAudio = useCallback(() => {
    const r = refs.current
    if (r.raf) cancelAnimationFrame(r.raf)
    r.raf = null
    if (r.timer) clearInterval(r.timer)
    r.timer = null
    try {
      r.processor?.disconnect()
    } catch {}
    try {
      r.source?.disconnect()
    } catch {}
    try {
      r.mutedGain?.disconnect()
    } catch {}
    try {
      r.analyser?.disconnect()
    } catch {}
    r.stream?.getTracks().forEach((t) => t.stop())
    if (r.audioCtx && r.audioCtx.state !== 'closed') {
      r.audioCtx.close().catch(() => {})
    }
    r.ws = null
    r.stream = null
    r.audioCtx = null
    r.source = null
    r.processor = null
    r.mutedGain = null
    r.analyser = null
  }, [])

  const stop = useCallback(async () => {
    const r = refs.current
    if (r.ws && r.ws.readyState === WebSocket.OPEN) {
      try {
        r.ws.close()
      } catch {}
    }
    teardownAudio()
    setConnectionState('idle')
  }, [teardownAudio])

  const start = useCallback(async () => {
    const r = refs.current
    setError(null)
    resetState()
    setConnectionState('connecting')

    try {
      // 1. Create a backend live session
      const session = await createLiveSession()
      if (!session?.success || !session.session_id) {
        throw new Error('Failed to initialize a live detection session.')
      }
      r.sessionId = session.session_id
      setSessionId(session.session_id)

      // 2. Microphone capture
      if (!navigator?.mediaDevices?.getUserMedia) {
        throw new Error('Microphone access is not supported in this browser (requires HTTPS or localhost).')
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      })
      r.stream = stream

      const AudioCtx = window.AudioContext || window.webkitAudioContext
      if (!AudioCtx) {
        throw new Error('Web Audio API is not supported in this browser.')
      }
      let audioCtx
      try {
        audioCtx = new AudioCtx({ sampleRate: SAMPLE_RATE })
      } catch {
        // Fallback to default hardware sample rate if 16000 is rejected by device
        audioCtx = new AudioCtx()
      }
      r.audioCtx = audioCtx

      // Ensure AudioContext is actively running (handles browser autoplay / suspended state)
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume()
      }

      const source = audioCtx.createMediaStreamSource(stream)
      r.source = source

      // 3. WebSocket (opened before we start sending audio)
      const ws = new WebSocket(buildLiveSocketUrl(r.sessionId))
      r.ws = ws

      await new Promise((resolve, reject) => {
        let settled = false
        ws.onopen = () => {
          settled = true
          resolve()
        }
        ws.onerror = () => {
          if (!settled) {
            settled = true
            reject(new Error('Could not establish the live audio connection. Check backend server.'))
          }
        }
        ws.onclose = (e) => {
          if (!settled) {
            settled = true
            reject(new Error(`Live audio connection closed (code: ${e.code || 'unknown'}).`))
          }
        }
      })

      ws.onmessage = (evt) => {
        let msg
        try {
          msg = JSON.parse(evt.data)
        } catch {
          return
        }

        if (msg.status === 'INDICATOR_UPDATED') {
          setIndicators(msg.indicators || [])
          return
        }

        if (msg.status !== 'ANALYZED' && msg.status !== 'SILENCE') return
        const win = msg.window || {}
        const call = msg.call_risk || {}

        setWindowCount((c) => c + 1)
        if (msg.status === 'ANALYZED') {
          setAnalyzedWindows((c) => c + 1)
          const scorePct = Math.round((win.fake_probability ?? 0) * 100)
          r.peak = Math.max(r.peak, scorePct)
          r.scores.push(scorePct)
          setPeakScore(r.peak)

          setTimeline((prev) => {
            const next = [
              ...prev,
              {
                id: win.window_id ?? prev.length,
                t: win.start_time ?? 0,
                fakeProb: win.fake_probability ?? 0,
                riskScore: win.risk_score ?? 0,
                riskLevel: win.risk_level || 'LOW_RISK',
                confidence: win.confidence ?? 0,
                status: msg.status,
              },
            ]
            return next.slice(-TIMELINE_CAP)
          })

          setLevelHistory((prev) => [
            ...prev,
            { t: win.start_time ?? prev.length * WINDOW_SECONDS, score: scorePct },
          ].slice(-TIMELINE_CAP))
        }

        if (call.risk_level) setRiskLevel(call.risk_level)
        if (typeof call.risk_score === 'number') setRiskScore(Math.round(call.risk_score))
        if (typeof win.confidence === 'number') setConfidence(Math.round(win.confidence))
        if (msg.incident) setIncident(msg.incident)
      }

      ws.onclose = () => {
        teardownAudio()
        setConnectionState((s) => (s === 'connecting' ? 'error' : 'idle'))
      }

      // 4. ScriptProcessor streams float32 PCM frames downsampled to 16kHz over the socket.
      //    A muted gain node connects to destination to keep the processor
      //    running without echoing the microphone back to the user.
      const processor = audioCtx.createScriptProcessor(4096, 1, 1)
      r.processor = processor
      const captureSampleRate = audioCtx.sampleRate
      processor.onaudioprocess = (e) => {
        if (ws.readyState === WebSocket.OPEN) {
          const input = e.inputBuffer.getChannelData(0)
          if (input && input.length) {
            const out16k = downsampleTo16k(input, captureSampleRate)
            ws.send(out16k.buffer)
          }
        }
      }
      const mutedGain = audioCtx.createGain()
      mutedGain.gain.value = 0
      r.mutedGain = mutedGain

      source.connect(processor)
      processor.connect(mutedGain)
      mutedGain.connect(audioCtx.destination)

      // Analyser for the visualizer (display only)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 256
      r.analyser = analyser
      source.connect(analyser)

      // 5. Elapsed clock
      r.startedAt = Date.now()
      r.timer = setInterval(() => {
        setElapsed(Math.floor((Date.now() - r.startedAt) / 1000))
      }, 1000)

      setConnectionState('connected')
    } catch (err) {
      teardownAudio()
      setConnectionState('error')
      setError(err?.message || 'Could not start live monitoring.')
    }
  }, [resetState, teardownAudio])

  const addIndicator = useCallback((indicator) => {
    const r = refs.current
    if (r.ws && r.ws.readyState === WebSocket.OPEN) {
      r.ws.send(JSON.stringify({ action: 'add_indicator', indicator }))
    }
  }, [])

  const exportEvidence = useCallback(async () => {
    const r = refs.current
    if (!r.sessionId) return { ok: false, error: 'No active or recent session to export.' }
    try {
      const rep = await generateIncidentReport(r.sessionId)
      if (rep?.success) return { ok: true, incident: rep.incident, incidentId: rep.incident_id }
      return { ok: false, error: 'Could not generate evidence report.' }
    } catch (e) {
      return { ok: false, error: e.message }
    }
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      const r = refs.current
      try {
        r.ws?.close()
      } catch {}
      if (r.raf) cancelAnimationFrame(r.raf)
      if (r.timer) clearInterval(r.timer)
      try {
        r.processor?.disconnect()
        r.source?.disconnect()
        r.mutedGain?.disconnect()
      } catch {}
      r.stream?.getTracks().forEach((t) => t.stop())
      if (r.audioCtx && r.audioCtx.state !== 'closed') r.audioCtx.close().catch(() => {})
    }
  }, [])

  // Derive averages/peak from reactive state (levelHistory updates each window)
  // rather than from refs, which don't trigger re-renders.
  const avgScore = levelHistory.length
    ? Math.round(levelHistory.reduce((a, p) => a + p.score, 0) / levelHistory.length)
    : 0
  const displayPeak = levelHistory.length
    ? Math.round(Math.max(...levelHistory.map((p) => p.score)))
    : 0

  return {
    connectionState,
    elapsed,
    riskLevel,
    riskScore,
    confidence,
    windowCount,
    analyzedWindows,
    peakScore: displayPeak,
    avgScore,
    timeline,
    levelHistory,
    incident,
    indicators,
    error,
    sessionId: sessionId || refs.current.sessionId,
    analyserRef: refs,
    start,
    stop,
    addIndicator,
    exportEvidence,
  }
}
