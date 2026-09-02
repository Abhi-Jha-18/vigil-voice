import { useEffect, useRef } from 'react'

/**
 * Real-time frequency-bar visualizer driven by an AnalyserNode.
 * `analyserRef` is the refs object from useLiveDetection (holds `.analyser`).
 */
export default function AudioVisualizer({ analyserRef, active, height = 96 }) {
  const canvasRef = useRef(null)
  const rafRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    const draw = () => {
      rafRef.current = requestAnimationFrame(draw)
      const { width, height: h } = canvas
      const dpr = window.devicePixelRatio || 1
      if (canvas.width !== width * dpr) {
        canvas.width = width * dpr
        canvas.height = h * dpr
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      }

      ctx.clearRect(0, 0, width, h)
      // baseline
      ctx.strokeStyle = 'rgba(148,163,184,0.15)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(0, h / 2)
      ctx.lineTo(width, h / 2)
      ctx.stroke()

      const analyser = analyserRef?.current?.analyser
      if (!analyser || !active) {
        // idle flat wave
        ctx.fillStyle = 'rgba(34,211,238,0.15)'
        for (let x = 0; x < width; x += 8) {
          const barH = 2 + Math.sin(x / 14 + Date.now() / 900) * 1.5
          ctx.fillRect(x, h / 2 - barH / 2, 3, barH)
        }
        return
      }

      const bins = analyser.frequencyBinCount
      const data = new Uint8Array(bins)
      analyser.getByteFrequencyData(data)
      const bars = 48
      const step = Math.floor(bins / bars)
      const barW = width / bars

      for (let i = 0; i < bars; i++) {
        let sum = 0
        for (let j = 0; j < step; j++) sum += data[i * step + j]
        const value = sum / step / 255
        const barH = Math.max(2, value * h * 0.9)
        const grad = ctx.createLinearGradient(0, h, 0, h - barH)
        grad.addColorStop(0, 'rgba(34,211,238,0.35)')
        grad.addColorStop(1, 'rgba(34,211,238,0.95)')
        ctx.fillStyle = grad
        const x = i * barW
        ctx.fillRect(x + barW * 0.18, h - barH, barW * 0.64, barH)
      }
    }
    draw()
    return () => cancelAnimationFrame(rafRef.current)
  }, [analyserRef, active])

  return (
    <div className="panel-soft overflow-hidden p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="eyebrow">Audio activity</span>
        <span className="font-mono text-[11px] text-slate-500">16 kHz · float32</span>
      </div>
      <canvas ref={canvasRef} style={{ height }} className="w-full" />
    </div>
  )
}
