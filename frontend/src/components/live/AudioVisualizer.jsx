import React, { useEffect, useRef } from 'react';

export function AudioVisualizer({ analyser }) {
  const canvasRef = useRef(null);
  const animationIdRef = useRef(null);

  useEffect(() => {
    if (!analyser || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      if (!analyser) return;
      animationIdRef.current = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);

      // Match background
      ctx.fillStyle = '#171530'; 
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 1.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height;
        const r = 6 + i * 2;
        const g = 182 - i;
        const b = 212 + i;
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
        x += barWidth;
      }
    };

    draw();

    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
    };
  }, [analyser]);

  return (
    <div className="w-full h-48 rounded-xl overflow-hidden border border-border bg-[#171530]">
      {analyser ? (
        <canvas ref={canvasRef} className="w-full h-full" width={800} height={200} />
      ) : (
        <div className="w-full h-full flex items-center justify-center text-secondary opacity-50 relative overflow-hidden">
          <div className="absolute inset-0 bg-glow-primary opacity-5"></div>
          <span>~ waveform inactive ~</span>
        </div>
      )}
    </div>
  );
}
