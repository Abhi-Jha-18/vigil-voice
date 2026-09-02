import { useState, useRef, useCallback } from 'react';

export function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [recordedFile, setRecordedFile] = useState(null);
  
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const recordIntervalRef = useRef(null);
  const audioChunksRef = useRef([]);
  const animationFrameIdRef = useRef(null);
  const processorRef = useRef(null);

  const cleanupAudioContext = useCallback(() => {
    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current);
      animationFrameIdRef.current = null;
    }
    if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
      audioCtxRef.current.close().catch(() => {});
    }
    audioCtxRef.current = null;
    analyserRef.current = null;
    sourceNodeRef.current = null;
    processorRef.current = null;
  }, []);

  const stopRecording = useCallback(() => {
    if (!isRecording) return;
    clearInterval(recordIntervalRef.current);
    setIsRecording(false);

    if (processorRef.current) processorRef.current.disconnect();
    if (sourceNodeRef.current) sourceNodeRef.current.disconnect();
    if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
      audioCtxRef.current.close();
    }

    const pcmChunks = audioChunksRef.current.filter(chunk => chunk && chunk.length > 0);
    if (!pcmChunks.length) {
      console.warn('No valid audio data captured.');
      cleanupAudioContext();
      return null;
    }

    const wavBlob = audioFloatChunksToWavBlob(pcmChunks);
    const file = new File([wavBlob], 'recorded_voice.wav', { type: 'audio/wav' });
    setRecordedFile(file);
    cleanupAudioContext();
    return file;
  }, [isRecording, cleanupAudioContext]);

  const startRecording = useCallback(async (canvasRef) => {
    audioChunksRef.current = [];
    setRecordSeconds(0);
    setRecordedFile(null);
    setIsRecording(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) throw new Error('Web Audio API not supported.');

      audioCtxRef.current = new AudioContextClass();
      sourceNodeRef.current = audioCtxRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioCtxRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;

      const processor = audioCtxRef.current.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      sourceNodeRef.current.connect(analyserRef.current);
      sourceNodeRef.current.connect(processor);
      processor.connect(audioCtxRef.current.destination);

      processor.onaudioprocess = (event) => {
        const inputBuffer = event.inputBuffer.getChannelData(0);
        if (inputBuffer && inputBuffer.length > 0) {
          audioChunksRef.current.push(new Float32Array(inputBuffer));
        }
      };

      if (canvasRef && canvasRef.current) {
        setupVisualizer(canvasRef.current);
      }

      recordIntervalRef.current = setInterval(() => {
        setRecordSeconds(prev => prev + 1);
      }, 1000);

    } catch (err) {
      console.error('Microphone access denied:', err);
      setIsRecording(false);
      throw err;
    }
  }, []);

  const setupVisualizer = (canvas) => {
    const canvasCtx = canvas.getContext('2d');
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    const draw = () => {
      if (!audioCtxRef.current) return;
      animationFrameIdRef.current = requestAnimationFrame(draw);
      
      analyserRef.current.getByteFrequencyData(dataArray);
      
      canvasCtx.fillStyle = '#0d0c1e';
      canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
      
      const barWidth = (canvas.width / bufferLength) * 1.5;
      let x = 0;
      
      for (let i = 0; i < bufferLength; i++) {
        const barHeight = dataArray[i] / 2;
        const r = 6 + i * 2;
        const g = 182 - i;
        const b = 212 + i;
        canvasCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
        x += barWidth;
      }
    };
    draw();
  };

  return {
    isRecording,
    recordSeconds,
    recordedFile,
    setRecordedFile,
    startRecording,
    stopRecording
  };
}

// Utility function to convert float chunks to wav blob
function audioFloatChunksToWavBlob(floatChunks) {
  const totalLength = floatChunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const buffer = new ArrayBuffer(44 + totalLength * 2);
  const view = new DataView(buffer);

  function writeString(offset, value) {
    for (let i = 0; i < value.length; i++) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  }

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + totalLength * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true);
  view.setUint32(28, 16000 * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, totalLength * 2, true);

  let offset = 44;
  for (const chunk of floatChunks) {
    for (let i = 0; i < chunk.length; i++) {
      const sample = Math.max(-1, Math.min(1, chunk[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([buffer], { type: 'audio/wav' });
}
