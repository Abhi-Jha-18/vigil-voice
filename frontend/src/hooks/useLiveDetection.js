import { useState, useRef, useCallback } from 'react';
import { api, API_BASE_URL } from '../services/api';

export function useLiveDetection() {
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState('DISCONNECTED'); // DISCONNECTED, CONNECTING, CONNECTED, ERROR
  const [riskLevel, setRiskLevel] = useState('LOW_RISK');
  const [riskScore, setRiskScore] = useState(0);
  const [events, setEvents] = useState([]);
  const [stats, setStats] = useState({ windows: 0, peak: 0, avg: 0, sum: 0 });
  const [error, setError] = useState(null);

  const audioCtxRef = useRef(null);
  const streamRef = useRef(null);
  const processorRef = useRef(null);
  const wsRef = useRef(null);
  const sessionIdRef = useRef(null);
  const analyserRef = useRef(null);

  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
    analyserRef.current = null;
    setIsRecording(false);
    setStatus('DISCONNECTED');
  }, []);

  const startMonitoring = useCallback(async () => {
    cleanup();
    setStatus('CONNECTING');
    setError(null);
    setEvents([]);
    setStats({ windows: 0, peak: 0, avg: 0, sum: 0 });
    setRiskLevel('LOW_RISK');
    setRiskScore(0);

    try {
      // 1. Create Session
      const sessionRes = await api.createLiveSession();
      sessionIdRef.current = sessionRes.session_id;

      // 2. Open WebSocket
      const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/api/live/audio/${sessionRes.session_id}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = async () => {
        setStatus('CONNECTED');
        try {
          // 3. Start Audio Capture
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          streamRef.current = stream;
          
          const AudioContextClass = window.AudioContext || window.webkitAudioContext;
          const ctx = new AudioContextClass({ sampleRate: 16000 });
          audioCtxRef.current = ctx;

          const source = ctx.createMediaStreamSource(stream);
          const analyser = ctx.createAnalyser();
          analyser.fftSize = 256;
          analyserRef.current = analyser;

          const processor = ctx.createScriptProcessor(4096, 1, 1);
          processorRef.current = processor;

          processor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(inputData.buffer); // Send Float32Array buffer
            }
          };

          source.connect(analyser);
          source.connect(processor);
          processor.connect(ctx.destination);
          
          setIsRecording(true);
        } catch (err) {
          setError('Microphone access denied or audio error.');
          cleanup();
        }
      };

      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.status === 'WINDOW_ANALYZED' && data.result) {
          const score = data.result.overall_probability;
          setRiskScore(score);
          setRiskLevel(data.call_risk.risk_level);
          
          setEvents(prev => {
            const newEvents = [{ time: new Date().toLocaleTimeString(), level: data.call_risk.risk_level, score }, ...prev].slice(0, 50);
            return newEvents;
          });

          setStats(prev => {
            const newWindows = prev.windows + 1;
            const newSum = prev.sum + score;
            return {
              windows: newWindows,
              peak: Math.max(prev.peak, score),
              sum: newSum,
              avg: newSum / newWindows
            };
          });
        }
      };

      ws.onerror = () => {
        setError('WebSocket connection error.');
        cleanup();
      };
      
      ws.onclose = () => {
        cleanup();
      };

    } catch (err) {
      setError(err.message);
      cleanup();
    }
  }, [cleanup]);

  const stopMonitoring = useCallback(() => {
    cleanup();
  }, [cleanup]);

  return {
    isRecording,
    status,
    error,
    riskLevel,
    riskScore,
    events,
    stats,
    analyser: analyserRef.current,
    startMonitoring,
    stopMonitoring
  };
}
