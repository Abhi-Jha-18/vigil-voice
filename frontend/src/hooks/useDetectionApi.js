import { useState, useEffect, useCallback } from 'react';

export function useDetectionApi() {
  const [serverStatus, setServerStatus] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [isOnline, setIsOnline] = useState(false);
  
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchStatus() {
      try {
        const res = await fetch('/api/status', { signal: AbortSignal.timeout(4000) });
        if (res.ok) {
          const data = await res.json();
          setServerStatus(data);
          setIsOnline(true);
        }
      } catch (err) {
        console.warn('Could not reach /api/status:', err);
        setIsOnline(false);
      }
    }

    async function fetchModelInfo() {
      try {
        const res = await fetch('/api/model/info');
        if (res.ok) {
          const data = await res.json();
          setModelInfo(data);
        }
      } catch (err) {
        console.warn('Could not reach /api/model/info:', err);
      }
    }

    fetchStatus();
    fetchModelInfo();
  }, []);

  const runDetection = useCallback(async (file, phase, forceVerdict = '') => {
    setIsDetecting(true);
    setDetectionResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('phase', phase);
    if (forceVerdict) {
      formData.append('force_verdict', forceVerdict);
    }

    try {
      const response = await fetch('/api/detect', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        let errorMsg = 'Server returned an error.';
        try {
          const errorData = await response.json();
          if (errorData.error && errorData.error.message) {
            errorMsg = errorData.error.message;
            if (errorData.error.code) {
              errorMsg = `[${errorData.error.code}] ${errorMsg}`;
            }
          } else if (errorData.detail) {
            errorMsg = errorData.detail;
          }
        } catch (e) {
          errorMsg = `HTTP ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorMsg);
      }

      const data = await response.json();
      setDetectionResult(data);
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsDetecting(false);
    }
  }, []);

  return {
    serverStatus,
    modelInfo,
    isOnline,
    isDetecting,
    detectionResult,
    error,
    runDetection
  };
}
