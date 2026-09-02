import { useState, useEffect } from 'react';
import { api } from '../services/api';

export function useSystemStatus() {
  const [systemInfo, setSystemInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;

    async function fetchSysStatus() {
      try {
        const data = await api.getStatus();
        if (isMounted) {
          setSystemInfo(data);
          setError(null);
        }
      } catch (err) {
        if (isMounted) setError(err.message);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }

    fetchSysStatus();
    const interval = setInterval(fetchSysStatus, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return { systemInfo, isLoading, error };
}
