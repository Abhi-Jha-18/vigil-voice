export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function fetchWithTimeout(url, options = {}, timeout = 30000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    
    if (!response.ok) {
      let errorDetail = 'API error';
      try {
        const errorData = await response.json();
        errorDetail = errorData.detail || errorData.error?.message || errorDetail;
      } catch (e) {
        errorDetail = `HTTP ${response.status} ${response.statusText}`;
      }
      throw new Error(errorDetail);
    }
    
    return await response.json();
  } finally {
    clearTimeout(id);
  }
}

export const api = {
  getHealth: () => fetchWithTimeout(`${API_BASE_URL}/api/health`),
  getStatus: () => fetchWithTimeout(`${API_BASE_URL}/api/status`),
  getModelInfo: () => fetchWithTimeout(`${API_BASE_URL}/api/model/info`),
  
  analyzeAudio: async (file, phase = 'phase2') => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('phase', phase);
    
    return fetchWithTimeout(`${API_BASE_URL}/api/detect`, {
      method: 'POST',
      body: formData
    }, 60000);
  },

  createLiveSession: () => fetchWithTimeout(`${API_BASE_URL}/api/live/session`, { method: 'POST' }),
  
  getLiveStatus: (sessionId) => fetchWithTimeout(`${API_BASE_URL}/api/live/status/${sessionId}`),
  
  getReportingResources: () => fetchWithTimeout(`${API_BASE_URL}/api/live/reporting-resources`)
};
