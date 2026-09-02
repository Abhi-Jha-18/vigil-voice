import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The built React app is emitted directly into the existing FastAPI-served
// `static/` directory so the current Docker / uvicorn deployment keeps working
// without a second web server. During `npm run dev`, Vite proxies /api (REST +
// WebSocket) to the FastAPI backend.
export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: '../static',
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 1400,
  },
  server: {
    host: true,
    port: 5173,
    // Accept the sandbox/preview proxy hosts in development. A leading-dot
    // entry matches all subdomains of that suffix (e.g. *.e2b.app).
    allowedHosts: ['.e2b.app', '.e2b.dev', 'localhost', '127.0.0.1'],
    proxy: {
      '/api': {
        target: process.env.VITE_PROXY_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
        ws: true,
      },
    },
  },
})
