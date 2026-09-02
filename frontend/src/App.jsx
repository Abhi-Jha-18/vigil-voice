import { Suspense, lazy, useState } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import Sidebar from './components/layout/Sidebar'
import Topbar from './components/layout/Topbar'
import { PageLoader } from './components/common/Loading'
import { useModelStatus } from './hooks/useModelStatus'

// Route-level code splitting keeps the initial bundle lean.
const Dashboard = lazy(() => import('./pages/Dashboard'))
const LiveDetection = lazy(() => import('./pages/LiveDetection'))
const UploadAnalysis = lazy(() => import('./pages/UploadAnalysis'))
const Incidents = lazy(() => import('./pages/Incidents'))
const IncidentDetails = lazy(() => import('./pages/IncidentDetails'))
const ModelCenter = lazy(() => import('./pages/ModelCenter'))
const Reports = lazy(() => import('./pages/Reports'))
const Settings = lazy(() => import('./pages/Settings'))

const TITLES = {
  '/': 'Security Dashboard',
  '/live': 'Live Detection',
  '/analyze': 'Upload & Analyze',
  '/incidents': 'Incident Center',
  '/model': 'Model & AI Center',
  '/reports': 'Reports & Evidence',
  '/settings': 'Settings',
}

function titleFor(pathname) {
  if (pathname.startsWith('/incidents/')) return 'Incident Details'
  return TITLES[pathname] || 'VigilVoice'
}

export default function App() {
  const [navOpen, setNavOpen] = useState(false)
  const location = useLocation()
  const model = useModelStatus()

  return (
    <div className="relative z-10 min-h-screen">
      <Sidebar open={navOpen} onClose={() => setNavOpen(false)} />

      <div className="lg:pl-64">
        <Topbar onMenu={() => setNavOpen(true)} title={titleFor(location.pathname)} model={model} />

        <main className="min-h-[calc(100vh-4rem)] pb-16">
          <Suspense fallback={<PageLoader label="Loading view…" />}>
            <Routes location={location} key={location.pathname}>
              <Route path="/" element={<Dashboard model={model} />} />
              <Route path="/live" element={<LiveDetection />} />
              <Route path="/analyze" element={<UploadAnalysis model={model} />} />
              <Route path="/incidents" element={<Incidents />} />
              <Route path="/incidents/:id" element={<IncidentDetails />} />
              <Route path="/model" element={<ModelCenter model={model} />} />
              <Route path="/reports" element={<Reports model={model} />} />
              <Route path="/settings" element={<Settings model={model} />} />
              <Route path="*" element={<Dashboard model={model} />} />
            </Routes>
          </Suspense>
        </main>

        <footer className="border-t border-line py-5 text-center text-xs text-slate-600">
          © 2026 VigilVoice · Smart India Hackathon prototype · Probabilistic risk indicators — not
          legal proof
        </footer>
      </div>
    </div>
  )
}
