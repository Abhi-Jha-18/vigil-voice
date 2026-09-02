import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Sidebar } from './components/layout/Sidebar';
import { Topbar } from './components/layout/Topbar';
import { 
  Dashboard, 
  LiveDetection, 
  UploadAnalysis, 
  Incidents, 
  ModelCenter, 
  Reports, 
  Settings 
} from './pages';

function App() {
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden relative z-0">
        <Topbar />
        <div className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/live" element={<LiveDetection />} />
            <Route path="/analyze" element={<UploadAnalysis />} />
            <Route path="/incidents" element={<Incidents />} />
            <Route path="/model" element={<ModelCenter />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </div>
      </div>
    </div>
  );
}

export default App;
