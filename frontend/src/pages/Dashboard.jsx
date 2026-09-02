import React from 'react';
import { PageContainer } from '../components/layout/PageContainer';
import { StatCard } from '../components/dashboard/StatCard';
import { useSystemStatus } from '../hooks/useSystemStatus';
import { useModelStatus } from '../hooks/useModelStatus';
import { Activity, Cpu, ShieldAlert, Clock, CheckCircle, XCircle } from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip
} from 'recharts';

// Fictional data representing an empty state or placeholder graph for visual completeness
// since the backend does not expose historical timeseries analytics yet.
const mockActivityData = [
  { time: '00:00', load: 12 },
  { time: '04:00', load: 18 },
  { time: '08:00', load: 45 },
  { time: '12:00', load: 60 },
  { time: '16:00', load: 35 },
  { time: '20:00', load: 20 },
  { time: '24:00', load: 15 },
];

export function Dashboard() {
  const { systemInfo, isLoading: sysLoading } = useSystemStatus();
  const { status: modelInfo, isLoading: modelLoading } = useModelStatus();

  function formatUptime(seconds) {
    if (!seconds) return '0h 0m';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
  }

  const isCnnAvailable = systemInfo?.models?.phase2_cnn?.available;
  const currentModelStatus = modelInfo?.model_status || 'UNKNOWN';
  
  return (
    <PageContainer title="System Dashboard">
      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard 
          title="Engine Status" 
          value={currentModelStatus === 'REAL_MODEL' ? 'Secure' : currentModelStatus} 
          subtitle={sysLoading ? 'Loading...' : `VigilVoice Core v${systemInfo?.version || '1.0'}`}
          icon={ShieldAlert}
          colorClass={currentModelStatus === 'REAL_MODEL' ? 'text-success' : 'text-warning'}
        />
        <StatCard 
          title="Uptime" 
          value={sysLoading ? '...' : formatUptime(systemInfo?.uptime_seconds)}
          subtitle="Since last restart"
          icon={Clock}
          colorClass="text-primary"
        />
        <StatCard 
          title="Phase 2 CNN" 
          value={isCnnAvailable ? 'Online' : 'Offline'}
          subtitle={systemInfo?.models?.phase2_cnn?.version || 'N/A'}
          icon={Cpu}
          colorClass={isCnnAvailable ? 'text-success' : 'text-danger'}
        />
        <StatCard 
          title="Live Analyses" 
          value="No Data"
          subtitle="Endpoint not available"
          icon={Activity}
          colorClass="text-secondary"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Details Panel */}
        <div className="lg:col-span-1 glass-panel">
          <h3 className="text-lg font-semibold mb-4 border-b border-border pb-2">Active Detection Models</h3>
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-surface/50 border border-border">
              <div className="flex justify-between items-center mb-2">
                <span className="font-medium text-white">Phase 1: Heuristic</span>
                {systemInfo?.models?.phase1_heuristic?.available ? 
                  <CheckCircle className="w-5 h-5 text-success" /> : 
                  <XCircle className="w-5 h-5 text-danger" />
                }
              </div>
              <p className="text-xs text-secondary">Acoustic variance & spectral centroid analysis</p>
            </div>
            
            <div className="p-4 rounded-xl bg-surface/50 border border-border">
              <div className="flex justify-between items-center mb-2">
                <span className="font-medium text-white">Phase 2: CNN</span>
                {isCnnAvailable ? 
                  <CheckCircle className="w-5 h-5 text-success" /> : 
                  <XCircle className="w-5 h-5 text-danger" />
                }
              </div>
              <p className="text-xs text-secondary">Deep spectrogram analysis network</p>
              {modelInfo?.evaluation?.eer !== undefined && (
                <div className="mt-2 text-xs text-primary flex gap-2">
                  <span>EER: {(modelInfo.evaluation.eer * 100).toFixed(2)}%</span>
                  <span>|</span>
                  <span>AUC: {modelInfo.evaluation.roc_auc?.toFixed(4)}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Activity Chart (Placeholder for now) */}
        <div className="lg:col-span-2 glass-panel flex flex-col">
          <h3 className="text-lg font-semibold mb-4">System Activity (24h)</h3>
          <p className="text-xs text-secondary mb-4">Note: Real-time historical plotting API is currently unavailable. This is a placeholder visualization.</p>
          <div className="flex-1 w-full h-64 min-h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={mockActivityData}>
                <defs>
                  <linearGradient id="colorLoad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="time" stroke="#a0a0c0" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#a0a0c0" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#191632', borderColor: 'rgba(255,255,255,0.1)' }}
                  itemStyle={{ color: '#06b6d4' }}
                />
                <Area type="monotone" dataKey="load" stroke="#06b6d4" strokeWidth={2} fillOpacity={1} fill="url(#colorLoad)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </PageContainer>
  );
}
