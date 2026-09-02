import React from 'react';
import { PageContainer } from '../components/layout/PageContainer';
import { RiskMeter } from '../components/live/RiskMeter';
import { AudioVisualizer } from '../components/live/AudioVisualizer';
import { DetectionTimeline } from '../components/live/DetectionTimeline';
import { useLiveDetection } from '../hooks/useLiveDetection';
import { Mic, Square, Wifi, WifiOff, AlertTriangle } from 'lucide-react';

export function LiveDetection() {
  const {
    isRecording,
    status,
    error,
    riskLevel,
    riskScore,
    events,
    stats,
    analyser,
    startMonitoring,
    stopMonitoring
  } = useLiveDetection();

  return (
    <PageContainer title="Live Voice Monitor">
      <div className="flex justify-between items-end mb-6">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-full ${status === 'CONNECTED' ? 'bg-success/20 text-success' : 'bg-surface border border-border text-secondary'}`}>
            {status === 'CONNECTED' ? <Wifi className="w-5 h-5" /> : <WifiOff className="w-5 h-5" />}
          </div>
          <div>
            <p className="text-sm text-secondary">Connection Status</p>
            <p className="font-semibold">{status}</p>
          </div>
        </div>

        <button
          onClick={isRecording ? stopMonitoring : startMonitoring}
          disabled={status === 'CONNECTING'}
          className={`flex items-center gap-2 font-semibold py-3 px-6 rounded-xl shadow-lg transition-all duration-300 disabled:opacity-50 ${
            isRecording 
              ? 'bg-danger text-white hover:bg-danger/80' 
              : 'bg-primary text-white hover:brightness-110'
          }`}
        >
          {isRecording ? <Square className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
          {isRecording ? 'STOP MONITORING' : 'START MONITORING'}
        </button>
      </div>

      {error && (
        <div className="mb-6 p-4 rounded-xl bg-danger/10 border border-danger/30 flex items-center gap-3 text-rose-300">
          <AlertTriangle className="w-5 h-5" />
          <p>{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-4">
          <RiskMeter level={riskLevel} score={riskScore} />
        </div>
        
        <div className="lg:col-span-8 flex flex-col gap-6">
          <div className="glass-panel p-4 flex-1">
            <h3 className="text-sm font-semibold tracking-wide text-secondary uppercase mb-4">Audio Visualization</h3>
            <AudioVisualizer analyser={analyser} />
            <div className="flex justify-around mt-4 py-3 bg-surface/50 rounded-lg border border-border">
              <div className="text-center">
                <p className="text-xs text-secondary uppercase">Windows</p>
                <p className="font-mono text-lg">{stats.windows}</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-secondary uppercase">Peak Score</p>
                <p className="font-mono text-lg text-danger">{Math.round(stats.peak * 100)}%</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-secondary uppercase">Avg Score</p>
                <p className="font-mono text-lg text-warning">{Math.round(stats.avg * 100)}%</p>
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-12 glass-panel h-80">
          <h3 className="text-sm font-semibold tracking-wide text-secondary uppercase mb-4">Detection Timeline</h3>
          <DetectionTimeline events={events} />
        </div>
      </div>
    </PageContainer>
  );
}
