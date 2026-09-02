import React from 'react';
import { Dashboard } from './Dashboard';
import { LiveDetection } from './LiveDetection';
import { UploadAnalysis } from './UploadAnalysis';
import { Incidents } from './Incidents';
import { ModelCenter } from './ModelCenter';
import { PageContainer } from '../components/layout/PageContainer';

// Placeholder for Reports
export function Reports() {
  return (
    <PageContainer title="Reports & Evidence">
      <div className="glass-panel p-16 text-center">
        <h3 className="text-xl font-medium text-white mb-2">Reports Repository</h3>
        <p className="text-secondary max-w-md mx-auto">
          Currently, report generation and historical access are handled locally on the server filesystem.
          API access to these artifacts is planned for a future update.
        </p>
      </div>
    </PageContainer>
  );
}

// Placeholder for Settings
export function Settings() {
  return (
    <PageContainer title="Settings">
      <div className="glass-panel max-w-2xl">
        <h3 className="text-lg font-semibold mb-6 border-b border-border pb-2">Privacy & Security</h3>
        <div className="space-y-4">
          <div className="p-4 bg-surface rounded-xl border border-border">
            <h4 className="font-medium text-white mb-2">Data Retention</h4>
            <p className="text-sm text-secondary">
              Raw live audio is not stored by default. Audio chunks are processed transiently in memory 
              during active sessions to preserve user privacy.
            </p>
          </div>
          <div className="p-4 bg-surface rounded-xl border border-border">
            <h4 className="font-medium text-white mb-2">Telemetry</h4>
            <p className="text-sm text-secondary">
              VigilVoice operates entirely offline/on-premise by default. No data is sent to third-party servers.
            </p>
          </div>
        </div>
      </div>
    </PageContainer>
  );
}

export { Dashboard, LiveDetection, UploadAnalysis, Incidents, ModelCenter };
