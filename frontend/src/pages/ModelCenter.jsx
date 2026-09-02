import React from 'react';
import { PageContainer } from '../components/layout/PageContainer';
import { useModelStatus } from '../hooks/useModelStatus';
import { Cpu, ShieldCheck, Database, Server, Fingerprint } from 'lucide-react';

export function ModelCenter() {
  const { status, isLoading } = useModelStatus();

  return (
    <PageContainer title="Model & AI Center">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="glass-panel md:col-span-2">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold flex items-center gap-2">
              <Cpu className="w-6 h-6 text-primary" />
              Primary Detection Engine
            </h3>
            {isLoading ? (
               <span className="badge bg-surface text-secondary">Loading...</span>
            ) : (
               <span className={`badge ${status?.model_status === 'REAL_MODEL' ? 'badge-success' : 'badge-warning'}`}>
                 {status?.model_status || 'UNAVAILABLE'}
               </span>
            )}
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-surface rounded-xl border border-border">
              <p className="text-xs text-secondary uppercase mb-1">Architecture</p>
              <p className="font-semibold text-lg">SimpleCNNDetector</p>
            </div>
            <div className="p-4 bg-surface rounded-xl border border-border">
              <p className="text-xs text-secondary uppercase mb-1">Framework</p>
              <p className="font-semibold text-lg">PyTorch</p>
            </div>
            <div className="p-4 bg-surface rounded-xl border border-border">
              <p className="text-xs text-secondary uppercase mb-1">Feature Extraction</p>
              <p className="font-semibold text-lg">MFCC Spectrograms</p>
            </div>
            <div className="p-4 bg-surface rounded-xl border border-border">
              <p className="text-xs text-secondary uppercase mb-1">Dataset / Source</p>
              <p className="font-semibold text-lg">{status?.metadata?.dataset_identifier || 'N/A'}</p>
            </div>
          </div>
        </div>

        <div className="glass-panel">
          <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-success" />
            Integrity Check
          </h3>
          <div className="space-y-6">
            <div>
              <p className="text-sm text-secondary mb-1">Model Version</p>
              <p className="font-mono bg-surface p-2 rounded border border-border text-xs">
                {status?.metadata?.model_version || 'UNKNOWN'}
              </p>
            </div>
            <div>
              <p className="text-sm text-secondary mb-1 flex items-center gap-1">
                <Fingerprint className="w-4 h-4" /> SHA-256 Checksum
              </p>
              <p className="font-mono bg-surface p-2 rounded border border-border text-xs break-all text-success">
                {status?.metadata?.model_sha256 || 'N/A'}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="glass-panel">
        <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
          <Database className="w-6 h-6 text-primary" />
          Evaluation Metrics (Held-out Test Set)
        </h3>
        
        {status?.evaluation?.eer !== undefined ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-6 text-center border-r border-border last:border-0">
              <p className="text-3xl font-bold text-white mb-2">{(status.evaluation.eer * 100).toFixed(2)}%</p>
              <p className="text-sm text-secondary uppercase">EER</p>
            </div>
            <div className="p-6 text-center border-r border-border last:border-0">
              <p className="text-3xl font-bold text-white mb-2">{status.evaluation.roc_auc?.toFixed(4)}</p>
              <p className="text-sm text-secondary uppercase">ROC-AUC</p>
            </div>
            <div className="p-6 text-center border-r border-border last:border-0">
              <p className="text-3xl font-bold text-white mb-2">{(status.evaluation.far * 100).toFixed(2)}%</p>
              <p className="text-sm text-secondary uppercase">FAR</p>
            </div>
            <div className="p-6 text-center border-r border-border last:border-0">
              <p className="text-3xl font-bold text-white mb-2">{(status.evaluation.frr * 100).toFixed(2)}%</p>
              <p className="text-sm text-secondary uppercase">FRR</p>
            </div>
          </div>
        ) : (
          <div className="p-8 text-center text-secondary opacity-70">
            Evaluation metrics not available. The REAL_MODEL has not been trained or evaluated on a genuine dataset yet.
          </div>
        )}
      </div>
    </PageContainer>
  );
}
