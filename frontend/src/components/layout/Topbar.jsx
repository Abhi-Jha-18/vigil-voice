import React from 'react';
import { Bell, User } from 'lucide-react';
import { useModelStatus } from '../../hooks/useModelStatus';

export function Topbar() {
  const { status, isLoading } = useModelStatus();

  return (
    <header className="h-16 border-b border-border bg-surface/50 backdrop-blur flex items-center justify-between px-6 sticky top-0 z-10">
      <div className="flex items-center">
        <h1 className="text-lg font-medium text-white/90">Command Center</h1>
      </div>
      
      <div className="flex items-center gap-6">
        <div className="hidden sm:flex items-center gap-2">
          <span className="text-xs text-secondary">Engine Status:</span>
          {isLoading ? (
            <span className="badge bg-surface text-secondary">Checking...</span>
          ) : (
            <span className={`badge ${
              status?.model_status === 'REAL_MODEL' ? 'badge-success' :
              status?.model_status === 'DEMO_MODEL' ? 'badge-warning' : 'badge-danger'
            }`}>
              {status?.model_status || 'UNAVAILABLE'}
            </span>
          )}
        </div>
        
        <button className="text-secondary hover:text-white transition-colors relative">
          <Bell className="w-5 h-5" />
          <span className="absolute top-0 right-0 w-2 h-2 bg-danger rounded-full ring-2 ring-[#0a0914]"></span>
        </button>
        
        <div className="w-8 h-8 rounded-full bg-surface border border-border flex items-center justify-center">
          <User className="w-4 h-4 text-secondary" />
        </div>
      </div>
    </header>
  );
}
