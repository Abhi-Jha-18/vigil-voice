import React from 'react';
import { PageContainer } from '../components/layout/PageContainer';
import { AlertTriangle, Filter, Search } from 'lucide-react';

export function Incidents() {
  return (
    <PageContainer title="Incident Center">
      <div className="glass-panel mb-6 flex flex-col md:flex-row gap-4 justify-between items-center">
        <div className="relative w-full md:w-96">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-secondary" />
          <input 
            type="text" 
            placeholder="Search incident ID..." 
            className="w-full bg-surface/50 border border-border rounded-lg pl-10 pr-4 py-2 text-sm text-white focus:outline-none focus:border-primary transition-colors"
            disabled
          />
        </div>
        <div className="flex gap-2 w-full md:w-auto">
          <button className="flex items-center gap-2 px-4 py-2 bg-surface border border-border rounded-lg text-sm text-secondary hover:text-white transition-colors" disabled>
            <Filter className="w-4 h-4" /> Filter
          </button>
        </div>
      </div>

      <div className="glass-panel overflow-hidden">
        <div className="p-16 flex flex-col items-center justify-center text-center">
          <AlertTriangle className="w-16 h-16 text-secondary/50 mb-4" />
          <h3 className="text-xl font-medium text-white mb-2">No historical data available</h3>
          <p className="text-secondary max-w-md">
            The FastAPI backend does not currently expose a historical global incident reporting API. 
            Live session incidents are processed transiently per session.
          </p>
        </div>
      </div>
    </PageContainer>
  );
}
