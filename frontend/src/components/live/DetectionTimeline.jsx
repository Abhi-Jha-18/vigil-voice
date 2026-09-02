import React from 'react';
import { AlertCircle, CheckCircle, HelpCircle } from 'lucide-react';

export function DetectionTimeline({ events }) {
  if (!events || events.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-secondary opacity-50 py-12">
        <p>Awaiting analysis...</p>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto pr-2 space-y-3">
      {events.map((evt, idx) => {
        let Icon = CheckCircle;
        let color = 'text-success';
        
        if (evt.level === 'HIGH_SPOOF_RISK') {
          Icon = AlertCircle;
          color = 'text-danger';
        } else if (evt.level === 'SUSPICIOUS') {
          Icon = HelpCircle;
          color = 'text-warning';
        }

        return (
          <div key={idx} className="flex items-center gap-4 p-3 rounded-xl bg-surface/30 border border-border/50 hover:bg-surface/50 transition-colors">
            <Icon className={`w-5 h-5 ${color}`} />
            <div className="flex-1">
              <p className={`text-sm font-semibold ${color}`}>{evt.level.replace(/_/g, ' ')}</p>
              <p className="text-xs text-secondary">{evt.time}</p>
            </div>
            <div className="text-sm font-medium text-white/80">
              {Math.round(evt.score * 100)}%
            </div>
          </div>
        );
      })}
    </div>
  );
}
