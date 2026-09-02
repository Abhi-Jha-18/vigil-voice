import React from 'react';

export function StatCard({ title, value, subtitle, icon: Icon, colorClass = "text-primary" }) {
  return (
    <div className="glass-panel flex items-start gap-4">
      <div className={`p-3 rounded-xl bg-surface border border-border ${colorClass}`}>
        <Icon className="w-6 h-6" />
      </div>
      <div>
        <p className="text-sm font-medium text-secondary">{title}</p>
        <h3 className="text-2xl font-bold mt-1 text-white">{value}</h3>
        {subtitle && (
          <p className="text-xs text-secondary mt-1">{subtitle}</p>
        )}
      </div>
    </div>
  );
}
