import React from 'react';

export function RiskMeter({ level, score }) {
  const percentage = Math.round(score * 100);
  
  let color = 'text-success';
  let bgColor = 'bg-success';
  let bgGlow = 'shadow-[0_0_30px_rgba(16,185,129,0.3)]';
  let label = 'Low Risk';
  let description = 'Normal voice characteristics detected.';

  if (level === 'SUSPICIOUS') {
    color = 'text-warning';
    bgColor = 'bg-warning';
    bgGlow = 'shadow-[0_0_30px_rgba(245,158,11,0.3)]';
    label = 'Suspicious';
    description = 'Suspicious voice characteristics detected.';
  } else if (level === 'HIGH_SPOOF_RISK') {
    color = 'text-danger';
    bgColor = 'bg-danger';
    bgGlow = 'shadow-[0_0_30px_rgba(244,63,94,0.4)] animate-pulse-ring';
    label = 'High Spoof Risk';
    description = 'Potential Voice Spoofing Detected.';
  }

  return (
    <div className={`flex flex-col items-center justify-center p-8 rounded-2xl bg-surface border border-border ${bgGlow} transition-all duration-500`}>
      <h3 className="text-sm font-semibold tracking-widest text-secondary uppercase mb-6">Risk Status</h3>
      
      <div className="relative w-48 h-48 flex items-center justify-center mb-6">
        <svg className="absolute inset-0 w-full h-full transform -rotate-90">
          <circle cx="96" cy="96" r="88" className="stroke-surface-hover fill-none stroke-[12]" />
          <circle 
            cx="96" cy="96" r="88" 
            className={`fill-none stroke-[12] transition-all duration-700 ease-out ${color.replace('text', 'stroke')}`}
            strokeDasharray="553"
            strokeDashoffset={553 - (553 * percentage) / 100}
            strokeLinecap="round"
          />
        </svg>
        <div className="flex flex-col items-center">
          <span className={`text-5xl font-bold ${color}`}>{percentage}%</span>
          <span className="text-xs text-secondary mt-1">Spoof Score</span>
        </div>
      </div>
      
      <div className={`text-2xl font-bold tracking-wide ${color}`}>{label}</div>
      <p className="text-sm text-secondary mt-2 text-center">{description}</p>
    </div>
  );
}
