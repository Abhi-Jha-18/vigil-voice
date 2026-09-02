import React from 'react';

export default function Header({ serverStatus }) {
  const isOnline = !!serverStatus;
  const cnnAvailable = serverStatus?.models?.phase2_cnn?.available;

  return (
    <header>
      <div className="logo-container">
        <svg className="logo-svg" width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ filter: 'drop-shadow(0 0 8px rgba(6, 182, 212, 0.6))' }}>
          <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13 17H11V15H13V17ZM13 13H11V7H13V13Z" fill="#06b6d4"/>
        </svg>
        <span className="logo-text">VIGILVOICE</span>
        <span className="badge">PROTOTYPE V1.0</span>
      </div>
      <div className="header-status">
        <span id="status-indicator" style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span id="status-dot" style={{
            display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%',
            background: isOnline ? 'var(--color-emerald)' : '#6b7280',
            boxShadow: isOnline ? 'var(--glow-emerald)' : 'none'
          }}></span>
          <span id="status-text">{isOnline ? 'SYSTEM ONLINE' : 'CONNECTING...'}</span>
        </span>
        
        {isOnline && (
          <span id="model-status-badge" style={{
            display: 'inline-block', marginLeft: '0.75rem', fontSize: '0.75rem', padding: '0.2rem 0.6rem', borderRadius: '999px', fontWeight: 600, letterSpacing: '0.05em',
            background: cnnAvailable ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.12)',
            color: cnnAvailable ? 'var(--color-emerald)' : 'var(--color-amber)',
            border: cnnAvailable ? '1px solid rgba(16,185,129,0.35)' : '1px solid rgba(245,158,11,0.3)'
          }}>
            {cnnAvailable ? '🧠 CNN MODEL READY' : '⚡ PHASE 1 HEURISTIC'}
          </span>
        )}
      </div>
    </header>
  );
}
