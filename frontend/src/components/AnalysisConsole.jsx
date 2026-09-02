import React from 'react';

export default function AnalysisConsole({ detectionResult, isDetecting }) {
  if (!detectionResult && !isDetecting) {
    return (
      <section className="glass-panel" id="analysis-console" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <h2 className="panel-title">
          <span>Real-Time AI Analysis Console</span>
          <span style={{ fontFamily: "'JetBrains Mono'", fontSize: '0.85rem', color: 'var(--color-cyan)' }}>READY</span>
        </h2>
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
          <div style={{ opacity: 0.25, maxWidth: 280, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
            <svg width="64" height="64" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 20C7.59 20 4 16.41 4 12C4 7.59 7.59 4 12 4C16.41 4 20 7.59 20 12C20 16.41 16.41 20 12 20ZM11 6H13V12H11V6ZM11 14H13V16H11V14Z"/>
            </svg>
            <p style={{ fontWeight: 500, fontSize: '1.05rem' }}>Console Awaiting Input</p>
            <p style={{ fontSize: '0.8rem', lineHeight: 1.4 }}>Provide a voice clip via upload or record using the microphone on the left, then trigger the AI engine.</p>
          </div>
        </div>
      </section>
    );
  }

  const result = detectionResult || {};
  const isReady = !!detectionResult;

  const riskLevel = result?.risk_level || 'UNKNOWN';
  const riskScore = result?.raw_score || 0;
  const verdict = result?.verdict?.toLowerCase() || 'uncertain';
  const confidence = result?.confidence || 0;
  const circumference = 220;
  const offset = circumference - (confidence / 100) * circumference;

  let progressColor = 'var(--color-amber)';
  if (verdict === 'real') progressColor = 'var(--color-emerald)';
  if (verdict === 'fake') progressColor = 'var(--color-rose)';

  return (
    <section className="glass-panel" id="analysis-console">
      <h2 className="panel-title">
        <span>Real-Time AI Analysis Console</span>
        <span style={{ fontFamily: "'JetBrains Mono'", fontSize: '0.85rem', color: 'var(--color-cyan)' }}>
          {isDetecting ? 'ANALYZING...' : `COMPLETED IN ${result.processing_time_ms}ms`}
        </span>
      </h2>

      <div className="analysis-stepper">
        {['VAD & Prep', 'Features', 'AI Engine', 'Decision'].map((label, idx) => (
          <div key={label} className={`step-item ${isReady || idx === 0 ? 'completed' : isDetecting ? 'active' : ''}`}>
            <div className="step-icon">{idx + 1}</div>
            <span className="step-label">{label}</span>
          </div>
        ))}
      </div>

      {isReady ? (
        <div className="results-container">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
            
            {/* Risk Assessment */}
            <div className={`verdict-card ${verdict}`} style={{ flex: 1 }}>
              <div className="verdict-title">Overall Risk Assessment</div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
                <div className="verdict-value" style={{ fontSize: '2.5rem' }}>{result.verdict}</div>
                <span className="risk-tag">{riskLevel} RISK</span>
              </div>
              <div style={{ marginTop: '0.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                Risk Score: <strong style={{ color: 'var(--text-primary)' }}>{riskScore}/100</strong>
              </div>
            </div>

            {/* Confidence Meter */}
            <div className="meter-card" style={{ flex: 1 }}>
              <div className="radial-container">
                <svg className="radial-svg" viewBox="0 0 80 80">
                  <circle className="radial-bg" cx="40" cy="40" r="35"></circle>
                  <circle className="radial-progress" cx="40" cy="40" r="35" style={{ strokeDashoffset: offset, stroke: progressColor }}></circle>
                </svg>
                <div className="radial-text">{confidence}%</div>
              </div>
              <div className="meter-info">
                <div className="meter-label">AI Confidence</div>
                <div className="meter-desc" style={{ marginBottom: '0.5rem' }}>
                  {verdict === 'fake' ? 'High probability of synthetic audio.' : 'Authentic speech patterns detected.'}
                </div>
              </div>
            </div>
          </div>

          {/* Spectrogram */}
          {result.spectrogram && (
            <div className="visual-section" style={{ marginTop: '1rem' }}>
              <div className="settings-label">Extracted Feature Maps (Mel Spectrogram)</div>
              <div className="spectrogram-wrapper">
                <img className="spectrogram-img" src={result.spectrogram} alt="Mel Spectrogram" />
              </div>
            </div>
          )}
        </div>
      ) : (
        <div style={{ padding: '2rem', textAlign: 'center' }}>
           {/* Loading state can show sweeping animation here */}
           <div className="spectrogram-wrapper">
              <div className="scan-line" style={{ display: 'block' }}></div>
              <div className="spectrogram-placeholder">
                  <span>Extracting Features...</span>
              </div>
           </div>
        </div>
      )}
    </section>
  );
}
