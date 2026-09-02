import React, { useRef, useState } from 'react';

export default function ControlConsole({ 
  selectedFile, 
  setSelectedFile,
  isRecording,
  recordSeconds,
  startRecording,
  stopRecording,
  phase,
  setPhase,
  forceVerdict,
  setForceVerdict,
  onDetect,
  isDetecting,
  serverStatus
}) {
  const [activeTab, setActiveTab] = useState('upload');
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleRecordToggle = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording(canvasRef);
    }
  };

  const cnnAvailable = serverStatus?.models?.phase2_cnn?.available;

  const getPhaseNote = () => {
    if (phase === 'phase1') return '⚡ Heuristic mode — uses MFCC variance + spectral centroid. No model required.';
    if (phase === 'phase2') return cnnAvailable ? '🧠 CNN model loaded and active — running real neural inference.' : '⚠️ CNN model checkpoint not found — will auto-fallback to Phase 1 heuristic.';
    if (phase === 'phase3') return '🔬 Phase 3 CNN-LSTM is a skeleton — falls back to CNN/heuristic.';
    if (phase === 'phase4') return '🚀 Phase 4 Wav2Vec requires pre-trained transformer weights — falls back to CNN/heuristic.';
    return '';
  };

  const formatTime = (seconds) => {
    const mins = String(Math.floor(seconds / 60)).padStart(2, '0');
    const secs = String(seconds % 60).padStart(2, '0');
    return `${mins}:${secs}`;
  };

  return (
    <section className="glass-panel" id="control-console">
      <h2 className="panel-title">
        <span>Audio Input Terminal</span>
        <span style={{ fontSize: '0.8rem', fontWeight: 'normal', color: 'var(--text-secondary)' }}>Select file or record live</span>
      </h2>

      <div className="control-tabs">
        <button className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`} onClick={() => setActiveTab('upload')}>Upload File</button>
        <button className={`tab-btn ${activeTab === 'record' ? 'active' : ''}`} onClick={() => setActiveTab('record')}>Microphone</button>
      </div>

      {activeTab === 'upload' && (
        <div className="tab-content active">
          {!selectedFile ? (
            <div className="upload-zone" onClick={() => fileInputRef.current?.click()}>
              <svg className="upload-icon" width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4C9.11 4 6.6 5.64 5.35 8.04C2.34 8.36 0 10.91 0 14C0 17.31 2.69 20 6 20H19C21.76 20 24 17.76 24 15C24 12.36 21.95 10.22 19.35 10.04ZM19 18H6C3.79 18 2 16.21 2 14C2 11.95 3.53 10.24 5.56 10.03L6.63 9.92L7.13 8.97C8.08 7.14 9.94 6 12 6C14.89 6 17.38 8.01 17.9 10.88L18.15 12.26L19.55 12.36C20.96 12.46 22 13.63 22 15C22 16.65 20.65 18 19 18ZM8 13H10.55V16H13.45V13H16L12 9L8 13Z" fill="currentColor"/>
              </svg>
              <p className="upload-text">Drag and drop audio or video file here or <span>browse files</span></p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Supports WAV, MP3, M4A, MP4, OGG up to 50MB</p>
              <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="audio/*,video/mp4,video/quicktime,.wav,.mp3,.m4a,.mp4,.ogg,.flac,.webm,.aac" style={{ display: 'none' }} />
            </div>
          ) : (
            <div className="file-info-badge" style={{ display: 'flex' }}>
              <div className="file-info-name">{selectedFile.name} ({(selectedFile.size / (1024*1024)).toFixed(2)} MB)</div>
              <button className="remove-file-btn" onClick={handleRemoveFile}>Remove</button>
            </div>
          )}
        </div>
      )}

      {activeTab === 'record' && (
        <div className="tab-content active">
          <div className="recorder-container">
            <div className="record-btn-wrapper">
              <button className={`record-btn ${isRecording ? 'recording' : ''}`} onClick={handleRecordToggle} title="Start Recording">
                {isRecording ? '' : (
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 14C13.66 14 15 12.66 15 11V5C15 3.34 13.66 2 12 2C10.34 2 9 3.34 9 5V11C9 12.66 10.34 14 12 14ZM17.3 11C17.3 14 14.76 16.1 12 16.1C9.24 16.1 6.7 14 6.7 11H5C5 14.41 7.72 17.23 11 17.72V21H13V17.72C16.28 17.23 19 14.41 19 11H17.3Z"/>
                  </svg>
                )}
              </button>
            </div>
            <div className="recording-timer">{formatTime(recordSeconds)}</div>
            <canvas ref={canvasRef} className="visualizer-canvas"></canvas>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {isRecording ? <span style={{ color: 'var(--color-rose)', fontWeight: 600 }}>● RECORDING AUDIO...</span> : 'Click red button to capture voice'}
            </p>
          </div>
        </div>
      )}

      <div className="settings-group">
        <label className="settings-label">Detection Engine Phase</label>
        <select className="settings-select" value={phase} onChange={e => setPhase(e.target.value)}>
          <option value="phase1">Phase 1: Stub (Heuristics Analysis)</option>
          <option value="phase2">Phase 2: CNN Spectrogram Classifier</option>
          <option value="phase3">Phase 3: CNN-LSTM Temporal Classifier</option>
          <option value="phase4">Phase 4: Wav2Vec Transformer Engine</option>
        </select>
        <div style={{ fontSize: '0.75rem', marginTop: '0.4rem', padding: '0.35rem 0.6rem', borderRadius: '6px', border: '1px solid rgba(6,182,212,0.25)', color: 'var(--text-secondary)' }}>
          {getPhaseNote()}
        </div>
      </div>

      <div className="settings-group">
        <label className="settings-label">Demo Simulation Injection</label>
        <select className="settings-select" value={forceVerdict} onChange={e => setForceVerdict(e.target.value)}>
          <option value="">None (Run Automated Classifier Heuristics)</option>
          <option value="real">Force REAL (Demonstrate authentic speech)</option>
          <option value="fake">Force FAKE (Demonstrate synthetic voice)</option>
          <option value="uncertain">Force UNCERTAIN (Demonstrate low confidence)</option>
        </select>
      </div>

      <button className="submit-btn" disabled={!selectedFile || isDetecting} onClick={onDetect}>
        {isDetecting ? 'RUNNING AI ENGINE...' : 'RUN AI DETECTION ENGINE'}
      </button>
    </section>
  );
}
