import React, { useState, useRef } from 'react';
import { PageContainer } from '../components/layout/PageContainer';
import { api } from '../services/api';
import { UploadCloud, FileAudio, AlertTriangle, CheckCircle, ShieldAlert, FileText } from 'lucide-react';

export function UploadAnalysis() {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const data = await api.analyzeAudio(file, 'phase2');
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <PageContainer title="Upload & Analyze Audio">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload Section */}
        <div className="flex flex-col gap-6">
          <div 
            className={`glass-panel border-2 border-dashed flex flex-col items-center justify-center p-12 cursor-pointer transition-all ${
              isDragging ? 'border-primary bg-primary/10' : 'border-border hover:border-secondary'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept="audio/*" 
              onChange={handleFileChange}
            />
            
            <div className="w-16 h-16 rounded-full bg-surface border border-border flex items-center justify-center mb-6">
              <UploadCloud className="w-8 h-8 text-primary" />
            </div>
            
            <h3 className="text-xl font-medium mb-2">Drag & Drop Audio</h3>
            <p className="text-secondary text-sm text-center mb-6">
              or click to browse your computer
            </p>
            
            <div className="flex gap-2">
              <span className="badge bg-surface text-secondary">WAV</span>
              <span className="badge bg-surface text-secondary">MP3</span>
              <span className="badge bg-surface text-secondary">M4A</span>
              <span className="badge bg-surface text-secondary">OGG</span>
            </div>
          </div>

          {file && (
            <div className="glass-panel p-4 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <FileAudio className="w-8 h-8 text-secondary" />
                <div>
                  <p className="font-medium text-white">{file.name}</p>
                  <p className="text-xs text-secondary">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                </div>
              </div>
              <button 
                className="btn-primary"
                onClick={handleAnalyze}
                disabled={isAnalyzing}
              >
                {isAnalyzing ? 'Analyzing...' : 'Analyze Audio'}
              </button>
            </div>
          )}
          
          {error && (
            <div className="p-4 rounded-xl bg-danger/10 border border-danger/30 flex items-center gap-3 text-rose-300">
              <AlertTriangle className="w-5 h-5" />
              <p>{error}</p>
            </div>
          )}
        </div>

        {/* Results Section */}
        <div className="glass-panel relative overflow-hidden flex flex-col">
          {isAnalyzing ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-surface/80 backdrop-blur z-10">
              <div className="w-16 h-16 border-4 border-surface border-t-primary rounded-full animate-spin mb-4"></div>
              <p className="text-lg font-medium animate-pulse">Running ML Pipeline...</p>
            </div>
          ) : result ? (
            <div className="flex flex-col h-full">
              <div className="flex items-center justify-between mb-6 pb-4 border-b border-border">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-success" />
                  ANALYSIS COMPLETE
                </h3>
                {result.error && (
                  <span className="text-xs text-secondary font-mono">
                    Req: {result.error.request_id}
                  </span>
                )}
              </div>
              
              <div className="grid grid-cols-2 gap-4 mb-8">
                <div className="p-4 rounded-xl bg-surface">
                  <p className="text-xs text-secondary uppercase mb-1">Classification</p>
                  <p className={`text-xl font-bold ${
                    result.verdict === 'SPOOF' ? 'text-danger' : 
                    result.verdict === 'BONAFIDE' ? 'text-success' : 'text-warning'
                  }`}>
                    {result.verdict === 'SPOOF' ? 'Potential Spoof' : 
                     result.verdict === 'BONAFIDE' ? 'Genuine Voice' : 'Uncertain'}
                  </p>
                </div>
                
                <div className="p-4 rounded-xl bg-surface">
                  <p className="text-xs text-secondary uppercase mb-1">Spoof Probability</p>
                  <p className="text-xl font-bold font-mono">
                    {Math.round(result.confidence * 100)}%
                  </p>
                </div>
                
                <div className="p-4 rounded-xl bg-surface">
                  <p className="text-xs text-secondary uppercase mb-1">Model Used</p>
                  <p className="text-lg font-semibold flex items-center gap-2">
                    <ShieldAlert className="w-4 h-4 text-primary" />
                    {result.model_status || 'UNKNOWN'}
                  </p>
                </div>
                
                <div className="p-4 rounded-xl bg-surface">
                  <p className="text-xs text-secondary uppercase mb-1">Processing Time</p>
                  <p className="text-lg font-semibold font-mono">
                    {result.processing_time_ms} ms
                  </p>
                </div>
              </div>

              {result.spectrogram && (
                <div className="mb-6">
                  <p className="text-xs text-secondary uppercase mb-2">Mel Spectrogram</p>
                  <img src={result.spectrogram} alt="Spectrogram" className="w-full h-32 object-cover rounded-lg border border-border" />
                </div>
              )}
              
              <div className="mt-auto pt-4 border-t border-border">
                <button className="flex items-center gap-2 text-sm text-primary hover:text-white transition-colors">
                  <FileText className="w-4 h-4" />
                  View Detailed Evidence Report
                </button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-secondary opacity-50 p-12 text-center">
              <ShieldAlert className="w-16 h-16 mb-4" />
              <p>Upload an audio file to see forensic analysis results.</p>
            </div>
          )}
        </div>
      </div>
    </PageContainer>
  );
}
