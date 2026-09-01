document.addEventListener('DOMContentLoaded', () => {
    // UI Elements - Tabs & General
    const tabUpload = document.getElementById('tab-upload');
    const tabRecord = document.getElementById('tab-record');
    const contentUpload = document.getElementById('content-upload');
    const contentRecord = document.getElementById('content-record');
    
    // UI Elements - Upload Tab
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileBadge = document.getElementById('file-badge');
    const fileName = document.getElementById('file-name');
    const btnRemoveFile = document.getElementById('btn-remove-file');
    
    // UI Elements - Record Tab
    const btnRecord = document.getElementById('btn-record');
    const recordTimer = document.getElementById('record-timer');
    const recordStatus = document.getElementById('record-status');
    const liveVisualizer = document.getElementById('live-visualizer');
    const canvasCtx = liveVisualizer.getContext('2d');
    
    // UI Elements - Configurations & Actions
    const selectPhase = document.getElementById('select-phase');
    const selectVerdict = document.getElementById('select-verdict');
    const btnDetect = document.getElementById('btn-detect');
    const analysisTimer = document.getElementById('analysis-timer');
    
    // UI Elements - Stepper
    const stepPreprocess = document.getElementById('step-preprocess');
    const stepExtract = document.getElementById('step-extract');
    const stepAI = document.getElementById('step-ai');
    const stepDecision = document.getElementById('step-decision');
    const steps = [stepPreprocess, stepExtract, stepAI, stepDecision];
    
    // UI Elements - Results Area
    const resultsArea = document.getElementById('results-area');
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const verdictContainer = document.getElementById('verdict-container');
    const txtVerdict = document.getElementById('txt-verdict');
    const txtRisk = document.getElementById('txt-risk');
    const txtConfidence = document.getElementById('txt-confidence');
    const circleProgress = document.getElementById('circle-progress');
    const txtExplanation = document.getElementById('txt-explanation');
    const imgSpectrogram = document.getElementById('img-spectrogram');
    const spectrogramPlace = document.getElementById('spectrogram-place');
    const scanningSweep = document.getElementById('scanning-sweep');
    

    // App State Variables
    let selectedAudioFile = null;
    let isRecording = false;
    let mediaRecorder = null;
    let audioChunks = [];
    let recordInterval = null;
    let recordSeconds = 0;
    let serverStatus = null;   // Cached /api/status response
    
    // Web Audio API Contexts
    let audioCtx = null;
    let analyser = null;
    let sourceNode = null;
    let animationFrameId = null;

    // ==========================================
    // SERVER STATUS FETCH (on page load)
    // ==========================================
    const statusDot    = document.getElementById('status-dot');
    const statusText   = document.getElementById('status-text');
    const modelBadge   = document.getElementById('model-status-badge');
    const phaseNote    = document.getElementById('phase-model-note');
    const uploadLimitNote = document.getElementById('upload-limit-note');

    async function fetchServerStatus() {
        try {
            const res  = await fetch('/api/status', { signal: AbortSignal.timeout(4000) });
            const data = await res.json();
            serverStatus = data;

            const maxUploadMb = data.configuration?.max_upload_mb;
            if (maxUploadMb && uploadLimitNote) {
                uploadLimitNote.textContent = `Supports WAV, MP3, M4A, OGG up to ${maxUploadMb}MB`;
            }

            // Update header indicator
            statusDot.style.background    = 'var(--color-emerald)';
            statusDot.style.boxShadow     = 'var(--glow-emerald)';
            statusText.textContent        = 'SYSTEM ONLINE';

            // Show CNN model badge
            const cnnAvailable = data.models?.phase2_cnn?.available;
            modelBadge.style.display = 'inline-block';
            if (cnnAvailable) {
                modelBadge.textContent         = '🧠 CNN MODEL READY';
                modelBadge.style.background    = 'rgba(16,185,129,0.15)';
                modelBadge.style.color         = 'var(--color-emerald)';
                modelBadge.style.border        = '1px solid rgba(16,185,129,0.35)';
            } else {
                modelBadge.textContent         = '⚡ PHASE 1 HEURISTIC';
                modelBadge.style.background    = 'rgba(245,158,11,0.12)';
                modelBadge.style.color         = 'var(--color-amber)';
                modelBadge.style.border        = '1px solid rgba(245,158,11,0.3)';
            }

            // Update phase note on current selection
            updatePhaseNote();

            // Fetch Evaluation & Robustness info
            const evalRes = await fetch('/api/model/info');
            const evalData = await evalRes.json();
            const evalBadge = document.getElementById('model-eval-badge');
            if (evalData && evalData.evaluation && evalData.evaluation.dataset_type === "REAL") {
                const eer = evalData.evaluation.eer * 100;
                const auc = evalData.evaluation.roc_auc;
                evalBadge.style.display = 'inline-block';
                evalBadge.textContent = `EER: ${eer.toFixed(2)}% | AUC: ${auc.toFixed(3)}`;
            } else {
                evalBadge.style.display = 'none';
            }

            // Render Robustness Scorecard
            const robContainer = document.getElementById('robustness-container');
            if (evalData && evalData.robustness && evalData.robustness.available && robContainer) {
                const rData = evalData.robustness.data;
                const baseAcc = (rData.baseline?.accuracy * 100).toFixed(1);
                const noise10 = (rData.conditions?.noise_10db?.accuracy * 100 || 0).toFixed(1);
                const resample8 = (rData.conditions?.resample_8khz?.accuracy * 100 || 0).toFixed(1);
                const baseEer = (rData.baseline?.eer * 100).toFixed(2);
                
                robContainer.innerHTML = `
                    <div style="display: flex; flex-direction: column; gap: 0.4rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>Clean Audio Baseline:</span>
                            <strong style="color: var(--color-emerald);">${baseAcc}%</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Background Noise (10 dB):</span>
                            <strong style="color: var(--color-cyan);">${noise10}%</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Resampled (8 kHz):</span>
                            <strong style="color: var(--color-amber);">${resample8}%</strong>
                        </div>
                        <div style="margin-top: 0.3rem; font-size: 0.75rem; color: var(--text-secondary); border-top: 1px solid rgba(255,255,255,0.1); padding-top: 0.3rem;">
                            Baseline EER: <strong style="color: var(--text-primary);">${baseEer}%</strong> | EER Thresh: <strong style="color: var(--text-primary);">${(rData.eer_recommended_threshold || 0.5).toFixed(2)}</strong>
                        </div>
                    </div>
                `;
            }

        } catch (err) {
            statusDot.style.background = 'var(--color-rose)';
            statusText.textContent     = 'API OFFLINE';
            console.warn('Could not reach /api/status:', err);
        }
    }

    function updatePhaseNote() {
        if (!serverStatus) return;
        const phase        = selectPhase.value;
        const cnnAvailable = serverStatus.models?.phase2_cnn?.available;

        if (phase === 'phase1') {
            phaseNote.style.display = 'block';
            phaseNote.textContent   = '⚡ Heuristic mode — uses MFCC variance + spectral centroid. No model required.';
        } else if (phase === 'phase2') {
            phaseNote.style.display = 'block';
            if (cnnAvailable) {
                phaseNote.textContent = '🧠 CNN model loaded and active — running real neural inference.';
                phaseNote.style.borderColor = 'rgba(16,185,129,0.3)';
                phaseNote.style.color       = 'var(--color-emerald)';
            } else {
                phaseNote.textContent = '⚠️ CNN model checkpoint not found — will auto-fallback to Phase 1 heuristic. Run training/train.py to generate it.';
                phaseNote.style.borderColor = 'rgba(245,158,11,0.35)';
                phaseNote.style.color       = 'var(--color-amber)';
            }
        } else if (phase === 'phase3') {
            phaseNote.style.display     = 'block';
            phaseNote.textContent       = '🔬 Phase 3 CNN-LSTM is a skeleton — falls back to CNN/heuristic.';
            phaseNote.style.borderColor = 'rgba(6,182,212,0.25)';
            phaseNote.style.color       = 'var(--text-secondary)';
        } else if (phase === 'phase4') {
            phaseNote.style.display     = 'block';
            phaseNote.textContent       = '🚀 Phase 4 Wav2Vec requires pre-trained transformer weights — falls back to CNN/heuristic.';
            phaseNote.style.borderColor = 'rgba(6,182,212,0.25)';
            phaseNote.style.color       = 'var(--text-secondary)';
        } else {
            phaseNote.style.display = 'none';
        }
    }

    // Wire phase selector
    selectPhase.addEventListener('change', updatePhaseNote);

    // Kick off status fetch on load
    fetchServerStatus();


    // ==========================================
    // TAB SWITCHING LOGIC
    // ==========================================
    tabUpload.addEventListener('click', () => {
        tabUpload.classList.add('active');
        tabRecord.classList.remove('active');
        contentUpload.classList.add('active');
        contentRecord.classList.remove('active');
        stopRecordingIfActive();
        updateDetectButtonState();
    });

    tabRecord.addEventListener('click', () => {
        tabRecord.classList.add('active');
        tabUpload.classList.remove('active');
        contentRecord.classList.add('active');
        contentUpload.classList.remove('active');
        updateDetectButtonState();
        drawVisualizerPlaceholder();
    });

    // ==========================================
    // FILE UPLOAD & DRAG/DROP LOGIC
    // ==========================================
    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // Drag and drop styles
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    btnRemoveFile.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedAudioFile = null;
        fileInput.value = '';
        fileBadge.style.display = 'none';
        dropZone.style.display = 'flex';
        updateDetectButtonState();
    });

    function handleSelectedFile(file) {
        if (!file.type.startsWith('audio/') && file.type !== 'video/mp4' && !file.name.toLowerCase().endsWith('.mp4')) {
            alert('Please select a valid audio or MP4 file.');
            return;
        }
        selectedAudioFile = file;
        fileName.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
        fileBadge.style.display = 'flex';
        dropZone.style.display = 'none';
        updateDetectButtonState();
    }

    function updateDetectButtonState() {
        if (tabUpload.classList.contains('active')) {
            btnDetect.disabled = !selectedAudioFile;
        } else {
            // In record tab, we enable detect only if we have a recorded file stored
            btnDetect.disabled = !selectedAudioFile;
        }
    }

    // ==========================================
    // LIVE MICROPHONE RECORDING LOGIC
    // ==========================================
    btnRecord.addEventListener('click', () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });

    async function startRecording() {
        audioChunks = [];
        recordSeconds = 0;
        recordTimer.textContent = '00:00';
        isRecording = true;
        btnRecord.classList.add('recording');
        recordStatus.innerHTML = '<span style="color:var(--color-rose); font-weight:600;">● RECORDING AUDIO...</span>';

        // Clear previously selected file
        selectedAudioFile = null;
        updateDetectButtonState();

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) {
                throw new Error('Web Audio API is not supported in this browser.');
            }

            audioCtx = new AudioContextClass();
            sourceNode = audioCtx.createMediaStreamSource(stream);
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256;

            const processor = audioCtx.createScriptProcessor(4096, 1, 1);
            sourceNode.connect(analyser);
            sourceNode.connect(processor);
            processor.connect(audioCtx.destination);

            processor.onaudioprocess = (event) => {
                const inputBuffer = event.inputBuffer.getChannelData(0);
                if (inputBuffer && inputBuffer.length > 0) {
                    audioChunks.push(new Float32Array(inputBuffer));
                }
            };

            // Keep reference for cleanup
            sourceNode._recordProcessor = processor;

            // Set up visualizer
            setupVisualizer(stream);

            // Start clock timer
            recordInterval = setInterval(() => {
                recordSeconds++;
                const mins = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
                const secs = String(recordSeconds % 60).padStart(2, '0');
                recordTimer.textContent = `${mins}:${secs}`;
            }, 1000);

        } catch (err) {
            console.error('Microphone access denied:', err);
            alert('Microphone access denied or unsupported. Please check device permissions.');
            stopRecordingIfActive();
        }
    }

    function stopRecording() {
        if (!isRecording) return;

        clearInterval(recordInterval);
        recordInterval = null;
        isRecording = false;
        btnRecord.classList.remove('recording');

        const processor = sourceNode && sourceNode._recordProcessor;
        if (processor) {
            processor.disconnect();
        }

        if (sourceNode) {
            sourceNode.disconnect();
        }

        if (audioCtx && audioCtx.state !== 'closed') {
            audioCtx.close();
        }

        const pcmChunks = audioChunks.filter(chunk => chunk && chunk.length > 0);
        if (!pcmChunks.length) {
            selectedAudioFile = null;
            recordStatus.innerHTML = '<span style="color:var(--color-amber); font-weight:600;">⚠ NO VALID AUDIO CAPTURED</span>';
            console.warn('No valid audio data captured. Please speak for a little longer.');
            cleanupAudioContext();
            updateDetectButtonState();
            return;
        }

        const wavBlob = audioFloatChunksToWavBlob(pcmChunks);
        selectedAudioFile = new File([wavBlob], 'recorded_voice.wav', { type: 'audio/wav' });
        recordStatus.innerHTML = '<span style="color:var(--color-emerald); font-weight:600;">✓ AUDIO CAPTURED</span>';
        updateDetectButtonState();

        if (audioCtx && audioCtx.state !== 'closed') {
            audioCtx.close();
        }
        cleanupAudioContext();
    }

    function stopRecordingIfActive() {
        if (isRecording) {
            stopRecording();
            recordStatus.textContent = 'Recording cancelled.';
        }
    }

    // Canvas real-time visualizer mapping
    function setupVisualizer(stream) {
        if (!audioCtx) {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            audioCtx = new AudioContextClass();
        }

        if (!analyser) {
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256;
        }

        if (!sourceNode) {
            sourceNode = audioCtx.createMediaStreamSource(stream);
        }

        if (!sourceNode._visualizerConnected) {
            sourceNode.connect(analyser);
            sourceNode._visualizerConnected = true;
        }

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const width = liveVisualizer.width = liveVisualizer.clientWidth;
        const height = liveVisualizer.height = liveVisualizer.clientHeight;
        
        function draw() {
            if (!isRecording) return;
            animationFrameId = requestAnimationFrame(draw);
            
            analyser.getByteFrequencyData(dataArray);
            
            canvasCtx.fillStyle = '#0d0c1e';
            canvasCtx.fillRect(0, 0, width, height);
            
            const barWidth = (width / bufferLength) * 1.5;
            let barHeight;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                barHeight = dataArray[i] / 2;
                
                // Color gradient from cyan to violet
                const r = 6 + i * 2;
                const g = 182 - i;
                const b = 212 + i;
                canvasCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                
                canvasCtx.fillRect(x, height - barHeight, barWidth - 1, barHeight);
                x += barWidth;
            }
        }
        
        draw();
    }

    function drawVisualizerPlaceholder() {
        const width = liveVisualizer.width = liveVisualizer.clientWidth;
        const height = liveVisualizer.height = liveVisualizer.clientHeight;
        canvasCtx.fillStyle = '#0d0c1e';
        canvasCtx.fillRect(0, 0, width, height);
        
        // Draw flat line
        canvasCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        canvasCtx.lineWidth = 2;
        canvasCtx.beginPath();
        canvasCtx.moveTo(0, height / 2);
        canvasCtx.lineTo(width, height / 2);
        canvasCtx.stroke();
    }

    function cleanupAudioContext() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        if (audioCtx && audioCtx.state !== 'closed') {
            audioCtx.close().catch(() => {});
        }
        audioCtx = null;
        analyser = null;
        sourceNode = null;
    }

    function audioFloatChunksToWavBlob(floatChunks) {
        const totalLength = floatChunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const buffer = new ArrayBuffer(44 + totalLength * 2);
        const view = new DataView(buffer);

        function writeString(offset, value) {
            for (let i = 0; i < value.length; i++) {
                view.setUint8(offset + i, value.charCodeAt(i));
            }
        }

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + totalLength * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, 16000, true);
        view.setUint32(28, 16000 * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, totalLength * 2, true);

        let offset = 44;
        for (const chunk of floatChunks) {
            for (let i = 0; i < chunk.length; i++) {
                const sample = Math.max(-1, Math.min(1, chunk[i]));
                view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
                offset += 2;
            }
        }

        return new Blob([buffer], { type: 'audio/wav' });
    }

    function getExtensionFromMime(mimeType) {
        if (!mimeType) return 'webm';
        if (mimeType.includes('wav')) return 'wav';
        if (mimeType.includes('mp3')) return 'mp3';
        if (mimeType.includes('m4a')) return 'm4a';
        if (mimeType.includes('ogg')) return 'ogg';
        if (mimeType.includes('webm')) return 'webm';
        return 'webm';
    }

    async function convertRecordedBlobToWav(blob) {
        const arrayBuffer = await blob.arrayBuffer();
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) {
            throw new Error('This browser does not support Web Audio API.');
        }

        const audioContext = new AudioContextClass();
        try {
            const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
            const wavBlob = audioBufferToWavBlob(decoded);
            return new File([wavBlob], 'recorded_voice.wav', { type: 'audio/wav' });
        } finally {
            await audioContext.close();
        }
    }

    function audioBufferToWavBlob(audioBuffer) {
        const channels = [];
        const sampleRate = audioBuffer.sampleRate;
        const format = 1;
        const bitDepth = 16;

        for (let channelIndex = 0; channelIndex < audioBuffer.numberOfChannels; channelIndex++) {
            channels.push(audioBuffer.getChannelData(channelIndex));
        }

        const blockAlign = audioBuffer.numberOfChannels * bitDepth / 8;
        const dataLength = audioBuffer.length * blockAlign;
        const buffer = new ArrayBuffer(44 + dataLength);
        const view = new DataView(buffer);

        function writeString(offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + dataLength, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, audioBuffer.numberOfChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        writeString(36, 'data');
        view.setUint32(40, dataLength, true);

        let offset = 44;
        for (let sampleIndex = 0; sampleIndex < audioBuffer.length; sampleIndex++) {
            for (let channelIndex = 0; channelIndex < channels.length; channelIndex++) {
                const sample = Math.max(-1, Math.min(1, channels[channelIndex][sampleIndex]));
                view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
                offset += 2;
            }
        }

        return new Blob([buffer], { type: 'audio/wav' });
    }

    // ==========================================
    // DETECTOR & API CALL LOGIC
    // ==========================================
    btnDetect.addEventListener('click', async () => {
        if (!selectedAudioFile) return;

        // Reset UI steps and setup loading console state
        resetStepper();
        resultsPlaceholder.style.display = 'none';
        resultsArea.style.display = 'flex';
        verdictContainer.className = 'verdict-card';
        txtVerdict.textContent = 'ANALYZING...';
        txtRisk.textContent = 'PROCESSING DATA';
        
        imgSpectrogram.style.display = 'none';
        spectrogramPlace.style.display = 'flex';
        scanningSweep.style.display = 'block';

        btnDetect.disabled = true;
        analysisTimer.textContent = 'ANALYZING';

        // Animate step 1 & 2 for simulation visual flow
        setStepActive(0); // Preprocess
        
        const phaseVal = selectPhase.value;
        const forceVerdictVal = selectVerdict.value;
        
        // Prepare API Payload
        const formData = new FormData();
        formData.append('file', selectedAudioFile);
        formData.append('phase', phaseVal);
        if (forceVerdictVal) {
            formData.append('force_verdict', forceVerdictVal);
        }

        try {
            // Delay feature transition slightly for nice stepper simulation
            await sleep(600);
            setStepCompleted(0);
            setStepActive(1); // Features
            
            const response = await fetch('/api/detect', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                let errorMsg = 'Server returned an error.';
                try {
                    const errorData = await response.json();
                    if (errorData.error && errorData.error.message) {
                        errorMsg = errorData.error.message;
                        if (errorData.error.code) {
                            errorMsg = `[${errorData.error.code}] ${errorMsg}`;
                        }
                    } else if (errorData.detail) {
                        errorMsg = errorData.detail;
                    }
                } catch (e) {
                    errorMsg = `HTTP ${response.status}: ${response.statusText}`;
                }
                throw new Error(errorMsg);
            }

            const data = await response.json();
            
            await sleep(600);
            setStepCompleted(1);
            setStepActive(2); // AI Inference
            
            await sleep(600);
            setStepCompleted(2);
            setStepActive(3); // Decision
            
            await sleep(400);
            setStepCompleted(3);
            
            // Rendering final response values
            populateResults(data);
            
        } catch (error) {
            console.error('Detection error:', error);
            alert(`Analysis failed: ${error.message}`);
            resultsArea.style.display = 'none';
            resultsPlaceholder.style.display = 'flex';
            resetStepper();
            analysisTimer.textContent = 'ERROR';
        } finally {
            btnDetect.disabled = false;
            updateDetectButtonState();
            scanningSweep.style.display = 'none';
        }
    });

    function populateResults(data) {
        analysisTimer.textContent = `COMPLETED IN ${data.processing_time_ms}ms`;
        
        // Update CNN active badge if the server told us
        if (data.cnn_model_active !== undefined && modelBadge) {
            if (data.cnn_model_active) {
                modelBadge.textContent      = '🧠 CNN INFERENCE ACTIVE';
                modelBadge.style.background = 'rgba(16,185,129,0.15)';
                modelBadge.style.color      = 'var(--color-emerald)';
                modelBadge.style.border     = '1px solid rgba(16,185,129,0.35)';
            }
        }

        // 1. Overall Risk Assessment (Phase 6)
        const riskLevel = data.risk_report?.risk_level || data.risk_level || 'UNKNOWN';
        const riskScore = data.risk_report?.risk_score || data.raw_score || 0;
        const fakeProb = data.risk_report?.fake_probability || 0;
        
        txtVerdict.textContent = data.verdict;
        txtRisk.textContent = `${riskLevel} RISK`;
        document.getElementById('txt-risk-score').textContent = `${riskScore}/100`;
        
        const verdict = data.verdict.toLowerCase(); // real, fake, uncertain
        verdictContainer.className = `verdict-card ${verdict}`;
        
        // 2. Confidence & Quality
        const confidence = data.confidence;
        txtConfidence.textContent = `${confidence}%`;
        
        const circumference = 220;
        const offset = circumference - (confidence / 100) * circumference;
        circleProgress.style.strokeDashoffset = offset;
        
        if (verdict === 'real') {
            circleProgress.style.stroke = 'var(--color-emerald)';
        } else if (verdict === 'fake') {
            circleProgress.style.stroke = 'var(--color-rose)';
        } else {
            circleProgress.style.stroke = 'var(--color-amber)';
        }
        
        txtExplanation.textContent = getExplanationText(data.verdict, data.confidence, data.features?.spectral_centroid || 0);
        document.getElementById('txt-audio-quality').textContent = data.risk_report?.confidence || 'UNKNOWN';

        // 3. Security Recommendation (Phase 8)
        if (data.security_recommendation) {
            document.getElementById('rec-action').textContent = data.security_recommendation.recommendation;
            document.getElementById('rec-reason').textContent = data.security_recommendation.reason;
            document.getElementById('rec-method').textContent = data.security_recommendation.recommended_verification_method;
            
            const recPanel = document.getElementById('recommendation-panel');
            if (riskLevel === 'LOW') {
                recPanel.style.background = 'rgba(16, 185, 129, 0.05)';
                recPanel.style.borderColor = 'rgba(16, 185, 129, 0.2)';
                recPanel.style.borderLeftColor = 'var(--color-emerald)';
            } else if (riskLevel === 'MEDIUM') {
                recPanel.style.background = 'rgba(245, 158, 11, 0.05)';
                recPanel.style.borderColor = 'rgba(245, 158, 11, 0.2)';
                recPanel.style.borderLeftColor = 'var(--color-amber)';
            } else {
                recPanel.style.background = 'rgba(225, 29, 72, 0.05)';
                recPanel.style.borderColor = 'rgba(225, 29, 72, 0.2)';
                recPanel.style.borderLeftColor = 'var(--color-rose)';
            }
        }
        
        // 4. Segment Timeline (Phase 5)
        if (data.segments && data.segments.length > 0) {
            renderTimeline(data.segments);
        }

        // 5. Explainability Evidence (Phase 7)
        if (data.risk_report?.signals) {
            renderEvidence(data.risk_report.signals);
        }

        // 6. Spectrogram rendering
        if (data.spectrogram) {
            imgSpectrogram.src = data.spectrogram;
            imgSpectrogram.style.display = 'block';
            spectrogramPlace.style.display = 'none';
        } else {
            imgSpectrogram.style.display = 'none';
            spectrogramPlace.style.display = 'flex';
        }
    }

    function renderTimeline(segments) {
        const container = document.getElementById('timeline-container');
        container.innerHTML = '';
        document.getElementById('timeline-status').textContent = `${segments.length} segments`;
        
        segments.forEach(seg => {
            const el = document.createElement('div');
            el.className = 'timeline-segment';
            el.style.height = '16px';
            el.style.flex = '1';
            el.style.minWidth = '20px';
            el.style.borderRadius = '4px';
            el.style.cursor = 'pointer';
            el.style.transition = 'all 0.2s';
            
            if (seg.fake_probability > 0.6) {
                el.style.background = 'var(--color-rose)';
            } else if (seg.fake_probability > 0.4) {
                el.style.background = 'var(--color-amber)';
            } else {
                el.style.background = 'var(--color-emerald)';
            }
            
            el.addEventListener('mouseover', () => { el.style.opacity = '0.7'; });
            el.addEventListener('mouseout', () => { el.style.opacity = '1'; });
            
            el.addEventListener('click', () => {
                document.getElementById('segment-inspector').style.display = 'block';
                document.getElementById('inspector-time').textContent = `${seg.start_time.toFixed(1)}s - ${seg.end_time.toFixed(1)}s`;
                document.getElementById('inspector-label').textContent = seg.predicted_label;
                document.getElementById('inspector-label').style.color = (seg.predicted_label === 'FAKE') ? 'var(--color-rose)' : 'var(--color-emerald)';
                document.getElementById('inspector-prob').textContent = `${(seg.fake_probability * 100).toFixed(1)}%`;
                document.getElementById('inspector-conf').textContent = `${(seg.confidence * 100).toFixed(1)}%`;
            });
            
            container.appendChild(el);
        });
    }

    function renderEvidence(signals) {
        const container = document.getElementById('evidence-container');
        container.innerHTML = '';
        
        if (signals.length === 0) {
            container.innerHTML = '<div style="font-size: 0.85rem; color: var(--text-secondary); font-style: italic;">No anomalous signals detected.</div>';
            return;
        }
        
        signals.forEach(sig => {
            const el = document.createElement('div');
            el.style.padding = '0.5rem';
            el.style.background = 'rgba(255,255,255,0.03)';
            el.style.border = '1px solid rgba(255,255,255,0.1)';
            el.style.borderRadius = '4px';
            el.style.fontSize = '0.85rem';
            
            const strengthColor = sig.strength === 'HIGH' ? 'var(--color-rose)' : (sig.strength === 'MEDIUM' ? 'var(--color-amber)' : 'var(--text-secondary)');
            
            el.innerHTML = `
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <strong style="text-transform: uppercase;">${sig.type}</strong>
                    <span style="color: ${strengthColor}; font-weight: bold; font-size: 0.75rem;">${sig.strength}</span>
                </div>
                <div style="color: var(--text-secondary);">${sig.description}</div>
            `;
            container.appendChild(el);
        });
    }

    function getExplanationText(verdict, confidence, centroid) {
        if (verdict === 'REAL') {
            if (confidence > 90) {
                return 'Biometric markers verify standard vocal cord articulation. Signal reveals natural high-frequency variability corresponding to authentic human voice layers.';
            } else {
                return 'Audio details indicate high similarity to authentic speech, though slight ambient noise or clipping reduces certainty score slightly.';
            }
        } else if (verdict === 'FAKE') {
            if (confidence > 90) {
                return 'Critical Detection Alert: Signal indicates synthetic voice replication. High spectral flatness and phase pattern anomalies typical of neural text-to-speech vocoders.';
            } else {
                return 'Potential Deepfake Spoof: Acoustic features exhibit traits common in neural voice cloning methods (such as artifact traces in higher order MFCC frequency channels).';
            }
        } else {
            return 'Analysis is inconclusive. Voice characteristics show mixed acoustic markers, or recording quality is too degraded for full authentication.';
        }
    }

    // ==========================================
    // STEPPER ANIMATION UTILS
    // ==========================================
    function setStepActive(index) {
        if (steps[index]) {
            steps[index].classList.add('active');
        }
    }

    function setStepCompleted(index) {
        if (steps[index]) {
            steps[index].classList.add('completed');
        }
    }

    function resetStepper() {
        steps.forEach(step => {
            step.classList.remove('active', 'completed');
        });
    }

    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    // ==========================================
    // PHASE 15: LIVE CALL & INCIDENT RESPONSE
    // ==========================================
    const tabLive = document.getElementById('tab-live');
    const contentLive = document.getElementById('content-live');
    const btnStartCall = document.getElementById('btn-start-call');
    const btnEndCall = document.getElementById('btn-end-call');
    const liveCallTimer = document.getElementById('live-call-timer');
    const liveCallVisualizer = document.getElementById('live-call-visualizer');
    const incidentCard = document.getElementById('incident-card');
    const btnExportEvidence = document.getElementById('btn-export-evidence');
    const seCheckboxes = document.querySelectorAll('.se-check');

    let liveSessionId = null;
    let liveWebSocket = null;
    let liveAudioContext = null;
    let liveAudioStream = null;
    let liveProcessorNode = null;
    let liveCallTimerInterval = null;
    let liveCallStartEpoch = 0;
    let liveTimelineWindows = [];

    if (tabLive) {
        tabLive.addEventListener('click', () => {
            tabUpload.classList.remove('active');
            tabRecord.classList.remove('active');
            tabLive.classList.add('active');

            contentUpload.classList.remove('active');
            contentRecord.classList.remove('active');
            contentLive.classList.add('active');
        });
    }

    if (btnStartCall) {
        btnStartCall.addEventListener('click', async () => {
            try {
                // 1. Create Live Session via REST
                const res = await fetch('/api/live/session', { method: 'POST' });
                const sessionData = await res.json();
                if (!sessionData.success) {
                    throw new Error('Failed to initialize live detection session.');
                }
                liveSessionId = sessionData.session_id;

                // 2. Open Microphone Stream
                liveAudioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                liveAudioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                const source = liveAudioContext.createMediaStreamSource(liveAudioStream);

                // 3. Connect WebSocket
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/api/live/audio/${liveSessionId}`;
                liveWebSocket = new WebSocket(wsUrl);

                liveWebSocket.onopen = () => {
                    btnStartCall.disabled = true;
                    btnEndCall.disabled = false;
                    liveCallStartEpoch = Date.now();
                    liveCallTimerInterval = setInterval(updateLiveCallTimer, 1000);
                    
                    // Show results area for live monitoring
                    resultsPlaceholder.style.display = 'none';
                    resultsArea.style.display = 'flex';
                    txtVerdict.textContent = 'CALL ACTIVE';
                    txtRisk.textContent = 'MONITORING VOICE';
                    verdictContainer.className = 'verdict-card';
                    analysisTimer.textContent = 'LIVE MONITORING';
                    liveTimelineWindows = [];

                    // Setup processor node (bufferSize 4096 = ~256ms at 16kHz)
                    liveProcessorNode = liveAudioContext.createScriptProcessor(4096, 1, 1);
                    liveProcessorNode.onaudioprocess = (e) => {
                        if (liveWebSocket && liveWebSocket.readyState === WebSocket.OPEN) {
                            const inputData = e.inputBuffer.getChannelData(0);
                            liveWebSocket.send(inputData.buffer);
                        }
                    };
                    source.connect(liveProcessorNode);
                    liveProcessorNode.connect(liveAudioContext.destination);
                };

                liveWebSocket.onmessage = (evt) => {
                    try {
                        const msg = JSON.parse(evt.data);
                        if (msg.status === 'ANALYZED') {
                            handleLiveDetectionWindow(msg);
                        }
                    } catch (e) {
                        console.error('Error parsing live WS message:', e);
                    }
                };

                liveWebSocket.onerror = (err) => {
                    console.error('Live WS Error:', err);
                };

                liveWebSocket.onclose = () => {
                    stopLiveCall();
                };

            } catch (err) {
                console.error('Error starting live call:', err);
                alert(`Could not start live call: ${err.message}`);
                stopLiveCall();
            }
        });
    }

    if (btnEndCall) {
        btnEndCall.addEventListener('click', () => {
            stopLiveCall();
        });
    }

    function stopLiveCall() {
        if (liveCallTimerInterval) {
            clearInterval(liveCallTimerInterval);
            liveCallTimerInterval = null;
        }
        if (liveProcessorNode) {
            liveProcessorNode.disconnect();
            liveProcessorNode = null;
        }
        if (liveAudioStream) {
            liveAudioStream.getTracks().forEach(t => t.stop());
            liveAudioStream = null;
        }
        if (liveAudioContext && liveAudioContext.state !== 'closed') {
            liveAudioContext.close();
            liveAudioContext = null;
        }
        if (liveWebSocket && liveWebSocket.readyState === WebSocket.OPEN) {
            liveWebSocket.close();
            liveWebSocket = null;
        }

        if (btnStartCall) btnStartCall.disabled = false;
        if (btnEndCall) btnEndCall.disabled = true;
        if (analysisTimer) analysisTimer.textContent = 'CALL ENDED';
    }

    function updateLiveCallTimer() {
        if (!liveCallStartEpoch) return;
        const elapsedSec = Math.floor((Date.now() - liveCallStartEpoch) / 1000);
        const mins = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
        const secs = String(elapsedSec % 60).padStart(2, '0');
        if (liveCallTimer) liveCallTimer.textContent = `${mins}:${secs}`;
    }

    function handleLiveDetectionWindow(data) {
        const risk = data.call_risk || {};
        const win = data.window || {};
        const riskLevel = risk.risk_level || 'LOW_RISK';
        const riskScore = Math.round(risk.risk_score || 0);

        // Update Verdict & Risk Score
        if (riskLevel === 'HIGH_SPOOF_RISK') {
            txtVerdict.textContent = 'POTENTIAL SPOOF';
            txtRisk.textContent = 'HIGH SPOOF RISK';
            verdictContainer.className = 'verdict-card fake';
            circleProgress.style.stroke = 'var(--color-rose)';
        } else if (riskLevel === 'SUSPICIOUS') {
            txtVerdict.textContent = 'SUSPICIOUS';
            txtRisk.textContent = 'MEDIUM RISK';
            verdictContainer.className = 'verdict-card uncertain';
            circleProgress.style.stroke = 'var(--color-amber)';
        } else {
            txtVerdict.textContent = 'AUTHENTIC';
            txtRisk.textContent = 'LOW RISK';
            verdictContainer.className = 'verdict-card real';
            circleProgress.style.stroke = 'var(--color-emerald)';
        }

        document.getElementById('txt-risk-score').textContent = `${riskScore}/100`;
        txtConfidence.textContent = `${Math.round(win.confidence || 0)}%`;

        const circumference = 220;
        const offset = circumference - (riskScore / 100) * circumference;
        circleProgress.style.strokeDashoffset = offset;

        // Update Timeline
        if (win.status === 'ANALYZED') {
            liveTimelineWindows.push({
                start_time: win.start_time,
                end_time: win.end_time,
                fake_probability: win.fake_probability,
                predicted_label: (win.fake_probability > 0.6 ? 'FAKE' : (win.fake_probability > 0.4 ? 'UNCERTAIN' : 'REAL')),
                confidence: win.confidence / 100.0
            });
            renderTimeline(liveTimelineWindows.slice(-15));
        }

        // Show Incident Card if triggered
        if (data.incident && incidentCard) {
            incidentCard.style.display = 'block';
            document.getElementById('inc-id').textContent = data.incident.incident_id || '--';
            document.getElementById('inc-peak').textContent = `${Math.round(data.incident.peak_spoof_score * 100)}%`;
            document.getElementById('inc-windows').textContent = data.incident.suspicious_windows || '0';
            document.getElementById('inc-status').textContent = data.incident.model_status || '--';
        }
    }

    // Social Engineering checkbox handling
    seCheckboxes.forEach(cb => {
        cb.addEventListener('change', () => {
            if (cb.checked && liveWebSocket && liveWebSocket.readyState === WebSocket.OPEN) {
                liveWebSocket.send(JSON.stringify({
                    action: 'add_indicator',
                    indicator: cb.value
                }));
            }
        });
    });

    // Evidence Export Button
    if (btnExportEvidence) {
        btnExportEvidence.addEventListener('click', async () => {
            if (!liveSessionId) {
                alert('No active or recent call session to export evidence from.');
                return;
            }
            try {
                const res = await fetch(`/api/live/incident/${liveSessionId}/report`, { method: 'POST' });
                const rep = await res.json();
                if (rep.success) {
                    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(rep.incident, null, 2));
                    const downloadAnchor = document.createElement('a');
                    downloadAnchor.setAttribute("href", dataStr);
                    downloadAnchor.setAttribute("download", `${rep.incident_id}_evidence.json`);
                    document.body.appendChild(downloadAnchor);
                    downloadAnchor.click();
                    downloadAnchor.remove();
                } else {
                    alert('Could not generate evidence report.');
                }
            } catch (err) {
                console.error('Evidence export failed:', err);
                alert(`Export failed: ${err.message}`);
            }
        });
    }

    // Draw placeholder initially
    drawVisualizerPlaceholder();
});
