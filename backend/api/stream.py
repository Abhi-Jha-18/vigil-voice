import asyncio
import time
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from backend.config import settings
from backend.audio.processor import normalize_audio
from backend.audio.features import extract_acoustic_features
from backend.detection.detector import run_ai_detection
from backend.detection.risk import generate_risk_report
from backend.detection.explainability import generate_evidence
from backend.detection.aggregator import SegmentPrediction, aggregate_predictions

router = APIRouter()

class StreamAnalyzer:
    def __init__(self, sample_rate: int = 16000, chunk_duration_sec: float = 3.0, hop_duration_sec: float = 1.0):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration_sec)
        self.hop_size = int(sample_rate * hop_duration_sec)
        self.buffer = np.array([], dtype=np.float32)
        
        self.segments_history = []
        self.segment_counter = 0
        self.total_processed_samples = 0
        
    def add_audio(self, audio_chunk: np.ndarray):
        """Append new float32 audio data to the rolling buffer."""
        self.buffer = np.concatenate((self.buffer, audio_chunk))
        
    def get_ready_window(self) -> np.ndarray:
        """Returns the oldest ready window of chunk_size, or None if not enough data."""
        if len(self.buffer) >= self.chunk_size:
            return self.buffer[:self.chunk_size]
        return None
        
    def advance_buffer(self):
        """Advance the buffer by the hop size."""
        self.buffer = self.buffer[self.hop_size:]
        self.total_processed_samples += self.hop_size

    def process_window(self, audio_data: np.ndarray, phase: str = "phase2") -> dict:
        """Runs the heavy ML inference pipeline on the window."""
        # 1. Very basic VAD (energy threshold) to skip silence
        rms_energy = np.sqrt(np.mean(audio_data**2))
        if rms_energy < 0.005:  # Arbitrary low silence threshold
            return {"status": "SILENCE"}
            
        # 2. Normalize and feature extraction
        y_clean = normalize_audio(audio_data)
        features = extract_acoustic_features(y_clean, sr=self.sample_rate)
        
        # 3. Model Inference
        real_prob = run_ai_detection(features, phase=phase)
        fake_prob = 1.0 - real_prob
        
        # 4. Create Segment
        start_time = self.total_processed_samples / self.sample_rate
        end_time = start_time + (self.chunk_size / self.sample_rate)
        
        pred_label = "REAL" if real_prob >= 0.75 else "FAKE" if real_prob <= 0.40 else "UNCERTAIN"
        conf = real_prob if pred_label == "REAL" else fake_prob if pred_label == "FAKE" else (1.0 - 2 * abs(real_prob - 0.5))
        
        seg = SegmentPrediction(
            segment_id=self.segment_counter,
            start_time=start_time,
            end_time=end_time,
            fake_probability=fake_prob,
            real_probability=real_prob,
            predicted_label=pred_label,
            confidence=conf
        )
        self.segment_counter += 1
        
        # 5. Maintain rolling history for risk engine (e.g. last 10 segments)
        self.segments_history.append(seg)
        if len(self.segments_history) > 10:
            self.segments_history.pop(0)
            
        # 6. Aggregate Rolling Risk
        aggregation = aggregate_predictions(self.segments_history, strategy="max_risk")
        risk_report = generate_risk_report(
            max_fake_prob=aggregation.maximum_fake_probability,
            avg_fake_prob=aggregation.average_fake_probability,
            temporal_variance=aggregation.temporal_consistency.get("variance", 0.0),
            suspicious_segments=aggregation.suspicious_segments,
            total_segments=len(self.segments_history),
            quality_status="GOOD"  # Assuming good for real-time simplicity
        )
        
        evidence = generate_evidence(
            features=features,
            temporal_variance=aggregation.temporal_consistency.get("variance", 0.0),
            suspicious_segments_count=len(aggregation.suspicious_segments),
            total_segments=len(self.segments_history)
        )
        
        from backend.detection.recommendations import get_recommendation
        security_rec = get_recommendation(risk_report["risk_level"], len(aggregation.suspicious_segments), evidence)
        
        return {
            "status": "ANALYZED",
            "timestamp": start_time,
            "segment": seg.to_dict(),
            "risk_report": risk_report,
            "security_recommendation": security_rec,
            "evidence": evidence
        }


@router.websocket("/api/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    
    analyzer = StreamAnalyzer(sample_rate=settings.sample_rate, chunk_duration_sec=3.0, hop_duration_sec=1.0)
    
    try:
        while True:
            # Expecting raw float32 PCM frames from the browser AudioContext
            data = await websocket.receive_bytes()
            
            # Convert bytes to float32 numpy array
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            analyzer.add_audio(audio_chunk)
            
            # Process as many overlapping windows as are ready
            while True:
                window = analyzer.get_ready_window()
                if window is None:
                    break
                    
                # Run CPU-bound ML in a threadpool so we don't block the asyncio event loop
                result = await run_in_threadpool(analyzer.process_window, window, phase="phase2")
                
                # Send back JSON result
                await websocket.send_json(result)
                
                # Step forward
                analyzer.advance_buffer()

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected naturally.")
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("WebSocket error")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
