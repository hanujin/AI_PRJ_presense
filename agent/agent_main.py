import threading
import time
import sys
import cv2
import numpy as np
from pathlib import Path

# 모듈 경로 추가
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path: sys.path.insert(0, str(_root))

from agent.core import RealTimeStreamer
from agent.model_wrapper import StressInferenceEngine
from agent.feedback_engine import FeedbackEngine

class StressAnalysisAgent:
    """
    AI 피드백 에이전트 통합 클래스.
    실시간 프리뷰(자막 Overlay) 기능 포함.
    """
    def __init__(self, model_path=None, interval=1.5):
        self.streamer = RealTimeStreamer()
        self.inference = StressInferenceEngine(model_path=model_path)
        self.feedback = FeedbackEngine()
        
        self.interval = interval 
        self.current_status = {
            "stress_score": 0.0,
            "status": "IDLE",
            "message": "시스템 준비 중...",
            "is_stressed": False
        }
        
        self.running = False
        self.inference_thread = None

    def _inference_loop(self):
        """배경에서 주기적으로 추론 수행"""
        print(f"🔄 Inference loop started (Interval: {self.interval}s)")
        while self.running:
            t0 = time.time()
            video, audio = self.streamer.get_current_data()
            
            if video is not None and audio is not None:
                raw_score = self.inference.predict(video, audio)
                result = self.feedback.process_score(raw_score)
                
                self.current_status = {
                    "stress_score": result["smoothed_score"],
                    "status": result["status"],
                    "message": result["message"],
                    "is_stressed": result["status"] == "HIGH",
                    "timestamp": time.time()
                }
            
            elapsed = time.time() - t0
            time.sleep(max(0, self.interval - elapsed))

    def start(self):
        """에이전트 구동 시작"""
        if self.running: return
        self.streamer.start()
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        print("✅ Stress Analysis Agent is now running.")

    def stop(self):
        """에이전트 구동 중지"""
        self.running = False
        self.streamer.stop()
        cv2.destroyAllWindows()
        print("🛑 Stress Analysis Agent stopped.")

    def run_with_preview(self):
        """
        화면에 실시간 자막을 입혀서 보여주는 프리뷰 모드 실행.
        이 메서드는 메인 쓰레드에서 실행되어야 함 (UI 제약).
        """
        self.start()
        print("\n📺 Preview Window opening...")
        print("⌨️  Commands: [c] Calibrate (Zero-set) | [q] Quit")
        
        try:
            while self.running:
                # 스트리머로부터 '고화질 프리뷰 프레임' 가져오기
                if self.streamer.preview_frame is not None:
                    with self.streamer.video_lock:
                        frame = self.streamer.preview_frame.copy()
                    
                    # 보기 좋게 크기 조절 (원본 비율 유지하며 조절 가능)
                    frame = cv2.resize(frame, (960, 720)) 

                    # 1. 상단 정보 오버레이
                    status = self.get_status()
                    score = status["stress_score"]
                    color = (0, 0, 255) if status["status"] == "HIGH" else (0, 255, 0)
                    
                    if status["status"] == "LEARNING":
                        label = "Learning State..."
                        color = (255, 255, 0)
                    else:
                        label = f"Stress (Z-Dev: {status.get('deviation', 0)})"

                    cv2.putText(frame, f"{label}: {score}%", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    if self.feedback.baseline:
                        cv2.putText(frame, f"User Baseline: {self.feedback.baseline:.1f}", (20, 85), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    # 2. 하단 자막 바
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 520), (800, 600), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    
                    msg = status["message"]
                    cv2.putText(frame, msg, (30, 570), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                    cv2.imshow("PreSense AI Agent - Real-time Feedback", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('c') or key == ord('C'):
                    if len(self.feedback.score_history) > 0:
                        self.feedback.set_baseline()
                        # 화면에 잠시 표시할 플래그 (선택 사항)
                        print(">>> Calibration Command Received")
                    else:
                        print(">>> Wait! Buffering data for calibration...")
        finally:
            self.stop()

    def get_status(self):
        return self.current_status

if __name__ == "__main__":
    agent = StressAnalysisAgent()
    # 자막이 포함된 실시간 화면 실행!
    agent.run_with_preview()
