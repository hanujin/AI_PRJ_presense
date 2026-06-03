import cv2
import numpy as np
import pyaudio
import threading
import time
from collections import deque

class RealTimeStreamer:
    """
    웹캠과 마이크로부터 실시간 데이터를 수집하고 버퍼링하는 클래스.
    Phase 1: Data Streamer
    """
    def __init__(self, 
                 n_frames=16, 
                 frame_size=(224, 224), 
                 audio_sr=16000, 
                 audio_duration=10):
        # 설정
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.audio_sr = audio_sr
        self.audio_duration = audio_duration
        self.audio_samples = audio_sr * audio_duration

        # 데이터 버퍼
        self.video_buffer = deque(maxlen=n_frames)
        self.preview_frame = None # 고화질 프리뷰용
        self.audio_buffer = np.zeros(self.audio_samples, dtype=np.float32)
        self.audio_lock = threading.Lock()
        self.video_lock = threading.Lock() # 비디오 버퍼 보호용

        # 상태 관리
        self.running = False
        self.video_thread = None
        self.audio_thread = None

    def _video_worker(self):
        cap = cv2.VideoCapture(0) 
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. 프리뷰용 고화질 프레임 저장 (원본)
            with self.video_lock:
                self.preview_frame = frame.copy()
            
            # 2. AI 모델용 전처리 (Resize)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.frame_size)
            
            with self.video_lock:
                self.video_buffer.append(frame_resized)
            
            time.sleep(0.033) 
            
        cap.release()

    def _audio_worker(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=self.audio_sr,
                        input=True,
                        frames_per_buffer=1024)

        while self.running:
            try:
                data = stream.read(1024, exception_on_overflow=False)
                new_samples = np.frombuffer(data, dtype=np.float32)
                
                with self.audio_lock:
                    # 슬라이딩 윈도우 업데이트
                    self.audio_buffer = np.roll(self.audio_buffer, -len(new_samples))
                    self.audio_buffer[-len(new_samples):] = new_samples
            except Exception as e:
                print(f"Audio error: {e}")
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

    def start(self):
        """데이터 수집 시작"""
        if self.running:
            return
        self.running = True
        self.video_thread = threading.Thread(target=self._video_worker, daemon=True)
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.video_thread.start()
        self.audio_thread.start()
        print("🚀 Real-time Data Streamer started.")

    def stop(self):
        """데이터 수집 중지"""
        self.running = False
        if self.video_thread: self.video_thread.join()
        if self.audio_thread: self.audio_thread.join()
        print("🛑 Real-time Data Streamer stopped.")

    def get_current_data(self):
        """현재 버퍼에 쌓인 16프레임 영상과 10초 오디오 반환"""
        if len(self.video_buffer) < self.n_frames:
            return None, None
            
        # 영상 데이터: (16, H, W, 3) -> (16, 3, H, W) Tensor 변환은 Inference 단계에서 수행
        video_stack = np.stack(list(self.video_buffer))
        
        with self.audio_lock:
            audio_copy = self.audio_buffer.copy()
            
        return video_stack, audio_copy

if __name__ == "__main__":
    # 간단한 테스트 코드
    streamer = RealTimeStreamer()
    streamer.start()
    
    try:
        for _ in range(10):
            time.sleep(2)
            v, a = streamer.get_current_data()
            if v is not None:
                print(f"Captured: Video Shape {v.shape}, Audio Shape {a.shape}")
            else:
                print("Buffering data...")
    finally:
        streamer.stop()
