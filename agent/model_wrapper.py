import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트 및 필요한 모듈 경로 추가
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path: sys.path.insert(0, str(_root))
# ai 내의 경로들도 추가
_ai = _root / "ai"
for p in [_ai, _ai/'models', _ai/'data']:
    if str(p) not in sys.path: sys.path.insert(0, str(p))

from models_e2e import E2EStudentModel

class StressInferenceEngine:
    """
    학습된 E2EStudentModel을 로드하여 실시간 추론을 수행하는 클래스.
    Phase 2: Inference Engine
    """
    def __init__(self, model_path=None, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 기본 모델 경로 설정 (가장 성능이 좋은 binary/Arch_1/hybrid_student_kd_best.pt 권장)
        if model_path is None:
            model_path = _root / "checkpoints" / "binary" / "Arch_1" / "hybrid_student_kd_best.pt"
            
        self.model = self._load_model(model_path)
        self.model.eval()

        # 정규화 파라미터 (ImageNet 기준)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _load_model(self, path):
        print(f"Loading model from: {path}")
        # E2EStudentModel 초기화 (binary 분류 기준 num_classes=2)
        model = E2EStudentModel(num_classes=2)
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.to(self.device)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def preprocess_video(self, video_np):
        """
        (16, H, W, 3) numpy -> (1, 16, 3, H, W) torch tensor
        정규화 및 차원 변환 수행
        """
        # 0~255 -> 0~1
        video = video_np.astype(np.float32) / 255.0
        # Normalize
        video = (video - self.mean) / self.std
        # (T, H, W, C) -> (T, C, H, W)
        video = np.transpose(video, (0, 3, 1, 2))
        # Batch 차원 추가
        video_tensor = torch.from_numpy(video).unsqueeze(0).to(self.device)
        return video_tensor

    def preprocess_audio(self, audio_np):
        """
        (160000,) numpy -> (1, 160000) torch tensor
        """
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(self.device)
        return audio_tensor

    def predict(self, video_np, audio_np):
        """
        실시간 데이터를 입력받아 스트레스 확률(0~100)을 반환
        """
        if self.model is None:
            return 0.0

        with torch.no_grad():
            v_tensor = self.preprocess_video(video_np)
            a_tensor = self.preprocess_audio(audio_np)
            
            # 모델 추론
            logits, _, _ = self.model(v_tensor, a_tensor)
            
            # Softmax로 확률 변환
            probs = F.softmax(logits, dim=1)
            stress_prob = probs[0][1].item() * 100 # 1번 인덱스가 스트레스 클래스
            
        return stress_prob

if __name__ == "__main__":
    # 테스트용 가짜 데이터
    engine = StressInferenceEngine()
    fake_video = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
    fake_audio = np.random.randn(160000).astype(np.float32)
    
    score = engine.predict(fake_video, fake_audio)
    print(f"Test Inference Result (Stress Score): {score:.2f}%")
