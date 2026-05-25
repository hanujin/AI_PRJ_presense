"""
Architecture ① — End-to-End Raw Input (ResNet18 + Wav2Vec 2.0)

⚠️  이 모듈은 raw 영상 프레임 / raw 음성 파형 / raw 생리신호 시계열을 직접 처리합니다.
    현재 파이프라인(pre-extracted features)과 독립적으로 동작합니다.

실행 전 필요 작업
─────────────────
1. 추가 패키지 설치:
       pip install torchvision torchaudio transformers

2. dataset_e2e.py 의 StressIDRawDataset 으로 raw 데이터 로드

3. 학습 루프는 기존 train.py 와 별개로 구성해야 합니다
   (Teacher/Student forward 인자가 다름: frames, audio_wave, physio_ts)

임베딩 차원 (다이어그램 기준)
──────────────────────────
  f_v  : 512   ResNet18 avgpool 출력
  f_a  : 256   Wav2Vec2 768 → Linear(256)
  f_p  : 128   1D-CNN (Teacher 전용)

KD 방식
───────
  Soft Label KD : Teacher logits → Student loss
  Feature KD    : f_v' ≈ f_v (MSE),  f_a' ≈ f_a (MSE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── 선택적 임포트 ────────────────────────────────────────────────────────────
try:
    import torchvision.models as tv_models
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False

try:
    from transformers import Wav2Vec2Model
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

# ─── 임베딩 차원 ──────────────────────────────────────────────────────────────
VIDEO_EMB_DIM  = 512   # ResNet18 feature map
AUDIO_EMB_DIM  = 256   # Wav2Vec2 → proj
PHYSIO_EMB_DIM = 128   # 1D-CNN


# ══════════════════════════════════════════════════════════════════════════════
#  모달리티별 인코더
# ══════════════════════════════════════════════════════════════════════════════

class VideoEncoder(nn.Module):
    """
    ResNet18 기반 영상 인코더.

    입력: (B, T, 3, H, W)  — T개 프레임, H=W=224 권장
    출력: (B, 512)          — 프레임별 특징을 시간 평균 풀링
    """
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        if not _HAS_TORCHVISION:
            raise ImportError("pip install torchvision 필요")
        weights = 'IMAGENET1K_V1' if pretrained else None
        resnet  = tv_models.resnet18(weights=weights)
        # fc 레이어 제거 → avgpool 직후 512차원 특징 사용
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)                  # (B*T, C, H, W)
        x = self.backbone(x).flatten(1)              # (B*T, 512)
        x = x.view(B, T, VIDEO_EMB_DIM).mean(dim=1)  # (B, 512) — 프레임 평균
        return x  # f_v


class AudioEncoder(nn.Module):
    """
    Wav2Vec 2.0 base 기반 음성 인코더.

    입력: (B, T_audio)  — 16 kHz raw waveform (padding 처리 필요)
    출력: (B, 256)
    """
    def __init__(self, freeze: bool = True):
        super().__init__()
        if not _HAS_TRANSFORMERS:
            raise ImportError("pip install transformers 필요")
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        if freeze:
            for p in self.wav2vec.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(768, AUDIO_EMB_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.wav2vec(x).last_hidden_state.mean(dim=1)  # (B, 768)
        return self.proj(h)  # (B, 256) — f_a


class PhysioEncoder(nn.Module):
    """
    1D-CNN 생리신호 인코더 (Teacher 전용).

    입력: (B, 3, T_physio)  — ECG / EDA / Respiration raw 시계열
    출력: (B, 128)
    """
    def __init__(self, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32,  kernel_size=7, padding=3), nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32,          64,  kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, PHYSIO_EMB_DIM, kernel_size=3, padding=1), nn.BatchNorm1d(PHYSIO_EMB_DIM), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.net(x).squeeze(-1))  # (B, 128) — f_p


# ══════════════════════════════════════════════════════════════════════════════
#  Teacher / Student 모델
# ══════════════════════════════════════════════════════════════════════════════

class E2ETeacherModel(nn.Module):
    """
    Architecture ① Teacher — End-to-End (raw inputs).

    입력
    ────
      frames  : (B, T, 3, 224, 224)   영상 프레임 시퀀스
      audio   : (B, T_audio)           raw 음성 파형 (16 kHz)
      physio  : (B, 3, T_physio)       raw 생리신호 시계열 (ECG, EDA, Resp)

    출력: (logits, physio_emb, video_emb, audio_emb)

    KD 정렬 대상
    ────────────
      video_emb (f_v, 512) ↔ Student video_emb (f_v', 512)
      audio_emb (f_a, 256) ↔ Student audio_emb (f_a', 256)
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.3,
                 freeze_backbone: bool = True):
        super().__init__()
        self.video_enc  = VideoEncoder(pretrained=True, freeze=freeze_backbone)
        self.audio_enc  = AudioEncoder(freeze=freeze_backbone)
        self.physio_enc = PhysioEncoder(in_channels=3, dropout=dropout)

        fusion_dim = VIDEO_EMB_DIM + AUDIO_EMB_DIM + PHYSIO_EMB_DIM  # 512+256+128 = 896
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, frames, audio, physio):
        video_emb  = self.video_enc(frames)    # (B, 512)
        audio_emb  = self.audio_enc(audio)     # (B, 256)
        physio_emb = self.physio_enc(physio)   # (B, 128)
        fused  = torch.cat([video_emb, audio_emb, physio_emb], dim=1)
        logits = self.classifier(fused)
        return logits, physio_emb, video_emb, audio_emb


class E2EStudentModel(nn.Module):
    """
    Architecture ① Student — End-to-End (배포 시 Webcam + Mic).

    Teacher 와 동일한 VideoEncoder / AudioEncoder 구조 → Feature KD 가능.

    입력
    ────
      frames  : (B, T, 3, 224, 224)
      audio   : (B, T_audio)

    출력: (logits, video_emb, audio_emb)
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.3,
                 freeze_backbone: bool = True):
        super().__init__()
        self.video_enc = VideoEncoder(pretrained=True, freeze=freeze_backbone)
        self.audio_enc = AudioEncoder(freeze=freeze_backbone)

        fusion_dim = VIDEO_EMB_DIM + AUDIO_EMB_DIM  # 512+256 = 768
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, frames, audio):
        video_emb = self.video_enc(frames)   # (B, 512)
        audio_emb = self.audio_enc(audio)    # (B, 256)
        fused  = torch.cat([video_emb, audio_emb], dim=1)
        logits = self.classifier(fused)
        return logits, video_emb, audio_emb


# ══════════════════════════════════════════════════════════════════════════════
#  빠른 구조 확인용 (torchvision / transformers 없이도 shape 체크 가능)
# ══════════════════════════════════════════════════════════════════════════════

def print_model_summary():
    """
    임베딩 차원 및 Fusion 구조 요약 출력 (실제 모델 로드 없이).
    """
    print("=" * 55)
    print("  Architecture ① — End-to-End Model Summary")
    print("=" * 55)
    print(f"  VideoEncoder  : ResNet18 → {VIDEO_EMB_DIM}-dim  (f_v)")
    print(f"  AudioEncoder  : Wav2Vec2 → {AUDIO_EMB_DIM}-dim  (f_a)")
    print(f"  PhysioEncoder : 1D-CNN   → {PHYSIO_EMB_DIM}-dim  (f_p, Teacher only)")
    print()
    print(f"  Teacher fusion: {VIDEO_EMB_DIM}+{AUDIO_EMB_DIM}+{PHYSIO_EMB_DIM} = {VIDEO_EMB_DIM+AUDIO_EMB_DIM+PHYSIO_EMB_DIM} → 256 → num_classes")
    print(f"  Student fusion: {VIDEO_EMB_DIM}+{AUDIO_EMB_DIM} = {VIDEO_EMB_DIM+AUDIO_EMB_DIM} → 256 → num_classes")
    print()
    print("  KD 정렬 대상:")
    print(f"    f_v' ≈ f_v  (MSE, {VIDEO_EMB_DIM}-dim)")
    print(f"    f_a' ≈ f_a  (MSE, {AUDIO_EMB_DIM}-dim)")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════════════════════
#  Architecture 1-Hybrid: Encoder + LSTM
# ══════════════════════════════════════════════════════════════════════════════

class HybridE2EModel(nn.Module):
    """
    Architecture ① E2E Encoder + LSTM
    각 인코더(ResNet/Wav2Vec/1D-CNN) → 시간 흐름 학습(LSTM) → Classifier
    """
    def __init__(self, num_classes=2, hidden_dim=256, num_layers=1, dropout=0.3):
        super().__init__()
        # 기존 인코더 재사용
        self.video_enc  = VideoEncoder(pretrained=True, freeze=True)
        self.audio_enc  = AudioEncoder(freeze=True)
        self.physio_enc = PhysioEncoder(in_channels=3)
        
        # 합쳐진 차원: 512 + 256 + 128 = 896
        input_dim = VIDEO_EMB_DIM + AUDIO_EMB_DIM + PHYSIO_EMB_DIM
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, frames, audio, physio):
        # 1. 인코딩: 각 모달리티별로 특징 추출
        v_emb = self.video_enc(frames) # (B, 512)
        a_emb = self.audio_enc(audio)  # (B, 256)
        p_emb = self.physio_enc(physio)# (B, 128)
        
        # 2. 결합 후 LSTM 입력 형태로 변환 (B, 1, 896)
        fused = torch.cat([v_emb, a_emb, p_emb], dim=1).unsqueeze(1)
        
        # 3. LSTM 통과
        _, (h_n, _) = self.lstm(fused)
        
        # 4. 분류
        logits = self.classifier(h_n[-1])
        return logits, p_emb, v_emb, a_emb
