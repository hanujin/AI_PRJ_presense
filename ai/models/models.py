
import torch
import torch.nn as nn
# Reorganized: presense root and subfolders added to path
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent.parent
_ai_root = _root / 'ai'
for p in [_root, _ai_root, _ai_root/'data', _ai_root/'models', _ai_root/'train', _ai_root/'evaluate', _ai_root/'feature_analysis']:
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import config
from pathlib import Path

# 모달리티 인코더 및 모델 구조들은 그대로 유지
# (이 부분은 기존 models.py 내용을 그대로 보존한다고 가정)

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = config.HIDDEN_DIM, embed_dim: int = config.EMBED_DIM, dropout: float = config.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TeacherModel(nn.Module):
    def __init__(self, physio_dim=config.PHYSIO_DIM, video_dim=config.VIDEO_DIM, audio_dim=config.AUDIO_DIM, embed_dim=config.EMBED_DIM, num_classes=2, hidden_dim=config.HIDDEN_DIM, dropout=config.DROPOUT):
        super().__init__()
        self.physio_enc = ModalityEncoder(physio_dim, hidden_dim, embed_dim, dropout)
        self.video_enc  = ModalityEncoder(video_dim,  hidden_dim, embed_dim, dropout)
        self.audio_enc  = ModalityEncoder(audio_dim,  hidden_dim, embed_dim, dropout)
        self.classifier = nn.Sequential(nn.Linear(embed_dim*3, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, phys, video, audio):
        p, v, a = self.physio_enc(phys), self.video_enc(video), self.audio_enc(audio)
        fused = torch.cat([p, v, a], dim=1)
        return self.classifier(fused), p, v, a

class StudentModel(nn.Module):
    def __init__(self, video_dim=config.VIDEO_DIM, audio_dim=config.AUDIO_DIM, embed_dim=config.EMBED_DIM, num_classes=2, hidden_dim=config.HIDDEN_DIM, dropout=config.DROPOUT):
        super().__init__()
        self.video_enc = ModalityEncoder(video_dim, hidden_dim, embed_dim, dropout)
        self.audio_enc = ModalityEncoder(audio_dim, hidden_dim, embed_dim, dropout)
        self.classifier = nn.Sequential(nn.Linear(embed_dim*2, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, video, audio):
        v, a = self.video_enc(video), self.audio_enc(audio)
        fused = torch.cat([v, a], dim=1)
        return self.classifier(fused), v, a

class LSTMTeacherModel(nn.Module):
    def __init__(self, physio_dim=config.PHYSIO_DIM, video_dim=config.VIDEO_DIM, audio_dim=config.AUDIO_DIM, hidden_dim=config.HIDDEN_DIM, num_layers=2, num_classes=2, dropout=config.DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(physio_dim + video_dim + audio_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, phys, video, audio):
        x = torch.cat([phys, video, audio], dim=1).unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])

class LSTMStudentModel(nn.Module):
    def __init__(self, video_dim=config.VIDEO_DIM, audio_dim=config.AUDIO_DIM, hidden_dim=config.HIDDEN_DIM, num_layers=2, num_classes=2, dropout=config.DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(video_dim + audio_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, video, audio):
        x = torch.cat([video, audio], dim=1).unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])

def build_models(arch: str = 'mlp', task: str = 'binary'):
    num_classes = config.NUM_CLASSES_BINARY if task == 'binary' else config.NUM_CLASSES_3WAY
    if arch == 'mlp':
        teacher = TeacherModel(num_classes=num_classes)
        student = StudentModel(num_classes=num_classes)
    elif arch == 'lstm':
        teacher = LSTMTeacherModel(num_classes=num_classes)
        student = LSTMStudentModel(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return teacher, student

def load_checkpoint(model: nn.Module, path: str, device='cpu') -> nn.Module:
    path = Path(path)
    if not path.exists():
        print(f"⚠️ 모델 파일 없음: {path}")
        return model
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[Checkpoint] Loaded from {path}")
    return model
