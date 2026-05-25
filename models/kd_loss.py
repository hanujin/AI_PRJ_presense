"""
PreSense - Knowledge Distillation Loss

Total Loss = α × Soft_KD + β × Feature_KD + γ × CE

- Soft Label KD  : Teacher softmax(T) 분포를 Student가 모방 (KL Divergence)
- Feature KD     : Teacher의 Video/Audio 임베딩과 Student 임베딩 정렬 (MSE)
- Cross-Entropy  : Ground Truth 레이블로 Student 직접 지도

Architecture ①의 경우 β=0 (Feature KD 미사용)
Architecture ③의 경우 세 항 모두 사용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
# Reorganized: presense root and subfolders added to path
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
for p in [_root, _root/'data', _root/'models', _root/'train', _root/'evaluate', _root/'analysis']:
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import config


class SoftLabelKDLoss(nn.Module):
    """
    Hinton et al. (2015) 방식의 Soft Label KD Loss.
    KL( Teacher_soft || Student_soft ) 계산.

    temperature T가 높을수록 확률 분포가 평탄해져
    클래스 간 상대적 유사성 정보가 잘 전달됩니다.
    """
    def __init__(self, temperature: float = config.KD_TEMPERATURE):
        super().__init__()
        self.T = temperature

    def forward(self,
                teacher_logits: torch.Tensor,
                student_logits: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        teacher_logits : (B, C) — Teacher 출력 (detach 권장)
        student_logits : (B, C) — Student 출력

        Returns
        -------
        loss : scalar
        """
        T = self.T
        # Soft probabilities
        p_teacher = F.softmax(teacher_logits.detach() / T, dim=1)
        log_p_student = F.log_softmax(student_logits / T, dim=1)

        # KL Divergence: T^2 스케일링 (Hinton et al.)
        loss = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (T ** 2)
        return loss


class FeatureKDLoss(nn.Module):
    """
    중간 임베딩 레벨의 Knowledge Distillation Loss.

    Teacher의 video_emb, audio_emb와
    Student의 video_emb, audio_emb 사이의 MSE를 최소화합니다.

    Architecture ③에서 Teacher와 Student가 같은 embed_dim을 공유하므로
    별도의 projection layer 없이 직접 정렬 가능합니다.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self,
                teacher_video_emb: torch.Tensor,
                teacher_audio_emb: torch.Tensor,
                student_video_emb: torch.Tensor,
                student_audio_emb: torch.Tensor,
                has_audio: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        teacher_video_emb : (B, embed_dim)
        teacher_audio_emb : (B, embed_dim)
        student_video_emb : (B, embed_dim)
        student_audio_emb : (B, embed_dim)
        has_audio         : (B,) bool — 0벡터로 채워진 샘플은 audio KD 제외
        """
        video_loss = self.loss_fn(student_video_emb, teacher_video_emb.detach())

        if has_audio is not None and has_audio.any():
            # 실제 audio가 있는 샘플에서만 audio 임베딩 정렬
            audio_loss = F.mse_loss(student_audio_emb[has_audio],
                                    teacher_audio_emb[has_audio].detach())
        elif has_audio is None:
            audio_loss = self.loss_fn(student_audio_emb, teacher_audio_emb.detach())
        else:
            audio_loss = torch.tensor(0.0, device=student_video_emb.device)

        return (video_loss + audio_loss) / 2.0


class TotalKDLoss(nn.Module):
    """
    최종 KD 통합 손실 함수.

    Total Loss = α × Soft_KD + β × Feature_KD + γ × CE

    arch='lstm' 인 경우 β=0 자동 적용.
    """
    def __init__(self,
                 alpha: float       = config.KD_ALPHA,
                 beta: float        = config.KD_BETA,
                 gamma: float       = config.KD_GAMMA,
                 temperature: float = config.KD_TEMPERATURE,
                 arch: str          = 'mlp'):
        super().__init__()
        self.alpha = alpha
        # Feature KD는 MLP와 E2E 아키텍처에서 사용 가능
        self.beta  = beta if arch in ['mlp', 'e2e'] else 0.0
        self.gamma = gamma

        # LSTM은 Feature KD 없으므로 alpha+gamma=1 로 재정규화
        if arch == 'lstm':
            total = alpha + gamma
            self.alpha = alpha / total
            self.gamma = gamma / total

        self.soft_kd_loss = SoftLabelKDLoss(temperature=temperature)
        self.feat_kd_loss = FeatureKDLoss()
        self.ce_loss      = nn.CrossEntropyLoss()

    def forward(self,
                teacher_logits: torch.Tensor,
                student_logits: torch.Tensor,
                labels: torch.Tensor,
                teacher_video_emb: torch.Tensor = None,
                teacher_audio_emb: torch.Tensor = None,
                student_video_emb: torch.Tensor = None,
                student_audio_emb: torch.Tensor = None,
                has_audio: torch.Tensor = None):
        """
        Parameters
        ----------
        teacher_logits     : (B, C)
        student_logits     : (B, C)
        labels             : (B,) — ground truth
        teacher_video_emb  : (B, E) — Feature KD용 (arch='mlp'일 때)
        teacher_audio_emb  : (B, E)
        student_video_emb  : (B, E)
        student_audio_emb  : (B, E)

        Returns
        -------
        total_loss  : scalar
        loss_dict   : {'soft_kd', 'feat_kd', 'ce', 'total'}
        """
        # 1. Soft Label KD
        L_soft = self.soft_kd_loss(teacher_logits, student_logits)

        # 2. Feature KD (MLP arch만)
        if (self.beta > 0 and
                teacher_video_emb is not None and
                student_video_emb is not None):
            L_feat = self.feat_kd_loss(teacher_video_emb, teacher_audio_emb,
                                        student_video_emb, student_audio_emb,
                                        has_audio=has_audio)
        else:
            L_feat = torch.tensor(0.0, device=student_logits.device)

        # 3. Cross-Entropy
        L_ce = self.ce_loss(student_logits, labels)

        total = self.alpha * L_soft + self.beta * L_feat + self.gamma * L_ce

        loss_dict = {
            'soft_kd': L_soft.item(),
            'feat_kd': L_feat.item(),
            'ce':      L_ce.item(),
            'total':   total.item(),
        }
        return total, loss_dict
