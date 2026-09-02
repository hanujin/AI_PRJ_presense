"""
PreSense — Backbone 모델 정의 (pre-extracted features 기반)
  Teacher, StudentCMABi, CBMWithResidual

입력 파이프라인:
  frame_feats : (B, T, 512)  ResNet18 per-frame  (cbm_resnet_feats.npy)
  audio_emb   : (B, 256)     Wav2Vec2 → Linear   (cbm_w2v_feats.npy)
  geometry    : (B, T, 24)   MediaPipe 파생 지표  (dual_geo_scaled.npy)
  ege_feats   : (B, 88)      eGeMAPS v02           (ege_maps_feats.npy, optional)
  physio      : (B, 132)     ECG/EDA/Resp CSV     (Teacher 학습 전용)
  task_id     : (B,)         발표 태스크 인덱스

활성 모델:
  Teacher        — KD Teacher (학습 전용, physio 사용)
  StudentCMABi — V2 Student (eGeMAPS + GeometryGRUAttn)  ← 배포 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 차원 상수 ─────────────────────────────────────────────────────────────────
VIDEO_EMB_DIM  = 512
AUDIO_EMB_DIM  = 256
EGEMAP_DIM     = 88
AUDIO_V2_DIM   = AUDIO_EMB_DIM + EGEMAP_DIM   # 344

NUM_TASKS    = 11
TASK_EMB_DIM = 8

GEO_INPUT_DIM  = 24
GEO_HIDDEN_DIM = 64
FUSED_EXP_DIM  = AUDIO_EMB_DIM + GEO_HIDDEN_DIM   # 320


# ══════════════════════════════════════════════════════════════════════════════
#  공용 모듈
# ══════════════════════════════════════════════════════════════════════════════

class ModalityGate(nn.Module):
    """
    학습 가능한 경량 모달리티 게이트.
    concat(v, a[, p]) → softmax → 가중합 → fused
    """
    def __init__(self, v_dim: int, a_dim: int, p_dim: int = None, proj_dim: int = 256):
        super().__init__()
        self.proj_dim   = proj_dim
        self.has_physio = p_dim is not None
        n = 3 if self.has_physio else 2

        self.v_proj = nn.Linear(v_dim, proj_dim)
        self.a_proj = nn.Linear(a_dim, proj_dim)
        if self.has_physio:
            self.p_proj = nn.Linear(p_dim, proj_dim)
        self.gate = nn.Linear(proj_dim * n, n)

    def forward(self, v, a, p=None):
        vp = self.v_proj(v)
        ap = self.a_proj(a)
        if self.has_physio and p is not None:
            pp    = self.p_proj(p)
            w     = torch.softmax(self.gate(torch.cat([vp, ap, pp], dim=1)), dim=1)
            fused = w[:, 0:1] * vp + w[:, 1:2] * ap + w[:, 2:3] * pp
        else:
            w     = torch.softmax(self.gate(torch.cat([vp, ap], dim=1)), dim=1)
            fused = w[:, 0:1] * vp + w[:, 1:2] * ap
        return fused, w  # (B, proj_dim), (B, n)


class CrossModalAttention(nn.Module):
    """
    Q: audio_emb (B, a_dim) — 무엇을 찾을지
    K, V: lstm_out (B, T, v_dim) — 영상 프레임 시퀀스
    → 음성이 관련 프레임에 주목 → audio-guided video representation
    """
    def __init__(self, v_dim: int = 256, a_dim: int = 256,
                 d_k: int = 64, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.q_proj   = nn.Linear(a_dim, d_k)
        self.k_proj   = nn.Linear(v_dim, d_k)
        self.v_proj   = nn.Linear(v_dim, out_dim)
        self.out_proj = nn.Linear(out_dim + a_dim, out_dim)
        self.norm     = nn.LayerNorm(out_dim)
        self.drop     = nn.Dropout(dropout)
        self.scale    = d_k ** -0.5

    def forward(self, lstm_out: torch.Tensor, a_emb: torch.Tensor):
        """
        lstm_out : (B, T, v_dim)
        a_emb    : (B, a_dim)
        → fused (B, out_dim), attn_weights (B, T)
        """
        Q = self.q_proj(a_emb).unsqueeze(1)              # (B, 1, d_k)
        K = self.k_proj(lstm_out)                         # (B, T, d_k)
        V = self.v_proj(lstm_out)                         # (B, T, out_dim)
        scores   = Q @ K.transpose(-1, -2) * self.scale  # (B, 1, T)
        weights  = torch.softmax(scores, dim=-1)          # (B, 1, T)
        attended = self.drop(weights @ V).squeeze(1)      # (B, out_dim)
        fused    = self.out_proj(torch.cat([attended, a_emb], dim=1))
        return self.norm(fused), weights.squeeze(1)       # (B, out_dim), (B, T)


class GeometryGRUAttn(nn.Module):
    """
    geometry 시퀀스 (B, T, 24) → g_emb (B, 64).
    GRU 전체 스텝 출력을 학습된 스칼라 어텐션으로 집계.
    발표 중 스트레스 피크 구간을 시점과 무관하게 포착.
    """
    def __init__(self, input_dim: int = GEO_INPUT_DIM,
                 hidden_dim: int = GEO_HIDDEN_DIM, dropout: float = 0.2):
        super().__init__()
        self.gru        = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.attn_drop  = nn.Dropout(dropout * 0.5)
        self.drop       = nn.Dropout(dropout)
        self.norm       = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)                                    # (B, T, 64)
        w      = torch.softmax(self.attn_score(out), dim=1)    # (B, T, 1)
        pooled = (self.attn_drop(w) * out).sum(dim=1)          # (B, 64)
        return self.norm(self.drop(pooled))


# ══════════════════════════════════════════════════════════════════════════════
#  모델
# ══════════════════════════════════════════════════════════════════════════════

class Teacher(nn.Module):
    """
    KD Teacher — video + audio + geometry + physio.

    입력 (모두 pre-extracted):
      frame_feats : (B, T, 512)
      audio_emb   : (B, 256)
      physio      : (B, physio_in)  ECG/EDA/Resp CSV 피처
      geometry    : (B, T, 24)
      task_id     : (B,)

    출력: logits (B, 2), fused_expanded (B, 320)
    """
    def __init__(self, physio_in: int = 132, lstm_hidden: int = 256,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.video_lstm = nn.LSTM(VIDEO_EMB_DIM, lstm_hidden, batch_first=True)
        self.geo_gru    = GeometryGRUAttn(GEO_INPUT_DIM, GEO_HIDDEN_DIM, dropout=dropout * 0.7)
        self.physio_enc = nn.Sequential(
            nn.Linear(physio_in, 128), nn.LayerNorm(128),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.modal_gate = ModalityGate(lstm_hidden, AUDIO_EMB_DIM, 128,
                                       proj_dim=lstm_hidden)
        self.task_emb   = nn.Embedding(NUM_TASKS, TASK_EMB_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(FUSED_EXP_DIM + TASK_EMB_DIM, 128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, frame_feats: torch.Tensor, audio_emb: torch.Tensor,
                physio: torch.Tensor, geometry: torch.Tensor,
                task_id: torch.Tensor):
        _, (h_v, _) = self.video_lstm(frame_feats)
        v_emb = h_v[-1]                             # (B, 256)
        g_emb = self.geo_gru(geometry)              # (B, 64)
        p_emb = self.physio_enc(physio)             # (B, 128)

        fused, _       = self.modal_gate(v_emb, audio_emb, p_emb)  # (B, 256)
        fused_expanded = torch.cat([fused, g_emb], dim=1)           # (B, 320)
        t_emb          = self.task_emb(task_id)
        logits = self.classifier(torch.cat([fused_expanded, t_emb], dim=1))
        return logits, fused_expanded


class StudentCMABi(nn.Module):
    """
    V2 Student (배포 모델) — eGeMAPS late fusion + GeometryGRUAttn.

    fused_expanded = 320 → Teacher와 KD loss 정렬 가능.
    ege_feats=None이면 eGeMAPS 경로를 건너뜀.
    """
    BI_HIDDEN    = 128
    EGE_PROJ_DIM = 64

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.video_lstm = nn.LSTM(
            VIDEO_EMB_DIM, self.BI_HIDDEN,
            batch_first=True, bidirectional=True,
        )
        self.geo_gru    = GeometryGRUAttn(GEO_INPUT_DIM, GEO_HIDDEN_DIM,
                                          dropout=dropout * 0.7)
        self.cross_attn = CrossModalAttention(
            v_dim=self.BI_HIDDEN * 2, a_dim=AUDIO_EMB_DIM,
            d_k=64, out_dim=self.BI_HIDDEN * 2, dropout=dropout * 0.3,
        )
        self.ege_branch = nn.Sequential(
            nn.Linear(EGEMAP_DIM, self.EGE_PROJ_DIM),
            nn.ReLU(),
            nn.LayerNorm(self.EGE_PROJ_DIM),
        )
        self.ege_fusion = nn.Sequential(
            nn.Linear(FUSED_EXP_DIM + self.EGE_PROJ_DIM, FUSED_EXP_DIM),
            nn.ReLU(),
            nn.LayerNorm(FUSED_EXP_DIM),
        )
        self.task_emb   = nn.Embedding(NUM_TASKS, TASK_EMB_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(FUSED_EXP_DIM + TASK_EMB_DIM, 128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, frame_feats: torch.Tensor, audio_emb: torch.Tensor,
                geometry: torch.Tensor, task_id: torch.Tensor,
                ege_feats: torch.Tensor | None = None,
                return_fused: bool = False):
        """
        frame_feats : (B, T, 512)
        audio_emb   : (B, 256)
        geometry    : (B, T, 24)
        task_id     : (B,)
        ege_feats   : (B, 88)  z-score 정규화된 eGeMAPS (없으면 skip)
        """
        lstm_out, _    = self.video_lstm(frame_feats)           # (B, T, 256)
        g_emb          = self.geo_gru(geometry)                  # (B, 64)
        fused, attn_w  = self.cross_attn(lstm_out, audio_emb)   # (B, 256), (B, T)
        fused_expanded = torch.cat([fused, g_emb], dim=1)       # (B, 320)

        if ege_feats is not None:
            e_emb = self.ege_branch(ege_feats)                   # (B, 64)
            fused_expanded = self.ege_fusion(
                torch.cat([fused_expanded, e_emb], dim=1))       # (B, 320)

        t_emb  = self.task_emb(task_id)
        logits = self.classifier(torch.cat([fused_expanded, t_emb], dim=1))
        # None 자리: 레거시 시그니처 (physio_emb, v_emb) 호환 — train 스크립트가 positional unpack
        if return_fused:
            return logits, None, None, audio_emb, attn_w, fused_expanded
        return logits, None, None, audio_emb, attn_w


# ══════════════════════════════════════════════════════════════════════════════
#  CBM (Concept Bottleneck Module — post-hoc XAI layer)
# ══════════════════════════════════════════════════════════════════════════════

class CBMWithResidual(nn.Module):
    """
    목적:
        train_egemap_kd.py로 학습된 StudentCMABi에 XAI용 CBM 레이어를 추가.
        모델이 '왜 스트레스인가'를 개념 7개로 설명할 수 있게 함.

    구조:
        student_kd_fused_expanded.npy (320차원)
            ↓
        concept_predictor : Linear(320→128) → ReLU → Linear(128→7)
            └ 320차원 내부 표현을 압축해서 개념 7개 수치로 변환
              각 개념은 0~1 범위 없이 실수값으로 출력
        concept_classifier : Linear(7→32) → ReLU → Linear(32→2)
            └ 개념 7개만 보고 스트레스/비스트레스 점수(logit) 계산

        final = concept_logits + residual_w × StudentCMABi_logits
            └ 개념 7개만으로는 정보가 부족해 StudentCMABi가 직접 계산한
              점수를 residual_w 비율로 섞어서 보완 (Koh et al. 2020 Hybrid CBM)

        residual_w=0 → 순수 CBM (완전 해석 가능, 성능 손실 위험)
        residual_w>0 → 잔류 경로로 성능 보전
    """
    def __init__(self, fused_dim: int = FUSED_EXP_DIM, n_concepts: int = 7,
                 residual_w: float = 0.7, dropout: float = 0.3):
        super().__init__()
        self.residual_w = residual_w
        self.concept_predictor = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_concepts))
        self.concept_classifier = nn.Sequential(
            nn.Linear(n_concepts, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, fused_exp: torch.Tensor,
                base_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        fused_exp   : (B, fused_dim)  StudentCMABi의 fused_expanded
        base_logits : (B, 2)          StudentCMABi의 분류 로짓
        → final_logits (B, 2), c_pred (B, n_concepts)
        """
        c_pred  = self.concept_predictor(fused_exp)
        c_logit = self.concept_classifier(c_pred)
        return c_logit + self.residual_w * base_logits.detach(), c_pred
