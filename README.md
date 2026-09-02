# PreSense: 발표 스트레스 실시간 감지 시스템

웹캠 + 마이크만으로 발표 중 스트레스를 실시간으로 감지하고, 인간 해석 가능한 개념 기반의 XAI 피드백을 제공하는 시스템입니다.

---

## 프로젝트 목표

| 목표 | 내용 |
|---|---|
| **배포 조건** | 웹캠 + 마이크만으로 추론 (생리신호 센서 불필요) |
| **학습 전략** | KD — 학습 시에만 생리신호(ECG/EDA/호흡) 사용 |
| **설명 가능성** | CBM(Concept Bottleneck Model) — 7개 개념으로 예측 근거 제공 |
| **데이터** | StressID (영상+음성+생리신호 멀티모달, 이진 분류 Stress/Non-stress) |

---

## 최종 모델 아키텍처

### 전체 파이프라인

```
[학습 시]
생리신호(132-dim) ─┐
영상 프레임        ─┤─ Teacher ─┐
음성 임베딩        ─┘          │ KD (Phase 1)
                               ▼
영상 프레임        ─┐         StudentCMABi
음성 임베딩        ─┤             │  backbone 동결
eGeMAPS(88-dim)   ─┘             ▼
                            CBMWithResidual (Phase 2)
                                 │
                                 ▼
                       7개 개념 + 스트레스 판정

[추론 시]
웹캠 영상 + 마이크 음성 → StudentCMABi → CBMWithResidual → 스트레스 판정 + XAI 설명
```

---

### Phase 1 — Knowledge Distillation

#### Teacher

학습 시에만 사용. 영상 + 음성 + 생리신호 3개 모달리티 입력.

```
frame_feats (B, 16, 512)  ─── LSTM(512→256) ──────────────────► h_T (B, 256) ─┐
audio_emb   (B, 256)      ───────────────────────────────────────────────────► ModalityGate
physio      (B, 132)      ─── Linear(132→128) + BN + ReLU ──────────────────►  (3-way softmax)
                                                                                 fused (B, 256)
geometry    (B, 16, 24)   ─── GRU(24→64) + Dropout + LayerNorm ─────────── g_emb (B, 64)

fused_expanded = cat[fused, g_emb]          (B, 320)
task_emb       = Embedding(11 tasks, 8)     (B, 8)
logits = Linear(328→128) → ReLU → Dropout → Linear(128→2)
```

#### StudentCMABi

배포용. 영상 + 음성 + eGeMAPS 입력 (생리신호 불필요).

```
frame_feats (B, 16, 512)
      │
      └── BiLSTM(512, hidden=128, bidirectional=True)
              lstm_out (B, 16, 256)
                    │
                    │          audio_emb (B, 256)
                    │                │
                    └── CrossModalAttention (a_dim=256)
                         Q = Linear(256→64)(audio)
                         K = Linear(256→64)(lstm_out)
                         V = Linear(256→256)(lstm_out)
                         fused (B, 256)

geometry    (B, 16, 24)
      └── GeometryGRUAttn ──────────────────────────── g_emb (B, 64)
            GRU(24→64) → all hidden states
            score=Linear(64,1) → softmax(T) → weighted sum

fused_expanded = cat[fused, g_emb]           (B, 320)

eGeMAPS     (B, 88)   [선택적 입력]
      └── ege_branch: Linear(88→64) → ReLU → LayerNorm   → (B, 64)
      └── ege_fusion: Linear(384→320) → ReLU → LayerNorm  → fused_expanded (B, 320)

task_emb = Embedding(11 tasks, 8)      (B, 8)
logits   = Linear(328→128) → ReLU → Dropout(0.3) → Linear(128→2)
```

#### KD 손실

```
L = 0.3 · CE(s_logits, labels)
  + 0.4 · KL(softmax(s/T) ‖ softmax(t/T)) · T²     (T=4.0)
  + 0.3 · MSE(s_fused_expanded, t_fused_expanded)
```

---

### Phase 2 — Residual CBM (XAI)

StudentCMABi backbone 동결 후, fused_expanded 위에 CBM head 학습.

```
fused_expanded (B, 320)   [StudentCMABi backbone 동결]
        │
        ├─ concept_predictor
        │     Linear(320→128) → ReLU → Dropout(0.3) → Linear(128→7)
        │     c_pred (B, 7)   ← 7개 개념 예측값
        │
        └─ concept_classifier
              Linear(7→32) → ReLU → Linear(32→2)
              c_logit (B, 2)

final_logit = c_logit + 0.80 × base_logits   [student_kd_logits.npy 캐시 사용]
```

**CBM 손실**

```
Stage 1: MSE(c_pred, concept_labels)                              # concept predictor warmup
Stage 2: CE(final_logit, labels) + 4.0 · MSE(c_pred, concept_labels)
```

---

### 7개 개념 정의

| # | 개념 | 도메인 | 원본 피처 | R² (V1, val) | R² (V2, val) |
|---|---|---|---|---|---|
| 1 | 목소리 거칠기 | 음성 | Spectral Contrast Mean | — | — |
| 2 | 음색 불안정성 | 음성 | Tonnetz Std + eGeMAPS Jitter/Shimmer | +0.264 | **+0.384** |
| 3 | 음성 두께 | 음성 | MFCC Mean (저주파) | — | — |
| 4 | 음성 피치 | 음성 | F0 Mean + eGeMAPS F0 통계 | +0.549 | **+0.824** |
| 5 | 표정 변화량 | 영상 | Expression Variability + TemporalAttn | +0.313 | **+0.463** |
| 6 | 입술 긴장도 | 영상 | Lip Compression (MediaPipe) | — | — |
| 7 | 눈썹 찡그림 | 영상 | Brow Tension (landmarks 55/285) | — | — |

> val set 기준 R² 평균: V1 = 0.5965 → **V2 = 0.6916** (+0.095)

---

## 최종 성능

### 모델별 성능 비교 (GroupShuffleSplit seed=42, val n=121)

| 모델 | Acc | Sensitivity | Specificity | R² avg (val) | 비고 |
|---|---|---|---|---|---|
| Teacher (영상+음성+생리신호) | 0.7917 | — | — | — | 학습 전용, 배포 불가 |
| StudentCMABi KD | 0.8099 | — | — | — | 배포용, 영상+음성만 |
| Residual CBM V1 (w=0.7) | 0.8182 | 0.8382 | 0.7925 | 0.5965 | V1 기준선 |
| **Residual CBM V2 (w=0.8, λ=4.0)** | **0.8182** | **0.8529** | 0.7736 | **0.6916** | **최종 채택** |

> V2 Confusion Matrix (val): TN=41, FP=12, FN=10, TP=58  
> R² 비교는 val set 전용 (n=121) 기준. 학습 스크립트 내 R²(전체 기준)와 다를 수 있음.

### 주요 Ablation 이력

| 방법 | Acc | 비고 |
|---|---|---|
| 기준선 (KD BiLSTM V1) | 0.8099 | CBM 이전 backbone |
| Residual CBM V1 (w=0.7) | 0.8182 | V1 최고 Acc |
| Residual CBM V1 (w=0.5) | 0.7934 | V1 최고 R² (스크립트 기준 0.8085) |
| Joint KD+CBM | 0.7934 | backbone gradient 충돌 |
| DANN (λ=0.2) | 0.8167* | 단일 seed noise |
| 5-seed 앙상블 | 0.8083 | Majority Vote 기준 |
| GeometryGRUAttn (V2 backbone) | 0.8017 | KD 단독, eGeMAPS 없이 |
| **Residual CBM V2 (w=0.8, λ=4.0)** | **0.8182** | R² +9.5%p (val), **최종** |

### 데이터셋 정보

| 항목 | 값 |
|---|---|
| 데이터셋 | StressID |
| 총 샘플 | 578개 (생리신호 포함 577개) |
| 분할 방법 | GroupShuffleSplit (피험자 단위, test_size=0.2, seed=42) |
| Train | 457개 / Val | 121개 |
| 피험자 | 53명 |
| 클래스 | Stress / Non-stress |

---

## 에이전트 파이프라인

```
실시간 입력 (웹캠 + 마이크)
        │
        ▼
StressInferenceEngine
  ├── ResNet18  → frame_feats (B, 16, 512)
  ├── Wav2Vec2  → audio_emb  (B, 256)
  ├── MediaPipe → geometry   (B, 16, 24)
  └── opensmile → eGeMAPS   (B, 88)
        │
        ▼
PreSenseModel
  └── StudentCMABi + CBMWithResidual
        │
        ├── stress_prob (float)
        ├── c_pred (7개 개념값)
        └── attn_w (16 프레임 가중치)
        │
        ▼
FeedbackEngine
  ├── DSPy ChainOfThought
  ├── ChromaDB RAG (data/papers/ 자동 처리)
  └── LuxiaLM (솔트룩스 Luxia API)
        │
        ▼
실시간 오버레이 피드백
```

---

## 프로젝트 구조

```
presense/
├── ai/
│   ├── models/
│   │   └── models_e2e.py          # Teacher, StudentCMABi, CBMWithResidual
│   ├── train/
│   │   ├── train_egemap_kd.py     # Phase 1: KD backbone 학습
│   │   └── train_egemap_cbm.py    # Phase 2: Residual CBM 학습
│   ├── data/
│   │   └── dataset_e2e.py         # StressID 데이터 로더
│   └── feature_analysis/
│       ├── extract_geometry_feats.py  # MediaPipe geometry 추출
│       └── extract_ege_maps.py        # opensmile eGeMAPS v02 추출
├── agent/
│   ├── model_wrapper.py           # StressInferenceEngine + PreSenseModel
│   ├── feedback_engine.py         # FeedbackEngine (베이스라인 캘리브레이션 + LLM)
│   ├── dspy_pipeline.py           # StressCoachModule (DSPy ChainOfThought)
│   └── rag_db.py                  # ChromaDB RAG (data/papers/ 자동 처리)
├── data/
│   └── papers/                    # PDF 논문 → PaperRAG() 초기화 시 자동 처리
├── checkpoints/binary/
│   ├── student_kd_best.pt             # Phase 1 최종 (Acc=0.8099)
│   ├── cbm_bi_residual_v2_w0.8_lc4.0_best.pt  # Phase 2 최종 ★
│   ├── student_kd_fused_expanded.npy  # fused_expanded 캐시 (N, 320)
│   └── student_kd_logits.npy          # logits 캐시 (N, 2)
├── config.py
└── README.md
```

---

## 빠른 시작

```bash
# Phase 1: StudentCMABi KD 학습
cd presense
python ai/feature_analysis/extract_ege_maps.py   # eGeMAPS 추출 (~3분)
python ai/train/train_egemap_kd.py               # KD backbone 학습

# Phase 2: Residual CBM 학습
python ai/train/train_egemap_cbm.py              # CBM 학습 (그리드 탐색)
```

---

## 환경

```
Python 3.12 / PyTorch 2.x / MPS(Apple Silicon) or CUDA
torchvision, torchaudio, transformers (Wav2Vec2)
mediapipe, opencv-python, opensmile
scikit-learn, numpy, pandas
dspy-ai, chromadb, pdfplumber
librosa
```
