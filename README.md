# PreSense: 발표 스트레스 실시간 감지 시스템

웹캠 + 마이크만으로 발표 중 스트레스를 실시간으로 감지하고, 인간 해석 가능한 개념 기반의 XAI 피드백을 제공하는 시스템입니다.

---

## 프로젝트 목표

| 목표 | 내용 |
|---|---|
| **배포 조건** | 웹캠 + 마이크만으로 추론 (생리신호 센서 불필요) |
| **학습 전략** | LUPI KD — 학습 시에만 생리신호(ECG/EDA/호흡) 사용 |
| **설명 가능성** | CBM(Concept Bottleneck Model) — 7개 개념으로 예측 근거 제공 |
| **데이터** | StressID (영상+음성+생리신호 멀티모달, 이진 분류 Stress/Non-stress) |

---

## 최종 모델 아키텍처 (V2)

### 전체 파이프라인

```
[학습 시]
생리신호(132-dim) ─┐
영상 프레임        ─┤─ Teacher ─┐
음성 임베딩        ─┘          │ LUPI KD (Phase 1)
                               ▼
영상 프레임        ─┐         Student V2 (DualBranchStudentCMABiV2)
음성 임베딩        ─┤             │  backbone 동결
eGeMAPS(88-dim)   ─┘             ▼
                            CBM Head (Phase 2)
                                 │
                                 ▼
                       7개 개념 + 스트레스 판정

[추론 시]
웹캠 영상 + 마이크 음성 → Student V2 → CBM → 스트레스 판정 + XAI 설명
```

---

### Phase 1 — LUPI Knowledge Distillation

#### Teacher: `DualBranchTeacher`

학습 시에만 사용. 영상 + 음성 + 생리신호 3개 모달리티 입력.

```
frame_feats (B, 16, 512)  ─── LSTM(512→256) ─────────────────► h_T (B, 256) ─┐
audio_emb   (B, 256)      ────────────────────────────────────────────────────► ModalityGate
physio      (B, 132)      ─── Linear(132→128) + BN + ReLU ───────────────────►  (3-way softmax)
                                                                                  fused (B, 256)
geometry    (B, 16, 24)   ─── GRU(24→64) + Dropout + LayerNorm ────────────── g_emb (B, 64)

fused_expanded = cat[fused, g_emb]          (B, 320)
task_emb       = Embedding(11 tasks, 8)     (B, 8)
logits = Linear(328→128) → ReLU → Dropout → Linear(128→2)
```

#### Student V2: `DualBranchStudentCMABiV2`

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
            TemporalAttentionPool: score=Linear(64,1) → softmax(T) → weighted sum
            (V1 GRU 마지막 hidden state → V2 attention-weighted pool로 교체)

fused_expanded = cat[fused, g_emb]           (B, 320)

eGeMAPS     (B, 88)   [선택적 입력]
      └── ege_branch: Linear(88→64) → ReLU → LayerNorm   → (B, 64)
      └── ege_fusion: Linear(384→320) → ReLU → LayerNorm  → fused_expanded (B, 320)

task_emb       = Embedding(11 tasks, 8)      (B, 8)
logits_LUPI    = Linear(328→128) → ReLU → Dropout(0.3) → Linear(128→2)
```

**V1 대비 변경점:**
- `GRU last hidden` → `GeometryGRUAttn` (TemporalAttentionPool): 발표 중 스트레스 피크 구간 포착
- `eGeMAPS late fusion`: Jitter/Shimmer/F0 통계 → 음색 불안정성·음성 피치 R² 대폭 개선
- CrossModalAttention `a_dim=256` 유지 → V1 가중치 완전 전이

#### LUPI KD 손실

```
L = 0.3 · CE(s_logits, labels)
  + 0.4 · KL(softmax(s/T) ‖ softmax(t/T)) · T²     (T=4.0)
  + 0.3 · MSE(s_fused_expanded, t_fused_expanded)
```

---

### Phase 2 — Residual CBM (XAI)

Student backbone 동결 후, fused_expanded 위에 CBM head 학습.

```
fused_expanded (B, 320)   [Student backbone 동결]
        │
        ├─ concept_predictor
        │     Linear(320→128) → ReLU → Dropout(0.3) → Linear(128→7)
        │     c_pred (B, 7)   ← 7개 개념 예측값
        │
        └─ concept_classifier
              Linear(7→32) → ReLU → Linear(32→2)
              c_logit (B, 2)

final_logit = c_logit + 0.80 × logits_LUPI
```

**CBM 손실**

```
L = CE(final_logit, labels) + 4.0 · MSE(c_pred, concept_labels)
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

### 입출력 형식 요약

| 구분 | 이름 | 형태 | 설명 |
|---|---|---|---|
| 입력 | frame_feats | (B, 16, 512) | ResNet18 per-frame feature, 16프레임 샘플링 |
| 입력 | audio_emb | (B, 256) | Wav2Vec2 768-dim → Linear(256) |
| 입력 | geometry | (B, 16, 24) | MediaPipe 파생 geometry 지표 24개 |
| 입력 | eGeMAPS | (B, 88) | opensmile eGeMAPS v02 Functionals (Optional) |
| 입력 | task_id | (B,) | 발표 과제 유형 (11종) |
| 내부 | fused_expanded | (B, 320) | BiLSTM+CMA+GeoGRUAttn(+eGeMAPS) 융합 표현 |
| 출력 | logits | (B, 2) | 스트레스 / 비스트레스 |
| 출력 | c_pred | (B, 7) | 7개 개념 예측값 (XAI) |
| 출력 | attn_w | (B, 16) | 프레임별 어텐션 가중치 (시각화용) |

---

## 최종 성능

### 모델별 성능 비교 (GroupShuffleSplit seed=42, val n=121)

| 모델 | Acc | Sensitivity | Specificity | R² avg (val) | 비고 |
|---|---|---|---|---|---|
| Teacher (영상+음성+생리신호) | 0.7917 | — | — | — | 학습 전용, 배포 불가 |
| Student V1 LUPI KD | 0.8099 | — | — | — | 배포용, 영상+음성만 |
| Residual CBM V1 (w=0.7) | 0.8182 | 0.8382 | 0.7925 | 0.5965 | V1 기준선 |
| **Residual CBM V2 (w=0.8, λ=4.0)** | **0.8182** | **0.8529** | 0.7736 | **0.6916** | **최종 채택** |

> V2 Confusion Matrix (val): TN=41, FP=12, FN=10, TP=58  
> R² 비교는 val set 전용 (n=121) 기준. 학습 스크립트 내 R²(전체 기준)와 다를 수 있음.

### 주요 Ablation 이력

| 방법 | Acc | 비고 |
|---|---|---|
| 기준선 (LUPI KD BiLSTM V1) | 0.8099 | CBM 이전 backbone |
| Residual CBM V1 (w=0.7) | 0.8182 | V1 최고 Acc |
| Residual CBM V1 (w=0.5) | 0.7934 | V1 최고 R² (스크립트 기준 0.8085) |
| Joint LUPI+CBM | 0.7934 | backbone gradient 충돌 |
| DANN (λ=0.2) | 0.8167* | 단일 seed noise |
| 5-seed 앙상블 | 0.8083 | Majority Vote 기준 |
| GeometryGRUAttn (V2 backbone) | 0.8017 | LUPI 단독, eGeMAPS 없이 |
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
PreSenseModel V2
  └── DualBranchStudentCMABiV2 + CBMWithLUPIResidual
        │
        ├── stress_prob (float)
        ├── c_pred (7개 개념값)
        └── attn_w (16 프레임 가중치)
        │
        ▼
SHAP Explainer → concept_contrib (attribution)
        │
        ▼
FeedbackEngine
  ├── DSPy ChainOfThought
  ├── ChromaDB RAG (논문 20편)
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
│   │   ├── models_e2e.py                      # DualBranchStudentCMABiV2, Teacher, CrossModalAttention
│   │   │                                      # TemporalAttentionPool, GeometryGRUAttn
│   │   └── models_cbm.py                      # CBM 구조, 개념 이름 정의
│   ├── train/
│   │   ├── train_lupi_kd_cma_bi.py            # Phase 1 V1: LUPI KD (BiLSTM)
│   │   ├── train_egemap_kd.py               # StudentCMABi KD 학습 (eGeMAPS + GeometryGRUAttn)
│   │   ├── train_cbm_bi_lupi_residual.py      # Phase 2 V1: Residual CBM
│   │   ├── train_egemap_cbm.py                # Phase 2 V2: Residual CBM (eGeMAPS)
│   │   ├── train_multiseed_eval.py            # 다중 seed 신뢰도 평가
│   │   └── train_ensemble_multiseed.py        # 동일 split 앙상블
│   ├── data/
│   │   └── dataset_e2e.py                     # StressID 데이터 로더, N_FRAMES=16
│   └── feature_analysis/
│       ├── extract_landmarks_per_frame.py     # MediaPipe geometry 추출
│       └── extract_ege_maps.py                # opensmile eGeMAPS v02 추출
├── agent/
│   ├── model_wrapper.py                       # 실시간 추론 엔진 (V2)
│   ├── agent_main.py                          # 웹캠 + 마이크 스트리밍
│   ├── feedback_engine.py                     # FeedbackEngine (DSPy + RAG)
│   ├── dspy_pipeline.py                       # StressCoachModule (ChainOfThought)
│   ├── rag_db.py                              # ChromaDB 논문 RAG
│   ├── rag_evaluator.py                       # RAGAS 오프라인 평가
│   ├── shap_explainer.py                      # SHAP KernelExplainer
│   └── luxia_client.py                        # 솔트룩스 Luxia API 클라이언트
├── checkpoints/binary/
│   ├── final/
│   │   ├── dual_lupi_v2_student_best.pt           # Phase 1 V2 최종 (Acc=0.8017)
│   │   ├── dual_lupi_cma_bi_student_best.pt       # Phase 1 V1 (참고용)
│   │   └── dual_lupi_cma_teacher_best.pt          # Teacher (학습 전용)
│   ├── cbm_bi_lupi_residual_v2_w0.8_lc4.0_best.pt  # Phase 2 V2 최종 ★
│   ├── cbm_bi_lupi_residual_w0.7_best.pt           # Phase 2 V1 (참고용)
│   ├── dual_lupi_v2_fused_expanded.npy            # V2 fused_expanded 캐시 (N, 320)
│   └── ege_maps_feats.npy                          # eGeMAPS 피처 캐시 (N, 88)
├── EXPERIMENTS.md                             # 전체 실험 이력
├── XAI_고민정리.md                             # CBM 설계 고민 기록
└── README.md
```

---

## 빠른 시작

```bash
# 실시간 에이전트 실행
cd presense && python agent/agent_main.py
# 'c': 영점 보정(Calibration), 'q': 종료

# StudentCMABi KD 학습 (eGeMAPS + GeometryGRUAttn)
cd presense && python ai/feature_analysis/extract_ege_maps.py   # eGeMAPS 추출 (~3분)
cd presense && python ai/train/train_egemap_kd.py               # KD backbone 학습

# Phase 2 V2: Residual CBM
cd presense && python ai/train/train_egemap_cbm.py              # CBM 학습 (그리드 탐색)
```

---

## 환경

```
Python 3.12 / PyTorch 2.x / MPS(Apple Silicon) or CUDA
torchvision, torchaudio, transformers (Wav2Vec2)
mediapipe, opencv-python, opensmile
scikit-learn, numpy, pandas
dspy-ai, chromadb
librosa, shap
```
