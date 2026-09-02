"""
PreSense - Configuration
프로젝트 전반에 사용되는 경로 및 하이퍼파라미터 설정
"""

from pathlib import Path

# ─── 디렉토리 경로 ────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.parent          # AI_Project/
STRESSID_DIR   = BASE_DIR / "stressID-main"
FEATURES_DIR   = STRESSID_DIR / "Feature Extraction" / "Features"
DATASET_DIR    = BASE_DIR / "Dataset"                  # 실제 StressID 데이터셋 위치
RESULTS_DIR    = Path(__file__).parent / "results"
CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"

# ─── 피처 파일 경로 ────────────────────────────────────────────────────────────
PHYSIO_FEATURES = FEATURES_DIR / "all_physiological_features.csv"
AUDIO_FEATURES  = FEATURES_DIR / "HCfeatures.csv"
VIDEO_FEATURES  = FEATURES_DIR / "video11tasks_aus_gaze_mean_std.csv"
LABELS_FILE     = FEATURES_DIR / "labels.csv"

# ─── 피처 차원 ────────────────────────────────────────────────────────────────
PHYSIO_DIM = 132   # ECG + EDA + Respiration 통계 피처
VIDEO_DIM  = 84    # Action Units (AU) + Eye Gaze 통계 피처
AUDIO_DIM  = 140   # MFCC 등 Handcrafted 음성 피처

# ─── 모델 아키텍처 하이퍼파라미터 (Architecture ③: Hybrid MLP Encoder) ───────
EMBED_DIM   = 64    # 각 Modality MLP Encoder 출력 차원
HIDDEN_DIM  = 128   # MLP 중간 레이어 차원
DROPOUT     = 0.3   # 드롭아웃 비율
NUM_CLASSES_BINARY = 2  # 이진 스트레스 분류 (0: 비스트레스, 1: 스트레스)
NUM_CLASSES_3WAY   = 3  # 3분류 (0: Relaxed, 1: Neutral, 2: Stressed)

# ─── 학습 하이퍼파라미터 ────────────────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 100
EARLY_STOPPING_PATIENCE = 15
RANDOM_SEED   = 42

# ─── KD (Knowledge Distillation) 손실 가중치 ──────────────────────────────────
# Total Loss = α × Soft_KD + β × Feature_KD + γ × CE
KD_ALPHA       = 0.4   # Soft Label KD (KL Divergence) 가중치
KD_BETA        = 0.3   # Feature KD (MSE) 가중치
KD_GAMMA       = 0.3   # Cross-Entropy 가중치
KD_TEMPERATURE = 4.0   # Soft Label KD 온도 파라미터

# ─── 데이터 분할 비율 (subject 기반 – 데이터 누수 방지) ────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ─── 추론 임계값 ──────────────────────────────────────────────────────────────
# 에러 분석 결과 FP(17) >> FN(5) → 0.40으로 낮춰 val_acc 0.8017 → 0.8099
STRESS_THRESHOLD = 0.40

# 스트레스 유발 태스크 (heuristic fallback labels 생성 시 사용)
STRESS_TASKS = {"Counting1", "Counting2", "Counting3", "Math", "Stroop", "Speaking", "Reading"}
RELAX_TASKS  = {"Breathing", "Video1", "Video2", "Relax"}
