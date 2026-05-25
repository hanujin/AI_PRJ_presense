# 🎤 PreSense: 발표 스트레스 실시간 감지 시스템

이 프로젝트는 인공지능을 활용하여 발표자가 느끼는 스트레스를 실시간으로 분석하고 피드백을 제공하기 위한 시스템입니다. **지식 증류(Knowledge Distillation)** 기법을 활용하여, 별도의 생리 신호 센서 없이 웹캠(영상)과 마이크(음성)만으로 고정밀 스트레스 감지를 구현합니다.

---

## 📂 프로젝트 구조 (Directory Structure)

모든 소스코드는 기능에 따라 5개의 폴더로 관리됩니다.

### 1. `data/` (데이터 처리)
*   `dataset.py`: 요약된 통계 피처(Handcrafted) 데이터를 로드하고 전처리합니다.
*   `dataset_e2e.py`: 비디오 프레임, 오디오 파형, 생리 신호 시계열 원본 데이터를 처리합니다.

### 2. `models/` (모델 정의)
*   `models.py`: 표준 MLP 및 LSTM 기반의 모델 구조가 정의되어 있습니다.
*   `models_e2e.py`: ResNet, Wav2Vec 2.0, 1D-CNN 등이 결합된 하위 하이브리드 모델 구조입니다.
*   `kd_loss.py`: 지식 증류를 위한 Soft Label 및 Feature KD 손실 함수를 포함합니다.

### 3. `train/` (학습 스크립트)
*   `train_e2e_teacher.py` / `train_e2e_hybrid_teacher.py`: 모든 모달리티를 사용하는 고성능 Teacher 모델을 학습합니다.
*   `train_e2e_student.py`: Teacher의 지식을 전수받는 배포용 Student 모델을 학습합니다. (`--mode kd` 옵션 사용)
*   `train_pca_teacher.py` / `train_pca_student.py`: PCA로 압축된 고효율 데이터를 사용하는 모델을 학습합니다.

### 4. `evaluate/` (성능 평가)
*   `evaluate_e2e.py`: 가장 고도화된 Arch 1(E2E/Hybrid) 모델들의 성능을 통합 평가하고 혼동 행렬을 생성합니다.
*   `evaluate.py`: MLP, LSTM 및 PCA 기반 모델들의 성능 비교 리포트를 생성합니다.

### 5. `feature_analysis/` (데이터 분석)
*   `analyze_features.py`: 어떤 피처가 스트레스 감지에 중요한지(호흡, 심박 등) 분석하고 시각화합니다.
*   `analyze_pca.py` / `analyze_pca_components.py`: 데이터 압축 효율 및 압축된 성분의 의미를 분석합니다.
*   `results/`: 피처 중요도(`feature_importance.csv`)와 압축된 피처셋(`final_features.csv`)이 저장됩니다.

---

## 🚀 빠른 시작 가이드 (Quick Start)

### 1. 모델 학습
가장 성능이 뛰어난 하이브리드 학생 모델을 학습시키려면 아래 명령어를 사용하세요.
```bash
python presense/train/train_e2e_student.py --mode kd
```

### 2. 성능 확인
학습된 모델들의 최종 정확도와 성적표를 확인하려면 아래 명령어를 사용하세요.
```bash
python presense/evaluate/evaluate_e2e.py
```
