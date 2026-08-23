"""
eGeMAPS v02 피처 추출 (opensmile)

Wav2Vec2 임베딩(256차원)이 잡지 못하는 명시적 음성 특징을 보완:
  - Jitter / Shimmer (성대 불규칙성 → 음색 불안정성, 목소리 거칠기)
  - Pitch 통계 (F0 mean, range, percentiles → 음성 피치)
  - HNR, spectral features (음성 두께, 공명)

출력: ege_maps_feats.npy  (N, 88)  — samples_all 순서와 동일
      ege_maps_names.npy  (88,)    — 피처명 배열

실행:
  pip install opensmile
  cd presense && python ai/feature_analysis/extract_ege_maps.py
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
for p in [_root, _root/'ai', _root/'ai'/'models', _root/'ai'/'data']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config
from dataset_e2e import build_sample_list, load_audio

SAVE_PATH  = config.CHECKPOINTS_DIR / 'binary' / 'ege_maps_feats.npy'
NAMES_PATH = config.CHECKPOINTS_DIR / 'binary' / 'ege_maps_names.npy'
SR = 16000


def extract_one(waveform_np: np.ndarray, smile, sr: int = SR) -> np.ndarray:
    """
    numpy waveform → eGeMAPS v02 Functionals (88차원).
    opensmile은 최소 0.5초 이상의 신호를 요구하므로 짧으면 패딩.
    """
    min_len = int(sr * 0.5)
    if len(waveform_np) < min_len:
        waveform_np = np.pad(waveform_np, (0, min_len - len(waveform_np)))

    try:
        feat = smile.process_signal(waveform_np, sr)
        return feat.values[0].astype(np.float32)
    except Exception:
        return np.zeros(88, dtype=np.float32)


def main():
    try:
        import opensmile
    except ImportError:
        print("opensmile 미설치. 실행: pip install opensmile")
        return

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    labels_df  = pd.read_csv(config.LABELS_FILE)
    labels_map = dict(zip(labels_df.iloc[:, 0], labels_df.iloc[:, 1]))
    samples    = build_sample_list(labels_map)
    N          = len(samples)

    print(f"eGeMAPS 추출: {N}개 샘플")
    print(f"  피처 차원 확인 중...")

    # 첫 샘플로 차원 확인
    wav0 = load_audio(samples[0]['video_path']).numpy()
    feat0 = extract_one(wav0, smile)
    n_feat = feat0.shape[0]
    print(f"  eGeMAPS 차원: {n_feat}")

    # 피처명 저장
    feat_names = smile.process_signal(wav0, SR).columns.tolist()
    np.save(NAMES_PATH, np.array(feat_names))

    all_feats = np.zeros((N, n_feat), dtype=np.float32)
    t0 = time.time()
    for i, s in enumerate(samples):
        wav = load_audio(s['video_path']).numpy()
        all_feats[i] = extract_one(wav, smile)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N - i - 1)
            print(f"  {i+1}/{N}  ({elapsed:.0f}s, ETA {eta:.0f}s)")

    np.save(SAVE_PATH, all_feats)
    print(f"\n✅ 저장: {SAVE_PATH}  shape={all_feats.shape}")
    print(f"   NaN 수: {np.isnan(all_feats).sum()}")

    # 간단한 통계
    print(f"\n  mean abs: {np.abs(all_feats).mean():.4f}")
    print(f"  std:      {all_feats.std():.4f}")


if __name__ == '__main__':
    main()
