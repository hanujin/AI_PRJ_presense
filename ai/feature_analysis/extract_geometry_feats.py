"""
Geometry 파생 지표 추출 (프레임당 ~25dim)

얼굴 크기 정규화된 파생 지표 — 좌표 원본 아님:

  EAR L/R                   (2)  눈 열림 비율
  눈썹 inner gap             (1)  내측 눈썹 간격 / 얼굴폭
  눈썹 raise L/R             (2)  눈썹 높이 - 눈 상단 거리 (상대적 높이)
  입술 H/W 비                (1)  입 높이 / 폭
  입 열림                    (1)  위아래 입술 간격
  홍채 편차 L/R (x, y)       (4)  홍채 중심 - 눈 중심 / 눈 폭
  head yaw / pitch / roll   (3)  facial_transformation_matrixes에서 추출
  AU proxy (blendshape 10개) (10) browDown, browInnerUp, cheekSquint,
                                  eyeSquint, jawOpen, mouthPress 등
  ─────────────────────────────
  합계                       25 dim

정규화:
  - 기하 지표: 얼굴 크기(IOD 또는 face_w/h)로 나눠 스케일 불변
  - head pose: 라디안 그대로 (이미 스케일 동일)
  - blendshape: 0~1 (MediaPipe 원본)
  - dataset-level z-score는 probe 스크립트에서 수행

face miss 프레임: 0 벡터 (probe 시 마스킹)

출력: cbm_geometry_feats.npy  (N, 16, 25)

실행:
  cd presense && python ai/feature_analysis/extract_geometry_feats.py
"""

import sys, time, cv2
import numpy as np
import pandas as pd
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
for p in [_root, _root/'ai', _root/'ai'/'models', _root/'ai'/'data']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config
from dataset_e2e import build_sample_list, N_FRAMES

SAVE_PATH = config.CHECKPOINTS_DIR / 'binary' / 'cbm_geometry_feats.npy'

# ── MediaPipe 랜드마크 인덱스 ──────────────────────────────────────────────────
_L_EYE  = [33,  160, 158, 133, 153, 144]   # EAR
_R_EYE  = [362, 385, 387, 263, 380, 373]   # EAR
_L_EYE_INNER = 133;  _L_EYE_OUTER = 33     # eye corners
_R_EYE_INNER = 362;  _R_EYE_OUTER = 263
_L_BROW_INNER = 55;  _R_BROW_INNER = 285   # inner brow
_L_TEMPLE = 234;     _R_TEMPLE = 454        # face width ref
_LIP_TOP  = 13;      _LIP_BOT  = 14        # inner lip
_MOUTH_L  = 61;      _MOUTH_R  = 291       # mouth width
_L_IRIS   = 468;     _R_IRIS   = 473       # iris (requires model with iris)

# AU proxy: blendshape 이름 목록 (순서가 feature dim 순서)
_AU_NAMES = [
    'browDownLeft',     # 0  눈썹 찡그림 L
    'browDownRight',    # 1  눈썹 찡그림 R
    'browInnerUp',      # 2  내측 눈썹 올림 (걱정)
    'cheekSquintLeft',  # 3  뺨 수축 L
    'cheekSquintRight', # 4  뺨 수축 R
    'eyeSquintLeft',    # 5  눈 가늘게 L
    'eyeSquintRight',   # 6  눈 가늘게 R
    'jawOpen',          # 7  입 벌림
    'mouthPressLeft',   # 8  입술 압박 L
    'mouthPressRight',  # 9  입술 압박 R
]
N_AU   = len(_AU_NAMES)
N_GEOM = 2 + 3 + 2 + 4 + 3 + N_AU   # = 25
FEAT_NAMES = (
    ['ear_L', 'ear_R',
     'brow_inner_gap', 'brow_raise_L', 'brow_raise_R',
     'lip_hw_ratio', 'mouth_open',
     'iris_dev_Lx', 'iris_dev_Ly', 'iris_dev_Rx', 'iris_dev_Ry',
     'head_yaw', 'head_pitch', 'head_roll']
    + _AU_NAMES
)
assert len(FEAT_NAMES) == N_GEOM, f"{len(FEAT_NAMES)} ≠ {N_GEOM}"


def _ear(lm, indices):
    """Eye Aspect Ratio (6 landmarks)."""
    p = np.array([[lm[i].x, lm[i].y] for i in indices])
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h  = np.linalg.norm(p[0] - p[3]) + 1e-6
    return (v1 + v2) / (2 * h)


def _rot_to_euler(mat4x4):
    """4×4 facial_transformation_matrix → yaw, pitch, roll (radians)."""
    R = np.array(mat4x4).reshape(4, 4)[:3, :3]
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1, 1)))
    yaw   = float(np.arctan2(R[1, 0], R[0, 0]))
    roll  = float(np.arctan2(R[2, 1], R[2, 2]))
    return yaw, pitch, roll


def _extract_frame_geom(lm, blendshapes, transform_mat) -> np.ndarray:
    """
    단일 프레임 → (25,) geometry 벡터.
    lm: face_landmarks[0]
    blendshapes: face_blendshapes[0] 또는 None
    transform_mat: facial_transformation_matrixes[0] 또는 None
    """
    feat = np.zeros(N_GEOM, dtype=np.float32)
    idx  = 0

    # ── 얼굴 참조 치수 ──────────────────────────────────────────────────────
    face_w = abs(lm[_R_TEMPLE].x - lm[_L_TEMPLE].x) + 1e-6
    face_h = abs(lm[152].y       - lm[10].y)         + 1e-6  # chin - forehead
    iod    = abs(lm[_R_EYE_INNER].x - lm[_L_EYE_INNER].x) + 1e-6  # inter-ocular

    # ── EAR ──────────────────────────────────────────────────────────────
    feat[idx]   = _ear(lm, _L_EYE)
    feat[idx+1] = _ear(lm, _R_EYE)
    idx += 2

    # ── 눈썹 inner_gap, brow_raise L/R ──────────────────────────────────
    inner_gap = abs(lm[_R_BROW_INNER].x - lm[_L_BROW_INNER].x) / face_w
    # brow_raise: 눈썹 y - 눈 상단 y  (양수=눈썹이 눈보다 위)
    l_eye_top  = min(lm[i].y for i in _L_EYE)
    r_eye_top  = min(lm[i].y for i in _R_EYE)
    l_brow_y   = lm[_L_BROW_INNER].y
    r_brow_y   = lm[_R_BROW_INNER].y
    brow_raise_L = (l_eye_top - l_brow_y) / face_h   # 높을수록 눈썹 위로
    brow_raise_R = (r_eye_top - r_brow_y) / face_h
    feat[idx]   = inner_gap
    feat[idx+1] = brow_raise_L
    feat[idx+2] = brow_raise_R
    idx += 3

    # ── 입술 H/W 비, 입 열림 ───────────────────────────────────────────
    lip_h     = abs(lm[_LIP_BOT].y   - lm[_LIP_TOP].y)
    mouth_w   = abs(lm[_MOUTH_R].x   - lm[_MOUTH_L].x) + 1e-6
    mouth_open= abs(lm[17].y          - lm[0].y)           # 아랫입술 - 윗입술
    feat[idx]   = lip_h / mouth_w
    feat[idx+1] = mouth_open / face_h
    idx += 2

    # ── 홍채 편차 ─────────────────────────────────────────────────────
    has_iris = len(lm) > 473
    if has_iris:
        l_eye_cx = (lm[_L_EYE_INNER].x + lm[_L_EYE_OUTER].x) / 2
        l_eye_cy = (lm[_L_EYE_INNER].y + lm[_L_EYE_OUTER].y) / 2
        l_eye_w  = abs(lm[_L_EYE_INNER].x - lm[_L_EYE_OUTER].x) + 1e-6
        r_eye_cx = (lm[_R_EYE_INNER].x + lm[_R_EYE_OUTER].x) / 2
        r_eye_cy = (lm[_R_EYE_INNER].y + lm[_R_EYE_OUTER].y) / 2
        r_eye_w  = abs(lm[_R_EYE_INNER].x - lm[_R_EYE_OUTER].x) + 1e-6
        feat[idx]   = (lm[_L_IRIS].x - l_eye_cx) / l_eye_w
        feat[idx+1] = (lm[_L_IRIS].y - l_eye_cy) / l_eye_w
        feat[idx+2] = (lm[_R_IRIS].x - r_eye_cx) / r_eye_w
        feat[idx+3] = (lm[_R_IRIS].y - r_eye_cy) / r_eye_w
    idx += 4

    # ── Head pose ─────────────────────────────────────────────────────
    if transform_mat is not None:
        try:
            yaw, pitch, roll = _rot_to_euler(transform_mat.data)
            feat[idx]   = yaw
            feat[idx+1] = pitch
            feat[idx+2] = roll
        except Exception:
            pass
    idx += 3

    # ── AU proxy (blendshapes) ────────────────────────────────────────
    if blendshapes is not None:
        bs_map = {b.category_name: b.score for b in blendshapes}
        for j, name in enumerate(_AU_NAMES):
            feat[idx + j] = float(bs_map.get(name, 0.0))
    idx += N_AU

    assert idx == N_GEOM
    return feat


def extract_video_geometry(video_path: str, face_landmarker) -> np.ndarray:
    """영상 → (N_FRAMES, N_GEOM) geometry 행렬. face miss → 0 벡터."""
    import mediapipe as mp_module

    cap        = cv2.VideoCapture(str(video_path))
    total      = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    sample_idx = np.linspace(0, total - 1, min(N_FRAMES, total), dtype=int)
    result     = np.zeros((N_FRAMES, N_GEOM), dtype=np.float32)

    for fi, fidx in enumerate(sample_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)
        res    = face_landmarker.detect(mp_img)
        if not res.face_landmarks:
            continue
        try:
            lm  = res.face_landmarks[0]
            bs  = res.face_blendshapes[0]       if res.face_blendshapes        else None
            tm  = res.facial_transformation_matrixes[0] \
                  if res.facial_transformation_matrixes else None
            result[fi] = _extract_frame_geom(lm, bs, tm)
        except Exception:
            pass

    cap.release()
    return result


def main():
    labels_df  = pd.read_csv(config.LABELS_FILE)
    labels_map = dict(zip(labels_df.iloc[:, 0], labels_df.iloc[:, 1]))
    samples    = build_sample_list(labels_map)
    N          = len(samples)
    labels     = np.array([s['label'] for s in samples])
    print(f"샘플: {N}개  →  출력: ({N}, {N_FRAMES}, {N_GEOM})\n")

    if SAVE_PATH.exists():
        arr = np.load(SAVE_PATH)
        if arr.shape == (N, N_FRAMES, N_GEOM):
            print(f"캐시 이미 존재: {SAVE_PATH.name}  {arr.shape}")
            _print_stats(arr, labels)
            return
        SAVE_PATH.unlink()

    import mediapipe as mp_module
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = _root / 'agent' / 'face_landmarker.task'
    if not model_path.exists():
        print(f"⛔  face_landmarker.task 없음: {model_path}"); return

    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,   # head pose 활성화
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    print("MediaPipe FaceLandmarker 초기화 완료 (blendshapes + transform ON)\n")

    out = np.zeros((N, N_FRAMES, N_GEOM), dtype=np.float32)
    t0  = time.time()
    err = 0

    for i, s in enumerate(samples):
        try:
            out[i] = extract_video_geometry(s['video_path'], face_landmarker)
        except Exception as e:
            err += 1
            print(f"  오류 [{i}] {s['task']}: {e}")
        if (i + 1) % 50 == 0:
            el  = time.time() - t0
            eta = el / (i + 1) * (N - i - 1)
            det = (out[:i+1].sum(axis=(1, 2)) != 0).sum()
            print(f"  {i+1}/{N}  ({el:.0f}s ETA {eta:.0f}s)  검출={det}/{i+1}")

    face_landmarker.close()
    np.save(SAVE_PATH, out)
    print(f"\n완료 ({time.time()-t0:.0f}s)  오류={err}  →  {SAVE_PATH.name}")
    _print_stats(out, labels)


def _print_stats(arr, labels):
    det = (arr.sum(axis=(1, 2)) != 0).sum()
    print(f"\n  face 검출률: {det}/{len(labels)} ({det/len(labels)*100:.1f}%)")
    flat = arr[arr.sum(axis=(1, 2)) != 0]
    if flat.shape[0] > 10:
        from scipy.stats import pearsonr
        frame_mean = arr.mean(axis=1)   # (N, 25)
        corrs = []
        for d in range(N_GEOM):
            try:
                r, _ = pearsonr(frame_mean[:, d], labels)
                corrs.append(r)
            except Exception:
                corrs.append(0.0)
        top5 = np.argsort(np.abs(corrs))[::-1][:5]
        print(f"\n  상위 5개 geometry 피처 (스트레스 상관관계):")
        for d in top5:
            bar = "█" * int(abs(corrs[d]) * 30)
            print(f"    {FEAT_NAMES[d]:20s}  r={corrs[d]:+.3f}  {bar}")


if __name__ == '__main__':
    main()
