import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_ai   = _root / 'ai'
for p in [_root, _ai, _ai/'models', _ai/'data', _ai/'train']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models_e2e import StudentCMABi, CBMWithResidual, GEO_INPUT_DIM, EGEMAP_DIM, FUSED_EXP_DIM
from dataset_e2e import TASK_TO_ID
import config as _cfg

_WEIGHTS_DIR = _root / 'weights'   # presense/weights/ — 추론용 가중치 폴더

CONCEPT_NAMES = ['목소리 거칠기', '음색 불안정성', '음성 두께', '음성 피치',
                 '표정 변화량',   '입술 긴장도',   '눈썹 찡그림']
N_CONCEPTS  = 7
FUSED_DIM   = FUSED_EXP_DIM   # 320

# MediaPipe landmark indices
_L_EYE  = [33,  160, 158, 133, 153, 144]
_R_EYE  = [362, 385, 387, 263, 380, 373]
_L_EYE_INNER = 133;  _L_EYE_OUTER = 33
_R_EYE_INNER = 362;  _R_EYE_OUTER = 263
_L_BROW_INNER = 55;  _R_BROW_INNER = 285
_L_TEMPLE = 234;     _R_TEMPLE = 454
_LIP_TOP  = 13;      _LIP_BOT  = 14
_MOUTH_L  = 61;      _MOUTH_R  = 291
_L_IRIS   = 468;     _R_IRIS   = 473
_AU_NAMES = [
    'browDownLeft', 'browDownRight', 'browInnerUp',
    'cheekSquintLeft', 'cheekSquintRight',
    'eyeSquintLeft', 'eyeSquintRight',
    'jawOpen', 'mouthPressLeft', 'mouthPressRight',
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── 단일 추론 모델 ─────────────────────────────────────────────────────────────

class PreSenseModel(nn.Module):
    """
    StudentCMABi + CBMWithResidual — 추론용 단일 모듈.

    V2 변경점:
      - GeometryGRUAttn (TemporalAttentionPool) → 스트레스 피크 구간 포착
      - eGeMAPS late fusion (88→64) → 음성 피치/Jitter/Shimmer 명시적 반영
    """

    def __init__(self, backbone_state, cbm_state, residual_w=0.8):
        super().__init__()
        self.backbone = StudentCMABi(dropout=0.0)
        self.backbone.load_state_dict(backbone_state)

        self.cbm = CBMWithResidual(residual_w=residual_w)
        self.cbm.load_state_dict(cbm_state)

    def forward(self, ff, ae, geo, ti, ege=None):
        """
        ff  : (1, 16, 512)  ResNet18 프레임 피처
        ae  : (1, 256)      Wav2Vec2 임베딩
        geo : (1, 16, 24)   MediaPipe geometry (z-scored)
        ti  : (1,)          task id
        ege : (1, 88)       eGeMAPS v02 (z-scored), Optional

        Returns:
            final_logits : (1, 2)
            c_pred       : (1, 7)  개념 예측값
            attn_w       : (1, 16) 프레임별 attention (시각화용)
        """
        logits, _, _, _, attn_w, fused_exp = self.backbone(
            ff, ae, geo, ti, ege_feats=ege, return_fused=True)
        final_logits, c_pred = self.cbm(fused_exp, logits)
        return final_logits, c_pred, attn_w


# ── 추론 엔진 ──────────────────────────────────────────────────────────────────

class StressInferenceEngine:
    """
    PreSenseModel V2 + 실시간 피처 추출기.

    predict(video_np, audio_np) →
        stress_prob, gate_w, dominant, audio_feat, video_feat, concept_contrib
    """

    def __init__(self, task_name='Speaking', device=None):
        self.device = device or (
            'mps'  if torch.backends.mps.is_available()  else
            'cuda' if torch.cuda.is_available()           else 'cpu')
        print(f"Using device: {self.device}")

        self._load_model()
        self._load_feature_extractors()
        self._init_mediapipe()
        self._load_geo_scaler()
        self._load_ege_scaler()

        self.task_id = self._get_task_id(task_name)
        self.session_log: list[dict] = []
        self._session_start = time.time()

    # ── 모델 로드 ──────────────────────────────────────────────────────────────

    def _load_model(self):
        backbone_path = _WEIGHTS_DIR / 'student_kd_best.pt'
        cbm_path      = _WEIGHTS_DIR / 'cbm_bi_residual_v2_w0.8_lc4.0_best.pt'

        if not backbone_path.exists():
            print(f"⛔ backbone 없음: {backbone_path}"); self.model = None; return
        if not cbm_path.exists():
            print(f"⛔ CBM 없음: {cbm_path}"); self.model = None; return

        backbone_state = torch.load(backbone_path, map_location='cpu')['model_state_dict']
        cbm_ckpt       = torch.load(cbm_path,      map_location='cpu')
        cbm_state      = cbm_ckpt['cbm_state_dict']
        residual_w         = cbm_ckpt.get('residual_w', 0.8)

        self.model = PreSenseModel(backbone_state, cbm_state, residual_w).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        acc = cbm_ckpt.get('val_acc', 0)
        r2  = cbm_ckpt.get('r2', 0)
        print(f"✅ PreSenseModel V2 로드  residual_w={residual_w}  val_acc={acc:.4f}  R²={r2:.4f}")

    # ── 피처 추출기 ────────────────────────────────────────────────────────────

    def _load_feature_extractors(self):
        """ResNet18 (프레임 피처) + Wav2Vec2 + 프로젝션 (오디오 임베딩)."""

        # ResNet18 — ImageNet 사전학습, avgpool 출력 (512-dim)
        try:
            import torchvision.models as tv
            resnet = tv.resnet18(weights='IMAGENET1K_V1')
            self.resnet = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
            self.resnet.eval()
            for p in self.resnet.parameters():
                p.requires_grad = False
            print("✅ ResNet18 로드")
        except Exception as e:
            print(f"⚠️  ResNet18 로드 실패: {e}"); self.resnet = None

        # Wav2Vec2 base + Linear(768→256) 프로젝션
        try:
            from transformers import Wav2Vec2Model
            self.wav2vec = Wav2Vec2Model.from_pretrained(
                'facebook/wav2vec2-base').to(self.device)
            self.wav2vec.eval()
            for p in self.wav2vec.parameters():
                p.requires_grad = False

            self.audio_proj = nn.Linear(768, 256).to(self.device)
            for proj_name in [
                'student_kd_best.pt',
                'kd_student_prev_best.pt',
            ]:
                proj_path = _WEIGHTS_DIR / proj_name
                if not proj_path.exists():
                    continue
                ckpt = torch.load(proj_path, map_location='cpu')
                sd   = ckpt.get('model_state_dict', ckpt)
                proj_sd = {
                    k.replace('audio_enc.proj.', ''): v
                    for k, v in sd.items() if k.startswith('audio_enc.proj.')
                }
                if proj_sd:
                    self.audio_proj.load_state_dict(proj_sd)
                    print(f"✅ Wav2Vec2 + proj 로드 ({proj_name})")
                    break
            else:
                print("⚠️  proj 가중치 없음 — 랜덤 초기화")

            self.audio_proj.eval()
            for p in self.audio_proj.parameters():
                p.requires_grad = False
        except Exception as e:
            print(f"⚠️  Wav2Vec2 로드 실패: {e}")
            self.wav2vec = None; self.audio_proj = None

    def _load_geo_scaler(self):
        mean_p = _WEIGHTS_DIR / 'geo_scaler_mean.npy'
        std_p  = _WEIGHTS_DIR / 'geo_scaler_std.npy'
        if mean_p.exists() and std_p.exists():
            self.geo_mean = np.load(mean_p)  # (24,)
            self.geo_std  = np.load(std_p)   # (24,)
            print("✅ Geometry scaler 로드")
        else:
            self.geo_mean = np.zeros(GEO_INPUT_DIM, dtype=np.float32)
            self.geo_std  = np.ones(GEO_INPUT_DIM,  dtype=np.float32)
            print("⚠️  Geometry scaler 없음 — 정규화 스킵")

    def _load_ege_scaler(self):
        """학습 시 사용된 eGeMAPS 통계 로드 (z-score 정규화용)."""
        ege_path = _WEIGHTS_DIR / 'ege_maps_feats.npy'
        if ege_path.exists():
            ege_np = np.load(ege_path).astype(np.float32)
            self._ege_mean = ege_np.mean(0)   # (88,)
            self._ege_std  = ege_np.std(0)    # (88,)
            print("✅ eGeMAPS scaler 로드")
        else:
            self._ege_mean = None
            self._ege_std  = None
            print("⚠️  eGeMAPS scaler 없음 — eGeMAPS 추론 스킵")



    # ── MediaPipe 초기화 ───────────────────────────────────────────────────────

    def _init_mediapipe(self):
        model_path = Path(__file__).parent / 'face_landmarker.task'
        if not model_path.exists():
            print(f"⚠️  face_landmarker.task 없음: {model_path}")
            self.face_mesh = None; self._mp = None; return
        try:
            import mediapipe as mp
            opts = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.3,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self._mp       = mp
            self.face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
            print("✅ MediaPipe FaceLandmarker 초기화 (blendshapes + transform ON)")
        except Exception as e:
            print(f"⚠️  MediaPipe 초기화 실패: {e}")
            self.face_mesh = None; self._mp = None

    def set_task(self, task_name: str):
        self.task_id = self._get_task_id(task_name)

    def _get_task_id(self, task_name: str):
        idx = TASK_TO_ID.get(task_name, 0)
        return torch.tensor([idx], dtype=torch.long).to(self.device)

    # ── 실시간 피처 추출 ───────────────────────────────────────────────────────

    def _extract_frame_features(self, video_np: np.ndarray) -> torch.Tensor:
        """(16, H, W, 3) uint8 BGR → (1, 16, 512) ResNet18 피처."""
        if self.resnet is None:
            return torch.zeros(1, 16, 512, device=self.device)

        frames = []
        for f in video_np:
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            f_res = cv2.resize(f_rgb, (224, 224)).astype(np.float32) / 255.0
            f_norm = (f_res - IMAGENET_MEAN) / IMAGENET_STD
            frames.append(torch.from_numpy(f_norm.transpose(2, 0, 1)))

        frames_t = torch.stack(frames).to(self.device)  # (16, 3, 224, 224)
        with torch.no_grad():
            feat = self.resnet(frames_t).flatten(1)     # (16, 512)
        return feat.unsqueeze(0)                         # (1, 16, 512)

    def _extract_audio_embedding(self, audio_np: np.ndarray) -> torch.Tensor:
        """(160000,) float32 16kHz → (1, 256) Wav2Vec2 임베딩."""
        if self.wav2vec is None:
            return torch.zeros(1, 256, device=self.device)

        audio_t = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            h = self.wav2vec(audio_t).last_hidden_state.mean(dim=1)  # (1, 768)
            ae = self.audio_proj(h)                                    # (1, 256)
        return ae

    def _extract_ege_maps(self, audio_np: np.ndarray) -> torch.Tensor | None:
        """raw waveform → eGeMAPS v02 Functionals (1, 88), z-score 정규화."""
        if self._ege_mean is None:
            return None
        try:
            import opensmile
            if not hasattr(self, '_smile'):
                self._smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.Functionals,
                )
            sr  = 16000
            wav = audio_np.astype(np.float32)
            min_len = int(sr * 0.5)
            if len(wav) < min_len:
                wav = np.pad(wav, (0, min_len - len(wav)))
            feat = self._smile.process_signal(wav, sr)
            ege  = feat.values[0].astype(np.float32)
            ege  = (ege - self._ege_mean) / (self._ege_std + 1e-6)
            return torch.from_numpy(ege).unsqueeze(0).to(self.device)  # (1, 88)
        except Exception:
            return None

    def _extract_geometry(self, video_np: np.ndarray) -> torch.Tensor:
        """(16, H, W, 3) uint8 BGR → (1, 16, 24) geometry (z-scored)."""
        if self.face_mesh is None:
            return torch.zeros(1, 16, GEO_INPUT_DIM, device=self.device)

        mp    = self._mp
        geo   = np.zeros((16, GEO_INPUT_DIM), dtype=np.float32)

        for fi, frame in enumerate(video_np):
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = self.face_mesh.detect(mp_img)
            if not res.face_landmarks:
                continue
            try:
                lm  = res.face_landmarks[0]
                bs  = res.face_blendshapes[0]              if res.face_blendshapes               else None
                tm  = res.facial_transformation_matrixes[0] if res.facial_transformation_matrixes else None
                geo[fi] = self._frame_geom(lm, bs, tm)
            except Exception:
                pass

        geo_scaled = (geo - self.geo_mean) / (self.geo_std + 1e-8)
        return torch.from_numpy(geo_scaled).unsqueeze(0).to(self.device)  # (1, 16, 24)

    @staticmethod
    def _rot_to_euler(mat4x4):
        R = np.array(mat4x4).reshape(4, 4)[:3, :3]
        pitch = float(np.arcsin(np.clip(-R[2, 0], -1, 1)))
        yaw   = float(np.arctan2(R[1, 0], R[0, 0]))
        roll  = float(np.arctan2(R[2, 1], R[2, 2]))
        return yaw, pitch, roll

    def _frame_geom(self, lm, blendshapes, transform_mat) -> np.ndarray:
        """단일 프레임 → (24,) geometry 벡터 (extract_geometry_feats.py 와 동일 로직)."""
        feat = np.zeros(GEO_INPUT_DIM, dtype=np.float32)
        idx  = 0

        face_w = abs(lm[_R_TEMPLE].x - lm[_L_TEMPLE].x) + 1e-6
        face_h = abs(lm[152].y - lm[10].y) + 1e-6

        # EAR L/R
        def ear(indices):
            p = np.array([[lm[i].x, lm[i].y] for i in indices])
            return (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / \
                   (2 * np.linalg.norm(p[0]-p[3]) + 1e-6)
        feat[idx]   = ear(_L_EYE)
        feat[idx+1] = ear(_R_EYE)
        idx += 2

        # 눈썹 inner_gap, brow_raise L/R
        inner_gap    = abs(lm[_R_BROW_INNER].x - lm[_L_BROW_INNER].x) / face_w
        l_eye_top    = min(lm[i].y for i in _L_EYE)
        r_eye_top    = min(lm[i].y for i in _R_EYE)
        feat[idx]    = inner_gap
        feat[idx+1]  = (l_eye_top - lm[_L_BROW_INNER].y) / face_h
        feat[idx+2]  = (r_eye_top - lm[_R_BROW_INNER].y) / face_h
        idx += 3

        # 입술 H/W, 입 열림
        lip_h       = abs(lm[_LIP_BOT].y - lm[_LIP_TOP].y)
        mouth_w     = abs(lm[_MOUTH_R].x - lm[_MOUTH_L].x) + 1e-6
        feat[idx]   = lip_h / mouth_w
        feat[idx+1] = abs(lm[17].y - lm[0].y) / face_h
        idx += 2

        # 홍채 편차 (iris landmarks 있을 때만)
        if len(lm) > 473:
            for iris_i, inner_i, outer_i in [
                    (_L_IRIS, _L_EYE_INNER, _L_EYE_OUTER),
                    (_R_IRIS, _R_EYE_INNER, _R_EYE_OUTER)]:
                cx  = (lm[inner_i].x + lm[outer_i].x) / 2
                cy  = (lm[inner_i].y + lm[outer_i].y) / 2
                ew  = abs(lm[inner_i].x - lm[outer_i].x) + 1e-6
                feat[idx]   = (lm[iris_i].x - cx) / ew
                feat[idx+1] = (lm[iris_i].y - cy) / ew
                idx += 2
        else:
            idx += 4

        # Head pose
        if transform_mat is not None:
            try:
                yaw, pitch, roll = self._rot_to_euler(transform_mat.data)
                feat[idx], feat[idx+1], feat[idx+2] = yaw, pitch, roll
            except Exception:
                pass
        idx += 3

        # AU proxy (blendshapes)
        if blendshapes is not None:
            bs_map = {b.category_name: b.score for b in blendshapes}
            for j, name in enumerate(_AU_NAMES):
                feat[idx + j] = float(bs_map.get(name, 0.0))
        idx += len(_AU_NAMES)

        return feat

    # ── 추론 ──────────────────────────────────────────────────────────────────

    def predict(self, video_np, audio_np):
        """
        video_np : (16, H, W, 3) uint8 BGR
        audio_np : (160000,) float32  16kHz

        Returns:
            stress_prob      : float  0~1
            gate_w           : [video_w, audio_w]
            dominant         : str
            audio_features   : dict
            video_features   : dict
            concept_contrib  : dict {개념명: 예측값}
        """
        if self.model is None:
            return 0.5, [0.5, 0.5], '알 수 없음', {}, {}, {}

        ff  = self._extract_frame_features(video_np)   # (1, 16, 512)
        ae  = self._extract_audio_embedding(audio_np)  # (1, 256)
        geo = self._extract_geometry(video_np)          # (1, 16, 24)
        ege = self._extract_ege_maps(audio_np)          # (1, 88) or None

        with torch.no_grad():
            final_logits, c_pred, attn = self.model(ff, ae, geo, self.task_id, ege=ege)

        stress_prob = F.softmax(final_logits, dim=1)[0][1].item()
        c_pred_np   = c_pred[0].cpu().numpy()  # (7,)

        concept_contrib = {
            name: round(float(v), 3)
            for name, v in zip(CONCEPT_NAMES, c_pred_np)
        }

        # 개념 기여도 → 모달리티 주도 계산
        _AUDIO_CONCEPTS = {'목소리 거칠기', '음색 불안정성', '음성 두께', '음성 피치'}
        _VIDEO_CONCEPTS = {'표정 변화량', '입술 긴장도', '눈썹 찡그림'}
        audio_w = sum(abs(concept_contrib[n]) for n in _AUDIO_CONCEPTS)
        video_w = sum(abs(concept_contrib[n]) for n in _VIDEO_CONCEPTS)
        total   = audio_w + video_w + 1e-8
        gate_w  = [round(video_w / total, 3), round(audio_w / total, 3)]
        if gate_w[0] > 0.6:
            dominant = '영상'
        elif gate_w[1] > 0.6:
            dominant = '음성'
        else:
            dominant = '복합'

        audio_feat = self.analyze_audio(audio_np)
        video_feat = self._analyze_video_from_geo(video_np, geo[0].cpu().numpy())

        self.session_log.append({
            'timestamp':       time.time() - self._session_start,
            'stress_prob':     stress_prob,
            'gate_w':          gate_w,
            'dominant':        dominant,
            'concept_contrib': concept_contrib.copy(),
            'audio_feat':      audio_feat.copy() if audio_feat else {},
            'video_feat':      video_feat.copy() if video_feat else {},
            'frame':           video_np[-1].copy(),
            'attn_weights':    attn[0].cpu().numpy().tolist() if attn is not None else [],
        })

        return stress_prob, gate_w, dominant, audio_feat, video_feat, concept_contrib

    # ── 음성 분석 ──────────────────────────────────────────────────────────────

    def analyze_audio(self, audio_np: np.ndarray) -> dict:
        """raw waveform → 피드백용 음성 지표."""
        try:
            import librosa
            sr = 16000
            audio_f32 = audio_np.astype(np.float32)

            f0 = librosa.yin(audio_f32,
                             fmin=librosa.note_to_hz('C2'),
                             fmax=librosa.note_to_hz('C6'),
                             sr=sr)
            hop = len(audio_f32) // len(f0)
            rms = np.array([
                np.sqrt(np.mean(audio_f32[i*hop:(i+1)*hop]**2))
                for i in range(len(f0))])
            voiced_mask = rms > max(rms.mean() * 0.3, 1e-4)
            speech_mask = voiced_mask & (f0 >= 80.0) & (f0 <= 500.0)
            voiced_f0   = f0[speech_mask]
            f0_mean     = float(np.mean(voiced_f0))  if len(voiced_f0) > 0 else 0.0
            f0_std      = float(np.std(voiced_f0))   if len(voiced_f0) > 0 else 0.0
            voiced_ratio= float(voiced_mask.mean())
        except Exception:
            f0_mean = f0_std = 0.0; voiced_ratio = 0.5

        try:
            win   = 16000
            n     = max(len(audio_np) // win, 1)
            rms_s = np.array([
                np.sqrt(np.mean(audio_np[i*win:(i+1)*win]**2))
                for i in range(n)])
            energy_variability = float(rms_s.std() / (rms_s.mean() + 1e-8))
        except Exception:
            energy_variability = 0.0

        return {
            'f0_mean':            round(f0_mean, 1),
            'f0_std':             round(f0_std, 1),
            'voiced_ratio':       round(voiced_ratio, 3),
            'energy_variability': round(energy_variability, 3),
            'pitch_level':    '높음' if f0_mean > 200 else '낮음' if f0_mean < 120 else '보통',
            'pitch_stability':'불안정' if f0_std > 40 else '안정',
            'speech_pace':    '빠름' if voiced_ratio > 0.75 else '느림' if voiced_ratio < 0.45 else '보통',
        }

    # ── 영상 분석 (이미 추출된 geo 활용) ──────────────────────────────────────

    def _analyze_video_from_geo(self, video_np: np.ndarray,
                                geo_scaled: np.ndarray) -> dict:
        """
        이미 추출한 geometry (16, 24) 에서 피드백용 지표 재활용.
        geometry는 z-scored이므로 역변환 후 사용.
        """
        geo_raw = geo_scaled * self.geo_std + self.geo_mean  # (16, 24) 원래 스케일

        # 움직임 (raw 프레임)
        diff   = np.abs(video_np[1:].astype(np.float32) - video_np[:-1].astype(np.float32))
        motion = float(diff.mean() / 255.0)

        result = {
            'motion':       round(motion, 4),
            'motion_level': '많음' if motion > 0.05 else '약간' if motion > 0.02 else '적음',
            'face_detected': bool((geo_raw.sum(axis=1) != 0).any()),
        }

        # 유효 프레임만
        valid = geo_raw[geo_raw.sum(axis=1) != 0]  # (n_valid, 24)
        if len(valid) == 0:
            return result

        # EAR → 눈 열림
        ear_mean = float(valid[:, :2].mean())
        result['mean_ear']    = round(ear_mean, 3)
        result['blink_count'] = int((valid[:, 0] < 0.15).sum() + (valid[:, 1] < 0.15).sum()) // 2
        result['blink_level'] = '많음' if result['blink_count'] >= 5 else \
                                '보통' if result['blink_count'] >= 2 else '적음'

        # 홍채 편차 (idx 7~10)
        iris_devs = valid[:, 7:11]
        gaze_dev  = float(np.abs(iris_devs).mean())
        gaze_stab = round(max(0.0, 1 - gaze_dev * 2), 2)
        result['gaze_stability']    = gaze_stab
        result['eye_contact_score'] = gaze_stab
        result['eye_contact_level'] = (
            '좋음' if gaze_dev < 0.15 else '약간 회피' if gaze_dev < 0.25 else '회피 중')

        # Head pose — yaw/pitch/roll (idx 11~13)
        if valid.shape[1] > 13:
            roll_arr = valid[:, 13]
            head_movement = float(np.std(np.diff(roll_arr))) if len(roll_arr) >= 2 else 0.0
            result['head_movement']       = round(head_movement, 4)
            result['head_movement_level'] = (
                '많음' if head_movement > 0.1 else '약간' if head_movement > 0.04 else '안정')

        # 입술 H/W (idx 5) → 긴장도
        lip_hw = float(valid[:, 5].mean())
        result['lip_tension']       = round(lip_hw, 3)
        result['mouth_level']       = '긴장' if lip_hw < 0.15 else '보통' if lip_hw < 0.35 else '이완'

        # AU: browDownLeft/Right (idx 14,15) → 눈썹 찡그림
        if valid.shape[1] > 15:
            brow_down = float(valid[:, 14:16].mean())
            result['brow_tension']  = round(brow_down, 3)
            result['brow_level']    = '높음' if brow_down > 0.3 else '보통' if brow_down > 0.1 else '낮음'

        return result
