"""
Architecture ① — Raw 데이터 로더 (StressID 영상 / 음성 / 생리신호)

⚠️  이 모듈은 models_e2e.py 와 함께 사용하며, 기존 dataset.py (pre-extracted features)
    와 완전히 독립적입니다.

StressID 데이터 구조 (예상)
────────────────────────────
  StressID Dataset/
  ├── Videos/
  │   └── {subject_id}/          예: k67g/
  │       └── {task}.mp4         예: Speaking.mp4
  └── Physiological/             (별도 문의 필요)
      └── {subject_id}/
          └── {task}.csv         ECG, EDA, Resp 컬럼 포함

실행 전 필요 패키지
────────────────────
  pip install torchvision torchaudio transformers opencv-python

TODO 항목
─────────
  [ ] StressID 실제 디렉토리 구조 확인 후 경로 상수 수정
  [ ] 생리신호 파일 포맷(.csv / .bdf 등) 확인 후 load_physio_ts() 구현
  [ ] 영상 프레임 추출 fps / 해상도 확인 후 VideoTransform 수정
  [ ] 음성 샘플링 레이트 확인 (Wav2Vec2 는 16 kHz 필요)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

# ─── 선택적 임포트 ────────────────────────────────────────────────────────────
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except ImportError:
    _HAS_TORCHAUDIO = False

try:
    import torchvision.transforms as T
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False

# ─── 경로 상수 ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent.parent.parent  # AI_Project/
VIDEOS_DIR  = BASE_DIR / "StressID Dataset" / "Videos"
PHYSIO_DIR  = BASE_DIR / "StressID Dataset" / "Physiological"

# ─── Task 매핑 (11개 task) ────────────────────────────────────────────────────
TASK_TYPES = ['Breathing', 'Counting1', 'Counting2', 'Counting3', 'Math',
              'Reading', 'Relax', 'Speaking', 'Stroop', 'Video1', 'Video2']
TASK_TO_ID = {t: i for i, t in enumerate(TASK_TYPES)}
NUM_TASKS  = len(TASK_TYPES)

# ─── 처리 상수 ────────────────────────────────────────────────────────────────
N_FRAMES       = 16      # 과제당 샘플링할 프레임 수
FRAME_SIZE     = 224     # ResNet 입력 크기
AUDIO_SR       = 16_000  # Wav2Vec2 요구 샘플링 레이트 (16 kHz)
MAX_AUDIO_LEN  = AUDIO_SR * 10   # 데모를 위해 10초로 단축 (원래 120초)
PHYSIO_LEN     = 1_000   # 생리신호 시계열 길이 (샘플 수) — TODO: 실제 값 확인


# ══════════════════════════════════════════════════════════════════════════════
#  헬퍼 함수
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames(video_path: Path, n_frames: int = N_FRAMES) -> torch.Tensor:
    """
    동영상에서 균일 간격으로 n_frames 개 프레임을 추출합니다.

    반환: (n_frames, 3, FRAME_SIZE, FRAME_SIZE)  float32, 정규화 완료
    """
    if not _HAS_CV2:
        raise ImportError("pip install opencv-python 필요")
    if not _HAS_TORCHVISION:
        raise ImportError("pip install torchvision 필요")

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames, dtype=int)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((FRAME_SIZE, FRAME_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std= [0.229, 0.224, 0.225]),
    ])

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(frame))
    cap.release()

    return torch.stack(frames)  # (T, 3, H, W)


def load_audio(video_path: Path,
               target_sr: int = AUDIO_SR,
               max_len: int = MAX_AUDIO_LEN) -> torch.Tensor:
    """
    av (PyAV) 라이브러리를 사용하여 동영상에서 음성을 추출합니다.
    """
    try:
        import av
        container = av.open(str(video_path))
        audio_stream = next(s for s in container.streams if s.type == 'audio')
        
        # Resampler 설정
        resampler = av.AudioResampler(
            format='fltp', # float planar
            layout='mono',
            rate=target_sr
        )
        
        all_frames = []
        for frame in container.decode(audio_stream):
            resampled_frames = resampler.resample(frame)
            for rf in resampled_frames:
                all_frames.append(rf.to_ndarray())
        
        if not all_frames:
            return torch.zeros(max_len)
            
        waveform = np.concatenate(all_frames, axis=1).reshape(-1)
        waveform = torch.from_numpy(waveform).float()
        
        # 길이 맞추기
        if waveform.shape[0] > max_len:
            waveform = waveform[:max_len]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, max_len - waveform.shape[0]))
            
        container.close()
        return waveform
    except Exception as e:
        # print(f"Audio loading error for {video_path}: {e}")
        return torch.zeros(max_len)


def load_physio_ts(physio_path: Path, length: int = PHYSIO_LEN) -> torch.Tensor:
    """
    StressID 생리신호 파일(.txt, CSV format)을 로드합니다.
    컬럼: ECG, EDA, RR (Respiration)
    """
    if not physio_path.exists():
        return torch.zeros(3, length)
    
    try:
        import pandas as pd
        df = pd.read_csv(physio_path)
        # 필요한 컬럼만 추출
        cols = ['ECG', 'EDA', 'RR']
        if all(c in df.columns for c in cols):
            data = df[cols].values.astype(np.float32) # (T, 3)
        else:
            # 컬럼명이 다를 경우 첫 3개 컬럼 사용
            data = df.iloc[:, :3].values.astype(np.float32)
            
        ts = torch.tensor(data).T # (3, T)
        
        # 길이 맞추기 (Padding or Truncate)
        if ts.shape[1] > length:
            ts = ts[:, :length]
        else:
            ts = torch.nn.functional.pad(ts, (0, length - ts.shape[1]))
        return ts
    except Exception as e:
        # print(f"Error loading physio {physio_path}: {e}")
        return torch.zeros(3, length)


# ══════════════════════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════

class StressIDRawDataset(Dataset):
    """
    Architecture ① 용 raw 데이터셋.

    samples: list of dict  {
        'subject':    str,
        'task':       str,
        'video_path': Path,
        'label':      int,
    }

    반환 (teacher 모드):
        frames      : (T, 3, 224, 224)
        audio       : (max_len,)
        physio      : (3, physio_len)
        label       : int

    반환 (student 모드):
        frames      : (T, 3, 224, 224)
        audio       : (max_len,)
        label       : int
    """
    CACHE_DIR = BASE_DIR / 'StressID Dataset' / '.raw_cache'

    def __init__(self, samples: list, mode: str = 'teacher', with_task: bool = False):
        assert mode in ('teacher', 'student')
        self.samples   = samples
        self.mode      = mode
        self.with_task = with_task
        self.CACHE_DIR.mkdir(exist_ok=True)

    def _get_task_id(self, idx: int) -> torch.Tensor:
        task_full = self.samples[idx]['task']
        task_type = task_full.split('_', 1)[1] if '_' in task_full else task_full
        return torch.tensor(TASK_TO_ID.get(task_type, 0), dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def _cache_path(self, idx):
        return self.CACHE_DIR / f"{idx}_{self.mode}.pt"

    def __getitem__(self, idx):
        cache = self._cache_path(idx)
        if cache.exists():
            data = torch.load(cache, weights_only=True)
        else:
            s     = self.samples[idx]
            path  = s['video_path']
            label = torch.tensor(s['label'], dtype=torch.long)
            frames = extract_frames(path)
            audio  = load_audio(path)
            if self.mode == 'teacher':
                physio_path = PHYSIO_DIR / s['subject'] / f"{s['task']}.txt"
                physio = load_physio_ts(physio_path)
                data = (frames, audio, physio, label)
            else:
                data = (frames, audio, label)
            torch.save(data, cache)

        if self.with_task:
            return (*data, self._get_task_id(idx))
        return data


# ══════════════════════════════════════════════════════════════════════════════
#  샘플 목록 생성 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def build_sample_list(labels_dict: dict) -> list:
    """
    VIDEOS_DIR 를 순회하며 샘플 목록을 생성합니다.
    실제 파일이 존재하고 labels_dict 에 있는 샘플만 포함합니다.
    """
    samples = []
    if not VIDEOS_DIR.exists():
        print(f"Directory {VIDEOS_DIR} not found.")
        return samples

    for subj_dir in sorted(VIDEOS_DIR.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith('.'):
            continue
        subject = subj_dir.name
        # .mp4 파일만 찾되, AppleDouble ._ 파일은 제외
        for vf in sorted(subj_dir.glob("*.mp4")):
            if vf.name.startswith('._'):
                continue
            task = vf.stem
            key  = task  # 이미 subject_task 형태임 (예: 2ea4_Baseline)
            if key in labels_dict:
                samples.append({
                    'subject':    subject,
                    'task':       task, # 사실상 2ea4_Baseline 전체가 들어감
                    'video_path': vf,
                    'label':      labels_dict[key],
                })
    print(f"[Dataset E2E] Total valid samples found: {len(samples)}")
    return samples
