"""
PreSense - Dataset
StressID 피처 로딩, 전처리, 레이블 생성, Subject 기반 Train/Val/Test 분할
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader

import sys
# Reorganized: presense root and subfolders added to path
from pathlib import Path
_root = Path(__file__).resolve().parent.parent.parent
_ai_root = _root / 'ai'
for p in [_root, _ai_root, _ai_root/'data', _ai_root/'models', _ai_root/'train', _ai_root/'evaluate', _ai_root/'feature_analysis']:
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import config


# ─── 피처 로딩 ────────────────────────────────────────────────────────────────

def load_physiological_features() -> pd.DataFrame:
    """생리 신호 피처 로드 (ECG, EDA, Respiration → 132개 통계 피처)"""
    df = pd.read_csv(config.PHYSIO_FEATURES, header=0, index_col=0)
    print(f"[Physio]  shape: {df.shape}")
    return df


def load_audio_features() -> pd.DataFrame:
    """음성 Handcrafted 피처 로드 (MFCC 등 → 140개 피처)"""
    df = pd.read_csv(config.AUDIO_FEATURES, header=None, index_col=0)
    # 파일명 확장자 제거 (e.g. "2ea4_Counting1.wav" → "2ea4_Counting1")
    df.index = [str(i).split('.')[0] for i in df.index]
    df.columns = range(1, len(df.columns) + 1)
    print(f"[Audio]   shape: {df.shape}")
    return df


def load_video_features() -> pd.DataFrame:
    """비디오 AU + Gaze 피처 로드 (OpenFace → 84개 통계 피처)"""
    df = pd.read_csv(config.VIDEO_FEATURES, header=0, index_col=0)
    print(f"[Video]   shape: {df.shape}")
    return df


def load_labels(task: str = 'binary', fallback: bool = True) -> pd.DataFrame:
    """
    레이블 로드. 두 컬럼을 모두 반환하여 align_and_preprocess가 필요에 따라 선택하게 함.
    """
    if config.LABELS_FILE.exists():
        df = pd.read_csv(config.LABELS_FILE, header=0)

        # labels.csv 포맷 처리
        if 'subject/task' in df.columns:
            df = df.set_index('subject/task')
        elif df.index.name == 'subject/task':
            pass
        else:
            df = df.set_index(df.columns[0])

        df = df[['binary-stress', 'affect3-class']].dropna()
        print(f"[Labels]  shape: {df.shape}  (from labels.csv)")
        return df

    if not fallback:
        raise FileNotFoundError(f"labels.csv not found at {config.LABELS_FILE}.")

    # ─── Fallback: 태스크 이름으로 근사 레이블 생성 ──────────────────────────
    print("[Labels]  labels.csv 없음 → 태스크 이름 기반 근사 레이블 생성")
    physio = load_physiological_features()
    indices = list(physio.index)

    binary, three_class = [], []
    for idx in indices:
        task_name = idx.split('_', 1)[1] if '_' in idx else idx
        if task_name in config.STRESS_TASKS:
            binary.append(1)
            three_class.append(2)
        elif task_name in config.RELAX_TASKS:
            binary.append(0)
            three_class.append(0)
        else:
            binary.append(0)
            three_class.append(1)

    df = pd.DataFrame({
        'binary-stress':  binary,
        'affect3-class':  three_class,
    }, index=indices)
    df.index.name = 'subject/task'
    return df


# ─── 전처리 & 정렬 ────────────────────────────────────────────────────────────

def align_and_preprocess(x_phys: pd.DataFrame,
                          x_video: pd.DataFrame,
                          x_audio: pd.DataFrame,
                          labels: pd.DataFrame):
    """
    physio+video+labels 교집합(~599개)으로 정렬합니다.
    audio가 없는 샘플(이완 과제 등)은 0벡터로 채우고 has_audio=False로 표시합니다.
    StandardScaler는 split_by_subject() 내에서 train 데이터에만 fit합니다.

    Returns
    -------
    X_phys, X_video, X_audio  : np.ndarray  (N, D)  — imputed, unscaled
    has_audio                  : np.ndarray  (N,) bool
    y_binary, y_3class         : np.ndarray  (N,)
    common_idx                 : list of str
    subjects                   : list of str
    """
    # physio+video+labels 교집합 (audio 제외) → 이완 과제 포함
    common = set(x_phys.index) & set(x_video.index) & set(labels.index)
    common_idx = sorted(common)
    print(f"[Align]   전체 샘플: {len(common_idx)}  "
          f"(audio 있음: {sum(idx in x_audio.index for idx in common_idx)})")

    x_phys  = x_phys.loc[common_idx]
    x_video = x_video.loc[common_idx]
    labels  = labels.loc[common_idx]

    # median imputation: 모달리티마다 피처 분포가 달라 각각 독립 fit
    X_phys  = SimpleImputer(strategy='median').fit_transform(x_phys.values.astype(float))
    X_video = SimpleImputer(strategy='median').fit_transform(x_video.values.astype(float))

    # audio: 있는 샘플만 impute 후 채우고, 없는 샘플은 0벡터 유지
    audio_set = set(x_audio.index)
    has_audio = np.array([idx in audio_set for idx in common_idx], dtype=bool)
    X_audio = np.zeros((len(common_idx), x_audio.shape[1]), dtype=float)
    if has_audio.any():
        valid = [idx for idx in common_idx if idx in audio_set]
        X_audio[has_audio] = SimpleImputer(strategy='median').fit_transform(
            x_audio.loc[valid].values.astype(float)
        )

    y_binary  = labels['binary-stress'].values.astype(int)
    y_3class  = labels['affect3-class'].values.astype(int)
    subjects  = [idx.split('_')[0] for idx in common_idx]

    return X_phys, X_video, X_audio, has_audio, y_binary, y_3class, common_idx, subjects


# ─── Subject 기반 Train / Val / Test 분할 ────────────────────────────────────

def split_by_subject(X_phys, X_video, X_audio, has_audio,
                     y_binary, y_3class, subjects):
    """
    데이터 누수 방지를 위해 subject 단위로 train/val/test를 분할하고
    StandardScaler를 train에만 fit합니다.
    audio는 has_audio=True인 샘플에만 scaler를 적용해 0벡터가 왜곡되지 않게 합니다.

    Returns
    -------
    splits : dict  {
        'train': (phys, video, audio, has_audio, y_bin, y_3c),
        'val':   (...),
        'test':  (...)
    }
    """
    subjects = np.array(subjects)
    idx_all  = np.arange(len(y_binary))

    gss1 = GroupShuffleSplit(n_splits=1,
                              test_size=(config.VAL_RATIO + config.TEST_RATIO),
                              random_state=config.RANDOM_SEED)
    train_idx, valtest_idx = next(gss1.split(idx_all, y_binary, groups=subjects))

    val_frac = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    gss2 = GroupShuffleSplit(n_splits=1,
                              test_size=(1 - val_frac),
                              random_state=config.RANDOM_SEED)
    sub_idx = np.arange(len(valtest_idx))
    val_sub, test_sub = next(gss2.split(sub_idx, y_binary[valtest_idx],
                                         groups=subjects[valtest_idx]))
    val_idx  = valtest_idx[val_sub]
    test_idx = valtest_idx[test_sub]

    def _scale(train, others):
        sc = StandardScaler()
        return (sc.fit_transform(train),) + tuple(sc.transform(o) for o in others)

    def _scale_audio(train_idx_, val_idx_, test_idx_):
        # has_audio=True인 샘플에만 scaler 적용, 0벡터는 그대로 유지
        tr, v, te = X_audio[train_idx_].copy(), X_audio[val_idx_].copy(), X_audio[test_idx_].copy()
        ha_tr = has_audio[train_idx_]
        if ha_tr.any():
            sc = StandardScaler().fit(tr[ha_tr])
            tr[ha_tr] = sc.transform(tr[ha_tr])
            ha_v, ha_te = has_audio[val_idx_], has_audio[test_idx_]
            if ha_v.any():  v[ha_v]   = sc.transform(v[ha_v])
            if ha_te.any(): te[ha_te] = sc.transform(te[ha_te])
        return tr, v, te

    X_phys_tr,  X_phys_v,  X_phys_te  = _scale(X_phys[train_idx],
                                                 [X_phys[val_idx],  X_phys[test_idx]])
    X_video_tr, X_video_v, X_video_te = _scale(X_video[train_idx],
                                                 [X_video[val_idx], X_video[test_idx]])
    X_audio_tr, X_audio_v, X_audio_te = _scale_audio(train_idx, val_idx, test_idx)

    splits = {
        'train': (X_phys_tr, X_video_tr, X_audio_tr, has_audio[train_idx],
                  y_binary[train_idx], y_3class[train_idx]),
        'val':   (X_phys_v,  X_video_v,  X_audio_v,  has_audio[val_idx],
                  y_binary[val_idx],   y_3class[val_idx]),
        'test':  (X_phys_te, X_video_te, X_audio_te, has_audio[test_idx],
                  y_binary[test_idx],  y_3class[test_idx]),
    }

    print(f"[Split]   Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print(f"          Train subjects: {len(set(subjects[train_idx]))}  "
          f"Test subjects: {len(set(subjects[test_idx]))}")
    return splits


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────

class StressIDDataset(Dataset):
    """
    Teacher / Student 모두에 사용하는 PyTorch Dataset.

    Parameters
    ----------
    phys      : np.ndarray (N, 132) — Teacher 전용 입력
    video     : np.ndarray (N, 84)
    audio     : np.ndarray (N, 140)  — 없는 샘플은 0벡터
    has_audio : np.ndarray (N,) bool — 실제 audio가 있는 샘플 여부
    labels    : np.ndarray (N,)
    mode      : 'teacher' → (phys, video, audio, has_audio, label)
                'student' → (video, audio, has_audio, label)
    """
    def __init__(self, phys, video, audio, has_audio, labels, mode: str = 'teacher'):
        assert mode in ('teacher', 'student')
        self.phys      = torch.tensor(phys,      dtype=torch.float32)
        self.video     = torch.tensor(video,     dtype=torch.float32)
        self.audio     = torch.tensor(audio,     dtype=torch.float32)
        self.has_audio = torch.tensor(has_audio, dtype=torch.bool)
        self.labels    = torch.tensor(labels,    dtype=torch.long)
        self.mode      = mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.mode == 'teacher':
            return (self.phys[idx], self.video[idx], self.audio[idx],
                    self.has_audio[idx], self.labels[idx])
        else:
            return self.video[idx], self.audio[idx], self.has_audio[idx], self.labels[idx]


def make_dataloaders(splits: dict,
                     task: str = 'binary',
                     mode: str = 'teacher',
                     batch_size: int = config.BATCH_SIZE):
    """
    Parameters
    ----------
    splits : split_by_subject() 결과
    task   : 'binary' | '3class'
    mode   : 'teacher' | 'student'

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    label_col = 4 if task == 'binary' else 5  # (phys,video,audio,has_audio,y_bin,y_3c)

    def _loader(split_name, shuffle):
        data = splits[split_name]
        phys, video, audio, ha = data[0], data[1], data[2], data[3]
        y  = data[label_col]
        ds = StressIDDataset(phys, video, audio, ha, y, mode=mode)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return (
        _loader('train', shuffle=True),
        _loader('val',   shuffle=False),
        _loader('test',  shuffle=False),
    )


# ─── 전체 파이프라인 한 번에 실행 ─────────────────────────────────────────────

def load_all(task: str = 'binary', fallback_labels: bool = True):
    """
    피처 로드 → 정렬/전처리 → 분할 → DataLoader 반환.

    Parameters
    ----------
    task            : 'binary' | '3class'
    fallback_labels : labels.csv 없을 때 근사 레이블 사용 여부

    Returns
    -------
    splits          : dict (train/val/test raw arrays)
    teacher_loaders : (train, val, test) DataLoader  [physio 포함]
    student_loaders : (train, val, test) DataLoader  [physio 미포함]
    """
    x_phys  = load_physiological_features()
    x_video = load_video_features()
    x_audio = load_audio_features()
    labels  = load_labels(task=task, fallback=fallback_labels)

    X_phys, X_video, X_audio, has_audio, y_bin, y_3c, common_idx, subjects = \
        align_and_preprocess(x_phys, x_video, x_audio, labels)

    splits = split_by_subject(X_phys, X_video, X_audio, has_audio, y_bin, y_3c, subjects)

    teacher_loaders = make_dataloaders(splits, task=task, mode='teacher')
    student_loaders = make_dataloaders(splits, task=task, mode='student')

    return splits, teacher_loaders, student_loaders
