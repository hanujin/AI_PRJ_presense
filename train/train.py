"""
PreSense - 학습 루프

1. train_teacher()  : Teacher 단독 학습
2. train_student_baseline() : KD 없이 Student 단독 학습 (baseline)
3. train_student_kd()       : Knowledge Distillation으로 Student 학습
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

import sys
# Reorganized: presense root and subfolders added to path
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
for p in [_root, _root/'data', _root/'models', _root/'train', _root/'evaluate', _root/'analysis']:
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import config
from models.kd_loss import TotalKDLoss

# 결과/체크포인트 디렉토리 생성
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── 유틸 ─────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():     # Apple Silicon
        return torch.device('mps')
    return torch.device('cpu')


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


class EarlyStopping:
    def __init__(self, patience: int = config.EARLY_STOPPING_PATIENCE,
                 min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float('inf')
        self.triggered = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.triggered = True
        return self.triggered


def save_checkpoint(model, optimizer, epoch, val_loss, path: str):
    torch.save({
        'epoch':               epoch,
        'model_state_dict':    model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss':            val_loss,
    }, path)


# ─── 1. Teacher 단독 학습 ─────────────────────────────────────────────────────

def train_teacher(teacher,
                  train_loader,
                  val_loader,
                  arch: str    = 'mlp',
                  task: str    = 'binary',
                  epochs: int  = config.EPOCHS,
                  lr: float    = config.LEARNING_RATE,
                  save_path: str = None,
                  verbose: bool = True):
    """
    Teacher 모델을 CrossEntropy로 단독 학습합니다.

    Parameters
    ----------
    teacher     : TeacherModel 또는 LSTMTeacherModel
    arch        : 'mlp' | 'lstm' (forward 인자 수 결정)
    task        : 'binary' | '3class'
    save_path   : 최적 모델 저장 경로 (None이면 checkpoints/teacher_best.pt)
    """
    if save_path is None:
        save_dir = config.CHECKPOINTS_DIR / task
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / f"teacher_{arch}_{task}_best.pt")

    device  = get_device()
    teacher = teacher.to(device)
    print(f"\n[Teacher Train] device={device}  arch={arch}  task={task}  epochs={epochs}")

    optimizer  = optim.Adam(teacher.parameters(), lr=lr,
                             weight_decay=config.WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion  = nn.CrossEntropyLoss()
    early_stop = EarlyStopping()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        # ── Train ──
        teacher.train()
        t_loss, t_acc, n_batches = 0.0, 0.0, 0
        t0 = time.time()

        for batch in train_loader:
            phys, video, audio, _, labels = [b.to(device) for b in batch]
            if arch == 'mlp':
                logits, *_ = teacher(phys, video, audio)
            else:
                logits = teacher(phys, video, audio)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
            optimizer.step()

            t_loss += loss.item()
            t_acc  += accuracy(logits, labels)
            n_batches += 1

        t_loss /= n_batches
        t_acc  /= n_batches

        # ── Validation ──
        val_loss, val_acc = _eval_teacher(teacher, val_loader, criterion, device, arch)
        scheduler.step(val_loss)

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss {t_loss:.4f} Acc {t_acc:.4f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | "
                  f"{elapsed:.1f}s")

        # 최적 모델 저장
        if val_loss <= early_stop.best_loss:
            save_checkpoint(teacher, optimizer, epoch, val_loss, save_path)

        if early_stop(val_loss):
            print(f"  Early stopping at epoch {epoch}.")
            break

    print(f"[Teacher Train] Best Val Loss: {early_stop.best_loss:.4f}")
    return history


def _eval_teacher(model, loader, criterion, device, arch):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            phys, video, audio, _, labels = [b.to(device) for b in batch]
            if arch == 'mlp':
                logits, *_ = model(phys, video, audio)
            else:
                logits = model(phys, video, audio)
            total_loss += criterion(logits, labels).item()
            total_acc  += accuracy(logits, labels)
            n += 1
    return total_loss / n, total_acc / n


# ─── 2. Student Baseline (KD 없음) ───────────────────────────────────────────

def train_student_baseline(student,
                            train_loader,
                            val_loader,
                            arch: str    = 'mlp',
                            task: str    = 'binary',
                            epochs: int  = config.EPOCHS,
                            lr: float    = config.LEARNING_RATE,
                            save_path: str = None,
                            verbose: bool = True):
    """
    KD 없이 Student를 CrossEntropy로만 학습합니다 (KD 효과 비교 baseline).
    """
    if save_path is None:
        save_dir = config.CHECKPOINTS_DIR / task
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / f"student_baseline_{arch}_{task}_best.pt")

    device  = get_device()
    student = student.to(device)
    print(f"\n[Student Baseline] device={device}  arch={arch}  task={task}  epochs={epochs}")

    optimizer  = optim.Adam(student.parameters(), lr=lr,
                             weight_decay=config.WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion  = nn.CrossEntropyLoss()
    early_stop = EarlyStopping()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        student.train()
        t_loss, t_acc, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            video, audio, _, labels = [b.to(device) for b in batch]
            if arch == 'mlp':
                logits, *_ = student(video, audio)
            else:
                logits = student(video, audio)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            t_loss += loss.item()
            t_acc  += accuracy(logits, labels)
            n_batches += 1

        t_loss /= n_batches
        t_acc  /= n_batches

        val_loss, val_acc = _eval_student(student, val_loader, criterion, device, arch)
        scheduler.step(val_loss)

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss {t_loss:.4f} Acc {t_acc:.4f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

        if val_loss <= early_stop.best_loss:
            save_checkpoint(student, optimizer, epoch, val_loss, save_path)

        if early_stop(val_loss):
            print(f"  Early stopping at epoch {epoch}.")
            break

    print(f"[Student Baseline] Best Val Loss: {early_stop.best_loss:.4f}")
    return history


# ─── 3. Student with Knowledge Distillation ──────────────────────────────────

def train_student_kd(teacher,
                     student,
                     train_loader,        # Teacher용 DataLoader (phys, video, audio, has_audio, label 모두 포함)
                     val_loader,           # Student 검증용 (video, audio, label)
                     arch: str    = 'mlp',
                     task: str    = 'binary',
                     epochs: int  = config.EPOCHS,
                     lr: float    = config.LEARNING_RATE,
                     alpha: float = config.KD_ALPHA,
                     beta: float  = config.KD_BETA,
                     gamma: float = config.KD_GAMMA,
                     temperature: float = config.KD_TEMPERATURE,
                     save_path: str = None,
                     verbose: bool  = True):
    """
    Teacher 지식을 Student로 Knowledge Distillation.
    (데이터 불일치 방지를 위해 단일 DataLoader를 사용하도록 수정됨)
    """
    if save_path is None:
        save_dir = config.CHECKPOINTS_DIR / task
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / f"student_kd_{arch}_{task}_best.pt")

    device  = get_device()
    teacher = teacher.to(device).eval()   # Teacher는 frozen
    student = student.to(device)

    # Teacher 가중치 고정
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"\n[Student KD] device={device}  arch={arch}  epochs={epochs}")
    print(f"             α={alpha}  β={beta}  γ={gamma}  T={temperature}")

    optimizer  = optim.Adam(student.parameters(), lr=lr,
                             weight_decay=config.WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    kd_loss_fn = TotalKDLoss(alpha, beta, gamma, temperature, arch=arch)
    ce_loss_fn = nn.CrossEntropyLoss()
    early_stop = EarlyStopping()

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [],   'val_acc': [],
        'soft_kd': [],    'feat_kd': [],  'ce': [],
    }

    # 하나의 DataLoader에서 모든 모달리티를 동시에 꺼냄
    for epoch in range(1, epochs + 1):
        student.train()
        ep_loss = {'soft_kd': 0, 'feat_kd': 0, 'ce': 0, 'total': 0}
        t_acc, n_batches = 0.0, 0

        for batch in train_loader:
            # 공통 배치 데이터 (Teacher용 DataLoader 기준)
            phys, video, audio, has_audio, labels = [b.to(device) for b in batch]

            # Teacher forward (no grad)
            with torch.no_grad():
                if arch == 'mlp':
                    t_logits, t_phys_emb, t_video_emb, t_audio_emb = \
                        teacher(phys, video, audio)
                else:
                    t_logits = teacher(phys, video, audio)
                    t_video_emb = t_audio_emb = None

            # Student forward
            if arch == 'mlp':
                s_logits, s_video_emb, s_audio_emb = student(video, audio)
            else:
                s_logits = student(video, audio)
                s_video_emb = s_audio_emb = None

            # KD Loss 계산
            total_loss, loss_dict = kd_loss_fn(
                t_logits, s_logits, labels,
                t_video_emb, t_audio_emb,
                s_video_emb, s_audio_emb,
                has_audio=has_audio,
            )

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            for k in ep_loss:
                ep_loss[k] += loss_dict.get(k, total_loss.item())
            t_acc    += accuracy(s_logits, labels)
            n_batches += 1

        for k in ep_loss:
            ep_loss[k] /= n_batches
        t_acc /= n_batches

        val_loss, val_acc = _eval_student(student, val_loader, ce_loss_fn, device, arch)
        scheduler.step(val_loss)

        history['train_loss'].append(ep_loss['total'])
        history['train_acc'].append(t_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['soft_kd'].append(ep_loss['soft_kd'])
        history['feat_kd'].append(ep_loss['feat_kd'])
        history['ce'].append(ep_loss['ce'])

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Total {ep_loss['total']:.4f} "
                  f"(SoftKD {ep_loss['soft_kd']:.4f} "
                  f"FeatKD {ep_loss['feat_kd']:.4f} "
                  f"CE {ep_loss['ce']:.4f}) | "
                  f"Val Acc {val_acc:.4f}")

        if val_loss <= early_stop.best_loss:
            save_checkpoint(student, optimizer, epoch, val_loss, save_path)

        if early_stop(val_loss):
            print(f"  Early stopping at epoch {epoch}.")
            break

    print(f"[Student KD] Best Val Loss: {early_stop.best_loss:.4f}")
    return history


def _eval_student(model, loader, criterion, device, arch):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            video, audio, _, labels = [b.to(device) for b in batch]
            if arch == 'mlp':
                logits, *_ = model(video, audio)
            else:
                logits = model(video, audio)
            total_loss += criterion(logits, labels).item()
            total_acc  += accuracy(logits, labels)
            n += 1
    return total_loss / n, total_acc / n
