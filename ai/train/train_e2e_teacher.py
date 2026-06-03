
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import sys
from pathlib import Path
import time
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# presense 폴더를 path에 추가
# Reorganized: presense root and subfolders added to path
_root = Path(__file__).resolve().parent.parent.parent
_ai_root = _root / 'ai'
for p in [_root, _ai_root, _ai_root/'data', _ai_root/'models', _ai_root/'train', _ai_root/'evaluate', _ai_root/'feature_analysis']:
    if str(p) not in sys.path: sys.path.insert(0, str(p))

import config
from models_e2e import E2ETeacherModel
from dataset_e2e import StressIDRawDataset, build_sample_list
from dataset import load_labels
from train import get_device, accuracy, EarlyStopping, save_checkpoint

def train_full_e2e_teacher(epochs=config.EPOCHS, batch_size=4):
    device = get_device()
    TASK = 'binary'
    print(f"🔥 Architecture 1 (E2E) 정식 학습 시작 (Device: {device})")

    # 1. 데이터 준비 및 Subject 기반 분할
    labels_df = load_labels(fallback=True)
    labels_dict = labels_df['binary-stress'].to_dict()
    samples = build_sample_list(labels_dict)
    
    # Subject 리스트 추출 (데이터 누수 방지용)
    subjects = [s['subject'] for s in samples]
    
    # 8:2 분할
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=config.RANDOM_SEED)
    train_idx, val_idx = next(gss.split(samples, groups=subjects))
    
    full_dataset = StressIDRawDataset(samples, mode='teacher')
    train_ds = Subset(full_dataset, train_idx)
    val_ds   = Subset(full_dataset, val_idx)
    
    # num_workers=0 (Mac MPS에서 안정적임)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"📊 데이터 분할 완료: Train {len(train_ds)}개, Val {len(val_ds)}개")

    # 2. 모델 및 옵티마이저 (학습률을 조금 낮게 설정하여 안정성 확보)
    model = E2ETeacherModel(num_classes=2, freeze_backbone=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=7)

    ARCH = "teacher"
    save_dir = config.CHECKPOINTS_DIR / TASK
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{ARCH}_{TASK}_best.pt"
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # 3. 학습 루프
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_acc, n = 0, 0, 0
        t0 = time.time()
        
        print(f"\n[Epoch {epoch}/{epochs}]")
        for i, (frames, audio, physio, labels) in enumerate(train_loader):
            frames, audio, physio, labels = frames.to(device), audio.to(device), physio.to(device), labels.to(device)
            
            logits, *_ = model(frames, audio, physio)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            t_loss += loss.item()
            t_acc += accuracy(logits, labels)
            n += 1
            
            if (i+1) % 10 == 0 or (i+1) == len(train_loader):
                print(f"  Batch {i+1:3d}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {accuracy(logits, labels):.4f}")

        t_loss /= n
        t_acc /= n
        
        # Validation
        model.eval()
        v_loss, v_acc, vn = 0, 0, 0
        with torch.no_grad():
            for frames, audio, physio, labels in val_loader:
                frames, audio, physio, labels = frames.to(device), audio.to(device), physio.to(device), labels.to(device)
                logits, *_ = model(frames, audio, physio)
                v_loss += criterion(logits, labels).item()
                v_acc += accuracy(logits, labels)
                vn += 1
        
        v_loss /= vn if vn > 0 else 1
        v_acc /= vn if vn > 0 else 1
        scheduler.step(v_loss)
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        
        print(f"✨ Epoch {epoch:2d} 요약 | Loss {t_loss:.4f} Acc {t_acc:.4f} | Val Loss {v_loss:.4f} Acc {v_acc:.4f} | {time.time()-t0:.1f}s")
        
        # 최적 모델 저장
        if v_loss < early_stop.best_loss:
            print(f"💾 최고 성능 갱신! 모델 저장 중: {save_path.name}")
            save_checkpoint(model, optimizer, epoch, v_loss, str(save_path))
            
        if early_stop(v_loss):
            print("🛑 조기 종료 (Early Stopping) 발생")
            break

    print(f"\n✅ Architecture 1 학습 완료! 최적 모델: {save_path}")

if __name__ == "__main__":
    train_full_e2e_teacher(epochs=50, batch_size=4)
