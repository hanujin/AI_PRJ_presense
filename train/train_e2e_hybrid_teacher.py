import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import sys
from pathlib import Path
import time
from sklearn.model_selection import GroupShuffleSplit

# presense 폴더를 path에 추가
# Reorganized: presense root and subfolders added to path
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
for p in [_root, _root/'data', _root/'models', _root/'train', _root/'evaluate', _root/'analysis']:
    if str(p) not in sys.path: sys.path.insert(0, str(p))

import config
from models_e2e import HybridE2EModel
from dataset_e2e import StressIDRawDataset, build_sample_list
from dataset import load_labels
from train import get_device, accuracy, save_checkpoint, EarlyStopping
from kd_loss import TotalKDLoss

def train_e2e_hybrid(epochs=50, batch_size=2):
    device = get_device()
    TASK = 'binary'
    print(f"🌀 Hybrid E2E 학습 시작 (Task: {TASK}, Device: {device})")

    # 1. 데이터 준비
    labels_df = load_labels(task=TASK, fallback=True)
    labels_dict = labels_df[labels_df.columns[0]].to_dict()
    samples = build_sample_list(labels_dict)
    subjects = [s['subject'] for s in samples]
    
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=config.RANDOM_SEED)
    train_idx, val_idx = next(gss.split(samples, groups=subjects))
    
    full_dataset = StressIDRawDataset(samples, mode='teacher')
    train_ds = Subset(full_dataset, train_idx)
    val_ds   = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. 모델 로드 및 설정
    model = HybridE2EModel(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=7)

    save_dir = config.CHECKPOINTS_DIR / TASK
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(save_dir / "e2e_hybrid_best.pt")

    # 3. 학습 루프
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_acc, n = 0, 0, 0
        t0 = time.time()
        
        for i, (frames, audio, physio, labels) in enumerate(train_loader):
            frames, audio, physio, labels = [b.to(device) for b in (frames, audio, physio, labels)]
            
            logits, *_ = model(frames, audio, physio)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()
            t_acc += accuracy(logits, labels)
            n += 1
            
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                print(f"  Batch {i+1:3d}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {accuracy(logits, labels):.4f}")
            
        t_loss /= n
        t_acc /= n
        
        # Validation
        model.eval()
        v_loss, v_acc, vn = 0, 0, 0
        with torch.no_grad():
            for frames, audio, physio, labels in val_loader:
                frames, audio, physio, labels = [b.to(device) for b in (frames, audio, physio, labels)]
                logits, *_ = model(frames, audio, physio)
                v_loss += criterion(logits, labels).item()
                v_acc += accuracy(logits, labels)
                vn += 1
        v_loss /= vn
        v_acc /= vn
        
        print(f"✨ Epoch {epoch} 요약 | Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} | Time: {time.time()-t0:.1f}s")
        
        # 모델 저장 및 조기 종료
        if v_loss < early_stop.best_loss:
            print(f"💾 최고 성능 갱신! 모델 저장 중: {Path(save_path).name}")
            save_checkpoint(model, optimizer, epoch, v_loss, save_path)
            
        if early_stop(v_loss):
            print("🛑 조기 종료 (Early Stopping) 발생")
            break
            
    print(f"\n✅ 학습 완료: {save_path}")

if __name__ == "__main__":
    train_e2e_hybrid()
