import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
# Reorganized: presense root and subfolders added to path
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent.parent
_ai_root = _root / 'ai'
for p in [_root, _ai_root, _ai_root/'data', _ai_root/'models', _ai_root/'train', _ai_root/'evaluate', _ai_root/'feature_analysis']:
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import config
from train import get_device, save_checkpoint, EarlyStopping

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=5, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, num_classes))
    def forward(self, x): return self.net(x)

def train_pca_teacher(epochs=50, batch_size=4):
    device = get_device()
    TASK = 'binary'
    ARCH = 'Arch_2_PCA'
    print(f"🎓 PCA Teacher 학습 시작 (Device: {device})")

    X = np.load(config.RESULTS_DIR / "final_features.npy")
    y = np.load(config.RESULTS_DIR / "final_labels.npy")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long()), batch_size=batch_size)

    model = SimpleMLP(input_dim=X.shape[1], num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=7)

    save_dir = config.CHECKPOINTS_DIR / TASK / ARCH
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(save_dir / f"teacher_{ARCH}_{TASK}_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y_b in train_loader:
            x, y_b = x.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y_b)
            loss.backward()
            optimizer.step()
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y_b in val_loader:
                x, y_b = x.to(device), y_b.to(device)
                v_loss += criterion(model(x), y_b).item()
        
        v_loss /= len(val_loader)
        if v_loss < early_stop.best_loss:
            save_checkpoint(model, optimizer, epoch, v_loss, save_path)
        if early_stop(v_loss): break
    
    print(f"✅ 학습 완료: {save_path}")

if __name__ == "__main__":
    train_pca_teacher()
