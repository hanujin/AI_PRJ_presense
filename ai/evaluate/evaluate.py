"""
PreSense - 통합 평가 유틸리티
- Architecture ②(MLP), ③(LSTM) 원본 피처 기반 평가 (Concise Names)
- Architecture ② PCA 기반 모델 평가 (Merged into Arch_2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

# Reorganized path setup
_root = Path(__file__).resolve().parent.parent.parent
_ai_root = _root / 'ai'
for p in [_root, _ai_root, _ai_root/'data', _ai_root/'models', _ai_root/'train', _ai_root/'evaluate', _ai_root/'feature_analysis']:
    if str(p) not in sys.path: sys.path.insert(0, str(p))

import config
from train import get_device
from models import build_models

# ─── PCA용 모델 정의 ────────────────────────────────────────────────────────
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=5, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, num_classes))
    def forward(self, x): return self.net(x)

def compute_metrics(y_true, y_pred):
    return {
        'accuracy':          accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_weighted':       f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

def plot_cm(y_true, y_pred, title, save_path, task='binary'):
    class_names = (['Not Stressed', 'Stressed'] if task == 'binary'
                   else ['Relaxed', 'Neutral', 'Stressed'])
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# ─── Architecture ②, ③ (원본 및 PCA) 통합 평가 ──────────────────────────────

def evaluate_arch(task='binary', arch='mlp'):
    from data.dataset import load_all
    device = get_device()
    print(f"\n📊 {task.upper()} - Arch {arch.upper()} 성능 평가 시작")
    
    # 1. 원본 모델 평가
    _, _, student_loaders = load_all(task=task)
    _, _, test_loader = student_loaders

    def load_model(arch_folder, filename):
        path = config.CHECKPOINTS_DIR / task / arch_folder / filename
        _, model = build_models(arch=arch, task=task)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])
            model.eval()
            print(f"✅ 로드 성공: {filename}")
            return model.to(device)
        return None

    arch_folder = "Arch_2" if arch == 'mlp' else "Arch_3"
    models = {
        "Teacher":  load_model(arch_folder, "teacher_best.pt"),
        "Baseline": load_model(arch_folder, "student_baseline_best.pt"),
        "KD":       load_model(arch_folder, "student_kd_best.pt")
    }
    
    results = []
    save_dir = config.RESULTS_DIR / task / arch_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        if model is None: continue
        all_labels, all_preds = [], []
        with torch.no_grad():
            for batch in test_loader:
                video, audio, has_audio, labels = [b.to(device) for b in batch]
                logits, *_ = model(video, audio) if arch=='mlp' else (model(video, audio),)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = compute_metrics(all_labels, all_preds)
        results.append({"Model": name, "Accuracy": metrics['accuracy'], "F1-Score": metrics['f1_weighted']})
        plot_cm(all_labels, all_preds, f"{name} ({arch.upper()})", save_dir / f"cm_{name.lower()}.png", task=task)

    # 2. PCA 모델 평가 (Arch 2 전용)
    if arch == 'mlp' and task == 'binary':
        print(f"🔎 Arch 2 PCA 기반 모델 평가")
        X = np.load(config.RESULTS_DIR / "final_features.npy")
        y = np.load(config.RESULTS_DIR / "final_labels.npy")
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pca_loader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long()), batch_size=4)

        pca_models = {
            "PCA_Teacher":  load_model(arch_folder, "pca_teacher_best.pt"),
            "PCA_Baseline": load_model(arch_folder, "pca_student_baseline_best.pt"),
            "PCA_KD":       load_model(arch_folder, "pca_student_kd_best.pt")
        }

        for name, model in pca_models.items():
            if model is None: continue
            all_labels, all_preds = [], []
            with torch.no_grad():
                for x_b, y_b in pca_loader:
                    logits = model(x_b.to(device))
                    all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    all_labels.extend(y_b.numpy())
            
            metrics = compute_metrics(all_labels, all_preds)
            results.append({"Model": name, "Accuracy": metrics['accuracy'], "F1-Score": metrics['f1_weighted']})
            plot_cm(all_labels, all_preds, name, save_dir / f"cm_{name.lower()}.png", task=task)

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    df_res.to_csv(save_dir / "comparison.csv", index=False)

if __name__ == "__main__":
    for task in ['binary', '3class']:
        for arch in ['mlp', 'lstm']:
            evaluate_arch(task=task, arch=arch)
