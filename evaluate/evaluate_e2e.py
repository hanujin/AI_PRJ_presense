import torch
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns

# Reorganized: presense root and subfolders added to path
_root = Path(__file__).resolve().parent.parent
for p in [_root, _root/'data', _root/'models', _root/'train', _root/'evaluate', _root/'analysis']:
    if str(p) not in sys.path: sys.path.insert(0, str(p))

import config
from models_e2e import E2ETeacherModel, E2EStudentModel, HybridE2EModel
from dataset_e2e import StressIDRawDataset, build_sample_list
from dataset import load_labels
from train import get_device

def evaluate_all_e2e():
    device = get_device()
    TASK = 'binary'
    print(f"📊 Arch 1 (E2E/Hybrid) 통합 성능 평가 시작 (Device: {device})")

    # 1. 데이터 준비
    labels_df = load_labels(task=TASK, fallback=True)
    labels_dict = labels_df[labels_df.columns[0]].to_dict()
    samples = build_sample_list(labels_dict)
    subjects = [s['subject'] for s in samples]
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_SEED)
    _, test_idx = next(gss.split(samples, groups=subjects))
    
    test_dataset = StressIDRawDataset([samples[i] for i in test_idx], mode='teacher')
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)

    # 2. 모델 로드 유틸 (Concise Filenames)
    def load_model(cls, arch_folder, filename):
        path = config.CHECKPOINTS_DIR / TASK / arch_folder / filename
        model = cls(num_classes=2).to(device)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])
            model.eval()
            print(f"✅ 로드 성공: {filename}")
            return model
        print(f"⚠️ 파일 없음: {path}")
        return None

    models = {
        "Teacher":          load_model(E2ETeacherModel,  "Arch_1", "teacher_best.pt"),
        "Student_Baseline": load_model(E2EStudentModel, "Arch_1", "student_baseline_best.pt"),
        "Student_KD":       load_model(E2EStudentModel, "Arch_1", "student_kd_best.pt"),
        "Hybrid_Teacher":   load_model(HybridE2EModel,  "Arch_1", "hybrid_teacher_best.pt"),
        "Hybrid_Baseline":  load_model(E2EStudentModel, "Arch_1", "hybrid_student_baseline_best.pt"),
        "Hybrid_KD":        load_model(E2EStudentModel, "Arch_1", "hybrid_student_kd_best.pt")
    }

    # 3. 평가 및 시각화
    results = []
    save_dir = config.RESULTS_DIR / TASK / "Arch_1"
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        if model is None: continue
        all_labels, all_preds = [], []
        with torch.no_grad():
            for frames, audio, physio, labels in loader:
                frames, audio, physio, labels = [b.to(device) for b in (frames, audio, physio, labels)]
                if isinstance(model, HybridE2EModel) or isinstance(model, E2ETeacherModel):
                    logits, *_ = model(frames, audio, physio)
                else:
                    logits, *_ = model(frames, audio)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        results.append({"Model": name, "Accuracy": acc, "F1-Score": f1})
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} Confusion Matrix")
        plt.savefig(save_dir / f"cm_{name.lower()}.png")
        plt.close()

    df_res = pd.DataFrame(results)
    print("\n최종 성적표:\n", df_res)
    df_res.to_csv(save_dir / "comparison.csv", index=False)

if __name__ == "__main__":
    evaluate_all_e2e()
