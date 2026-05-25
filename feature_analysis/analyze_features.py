import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
import config

def analyze_features():
    # 1. 데이터 로드
    features_dir = Path("stressID-main/Feature Extraction/Features")
    files = {
        "Physio": "all_physiological_features.csv",
        "Audio": "HCfeatures.csv",
        "Video": "video11tasks_aus_gaze_mean_std.csv"
    }
    
    dfs = []
    for mod, f in files.items():
        path = features_dir / f
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            df.index = df.index.astype(str)
            dfs.append(df)
        else:
            print(f"Warning: {path} not found.")
    
    if not dfs:
        return

    combined = pd.concat(dfs, axis=1).fillna(0)
    labels_df = pd.read_csv("StressID Dataset/labels.csv", index_col=0)
    labels_df.index = labels_df.index.astype(str)
    
    common_idx = labels_df.index.intersection(combined.index)
    X = combined.loc[common_idx].values
    y = labels_df.loc[common_idx, 'binary-stress'].values
    
    # 2. 피처 중요도 분석
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    importance_df = pd.DataFrame({'Feature': combined.columns, 'Importance': clf.feature_importances_})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    importance_df.to_csv(Path(__file__).parent / "results" / "feature_importance.csv", index=False)
    
    # 3. 중요도 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(importance_df['Importance'].values)
    plt.axvline(x=50, color='r', linestyle='--', label='Cutoff (50)')
    plt.title("Feature Importance Curve")
    plt.xlabel("Number of Features (sorted)")
    plt.ylabel("Importance Score")
    plt.grid(True)
    plt.savefig(Path(__file__).parent / "results" / "feature_importance_plot.png")
    
    print(f"\n[분석 완료]")
    print(f"저장 완료: {config.RESULTS_DIR}/feature_importance.csv 및 feature_importance_plot.png")
    print(importance_df.head(20).to_string(index=False))

if __name__ == "__main__":
    analyze_features()
