import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
# Reorganized: presense root and subfolders added to path
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
for p in [_root, _root/'data', _root/'models', _root/'train', _root/'evaluate', _root/'analysis']:
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import config

def analyze_hybrid():
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
            # CSV 첫 번째 컬럼을 인덱스로 사용
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
    
    # 2. 정렬된 공통 인덱스 추출 (데이터 정렬의 핵심)
    common_idx = labels_df.index.intersection(combined.index)
    X = combined.loc[common_idx].values
    y = labels_df.loc[common_idx, 'binary-stress'].values
    
    print(f"매칭된 샘플 수: {len(common_idx)}")
    
    # 3. 중요도 필터링 및 PCA
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    mask = clf.feature_importances_ > 0
    X_important = X[:, mask]
    
    X_scaled = StandardScaler().fit_transform(X_important)
    pca = PCA(n_components=0.90)
    X_final = pca.fit_transform(X_scaled)
    
    # 4. 피처와 레이블을 짝지어 저장
    save_dir = config.RESULTS_DIR
    np.save(save_dir / "final_features.npy", X_final)
    np.save(save_dir / "final_labels.npy", y)
    
    print(f"\n[Hybrid 분석 완료]")
    print(f"최종 저장 경로: {save_dir}/final_features.npy, final_labels.npy")

if __name__ == "__main__":
    analyze_hybrid()
