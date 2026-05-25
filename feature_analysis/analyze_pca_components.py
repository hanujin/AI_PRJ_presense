import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import config

def analyze_pca_components():
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
    feature_names = combined.columns

    # 2. 중요도 분석 (중요도 0 제거)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    mask = clf.feature_importances_ > 0
    X_important = X[:, mask]
    important_features = feature_names[mask]
    
    # 3. PCA 수행 (필터링된 피처들에 대해서만)
    X_scaled = StandardScaler().fit_transform(X_important)
    pca = PCA(n_components=5) # 상위 5개 성분만 로딩 확인
    pca.fit(X_scaled)
    
    # 4. 주성분 로딩(Loading) 확인
    components = pd.DataFrame(pca.components_, columns=important_features, index=[f'PC{i}' for i in range(5)])
    
    print("\n[중요도 필터링 후 주성분별 핵심 기여 피처 Top 3]")
    for i in range(5):
        top3 = components.iloc[i].abs().sort_values(ascending=False).head(3)
        print(f"\nPC{i} 구성요소:")
        print(top3)

    # 전체 로딩 데이터 저장
    components.to_csv(config.RESULTS_DIR / "pca_components.csv")
    print(f"\n상세 정보 저장 완료: {config.RESULTS_DIR}/pca_components.csv")

if __name__ == "__main__":
    analyze_pca_components()
