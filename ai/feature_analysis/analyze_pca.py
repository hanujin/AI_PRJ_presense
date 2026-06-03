import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
import config

def analyze_pca():
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
    
    labels_path = Path("StressID Dataset/labels.csv")
    labels_df = pd.read_csv(labels_path, index_col=0)
    labels_df.index = labels_df.index.astype(str)
    
    common_idx = labels_df.index.intersection(combined.index)
    
    # 1. 표준화 (필수)
    X = StandardScaler().fit_transform(combined.loc[common_idx].values)
    
    # 2. PCA 수행
    pca = PCA()
    pca.fit(X)
    
    # 90% 분산 설명에 필요한 주성분 개수 찾기
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    
    print(f"\n[PCA 분석 결과]")
    print(f"전체 정보의 90%를 설명하는 주성분 개수: {n_components_90}개")
    
    # 3. 그래프 저장
    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_variance)
    plt.axhline(y=0.90, color='r', linestyle='--')
    plt.title("Cumulative Explained Variance by PCA")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance")
    plt.grid(True)
    
    save_path = config.RESULTS_DIR / "pca_variance.png"
    plt.savefig(save_path)
    print(f"그래프 저장 완료: {save_path}")

if __name__ == "__main__":
    analyze_pca()
