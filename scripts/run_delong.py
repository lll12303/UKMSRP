"""
Run DeLong-based Feature Selection
"""
import pandas as pd
from src.feature_selection.feature selection pipeline import feature selection pipeline

if __name__ == "__main__":
    df = pd.read_csv("data/metadata_example.csv")
    X = pd.read_csv("data/bio_features_example.csv")
    y = df["status"]

    results = feature selection pipeline(X, y, p_threshold=0.05)
    results.to_csv("output/delong_feature_ranking.csv", index=False)

    print("Feature selection done. Results saved to output/delong_feature_ranking.csv")
