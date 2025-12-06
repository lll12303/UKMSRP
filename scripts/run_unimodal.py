"""
Run Unimodal XGBoost Training
-----------------------------
Steps:
    1. Load full dataset
    2. Split England vs Scotland+Wales
    3. Train model with 5-fold CV
    4. Save best model
    5. Evaluate external dataset
"""

import yaml
import pandas as pd
import joblib
from models.train_unimodal import train_unimodal_model, external_validation


if __name__ == "__main__":

    # Load configuration
    config = yaml.safe_load(open("src/configs/config_example.yaml", "r"))

    df = pd.read_csv(config["raw_file"])
    X_full = pd.read_csv(config["feature_file"])

    # Split region
    eng_idx = df[df["Region"] == "England"].index
    ext_idx = df[df["Region"].isin(["Scotland", "Wales"])].index

    X_eng = X_full.loc[eng_idx]
    y_eng = df.loc[eng_idx, config["label_column"]]

    X_ext = X_full.loc[ext_idx]
    y_ext = df.loc[ext_idx, config["label_column"]]

    print(f"England training shape: {X_eng.shape}")
    print(f"External validation shape: {X_ext.shape}")

    # Train model
    best_model, summary, auc_list = train_unimodal_model(
        X_eng,
        y_eng,
        params=config["params"],
        n_splits=config["n_splits"]
    )

    # Save model
    joblib.dump(best_model, config["output_model"])
    print(f"Best model saved → {config['output_model']}")

    # External validation
    external_validation(best_model, X_ext, y_ext)
