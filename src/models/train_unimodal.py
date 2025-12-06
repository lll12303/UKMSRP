import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def train_unimodal_from_config(config: dict):
    """
    Train an XGBoost classifier using a configuration dict.

    Required keys:
        - feature_file          CSV path
        - label_column          "status"
        - index_col             "Participant ID"
        - output_model          "xxx.joblib"
        - n_folds               integer
    """

    feature_file = config["feature_file"]
    label_col = config["label_column"]
    index_col = config["index_col"]
    output_model = config["output_model"]
    n_folds = config.get("n_folds", 5)

    # ============================
    # Load Data
    # ============================
    df = pd.read_csv(feature_file)
    df = df.set_index(index_col)

    X = df.drop(columns=[label_col]).values
    y = df[label_col].values

    # ============================
    # Model
    # ============================
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )

    # ============================
    # CV Training
    # ============================
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, pred)
        aucs.append(auc)

        print(f"[Fold {fold+1}/{n_folds}] AUC = {auc:.4f}")

    print("\nMean CV AUC =", np.mean(aucs))

    # ============================
    # Save Model
    # ============================
    joblib.dump(model, output_model)
    print(f"Model saved => {output_model}")

    return model
