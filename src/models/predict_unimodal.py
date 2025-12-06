import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


def compute_metrics(y_true, y_prob):
    """Compute basic evaluation metrics"""
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, (y_prob > 0.5).astype(int))

    return {
        "AUC": float(auc),
        "Average Precision": float(ap),
        "Accuracy": float(acc)
    }


def main_from_config(config):
    """
    Predict + compute metrics for unimodal models.

    Required config keys:
        - feature_file
        - model_file
        - output_predictions
        - save_metrics
        - index_col
        - label_column
    """

    df = pd.read_csv(config["feature_file"])
    df = df.set_index(config["index_col"])

    label_col = config["label_column"]
    y_true = df[label_col]
    X = df.drop(columns=[label_col])

    # Load model
    model = joblib.load(config["model_file"])

    # Predict
    y_prob = model.predict_proba(X)[:, 1]

    pred_df = pd.DataFrame({
        config["index_col"]: df.index,
        "prediction": y_prob,
        "label": y_true.values
    })

    pred_df.to_csv(config["output_predictions"], index=False)
    print(f"Predictions saved => {config['output_predictions']}")

    # Metrics
    metrics = compute_metrics(y_true, y_prob)

    import json
    with open(config["save_metrics"], "w") as f:
        json.dump(metrics, f, indent=4)

    print("Metrics saved =>", config["save_metrics"])
    print(metrics)

    return pred_df, metrics
