"""
Feature Selection using DeLong Test
-----------------------------------
This module implements:
    - Single-feature AUC computation
    - DeLong test for comparing each feature vs baseline model
    - p-value filtering
    - Optional multiple-testing correction
    - Output ranked feature list

This script is modality-independent and can be reused for:
    * Proteomics
    * Metabolomics
    * Biochemistry
    * Clinical features
    * Lifestyle factors

Reference:
    DeLong ER, DeLong DM, Clarke-Pearson DL.
    Comparing the areas under two or more correlated ROC curves.
    Biometrics. 1988.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import norm


# -------------------------
# --- DeLong core methods
# -------------------------

def compute_midrank(x):
    """Compute midranks for DeLong test."""
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]
    N = len(x)
    T = np.zeros(N, dtype=float)

    i = 0
    while i < N:
        j = i
        while j < N and sorted_x[j] == sorted_x[i]:
            j += 1
        mid = 0.5 * (i + j - 1)
        for k in range(i, j):
            T[k] = mid
        i = j

    T2 = np.empty(N, dtype=float)
    T2[sorted_idx] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    Fast implementation of DeLong test.
    Reference implementation adapted from:
    https://github.com/yandexdataschool/roc_comparison
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    positive_predictions = predictions_sorted_transposed[:, :m]
    negative_predictions = predictions_sorted_transposed[:, m:]

    tx = np.apply_along_axis(compute_midrank, 1, predictions_sorted_transposed)
    tx_pos = tx[:, :m]
    tx_neg = tx[:, m:]

    aucs = (tx_pos.sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tx_pos - tx_pos.mean(axis=1, keepdims=True)) / n
    v10 = (tx_neg - tx_neg.mean(axis=1, keepdims=True)) / n

    S = np.cov(v01, v10, bias=True)
    return aucs, S


def delong_roc_test(y_true, pred1, pred2):
    """
    Compare two ROC AUCs using DeLong test.
    Returns p-value.
    """
    y_true = np.array(y_true)
    preds = np.vstack((pred1, pred2))

    order = np.argsort(-y_true)
    preds_sorted = preds[:, order]
    y_sorted = y_true[order]

    aucs, cov = fastDeLong(preds_sorted, y_sorted.sum())
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return aucs, p


# -------------------------
# --- Feature Screening
# -------------------------

def delong_feature_selection(X, y, baseline_pred=None, p_threshold=0.05):
    """
    Perform DeLong-based feature selection.

    Args:
        X (DataFrame): Feature matrix (all single features).
        y (Series): Binary outcome.
        baseline_pred (Array): Baseline model scores (optional).
                              If None → compare each feature vs null model.
        p_threshold (float): Significance cutoff.

    Returns:
        DataFrame: Ranked features with AUCs and p-values.
    """

    results = []

    # Baseline: if not provided, use random (null model)
    if baseline_pred is None:
        baseline_pred = np.random.normal(0, 1, size=len(y))

    for f in X.columns:
        feature_pred = np.array(X[f])

        # Compute AUC
        auc_f = roc_auc_score(y, feature_pred)

        # DeLong test (feature vs baseline)
        aucs, p_val = delong_roc_test(y, feature_pred, baseline_pred)

        results.append({
            "feature": f,
            "auc": auc_f,
            "p_value": p_val,
        })

    df_results = pd.DataFrame(results)
    df_results.sort_values("auc", ascending=False, inplace=True)

    # Add significance flag
    df_results["significant"] = df_results["p_value"] < p_threshold

    return df_results
