"""
Utility Metrics Functions
-------------------------
This module contains reusable evaluation metrics for
binary classification models used in the UKMSRP project.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)


def compute_auc(y_true, y_prob):
    """Compute ROC AUC."""
    return roc_auc_score(y_true, y_prob)


def compute_accuracy(y_true, y_pred):
    """Compute accuracy."""
    return accuracy_score(y_true, y_pred)


def compute_f1(y_true, y_pred):
    """Compute F1 score."""
    return f1_score(y_true, y_pred)


def compute_brier(y_true, y_prob):
    """Compute Brier score (calibration)."""
    return brier_score_loss(y_true, y_prob)


def compute_sensitivity_specificity(y_true, y_pred):
    """
    Compute sensitivity and specificity from confusion matrix.
    Returns: sensitivity, specificity
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def full_metrics_report(y_true, y_pred, y_prob):
    """
    Generate a complete metrics report useful for model comparison.
    """
    auc = compute_auc(y_true, y_prob)
    acc = compute_accuracy(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred)
    brier = compute_brier(y_true, y_prob)
    sen, spe = compute_sensitivity_specificity(y_true, y_pred)

    return {
        "auc": auc,
        "accuracy": acc,
        "f1_score": f1,
        "brier_score": brier,
        "sensitivity": sen,
        "specificity": spe,
    }
