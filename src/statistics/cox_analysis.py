"""
Cox Proportional Hazards Analysis Module
=======================================

This script implements large-scale Cox regression for omics data
(e.g., proteomics or metabolomics). It includes:

- Standardization of features
- Individual Cox PH model per variable
- Parallel computation for efficiency
- Extraction of HR, CI, and p-values
- FDR correction (Benjamini–Hochberg)

Author: Your Name
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
import os
import time


def _fit_single_cox(df, variable, time_col, status_col):
    """Fit Cox model for a single variable."""
    try:
        data = df[[time_col, status_col, variable]].dropna()

        if len(data) < 100:
            raise ValueError(f"Insufficient sample size (n={len(data)})")

        # Standardization
        data[variable] = (data[variable] - data[variable].mean()) / data[variable].std()

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(data, duration_col=time_col, event_col=status_col)

        hr = cph.hazard_ratios_[variable]
        ci = np.exp(cph.confidence_intervals_.loc[variable])
        p_value = cph.summary.loc[variable, "p"]

        return {
            "variable": variable,
            "HR": hr,
            "CI_lower": ci[0],
            "CI_upper": ci[1],
            "p_value": p_value,
            "n_samples": len(data),
            "-log10(p)": -np.log10(p_value)
        }

    except Exception as e:
        return {"variable": variable, "error": str(e)}



def run_large_scale_cox(df, time_col, status_col, exclude_cols=None, n_jobs=-1, chunk_size=100):
    """
    Run Cox regression on omics data.

    Parameters
    ----------
    df : DataFrame
        Input data containing survival time, status, and omics variables.
    time_col : str
        Name of follow-up time column.
    status_col : str
        Name of event status column (0/1).
    exclude_cols : list
        Columns to exclude from Cox modeling.
    n_jobs : int
        Number of CPU cores for parallel computing.
    chunk_size : int
        Batch size for processing variables.

    Returns
    -------
    results_df : DataFrame
        Cox regression summary table.
    """

    start = time.time()
    exclude_cols = exclude_cols or []

    variables = [col for col in df.columns if col not in [time_col, status_col] + exclude_cols]

    batches = [variables[i:i + chunk_size] for i in range(0, len(variables), chunk_size)]

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(lambda batch: [
            _fit_single_cox(df, var, time_col, status_col) for var in batch
        ])(batch) for batch in batches
    )

    flat_results = [item for sublist in results for item in sublist]
    results_df = pd.DataFrame([item for item in flat_results if "error" not in item])

    # FDR correction
    _, fdr, _, _ = multipletests(results_df["p_value"], method="fdr_bh")
    results_df["FDR"] = fdr
    results_df["significance"] = np.where(results_df["FDR"] < 0.05, "**",
                                          np.where(results_df["p_value"] < 0.05, "*", ""))

    elapsed = (time.time() - start) / 3600
    print(f"Cox analysis successfully finished. Time used: {elapsed:.2f} hours.")

    return results_df.sort_values("p_value")
