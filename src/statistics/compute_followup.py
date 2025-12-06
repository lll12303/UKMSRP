import pandas as pd
import numpy as np

# Hard-coded censoring dates for synthetic dataset
REGION_CUTOFF = {
    "England": pd.to_datetime("2022-10-31"),
    "Scotland": pd.to_datetime("2022-08-31"),
    "Wales": pd.to_datetime("2022-05-31"),
}


def compute_followup_pipeline(df):
    """
    Compute follow-up for synthetic dataset.

    Required columns:
        - Participant ID
        - Region
        - baseline time
        - Date of death
        - status
    """

    df = df.copy()

    # Ensure datetime format
    df["baseline time"] = pd.to_datetime(df["baseline time"], errors="coerce")
    df["Date of death"] = pd.to_datetime(df["Date of death"], errors="coerce")

    # If status missing, derive it
    if "status" not in df.columns:
        df["status"] = df["Date of death"].notna().astype(int)

    # Map censoring date by region
    df["cutoff_date"] = df["Region"].map(REGION_CUTOFF)

    # Determine final follow-up time
    df["final_time"] = np.where(
        df["status"] == 1,
        df["Date of death"],
        df["cutoff_date"]
    )

    df["final_time"] = pd.to_datetime(df["final_time"], errors="coerce")

    # Compute follow-up in days
    df["time"] = (df["final_time"] - df["baseline time"]).dt.days
    df["time"] = df["time"].clip(lower=0)

    return df[["Participant ID", "time", "status"]]
