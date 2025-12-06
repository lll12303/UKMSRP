"""
Run Cox survival analysis based on config file.
"""

import pandas as pd
import yaml
from src.statistics.cox_analysis import run_large_scale_cox


if __name__ == "__main__":

    # Load config
    with open("src/configs/cox_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(cfg["data_file"])

    # Remove columns if needed
    for col in cfg["drop_columns"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Run Cox analysis
    results = run_large_scale_cox(
        df=df,
        time_col=cfg["time_column"],
        status_col=cfg["status_column"],
        exclude_cols=cfg["exclude_columns"],
        n_jobs=cfg["n_jobs"],
        chunk_size=cfg["chunk_size"]
    )

    # Save results
    results.to_csv(cfg["output_file_full"], index=False)
    results[results["FDR"] < 0.05].to_csv(cfg["output_file_significant"], index=False)

    print(f"Results saved to:\n{cfg['output_file_full']}\n{cfg['output_file_significant']}")
