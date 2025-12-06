"""
Data Loading Utilities
-----------------------
Minimal module for reading metadata and feature matrices.
"""

import pandas as pd

def load_feature_data(feature_file):
    return pd.read_csv(feature_file)

def load_metadata(raw_file):
    return pd.read_csv(raw_file)
