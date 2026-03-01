import numpy as np
import pandas as pd


# ======================================================
# Rolling MAPE
# ======================================================
def compute_rolling_mape(actual, pred, window=8):
    """
    Computes rolling MAPE over time.

    Parameters
    ----------
    actual : pd.Series
    pred : pd.Series
    window : int

    Returns
    -------
    pd.Series
    """

    ape = np.abs((actual - pred) / actual) * 100
    rolling_mape = ape.rolling(window=window).mean()

    return rolling_mape


# ======================================================
# Drift Detection Logic
# ======================================================
def detect_drift(actual, pred, window=8, threshold_increase=20):
    """
    Detects model performance drift.

    Rule:
    If recent rolling MAPE increases by threshold → drift alert.

    Returns
    -------
    drift_flag : bool
    message : str
    rolling_mape : pd.Series
    """

    rolling_mape = compute_rolling_mape(actual, pred, window)

    if len(rolling_mape.dropna()) < 2:
        return False, "Not enough data for drift detection", rolling_mape

    recent = rolling_mape.dropna().iloc[-1]
    previous = rolling_mape.dropna().iloc[0]

    increase_pct = ((recent - previous) / previous) * 100

    if increase_pct > threshold_increase:
        return True, "⚠️ Drift detected — consider retraining", rolling_mape
    else:
        return False, "✅ No significant drift", rolling_mape