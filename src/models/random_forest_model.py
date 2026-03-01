import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# ======================================================
# Helper — prepare ML matrices safely
# ======================================================

def _prepare_ml_data(df_feat, train_index, test_index):
    """
    Prepares train/test matrices for ML models.
    
    Parameters
    ----------
    df_feat : pd.DataFrame
        Feature dataframe containing Weekly_Sales
    train_index : index
        Training index
    test_index : index
        Test index

    Returns
    -------
    X_train, X_test, y_train
    """

    # defensive copy
    df = df_feat.copy()

    # target
    if "Weekly_Sales" not in df.columns:
        raise ValueError("Weekly_Sales column missing in df_feat")

    y = df["Weekly_Sales"]
    X = df.drop(columns=["Weekly_Sales"])

    # split using time index (VERY IMPORTANT for time series)
    X_train = X.loc[train_index]
    X_test = X.loc[test_index]
    y_train = y.loc[train_index]

    return X_train, X_test, y_train


# ======================================================
# RANDOM FOREST MODEL
# ======================================================

def run_random_forest(df_feat, train_series, test_series):
    """
    Trains Random Forest and returns predictions aligned to test index.
    """

    print("\n===== RANDOM FOREST MODEL =====")

    # ------------------------------
    # Prepare data
    # ------------------------------
    X_train, X_test, y_train = _prepare_ml_data(
        df_feat,
        train_series.index,
        test_series.index,
    )

    # ------------------------------
    # Model (stable + interview-safe config)
    # ------------------------------
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    # ------------------------------
    # Train
    # ------------------------------
    model.fit(X_train, y_train)

    # ------------------------------
    # Predict
    # ------------------------------
    preds = model.predict(X_test)

    # no negative sales
    preds = np.clip(preds, a_min=0, a_max=None)

    # return as Series aligned with test index
    preds_series = pd.Series(preds, index=test_series.index, name="RF_Prediction")

    return preds_series