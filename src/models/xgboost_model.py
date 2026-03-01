import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError(
        "XGBoost is not installed. Run: pip install xgboost"
    )


# ======================================================
# Helper — prepare ML matrices safely
# ======================================================

def _prepare_ml_data(df_feat, train_index, test_index):
    df = df_feat.copy()

    if "Weekly_Sales" not in df.columns:
        raise ValueError("Weekly_Sales column missing in df_feat")

    y = df["Weekly_Sales"]
    X = df.drop(columns=["Weekly_Sales"])

    X_train = X.loc[train_index]
    X_test = X.loc[test_index]
    y_train = y.loc[train_index]

    return X_train, X_test, y_train


# ======================================================
# XGBOOST MODEL
# ======================================================

def run_xgboost(df_feat, train_series, test_series):

    print("\n===== XGBOOST MODEL =====")

    # prepare data
    X_train, X_test, y_train = _prepare_ml_data(
        df_feat,
        train_series.index,
        test_series.index,
    )

    # strong baseline config
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = np.clip(preds, a_min=0, a_max=None)

    return pd.Series(preds, index=test_series.index, name="XGB_Prediction")