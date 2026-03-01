import pandas as pd


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag, rolling, and calendar features.
    Assumes Date is index and Weekly_Sales exists.
    """
    df = df.copy()

    # -----------------------
    # LAG FEATURES
    # -----------------------
    df["lag1"] = df["Weekly_Sales"].shift(1)
    df["lag7"] = df["Weekly_Sales"].shift(7)
    df["lag14"] = df["Weekly_Sales"].shift(14)

    # -----------------------
    # ROLLING FEATURES
    # -----------------------
    df["rolling_mean_7"] = df["Weekly_Sales"].rolling(window=7).mean()
    df["rolling_std_7"] = df["Weekly_Sales"].rolling(window=7).std()

    # -----------------------
    # DATE FEATURES
    # -----------------------
    df["month"] = df.index.month
    df["week"] = df.index.isocalendar().week.astype(int)
    df["dayofweek"] = df.index.dayofweek

    return df