import numpy as np
import pandas as pd


def naive_forecast(train: pd.Series, test: pd.Series) -> np.ndarray:
    """Forecast = last observed value"""
    last_value = train.iloc[-1]
    return np.repeat(last_value, len(test))


def moving_average_forecast(
    train: pd.Series,
    test: pd.Series,
    window: int = 7
) -> np.ndarray:
    """Forecast = mean of last window values"""
    mean_value = train.iloc[-window:].mean()
    return np.repeat(mean_value, len(test))