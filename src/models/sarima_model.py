import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


# =====================================================
# Helper: find non-seasonal differencing d
# =====================================================
def find_d(series, max_d=2):
    d = 0
    temp = series.copy()

    while d <= max_d:
        try:
            if adfuller(temp.dropna())[1] < 0.05:
                return d
            temp = temp.diff().dropna()
            d += 1
        except:
            break

    return min(d, max_d)


# =====================================================
# Helper: find seasonal differencing D
# (SMART — avoids over-differencing on small data)
# =====================================================
def find_D(series, seasonal_period, max_D=1):
    """
    Only apply seasonal differencing if enough seasonal cycles exist.
    Rule: need at least 3 seasonal cycles.
    """
    if len(series) < seasonal_period * 3:
        return 0  # 🔥 CRITICAL FIX

    D = 0
    temp = series.copy()

    while D <= max_D:
        try:
            if adfuller(temp.dropna())[1] < 0.05:
                return D
            temp = temp.diff(seasonal_period).dropna()
            D += 1
        except:
            break

    return min(D, max_D)


# =====================================================
# AUTO SARIMA
# =====================================================
def run_sarima(train_series, test_series, seasonal_period=52):

    print("\n===== AUTO SARIMA =====")

    # -------------------------------------------------
    # STEP 1 — ensure weekly frequency
    # -------------------------------------------------
    train_series = train_series.asfreq("W-FRI")
    test_series = test_series.asfreq("W-FRI")

    # fill small gaps if any
    train_series = train_series.fillna(method="ffill")
    test_series = test_series.fillna(method="ffill")

    # -------------------------------------------------
    # STEP 2 — find differencing orders
    # -------------------------------------------------
    d = find_d(train_series)
    D = find_D(train_series, seasonal_period)

    print(f"Selected d = {d}")
    print(f"Selected D = {D}")

    # -------------------------------------------------
    # STEP 3 — grid search (AUTO TUNING)
    # -------------------------------------------------
    best_aic = float("inf")
    best_model = None
    best_order = None
    best_seasonal = None

    print("Searching best SARIMA parameters...")

    # search space (balanced for speed + accuracy)
    p_range = range(0, 3)
    q_range = range(0, 3)
    P_range = range(0, 2)
    Q_range = range(0, 2)

    for p in p_range:
        for q in q_range:
            for P in P_range:
                for Q in Q_range:

                    # skip totally empty model
                    if p == q == P == Q == 0:
                        continue

                    try:
                        model = SARIMAX(
                            train_series,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, seasonal_period),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )

                        fit = model.fit(disp=False)

                        if fit.aic < best_aic:
                            best_aic = fit.aic
                            best_model = fit
                            best_order = (p, d, q)
                            best_seasonal = (P, D, Q, seasonal_period)

                    except Exception:
                        continue

    # -------------------------------------------------
    # STEP 4 — safety fallback
    # -------------------------------------------------
    if best_model is None:
        print("⚠️ SARIMA fallback triggered")

        best_model = SARIMAX(
            train_series,
            order=(1, d, 1),
            seasonal_order=(0, D, 0, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        best_order = (1, d, 1)
        best_seasonal = (0, D, 0, seasonal_period)
        best_aic = best_model.aic

    print(f"Best order: {best_order}")
    print(f"Best seasonal order: {best_seasonal}")
    print(f"Best AIC: {best_aic:.2f}")

    # -------------------------------------------------
    # STEP 5 — forecast
    # -------------------------------------------------
    forecast = best_model.forecast(steps=len(test_series))

    # retail safety (no negative sales)
    forecast = forecast.clip(lower=0)

    return forecast