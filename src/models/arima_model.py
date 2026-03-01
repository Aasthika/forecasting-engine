import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA


# =====================================================
# Find differencing order d
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
# AUTO ARIMA
# =====================================================
def run_auto_arima(train_series, test_series):

    print("\n===== AUTO ARIMA =====")

    # ⭐ IMPORTANT — set frequency
    train_series = train_series.asfreq("W-FRI")
    test_series = test_series.asfreq("W-FRI")

    # -----------------------------
    # find d
    # -----------------------------
    d = find_d(train_series)
    print(f"Selected d = {d}")

    best_aic = float("inf")
    best_model = None
    best_order = None

    print("Searching best (p,d,q)...")

    for p in range(4):
        for q in range(4):

            # ⭐ stability guard (optional but good)
            if (p + q) > 5:
                continue

            try:
                model = ARIMA(train_series, order=(p, d, q))
                fit = model.fit()

                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_model = fit
                    best_order = (p, d, q)

            except Exception:
                continue

    print(f"Best ARIMA order: {best_order}")
    print(f"Best AIC: {best_aic:.2f}")

    forecast = best_model.forecast(steps=len(test_series))
    forecast = forecast.clip(lower=0)

    return forecast