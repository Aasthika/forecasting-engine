import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from prophet import Prophet


def run_prophet(train_series, test_series):
    print("\n===== PROPHET MODEL =====")

    # -------------------------------------------------
    # STEP 1 — Ensure weekly frequency (IMPORTANT)
    # -------------------------------------------------
    train_series = train_series.asfreq("W-FRI")
    test_series = test_series.asfreq("W-FRI")

    # fill small gaps if any
    train_series = train_series.fillna(method="ffill")
    test_series = test_series.fillna(method="ffill")

    # -------------------------------------------------
    # STEP 2 — Prepare Prophet dataframe
    # Prophet requires columns: ds, y
    # -------------------------------------------------
    train_df = pd.DataFrame({
        "ds": train_series.index,
        "y": train_series.values
    })

    # -------------------------------------------------
    # STEP 3 — Build improved Prophet model ⭐
    # -------------------------------------------------
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",      # ⭐ BIG improvement for retail
        changepoint_prior_scale=0.1,            # ⭐ tuned flexibility
    )

    # OPTIONAL (safe to keep commented if not needed)
    # model.add_country_holidays(country_name='US')

    # -------------------------------------------------
    # STEP 4 — Fit model
    # -------------------------------------------------
    model.fit(train_df)

    # -------------------------------------------------
    # STEP 5 — Create future dataframe
    # -------------------------------------------------
    future = model.make_future_dataframe(
        periods=len(test_series),
        freq="W-FRI"
    )

    forecast_df = model.predict(future)

    # -------------------------------------------------
    # STEP 6 — Extract only test horizon
    # -------------------------------------------------
    forecast = forecast_df.set_index("ds")["yhat"]
    forecast = forecast.iloc[-len(test_series):]

    # -------------------------------------------------
    # STEP 7 — Retail safety
    # -------------------------------------------------
    forecast = forecast.clip(lower=0)

    return forecast