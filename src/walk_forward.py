import numpy as np
import pandas as pd


# =========================================================
# METRICS
# =========================================================
def compute_metrics(actual, pred):
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return mae, rmse, mape


# =========================================================
# UNIVERSAL WALK-FORWARD ENGINE
# =========================================================
def walk_forward_validation(
    full_series,
    model_func,
    df_feat=None,
    initial_train_size=0.6,
    step=4,
    model_name="Model",
):
    """
    Universal walk-forward validation.

    Works for:
    - ARIMA
    - SARIMA
    - Prophet
    - Random Forest
    - XGBoost
    """

    print("\n===== WALK-FORWARD VALIDATION =====")

    n_total = len(full_series)
    train_end = int(n_total * initial_train_size)

    all_preds = []
    all_actuals = []
    all_index = []

    iteration = 1

    # -----------------------------------------------------
    # EXPANDING WINDOW LOOP
    # -----------------------------------------------------
    while train_end < n_total:

        print(f"🔁 Iteration {iteration}")

        train_series = full_series.iloc[:train_end]
        test_series = full_series.iloc[train_end : train_end + step]

        if len(test_series) == 0:
            break

        # ---------------------------------------------
        # MODEL CALL (SMART ROUTING)
        # ---------------------------------------------
        try:
            # ML models need df_feat
            if df_feat is not None and model_name in [
                "Random Forest",
                "XGBoost",
            ]:
                preds = model_func(df_feat, train_series, test_series)

            else:
                preds = model_func(train_series, test_series)

        except Exception as e:
            print(f"⚠️ Iteration failed: {e}")
            break

        # ---------------------------------------------
        # STORE RESULTS
        # ---------------------------------------------
        preds = pd.Series(preds, index=test_series.index)

        all_preds.extend(preds.values)
        all_actuals.extend(test_series.values)
        all_index.extend(test_series.index)

        # expand window
        train_end += step
        iteration += 1

    # -----------------------------------------------------
    # FINAL METRICS
    # -----------------------------------------------------
    preds_series = pd.Series(all_preds, index=all_index, name="WF_Pred")

    mae, rmse, mape = compute_metrics(
        np.array(all_actuals),
        np.array(all_preds),
    )

    results_df = pd.DataFrame(
        {
            "Model": [model_name],
            "MAE": [mae],
            "RMSE": [rmse],
            "MAPE (%)": [mape],
        }
    )

    print("\n===== WALK-FORWARD RESULTS =====")
    print(results_df)

    return results_df, preds_series