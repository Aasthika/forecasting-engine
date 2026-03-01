# ===================== IMPORTS =====================
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# allow src imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from src.data_processing import load_and_filter_data
from src.features import create_time_features
from src.models.baseline_models import (
    naive_forecast,
    moving_average_forecast,
)
from src.models.arima_model import run_auto_arima
from src.models.sarima_model import run_sarima

# ✅ NEW — Prophet import (ADDED ONLY)
from src.models.prophet_model import run_prophet
from src.models.random_forest_model import run_random_forest
from src.models.xgboost_model import run_xgboost
from src.walk_forward import walk_forward_validation
from src.model_selection import select_best_model
from src.drift import detect_drift

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Retail Forecasting Engine",
    page_icon="📈",
    layout="wide",
)

st.title("📊 Retail Sales Forecasting Platform")

# =========================================================
# ===================== METRIC FUNCTION ===================
# =========================================================
def compute_metrics(actual, pred):
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return mae, rmse, mape


def show_metrics(actual, pred, model_name):
    mae, rmse, mape = compute_metrics(actual, pred)

    st.subheader(f"📊 {model_name} Performance")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:,.2f}")
    c2.metric("RMSE", f"{rmse:,.2f}")
    c3.metric("MAPE (%)", f"{mape:.2f}%")

    return mae, rmse, mape


# ===================== DATA LOADING =====================
@st.cache_data
def load_pipeline_data():
    data_path = os.path.join(PROJECT_ROOT, "data", "walmart_sales.csv")

    df = load_and_filter_data(data_path)
    df_feat = create_time_features(df).dropna()

    target = df_feat["Weekly_Sales"]

    train_size = int(len(target) * 0.8)
    train = target.iloc[:train_size]
    test = target.iloc[train_size:]

    return df, df_feat, target, train, test


(df, df_feat, target, train, test) = load_pipeline_data()

# ===================== SIDEBAR =====================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Data Overview",
        "Exploratory Analysis",
        "Baseline Models",
        "ARIMA Model",
        "SARIMA Model",
        "Prophet Model",
        "Random Forest Model",
        "XGBoost Model",
        "Walk-Forward Validation",
        "Model Comparison",
        "Auto Model Selection (Walk-Forward)",  

    ],
)

# =========================================================
# ===================== DATA OVERVIEW =====================
# =========================================================
if page == "Data Overview":
    st.header("📋 Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(df))
    c2.metric("Date Start", str(df.index.min().date()))
    c3.metric("Date End", str(df.index.max().date()))

    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Statistics")
    st.dataframe(df.describe())


# =========================================================
# ===================== EDA ===============================
# =========================================================
elif page == "Exploratory Analysis":
    st.header("🔍 Time Series Analysis")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Weekly_Sales"])
    ax.set_title("Weekly Sales Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Outlier Check")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=df["Weekly_Sales"], ax=ax)
    st.pyplot(fig)


# =========================================================
# ===================== BASELINES =========================
# =========================================================
elif page == "Baseline Models":
    st.header("📉 Baseline Forecasts")

    if st.button("Run Baselines"):

        naive_pred = naive_forecast(train, test)
        ma_pred = moving_average_forecast(train, test)

        show_metrics(test, naive_pred, "Naive")
        show_metrics(test, ma_pred, "Moving Average")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(test.index, test, label="Actual")
        ax.plot(test.index, naive_pred, label="Naive")
        ax.plot(test.index, ma_pred, label="Moving Avg")
        ax.legend()
        ax.set_title("Baseline Comparison")
        st.pyplot(fig)


# =========================================================
# ===================== ARIMA ==============================
# =========================================================
elif page == "ARIMA Model":
    st.header("🤖 Auto ARIMA")

    if st.button("Run ARIMA"):

        arima_pred = run_auto_arima(train, test)
        show_metrics(test, arima_pred, "Auto ARIMA")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(test.index, test, label="Actual")
        ax.plot(test.index, arima_pred, label="ARIMA Forecast")
        ax.legend()
        ax.set_title("ARIMA Forecast vs Actual")
        st.pyplot(fig)


# =========================================================
# ===================== SARIMA =============================
# =========================================================
elif page == "SARIMA Model":
    st.header("📊 Seasonal ARIMA")

    if st.button("Run SARIMA"):

        sarima_pred = run_sarima(train, test)
        show_metrics(test, sarima_pred, "SARIMA")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(test.index, test, label="Actual")
        ax.plot(test.index, sarima_pred, label="SARIMA Forecast")
        ax.legend()
        ax.set_title("SARIMA Forecast vs Actual")
        st.pyplot(fig)


# =========================================================
# ===================== PROPHET ============================
# =========================================================
elif page == "Prophet Model":
    st.header("🔮 Prophet Forecast")

    if st.button("Run Prophet"):

        prophet_pred = run_prophet(train, test)
        show_metrics(test, prophet_pred, "Prophet")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(test.index, test, label="Actual")
        ax.plot(test.index, prophet_pred, label="Prophet Forecast")
        ax.legend()
        ax.set_title("Prophet Forecast vs Actual")
        st.pyplot(fig)


# =========================================================
# ===================== RANDOM FOREST =====================
# =========================================================
elif page == "Random Forest Model":
    st.header("🌲 Random Forest Forecast")

    if st.button("Run Random Forest"):

        rf_pred = run_random_forest(df_feat, train, test)
        show_metrics(test, rf_pred, "Random Forest")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(test.index, test, label="Actual")
        ax.plot(test.index, rf_pred, label="Random Forest")
        ax.legend()
        ax.set_title("Random Forest Forecast vs Actual")
        st.pyplot(fig)


# =========================================================
# ===================== XGBOOST ===========================
# =========================================================
elif page == "XGBoost Model":
    st.header("🚀 XGBoost Forecast")

    if st.button("Run XGBoost"):

        xgb_pred = run_xgboost(df_feat, train, test)
        show_metrics(test, xgb_pred, "XGBoost")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(test.index, test, label="Actual")
        ax.plot(test.index, xgb_pred, label="XGBoost")
        ax.legend()
        ax.set_title("XGBoost Forecast vs Actual")
        st.pyplot(fig)


# =========================================================
# ===================== WALK-FORWARD ======================
# =========================================================
elif page == "Walk-Forward Validation":
    st.header("🔁 Walk-Forward Validation")

    model_choice = st.selectbox(
        "Select model",
        [
            "SARIMA",
            "ARIMA",
            "Prophet",
            "Random Forest",
            "XGBoost",
        ],
    )

    if st.button("Run Walk-Forward"):

        if model_choice == "SARIMA":
            wf_results, wf_preds = walk_forward_validation(
                target,
                run_sarima,
                model_name="SARIMA",
            )

        elif model_choice == "ARIMA":
            wf_results, wf_preds = walk_forward_validation(
                target,
                run_auto_arima,
                model_name="ARIMA",
            )

        elif model_choice == "Prophet":
            wf_results, wf_preds = walk_forward_validation(
                target,
                run_prophet,
                model_name="Prophet",
            )

        elif model_choice == "Random Forest":
            wf_results, wf_preds = walk_forward_validation(
                target,
                run_random_forest,
                df_feat=df_feat,
                model_name="Random Forest",
            )

        elif model_choice == "XGBoost":
            wf_results, wf_preds = walk_forward_validation(
                target,
                run_xgboost,
                df_feat=df_feat,
                model_name="XGBoost",
            )

        st.subheader("📊 Walk-Forward Metrics")
        st.dataframe(wf_results)

        if not wf_preds.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(target.index, target, label="Actual")
            ax.plot(wf_preds.index, wf_preds, label="WF Prediction")
            ax.legend()
            ax.set_title("Walk-Forward Forecast")
            st.pyplot(fig)


# =========================================================
# ===================== MODEL COMPARISON ==================
# =========================================================
elif page == "Model Comparison":
    st.header("🏆 Model Comparison")

    if st.button("Run All Models"):

        naive_pred = naive_forecast(train, test)
        arima_pred = run_auto_arima(train, test)
        sarima_pred = run_sarima(train, test)
        prophet_pred = run_prophet(train, test)
        rf_pred = run_random_forest(df_feat, train, test)
        xgb_pred = run_xgboost(df_feat, train, test)

        results = []

        for name, pred in [
            ("Naive", naive_pred),
            ("ARIMA", arima_pred),
            ("SARIMA", sarima_pred),
            ("Prophet", prophet_pred),
            ("Random Forest", rf_pred),
            ("XGBoost", xgb_pred),
        ]:
            mae, rmse, mape = compute_metrics(test, pred)
            results.append([name, mae, rmse, mape])

        results_df = pd.DataFrame(
            results, columns=["Model", "MAE", "RMSE", "MAPE (%)"]
        )

        st.subheader("📊 Metrics Comparison")
        st.dataframe(results_df)

        best_model, best_row = select_best_model(results_df)

        st.success(f"🏆 Best Model: **{best_model}**")
        st.info(f"Best MAPE: {best_row['MAPE (%)']:.2f}%")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(test.index, test, label="Actual")
        ax.plot(test.index, naive_pred, label="Naive")
        ax.plot(test.index, arima_pred, label="ARIMA")
        ax.plot(test.index, sarima_pred, label="SARIMA")
        ax.plot(test.index, prophet_pred, label="Prophet")
        ax.plot(test.index, rf_pred, label="Random Forest")
        ax.plot(test.index, xgb_pred, label="XGBoost")
        ax.legend()
        ax.set_title("All Models Comparison")
        st.pyplot(fig)

        st.success("✅ Pipeline executed successfully!")

        # ================= DRIFT CHECK =================
        st.subheader("🚨 Drift Monitoring")

        best_pred_map = {
            "Naive": naive_pred,
            "ARIMA": arima_pred,
            "SARIMA": sarima_pred,
            "Prophet": prophet_pred,
            "Random Forest": rf_pred,
            "XGBoost": xgb_pred,
        }

        best_pred_series = best_pred_map[best_model]

        drift_flag, drift_msg, rolling_mape = detect_drift(test, best_pred_series)

        if drift_flag:
            st.error(drift_msg)
        else:
            st.success(drift_msg)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(rolling_mape.index, rolling_mape)
        ax.set_title("Rolling MAPE (Drift Monitor)")
        ax.set_ylabel("MAPE (%)")
        ax.grid(True)
        st.pyplot(fig)

# =========================================================
# ========== AUTO MODEL SELECTION (WALK-FORWARD) ==========
# =========================================================
elif page == "Auto Model Selection (Walk-Forward)":
    st.header("🤖 Auto Model Selection (Walk-Forward)")

    if st.button("Run Auto Selection (WF)"):

        model_results = []

        # ---------------- SARIMA ----------------
        with st.spinner("Running SARIMA walk-forward..."):
            sarima_res, _ = walk_forward_validation(
                target,
                run_sarima,
                model_name="SARIMA",
            )
            model_results.append(sarima_res.iloc[0])

        # ---------------- ARIMA ----------------
        with st.spinner("Running ARIMA walk-forward..."):
            arima_res, _ = walk_forward_validation(
                target,
                run_auto_arima,
                model_name="ARIMA",
            )
            model_results.append(arima_res.iloc[0])

        # ---------------- Prophet ----------------
        with st.spinner("Running Prophet walk-forward..."):
            prophet_res, _ = walk_forward_validation(
                target,
                run_prophet,
                model_name="Prophet",
            )
            model_results.append(prophet_res.iloc[0])

        # ---------------- Random Forest ----------------
        with st.spinner("Running Random Forest walk-forward..."):
            rf_res, _ = walk_forward_validation(
                target,
                run_random_forest,
                df_feat=df_feat,
                model_name="Random Forest",
            )
            model_results.append(rf_res.iloc[0])

        # ---------------- XGBoost ----------------
        with st.spinner("Running XGBoost walk-forward..."):
            xgb_res, _ = walk_forward_validation(
                target,
                run_xgboost,
                df_feat=df_feat,
                model_name="XGBoost",
            )
            model_results.append(xgb_res.iloc[0])

        # ================= RESULTS =================
        wf_df = pd.DataFrame(model_results)

        st.subheader("📊 Walk-Forward Model Comparison")
        st.dataframe(wf_df)

        # ================= BEST MODEL =================
        best_model = wf_df.sort_values("MAPE (%)").iloc[0]["Model"]
        best_row = wf_df.sort_values("MAPE (%)").iloc[0]

        st.success(f"🏆 Best Model (Walk-Forward): **{best_model}**")
        st.info(f"Best MAPE: {best_row['MAPE (%)']:.2f}%")

        st.markdown(
            """
            
            This selection is based on walk-forward validation, which
            simulates real production forecasting.
            """
        )