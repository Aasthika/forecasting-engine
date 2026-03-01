import os
import sys
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from src.data_processing import load_and_filter_data
from src.features import create_time_features
from src.models.baseline_models import naive_forecast, moving_average_forecast
from src.models.arima_model import run_auto_arima
from src.models.sarima_model import run_sarima
from src.evaluation import evaluate_forecast

# =============================
# LOAD
# =============================

data_path = os.path.join(PROJECT_ROOT, "data", "walmart_sales.csv")
df = load_and_filter_data(data_path)

# =============================
# FEATURES
# =============================

df_feat = create_time_features(df).dropna()

target = df_feat["Weekly_Sales"]

train_size = int(len(target) * 0.8)
train = target.iloc[:train_size]
test = target.iloc[train_size:]

print("Train:", len(train), "Test:", len(test))

# =============================
# BASELINES
# =============================

naive_pred = naive_forecast(train, test)
ma_pred = moving_average_forecast(train, test)

evaluate_forecast(test, naive_pred, "Naive")
evaluate_forecast(test, ma_pred, "Moving Average")

# =============================
# ARIMA
# =============================

arima_pred = run_auto_arima(train, test)
evaluate_forecast(test, arima_pred, "Auto ARIMA")

# =============================
# SARIMA
# =============================

sarima_pred = run_sarima(train, test)
evaluate_forecast(test, sarima_pred, "SARIMA")

# =============================
# PLOT
# =============================

plt.figure(figsize=(12, 5))
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, arima_pred, label="ARIMA")
plt.plot(test.index, sarima_pred, label="SARIMA")
plt.legend()
plt.title("Model Comparison")
plt.show()