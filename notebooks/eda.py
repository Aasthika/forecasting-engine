import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use("default")

# --------------------------------------------------
# STEP 1 — Load dataset (ROBUST PATH)
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, "data", "walmart_sales.csv")

print("Reading from:", data_path)

df = pd.read_csv(data_path)

# --------------------------------------------------
# STEP 2 — Preview data
# --------------------------------------------------

print("\n===== HEAD =====")
print(df.head())

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== INFO =====")
print(df.info())

print("\n===== DESCRIBE =====")
print(df.describe())

# --------------------------------------------------
# STEP 3 — Convert Date
# --------------------------------------------------

df["Date"] = pd.to_datetime(df["Date"])

print("\n===== DTYPES AFTER CONVERSION =====")
print(df.dtypes)

# --------------------------------------------------
# STEP 4 — Filter single time series
# --------------------------------------------------

df = df[(df["Store"] == 1) & (df["Dept"] == 1)]

df = df.sort_values("Date")
df = df.set_index("Date")

print("\n===== AFTER FILTERING =====")
print(df.head())

print("\n===== DATE RANGE =====")
print(df.index.min(), "to", df.index.max())

print("\n===== FINAL SHAPE =====")
print(df.shape)

# --------------------------------------------------
# STEP 5 — Plot time series
# --------------------------------------------------

plt.figure(figsize=(12,5))
plt.plot(df["Weekly_Sales"])
plt.title("Weekly Sales — Store 1 Dept 1")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()


print("\n===== MISSING DATE CHECK =====")

full_range = pd.date_range(start=df.index.min(),
                           end=df.index.max(),
                           freq="W-FRI")

missing_dates = full_range.difference(df.index)

print("Number of missing weeks:", len(missing_dates))
print(missing_dates[:5])

print("\n===== DUPLICATE CHECK =====")
print("Duplicate timestamps:", df.index.duplicated().sum())

plt.figure(figsize=(6,4))
sns.boxplot(x=df["Weekly_Sales"])
plt.title("Outlier Check — Weekly Sales")
plt.show()

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(df["Weekly_Sales"], lags=60)
plt.title("ACF — Seasonality Check")
plt.show()


df["rolling_mean_12"] = df["Weekly_Sales"].rolling(12).mean()

plt.figure(figsize=(12,5))
plt.plot(df["Weekly_Sales"], label="Actual")
plt.plot(df["rolling_mean_12"], label="Rolling Mean (12)", color="red")
plt.legend()
plt.title("Trend Inspection")
plt.show()