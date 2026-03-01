'''import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# ======================================================
# Helper — create sequences
# ======================================================
def create_sequences(series, window=12):
    """
    Convert time series into supervised LSTM sequences.
    """
    X, y = [], []

    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])

    return np.array(X), np.array(y)


# ======================================================
# LSTM MODEL
# ======================================================
def run_lstm(train_series, test_series, window=12):

    print("\n===== LSTM MODEL =====")

    # --------------------------------------------------
    # STEP 1 — Scale data (VERY IMPORTANT)
    # --------------------------------------------------
    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(
        train_series.values.reshape(-1, 1)
    )

    # --------------------------------------------------
    # STEP 2 — Create training sequences
    # --------------------------------------------------
    X_train, y_train = create_sequences(train_scaled, window)

    # Safety check (important for small data)
    if len(X_train) == 0:
        raise ValueError(
            "Not enough data to create LSTM sequences. "
            "Try reducing window size."
        )

    # reshape for LSTM → [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # --------------------------------------------------
    # STEP 3 — Build model (smaller = better for your data)
    # --------------------------------------------------
    model = Sequential([
        LSTM(32, activation="tanh", input_shape=(window, 1)),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    # --------------------------------------------------
    # STEP 4 — Early stopping (PRODUCTION STYLE ⭐)
    # --------------------------------------------------
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # --------------------------------------------------
    # STEP 5 — Train
    # --------------------------------------------------
    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=16,
        verbose=1,
        validation_split=0.2,   # ⭐ CRITICAL FIX
        callbacks=[early_stop]
    )

    # --------------------------------------------------
    # STEP 6 — Rolling forecast
    # --------------------------------------------------
    history = list(train_scaled.flatten())
    preds = []

    for _ in range(len(test_series)):
        x_input = np.array(history[-window:])
        x_input = x_input.reshape((1, window, 1))

        yhat = model.predict(x_input, verbose=0)[0][0]
        preds.append(yhat)

        history.append(yhat)

    # --------------------------------------------------
    # STEP 7 — Inverse scale
    # --------------------------------------------------
    preds = scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    ).flatten()

    # retail safety
    preds = np.clip(preds, a_min=0, a_max=None)

    return pd.Series(
        preds,
        index=test_series.index,
        name="LSTM_Prediction"
    )
'''