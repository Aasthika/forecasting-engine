import pandas as pd


# ======================================================
# AUTO MODEL SELECTOR
# ======================================================

def select_best_model(results_df):
    """
    Selects best model based on lowest MAPE.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns:
        ['Model', 'MAE', 'RMSE', 'MAPE (%)']

    Returns
    -------
    best_model_name : str
    best_row : pd.Series
    """

    if results_df.empty:
        raise ValueError("results_df is empty")

    best_row = results_df.sort_values("MAPE (%)").iloc[0]
    best_model_name = best_row["Model"]

    return best_model_name, best_row