import pandas as pd


def load_and_filter_data(path: str) -> pd.DataFrame:
    """
    Load Walmart data and filter single time series.
    """
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df[(df["Store"] == 1) & (df["Dept"] == 1)]
    df = df.sort_values("Date").set_index("Date")

    return df