from pathlib import Path

import numpy as np
import pandas as pd


CSV_PATH = Path("005930_2016to2026.csv")


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    values = np.full(len(close), np.nan)

    if len(close) <= period:
        return pd.Series(values, index=close.index)

    average_gain = gain.iloc[1 : period + 1].mean()
    average_loss = loss.iloc[1 : period + 1].mean()

    def rsi_value(avg_gain: float, avg_loss: float) -> float:
        if avg_gain == 0 and avg_loss == 0:
            return 50.0
        if avg_loss == 0:
            return 100.0
        relative_strength = avg_gain / avg_loss
        return 100 - (100 / (1 + relative_strength))

    values[period] = rsi_value(average_gain, average_loss)

    for position in range(period + 1, len(close)):
        average_gain = (
            average_gain * (period - 1) + gain.iloc[position]
        ) / period
        average_loss = (
            average_loss * (period - 1) + loss.iloc[position]
        ) / period
        values[position] = rsi_value(average_gain, average_loss)

    return pd.Series(values, index=close.index)


def main() -> None:
    data = pd.read_csv(CSV_PATH, dtype=str, encoding="utf-8-sig")
    original_columns = [
        column
        for column in data.columns
        if column not in {"RSI(14)", "MACD(12,26)", "MACD_SIGNAL(9)", "MACD_HIST"}
    ]

    data["_date"] = pd.to_datetime(
        data["날짜"].str.replace(" ", "", regex=False), format="%Y-%m-%d"
    )
    data["_close"] = pd.to_numeric(
        data["종가"].str.replace(",", "", regex=False), errors="raise"
    )

    chronological = data.sort_values("_date").copy()
    close = chronological["_close"]

    chronological["RSI(14)"] = calculate_rsi(close, period=14)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    chronological["MACD(12,26)"] = ema_12 - ema_26
    chronological["MACD_SIGNAL(9)"] = chronological["MACD(12,26)"].ewm(
        span=9, adjust=False
    ).mean()
    chronological["MACD_HIST"] = (
        chronological["MACD(12,26)"] - chronological["MACD_SIGNAL(9)"]
    )

    result = chronological.sort_index()
    indicator_columns = ["RSI(14)", "MACD(12,26)", "MACD_SIGNAL(9)", "MACD_HIST"]
    for column in indicator_columns:
        result[column] = result[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )

    result[original_columns + indicator_columns].to_csv(
        CSV_PATH,
        index=False,
        encoding="utf-8-sig",
        quoting=1,
    )


if __name__ == "__main__":
    main()
