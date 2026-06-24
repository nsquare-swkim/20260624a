"""Shared dataset paths and column definitions."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


CLEAN_DATA_PATH = Path("data/processed/005930_2016to2026_clean.csv")
MODEL_DATA_PATH = Path("data/processed/005930_model_dataset.csv")
LATEST_FEATURES_PATH = Path("data/processed/005930_latest_features.csv")
FEATURE_SUMMARY_PATH = Path("reports/feature_summary.json")
MODELS_DIR = Path("models")
METRICS_DIR = Path("reports/metrics")
PREDICTIONS_DIR = Path("reports/predictions")

TARGET_COLUMNS = [
    "target_next_open",
    "target_next_close",
    "target_next_open_return",
    "target_next_close_return",
]

METADATA_COLUMNS = [
    "date",
    "next_date",
    "current_close",
]

FEATURE_COLUMNS = [
    "return_1d",
    "return_2d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "close_to_ma_5",
    "close_to_ma_10",
    "close_to_ma_20",
    "close_to_ma_60",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volume_to_ma_5",
    "volume_to_ma_10",
    "volume_to_ma_20",
    "intraday_return",
    "open_gap_return",
    "intraday_range",
    "close_position_in_range",
    "rsi_14",
    "rsi_change_1d",
    "macd",
    "macd_signal",
    "macd_hist",
    "macd_change_1d",
    "macd_signal_change_1d",
    "macd_hist_change_1d",
    "foreign_ownership_pct",
    "foreign_ownership_change_1d",
    "day_of_week",
    "month",
]


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    raw_train_end: int
    raw_validation_end: int
    purge_rows: int


def temporal_split(
    data: pd.DataFrame,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    purge_rows: int = 1,
) -> TemporalSplit:
    """Split chronologically and purge labels crossing split boundaries."""
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("Split ratios must be positive")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("A positive test fraction is required")
    if purge_rows < 0:
        raise ValueError("purge_rows cannot be negative")

    ordered = data.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="raise")
    ordered["next_date"] = pd.to_datetime(
        ordered["next_date"], errors="raise"
    )
    ordered = ordered.sort_values("date", kind="stable").reset_index(drop=True)

    row_count = len(ordered)
    raw_train_end = int(row_count * train_ratio)
    raw_validation_end = int(
        row_count * (train_ratio + validation_ratio)
    )
    if raw_train_end <= purge_rows:
        raise ValueError("Training split is too small for the purge")
    if raw_validation_end - raw_train_end <= purge_rows:
        raise ValueError("Validation split is too small for the purge")

    train = ordered.iloc[: raw_train_end - purge_rows].copy()
    validation = ordered.iloc[
        raw_train_end : raw_validation_end - purge_rows
    ].copy()
    test = ordered.iloc[raw_validation_end:].copy()

    if train["next_date"].max() >= validation["date"].min():
        raise ValueError("Training labels overlap the validation period")
    if validation["next_date"].max() >= test["date"].min():
        raise ValueError("Validation labels overlap the test period")

    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
        raw_train_end=raw_train_end,
        raw_validation_end=raw_validation_end,
        purge_rows=purge_rows,
    )
