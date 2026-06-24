"""Create leakage-safe features and next-trading-day regression targets."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.dataset import (
    CLEAN_DATA_PATH,
    FEATURE_COLUMNS,
    FEATURE_SUMMARY_PATH,
    LATEST_FEATURES_PATH,
    METADATA_COLUMNS,
    MODEL_DATA_PATH,
    TARGET_COLUMNS,
)


REQUIRED_COLUMNS = [
    "날짜",
    "종가",
    "시가",
    "고가",
    "저가",
    "거래량",
    "RSI(14)",
    "MACD(12,26)",
    "MACD_SIGNAL(9)",
    "MACD_HIST",
    "외국인소진율",
]


def load_clean_data(path=CLEAN_DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data["날짜"] = pd.to_datetime(data["날짜"], errors="raise")
    data = data.sort_values("날짜", kind="stable").reset_index(drop=True)

    if data["날짜"].duplicated().any():
        raise ValueError("Duplicate dates found in clean data")
    if not data["날짜"].is_monotonic_increasing:
        raise ValueError("Clean data is not chronological")
    return data


def build_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return one row per source date before dropping warm-up/target NaNs."""
    close = data["종가"].astype(float)
    open_price = data["시가"].astype(float)
    high = data["고가"].astype(float)
    low = data["저가"].astype(float)
    volume = data["거래량"].astype(float)

    frame = pd.DataFrame(index=data.index)
    frame["date"] = data["날짜"].dt.strftime("%Y-%m-%d")
    frame["next_date"] = data["날짜"].shift(-1).dt.strftime("%Y-%m-%d")
    frame["current_close"] = close

    for period in (1, 2, 3, 5, 10, 20):
        frame[f"return_{period}d"] = close.pct_change(period)

    for period in (5, 10, 20, 60):
        moving_average = close.rolling(period, min_periods=period).mean()
        frame[f"close_to_ma_{period}"] = close / moving_average - 1

    daily_return = close.pct_change()
    for period in (5, 10, 20):
        frame[f"volatility_{period}d"] = daily_return.rolling(
            period, min_periods=period
        ).std(ddof=0)

    for period in (5, 10, 20):
        volume_average = volume.rolling(period, min_periods=period).mean()
        frame[f"volume_to_ma_{period}"] = volume / volume_average

    frame["intraday_return"] = close / open_price - 1
    frame["open_gap_return"] = open_price / close.shift(1) - 1
    frame["intraday_range"] = (high - low) / close

    price_range = high - low
    frame["close_position_in_range"] = np.where(
        price_range.eq(0),
        0.5,
        (close - low) / price_range,
    )

    frame["rsi_14"] = data["RSI(14)"].astype(float)
    frame["rsi_change_1d"] = frame["rsi_14"].diff()
    frame["macd"] = data["MACD(12,26)"].astype(float)
    frame["macd_signal"] = data["MACD_SIGNAL(9)"].astype(float)
    frame["macd_hist"] = data["MACD_HIST"].astype(float)
    frame["macd_change_1d"] = frame["macd"].diff()
    frame["macd_signal_change_1d"] = frame["macd_signal"].diff()
    frame["macd_hist_change_1d"] = frame["macd_hist"].diff()
    frame["foreign_ownership_pct"] = data["외국인소진율"].astype(float)
    frame["foreign_ownership_change_1d"] = frame[
        "foreign_ownership_pct"
    ].diff()
    frame["day_of_week"] = data["날짜"].dt.dayofweek
    frame["month"] = data["날짜"].dt.month

    frame["target_next_open"] = open_price.shift(-1)
    frame["target_next_close"] = close.shift(-1)
    frame["target_next_open_return"] = (
        frame["target_next_open"] / close - 1
    )
    frame["target_next_close_return"] = (
        frame["target_next_close"] / close - 1
    )

    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def build_outputs(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    full_frame = build_feature_frame(data)
    features_ready = full_frame[FEATURE_COLUMNS].notna().all(axis=1)
    targets_ready = full_frame[TARGET_COLUMNS].notna().all(axis=1)

    model_data = full_frame.loc[features_ready & targets_ready].copy()
    latest_candidates = full_frame.loc[features_ready & ~targets_ready].copy()
    if len(latest_candidates) != 1:
        raise ValueError(
            "Expected exactly one latest prediction row without targets, "
            f"found {len(latest_candidates)}"
        )

    latest_features = latest_candidates[
        ["date", "current_close", *FEATURE_COLUMNS]
    ].copy()
    model_data = model_data[
        [*METADATA_COLUMNS, *FEATURE_COLUMNS, *TARGET_COLUMNS]
    ]

    formulae = {
        "target_next_open_return": "target_next_open / current_close - 1",
        "target_next_close_return": "target_next_close / current_close - 1",
        "return_Nd": "current_close / close_N_trading_days_ago - 1",
        "close_to_ma_N": "current_close / rolling_mean_close_N - 1",
        "volatility_Nd": "population_std(return_1d, rolling_N)",
        "volume_to_ma_N": "current_volume / rolling_mean_volume_N",
        "intraday_return": "current_close / current_open - 1",
        "open_gap_return": "current_open / previous_close - 1",
        "intraday_range": "(current_high - current_low) / current_close",
        "close_position_in_range": (
            "(current_close - current_low) / "
            "(current_high - current_low); flat day = 0.5"
        ),
        "indicator_change_1d": "current_indicator - previous_indicator",
        "foreign_ownership_change_1d": (
            "current_foreign_ownership_pct - "
            "previous_foreign_ownership_pct"
        ),
    }
    summary = {
        "source_file": str(CLEAN_DATA_PATH),
        "model_dataset_file": str(MODEL_DATA_PATH),
        "latest_features_file": str(LATEST_FEATURES_PATH),
        "source_rows": int(len(data)),
        "model_rows": int(len(model_data)),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "metadata_columns": METADATA_COLUMNS,
        "warmup_or_incomplete_rows_removed": int(
            (~features_ready).sum()
        ),
        "rows_without_targets": int((~targets_ready).sum()),
        "model_date_min": model_data["date"].min(),
        "model_date_max": model_data["date"].max(),
        "latest_feature_date": latest_features.iloc[0]["date"],
        "latest_current_close": float(
            latest_features.iloc[0]["current_close"]
        ),
        "chronological_order": bool(
            pd.to_datetime(model_data["date"]).is_monotonic_increasing
        ),
        "remaining_missing_values_model": {
            column: int(count)
            for column, count in model_data.isna().sum().items()
        },
        "formulae": formulae,
        "leakage_controls": [
            "All rolling features use the current or earlier rows only.",
            "Targets are created only with shift(-1).",
            "The final row is excluded from training and saved for prediction.",
            "No backward fill or future-derived imputation is used.",
        ],
    }
    return model_data, latest_features, summary


def main() -> None:
    data = load_clean_data()
    model_data, latest_features, summary = build_outputs(data)

    MODEL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(MODEL_DATA_PATH, index=False, encoding="utf-8-sig")
    latest_features.to_csv(
        LATEST_FEATURES_PATH, index=False, encoding="utf-8-sig"
    )
    FEATURE_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
