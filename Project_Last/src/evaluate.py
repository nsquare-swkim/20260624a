"""Evaluation helpers for next-day return and restored-price forecasts."""

from __future__ import annotations

import numpy as np


def regression_metrics(
    actual_return,
    predicted_return,
    current_close,
    actual_price,
) -> dict[str, float]:
    actual_return = np.asarray(actual_return, dtype=float)
    predicted_return = np.asarray(predicted_return, dtype=float)
    current_close = np.asarray(current_close, dtype=float)
    actual_price = np.asarray(actual_price, dtype=float)
    predicted_price = current_close * (1 + predicted_return)

    price_error = predicted_price - actual_price
    denominator = np.abs(actual_price) + np.abs(predicted_price)
    smape = np.mean(
        np.where(denominator == 0, 0.0, 2 * np.abs(price_error) / denominator)
    )

    return {
        "price_mae": float(np.mean(np.abs(price_error))),
        "price_rmse": float(np.sqrt(np.mean(np.square(price_error)))),
        "price_smape_pct": float(smape * 100),
        "return_mae": float(
            np.mean(np.abs(predicted_return - actual_return))
        ),
        "direction_accuracy": float(
            np.mean(np.sign(predicted_return) == np.sign(actual_return))
        ),
    }
