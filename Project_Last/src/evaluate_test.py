"""One-time evaluation on the reserved independent test period."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import (
    FEATURE_COLUMNS,
    METRICS_DIR,
    MODEL_DATA_PATH,
    MODELS_DIR,
    PREDICTIONS_DIR,
    temporal_split,
)
from src.evaluate import regression_metrics
from src.train_baseline import TARGETS, rolling_mean_predictions


SELECTED_MODELS = {
    "open": "rolling_mean_20",
    "close": "naive",
}
RIDGE_ALPHAS = {
    "open": 1000.0,
    "close": 1000.0,
}


def make_ridge(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def main() -> None:
    data = pd.read_csv(MODEL_DATA_PATH, encoding="utf-8-sig")
    split = temporal_split(
        data,
        train_ratio=0.8,
        validation_ratio=0.1,
        purge_rows=1,
    )

    # Reuse all labels available before the test boundary. The raw row
    # immediately before the test is omitted because its target is in test.
    pretest_end = split.raw_validation_end - split.purge_rows
    pretest = data.iloc[:pretest_end].copy()
    test = split.test.copy()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, object]] = []
    predictions = test[["date", "next_date", "current_close"]].copy()

    for target_name, columns in TARGETS.items():
        return_column = columns["return"]
        price_column = columns["price"]
        y_pretest = pretest[return_column]
        y_test = test[return_column]
        actual_price = test[price_column]
        current_close = test["current_close"]

        ridge = make_ridge(RIDGE_ALPHAS[target_name])
        ridge.fit(pretest[FEATURE_COLUMNS], y_pretest)
        joblib.dump(
            ridge,
            MODELS_DIR / f"final_baseline_{target_name}_ridge.joblib",
        )

        model_predictions = {
            "naive": np.zeros(len(test)),
            "rolling_mean_20": rolling_mean_predictions(
                y_pretest, y_test, window=20
            ),
            "ridge": ridge.predict(test[FEATURE_COLUMNS]),
        }

        predictions[f"actual_{target_name}_return"] = y_test.to_numpy()
        predictions[f"actual_{target_name}_price"] = actual_price.to_numpy()

        for model_name, predicted_return in model_predictions.items():
            metrics = regression_metrics(
                actual_return=y_test,
                predicted_return=predicted_return,
                current_close=current_close,
                actual_price=actual_price,
            )
            metrics_rows.append(
                {
                    "model": model_name,
                    "target": target_name,
                    "selected_from_validation": (
                        SELECTED_MODELS[target_name] == model_name
                    ),
                    **metrics,
                }
            )
            predictions[
                f"predicted_{target_name}_return_{model_name}"
            ] = predicted_return
            predictions[
                f"predicted_{target_name}_price_{model_name}"
            ] = current_close.to_numpy() * (1 + predicted_return)

    metrics = pd.DataFrame(metrics_rows).sort_values(
        ["target", "price_mae", "model"]
    )
    metrics.to_csv(
        METRICS_DIR / "independent_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        PREDICTIONS_DIR / "independent_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected = metrics.loc[metrics["selected_from_validation"]].copy()
    summary = {
        "evaluation_type": "one_time_independent_test",
        "selection_rule": (
            "Models were selected by validation price MAE before test access."
        ),
        "selected_models": SELECTED_MODELS,
        "pretest_training": {
            "rows": int(len(pretest)),
            "date_min": str(pretest["date"].iloc[0]),
            "date_max": str(pretest["date"].iloc[-1]),
            "last_target_date": str(pretest["next_date"].iloc[-1]),
        },
        "test": {
            "rows": int(len(test)),
            "date_min": str(test["date"].iloc[0]),
            "date_max": str(test["date"].iloc[-1]),
        },
        "selected_model_test_metrics": selected.to_dict(orient="records"),
        "all_test_metrics": metrics.to_dict(orient="records"),
        "warning": (
            "The independent test has now been opened. It must not be reused "
            "for further model selection or hyperparameter tuning."
        ),
    }
    (METRICS_DIR / "independent_test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
