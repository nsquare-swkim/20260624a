"""Train and validate leakage-safe baseline and Ridge models."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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


RANDOM_SEED = 42
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
TARGETS = {
    "open": {
        "return": "target_next_open_return",
        "price": "target_next_open",
    },
    "close": {
        "return": "target_next_close_return",
        "price": "target_next_close",
    },
}


def ridge_search(
    features: pd.DataFrame, target: pd.Series
) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )
    time_series_cv = TimeSeriesSplit(n_splits=5, gap=1)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"ridge__alpha": ALPHAS},
        scoring="neg_mean_absolute_error",
        cv=time_series_cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    search.fit(features, target)
    return search


def rolling_mean_predictions(
    train_target: pd.Series,
    validation_target: pd.Series,
    window: int = 20,
) -> np.ndarray:
    """One-step walk-forward baseline using only already observed targets."""
    history = list(train_target.astype(float))
    predictions: list[float] = []
    for actual in validation_target.astype(float):
        predictions.append(float(np.mean(history[-window:])))
        history.append(float(actual))
    return np.asarray(predictions)


def split_summary(split) -> dict[str, object]:
    return {
        "strategy": "chronological_80_10_10_with_boundary_purge",
        "raw_boundaries": {
            "train_end_index_exclusive": split.raw_train_end,
            "validation_end_index_exclusive": split.raw_validation_end,
        },
        "purge_rows_per_train_validation_boundary": split.purge_rows,
        "train": {
            "rows": len(split.train),
            "date_min": split.train["date"].min().strftime("%Y-%m-%d"),
            "date_max": split.train["date"].max().strftime("%Y-%m-%d"),
            "last_target_date": split.train["next_date"]
            .max()
            .strftime("%Y-%m-%d"),
        },
        "validation": {
            "rows": len(split.validation),
            "date_min": split.validation["date"]
            .min()
            .strftime("%Y-%m-%d"),
            "date_max": split.validation["date"]
            .max()
            .strftime("%Y-%m-%d"),
            "last_target_date": split.validation["next_date"]
            .max()
            .strftime("%Y-%m-%d"),
        },
        "test": {
            "rows": len(split.test),
            "date_min": split.test["date"].min().strftime("%Y-%m-%d"),
            "date_max": split.test["date"].max().strftime("%Y-%m-%d"),
            "status": "reserved_not_evaluated",
        },
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "random_seed": RANDOM_SEED,
        "ridge_alpha_candidates": ALPHAS,
        "ridge_cv": "TimeSeriesSplit(n_splits=5, gap=1)",
    }


def main() -> None:
    data = pd.read_csv(MODEL_DATA_PATH, encoding="utf-8-sig")
    split = temporal_split(
        data,
        train_ratio=0.8,
        validation_ratio=0.1,
        purge_rows=1,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    x_train = split.train[FEATURE_COLUMNS]
    x_validation = split.validation[FEATURE_COLUMNS]
    metric_rows: list[dict[str, object]] = []
    cv_rows: list[pd.DataFrame] = []
    predictions = split.validation[
        ["date", "next_date", "current_close"]
    ].copy()

    for target_name, columns in TARGETS.items():
        return_column = columns["return"]
        price_column = columns["price"]
        y_train = split.train[return_column]
        y_validation = split.validation[return_column]
        actual_price = split.validation[price_column]
        current_close = split.validation["current_close"]

        model_predictions = {
            "naive": np.zeros(len(split.validation)),
            "rolling_mean_20": rolling_mean_predictions(
                y_train, y_validation, window=20
            ),
        }

        search = ridge_search(x_train, y_train)
        model_predictions["ridge"] = search.predict(x_validation)
        joblib.dump(
            search.best_estimator_,
            MODELS_DIR / f"baseline_{target_name}_ridge.joblib",
        )

        cv_result = pd.DataFrame(search.cv_results_)
        cv_result.insert(0, "target", target_name)
        cv_rows.append(
            cv_result[
                [
                    "target",
                    "param_ridge__alpha",
                    "mean_test_score",
                    "std_test_score",
                    "mean_train_score",
                    "rank_test_score",
                ]
            ].rename(
                columns={
                    "mean_test_score": "mean_cv_negative_return_mae",
                    "std_test_score": "std_cv_negative_return_mae",
                    "mean_train_score": "mean_train_negative_return_mae",
                }
            )
        )

        predictions[f"actual_{target_name}_return"] = y_validation.to_numpy()
        predictions[f"actual_{target_name}_price"] = actual_price.to_numpy()

        for model_name, predicted_return in model_predictions.items():
            metrics = regression_metrics(
                actual_return=y_validation,
                predicted_return=predicted_return,
                current_close=current_close,
                actual_price=actual_price,
            )
            metric_rows.append(
                {
                    "model": model_name,
                    "target": target_name,
                    **metrics,
                    "selected_alpha": (
                        float(search.best_params_["ridge__alpha"])
                        if model_name == "ridge"
                        else np.nan
                    ),
                }
            )
            predictions[
                f"predicted_{target_name}_return_{model_name}"
            ] = predicted_return
            predictions[
                f"predicted_{target_name}_price_{model_name}"
            ] = current_close.to_numpy() * (1 + predicted_return)

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["target", "price_mae", "model"]
    )
    metrics.to_csv(
        METRICS_DIR / "baseline_validation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(cv_rows, ignore_index=True).to_csv(
        METRICS_DIR / "ridge_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        PREDICTIONS_DIR / "baseline_validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = split_summary(split)
    summary["validation_metrics"] = metrics.to_dict(orient="records")
    (METRICS_DIR / "baseline_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
