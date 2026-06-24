import unittest

import numpy as np
import pandas as pd

from src.dataset import FEATURE_COLUMNS, TARGET_COLUMNS
from src.feature_engineering import build_feature_frame, build_outputs


def sample_data(rows: int = 80) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 100 + index
    return pd.DataFrame(
        {
            "날짜": pd.bdate_range("2020-01-01", periods=rows),
            "종가": close,
            "시가": close - 1,
            "고가": close + 2,
            "저가": close - 2,
            "거래량": 1_000 + index * 10,
            "RSI(14)": 40 + index * 0.1,
            "MACD(12,26)": index * 0.2,
            "MACD_SIGNAL(9)": index * 0.15,
            "MACD_HIST": index * 0.05,
            "외국인소진율": 50 + index * 0.01,
        }
    )


class FeatureEngineeringTests(unittest.TestCase):
    def test_next_day_targets_match_next_source_row(self):
        data = sample_data()
        frame = build_feature_frame(data)
        position = 65
        self.assertEqual(
            frame.loc[position, "target_next_open"],
            data.loc[position + 1, "시가"],
        )
        self.assertEqual(
            frame.loc[position, "target_next_close"],
            data.loc[position + 1, "종가"],
        )
        self.assertAlmostEqual(
            frame.loc[position, "target_next_close_return"],
            data.loc[position + 1, "종가"] / data.loc[position, "종가"] - 1,
        )

    def test_last_row_is_prediction_only(self):
        data = sample_data()
        model_data, latest, _ = build_outputs(data)
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest.iloc[0]["date"], "2020-04-21")
        self.assertNotIn("2020-04-21", set(model_data["date"]))

    def test_expected_features_and_targets_have_no_missing_values(self):
        model_data, latest, _ = build_outputs(sample_data())
        self.assertFalse(model_data[FEATURE_COLUMNS + TARGET_COLUMNS].isna().any().any())
        self.assertFalse(latest[FEATURE_COLUMNS].isna().any().any())

    def test_rolling_feature_does_not_change_when_future_rows_change(self):
        original = sample_data()
        changed = original.copy()
        changed.loc[70:, ["종가", "시가", "고가", "저가", "거래량"]] *= 100

        original_frame = build_feature_frame(original)
        changed_frame = build_feature_frame(changed)
        pd.testing.assert_series_equal(
            original_frame.loc[65, FEATURE_COLUMNS],
            changed_frame.loc[65, FEATURE_COLUMNS],
            check_names=False,
        )


if __name__ == "__main__":
    unittest.main()
