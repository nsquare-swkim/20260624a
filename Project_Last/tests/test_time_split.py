import unittest

import pandas as pd

from src.dataset import temporal_split


def sample_model_data(rows: int = 100) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=rows + 1)
    return pd.DataFrame(
        {
            "date": dates[:-1],
            "next_date": dates[1:],
            "value": range(rows),
        }
    )


class TemporalSplitTests(unittest.TestCase):
    def test_80_10_10_split_with_one_row_purge(self):
        split = temporal_split(sample_model_data(), purge_rows=1)
        self.assertEqual(len(split.train), 79)
        self.assertEqual(len(split.validation), 9)
        self.assertEqual(len(split.test), 10)

    def test_labels_do_not_cross_boundaries(self):
        split = temporal_split(sample_model_data(), purge_rows=1)
        self.assertLess(
            split.train["next_date"].max(),
            split.validation["date"].min(),
        )
        self.assertLess(
            split.validation["next_date"].max(),
            split.test["date"].min(),
        )

    def test_each_split_is_chronological(self):
        split = temporal_split(sample_model_data(), purge_rows=1)
        for frame in (split.train, split.validation, split.test):
            self.assertTrue(frame["date"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
