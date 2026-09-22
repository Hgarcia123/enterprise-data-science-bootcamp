"""Tests for the code in src. Run them from the project folder with:

    python -m unittest

A test gives the code a small input where we KNOW the right answer, and checks it.
Each test is a method whose name starts with `test_`, inside a class that extends
`unittest.TestCase`. The last two lines let you also run this one file, with
`python -m tests.test_data`. Always from the project folder and always with `-m`:
`python tests/test_data.py` fails with "No module named src", because Python then
looks for `src` inside `tests`.
"""

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import FEATURES, TARGET, load_env
from src.features import make_features, make_X


class TestFeatures(unittest.TestCase):
    """Checks on `src.features.make_features`."""

    def setUp(self):
        """Build, before each test, a tiny table made by hand: two loans."""
        self.df = pd.DataFrame({name: [1.0, 2.0] for name in FEATURES})
        self.df[TARGET] = [1.0, 0.0]
        self.df["home_ownership"] = ["RENT", "OWN"]

    def test_X_has_only_the_feature_columns(self):
        """X keeps the FEATURES columns, in order, and nothing else."""
        X, _ = make_features(self.df)
        self.assertEqual(list(X.columns), FEATURES)

    def test_y_is_whole_numbers(self):
        """y holds whole numbers, 1 and 0, not the decimals 1.0 and 0.0.

        Two checks, because in Python `[1.0, 0.0] == [1, 0]` is True: comparing
        the values alone would not notice decimals. The second line checks the type.
        """
        _, y = make_features(self.df)
        self.assertEqual(y.tolist(), [1, 0])
        self.assertTrue(pd.api.types.is_integer_dtype(y))

    def test_X_does_not_need_the_target(self):
        """make_X works on a new application, where the outcome is not known."""
        applications = self.df.drop(columns=[TARGET])
        X = make_X(applications)
        self.assertEqual(list(X.columns), FEATURES)

    def test_no_rows_are_lost(self):
        """X and y have one row for each row that went in."""
        X, y = make_features(self.df)
        self.assertEqual(len(X), len(self.df))
        self.assertEqual(len(y), len(self.df))


class TestLoadEnv(unittest.TestCase):
    """Checks on `src.config.load_env`."""

    def test_reads_a_name_and_a_value(self):
        """A NAME = value line ends up in os.environ; a comment line does not break it."""
        with tempfile.TemporaryDirectory() as folder:
            settings = Path(folder) / "settings.txt"
            settings.write_text("# a comment\nCLASS03_TEST_SETTING = hello\n")
            load_env(settings)
        self.assertEqual(os.environ["CLASS03_TEST_SETTING"], "hello")


if __name__ == "__main__":
    unittest.main()
