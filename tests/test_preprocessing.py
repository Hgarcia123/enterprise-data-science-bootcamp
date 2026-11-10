"""Tests for src/preprocessing.py. Run them from the project folder with:

    python -m unittest

or only this file with `python -m tests.test_preprocessing`.
"""

import unittest

import pandas as pd

from src.preprocessing import (
    DRUG_COLS,
    MED_DROP,
    MED_KEEP,
    build_med_features,
    preprocess_medications,
)

FEATURES = ["n_meds_taken", "n_dose_changes", "n_other_meds", "on_insulin_only"]


class TestPreprocessMedications(unittest.TestCase):
    """Checks on `build_med_features` and `preprocess_medications`."""

    def setUp(self):
        """Build, before each test, a tiny table made by hand: six encounters.

        Every one of the 23 drug columns is present and "No" unless set below.
        """
        self.df = pd.DataFrame({col: ["No"] * 6 for col in DRUG_COLS})
        self.df["age"] = ["[50-60)"] * 6  # a non-drug column, must be untouched

        # Row 0: no drugs at all.
        # Row 1: insulin only, steady.
        self.df.loc[1, "insulin"] = "Steady"
        # Row 2: insulin Up + metformin Down.
        self.df.loc[2, "insulin"] = "Up"
        self.df.loc[2, "metformin"] = "Down"
        # Row 3: a MED_DROP drug only.
        self.df.loc[3, "acarbose"] = "Steady"
        # Row 4: insulin + a MED_DROP drug with a dose change (so NOT insulin only).
        self.df.loc[4, "insulin"] = "Steady"
        self.df.loc[4, "nateglinide"] = "Up"
        # Row 5: metformin only.
        self.df.loc[5, "metformin"] = "Steady"

        self.expected = pd.DataFrame(
            {
                "n_meds_taken": [0, 1, 2, 1, 2, 1],
                "n_dose_changes": [0, 0, 2, 0, 1, 0],
                "n_other_meds": [0, 0, 0, 1, 1, 0],
                "on_insulin_only": [0, 1, 0, 0, 0, 0],
            }
        )

    def test_feature_values_are_right_for_each_row(self):
        """The 4 features match the hand-computed values, as whole numbers."""
        out = preprocess_medications(self.df)
        for col in FEATURES:
            self.assertEqual(out[col].tolist(), self.expected[col].tolist(), col)
            self.assertTrue(pd.api.types.is_integer_dtype(out[col]), col)

    def test_build_med_features_matches(self):
        """build_med_features gives the same values (on_insulin_only as bool)."""
        feats = build_med_features(self.df)
        self.assertEqual(list(feats.columns), FEATURES)
        for col in FEATURES:
            self.assertEqual(feats[col].astype(int).tolist(), self.expected[col].tolist(), col)

    def test_med_drop_columns_are_gone(self):
        """None of the 15 MED_DROP columns survive."""
        out = preprocess_medications(self.df)
        self.assertEqual([c for c in MED_DROP if c in out.columns], [])

    def test_med_keep_columns_remain_unchanged(self):
        """The 8 MED_KEEP columns are still there, with their raw string values."""
        out = preprocess_medications(self.df)
        for col in MED_KEEP:
            self.assertIn(col, out.columns)
            self.assertEqual(out[col].tolist(), self.df[col].tolist(), col)
        self.assertEqual(out["age"].tolist(), self.df["age"].tolist())

    def test_input_is_not_modified(self):
        """The DataFrame passed in is left exactly as it was."""
        before = self.df.copy()
        preprocess_medications(self.df)
        pd.testing.assert_frame_equal(self.df, before)

    def test_missing_column_raises_keyerror(self):
        """A missing drug column raises a KeyError that names it."""
        df = self.df.drop(columns=["insulin", "examide"])
        with self.assertRaises(KeyError) as ctx:
            preprocess_medications(df)
        self.assertIn("insulin", str(ctx.exception))
        self.assertIn("examide", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
