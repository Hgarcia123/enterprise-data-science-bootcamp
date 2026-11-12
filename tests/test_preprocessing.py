"""Tests for src/preprocessing.py. Run them from the project folder with:

    python -m unittest

or only this file with `python -m tests.test_preprocessing`.
"""

import unittest

import pandas as pd

from src.preprocessing import (
    AGE_ORDER,
    DRUG_COLS,
    MED_DROP,
    MED_KEEP,
    build_med_features,
    group_icd9,
    preprocess_demographics,
    preprocess_diagnoses,
    preprocess_labs,
    preprocess_medication_summary,
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


class TestPreprocessDemographics(unittest.TestCase):
    """Checks on `preprocess_demographics`."""

    def setUp(self):
        """Build, before each test, a tiny table made by hand: five encounters.

        Missing race and weight are NaN, as after reading the CSV with na_values="?".
        """
        self.df = pd.DataFrame(
            {
                "race": [None, "Caucasian", "AfricanAmerican", None, "Asian"],
                "gender": ["Female", "Unknown/Invalid", "Male", "Female", "Male"],
                "age": ["[0-10)", "[90-100)", "[50-60)", "[0-10)", "[70-80)"],
                "weight": [None, "[75-100)", None, "[50-75)", None],
                "num_medications": [1, 2, 3, 4, 5],  # a column the block must not touch
            },
            index=pd.Index([101, 102, 103, 104, 105], name="encounter_id"),
        )

    def test_values_are_right_for_each_row(self):
        """race, gender, age and weight_recorded match the hand-computed values."""
        out = preprocess_demographics(self.df)
        self.assertEqual(
            out["race"].tolist(),
            ["Unknown", "Caucasian", "AfricanAmerican", "Unknown", "Asian"],
        )
        self.assertEqual(out["gender"].tolist(), ["Female", "Unknown", "Male", "Female", "Male"])
        self.assertEqual(out["age"].tolist(), [0, 9, 5, 0, 7])
        self.assertTrue(pd.api.types.is_integer_dtype(out["age"]))
        self.assertEqual(out["weight_recorded"].tolist(), [0, 1, 0, 1, 0])
        self.assertTrue(pd.api.types.is_integer_dtype(out["weight_recorded"]))

    def test_weight_is_dropped_and_rest_kept(self):
        """weight is gone; the index and the other columns are unchanged."""
        out = preprocess_demographics(self.df)
        self.assertNotIn("weight", out.columns)
        self.assertEqual(out.index.tolist(), self.df.index.tolist())
        self.assertEqual(out.index.name, "encounter_id")
        self.assertEqual(out["num_medications"].tolist(), [1, 2, 3, 4, 5])

    def test_age_order_has_ten_bands(self):
        """AGE_ORDER goes from [0-10) to [90-100)."""
        self.assertEqual(len(AGE_ORDER), 10)
        self.assertEqual(AGE_ORDER[0], "[0-10)")
        self.assertEqual(AGE_ORDER[-1], "[90-100)")

    def test_input_is_not_modified(self):
        """The DataFrame passed in is left exactly as it was."""
        before = self.df.copy()
        preprocess_demographics(self.df)
        pd.testing.assert_frame_equal(self.df, before)

    def test_wrong_age_value_raises_valueerror(self):
        """An age that is not a band label raises a ValueError that names it."""
        self.df.loc[103, "age"] = "[100-110)"
        with self.assertRaises(ValueError) as ctx:
            preprocess_demographics(self.df)
        self.assertIn("[100-110)", str(ctx.exception))

    def test_already_encoded_age_raises_valueerror(self):
        """Ages already encoded as 0-9 raise instead of becoming -1.

        Guards against the notebook bug, where age was encoded twice.
        """
        self.df["age"] = [0, 9, 5, 0, 7]
        with self.assertRaises(ValueError):
            preprocess_demographics(self.df)


class TestPreprocessDiagnoses(unittest.TestCase):
    """Checks on `group_icd9` and `preprocess_diagnoses`."""

    # ICD-9 code -> expected group, including the edges of each range
    CASES = {
        "250.83": "Diabetes",
        "428": "Circulatory",
        "785": "Circulatory",
        "786": "Respiratory",
        "787": "Digestive",
        "788": "Genitourinary",
        "140": "Neoplasms",
        "239": "Neoplasms",
        "240": "Other",
        "800": "Injury",
        "999": "Injury",
        "710": "Musculoskeletal",
        "739": "Musculoskeletal",
        "V57": "Other",
        "E909": "Other",
    }

    def setUp(self):
        """Build, before each test, a tiny table made by hand: four encounters.

        Missing codes are NaN, as after reading the CSV with na_values="?".
        """
        self.df = pd.DataFrame(
            {
                "diag_1": ["250.83", "428", "V57", "800"],
                "diag_2": ["786", None, "140", "E909"],
                "diag_3": ["240", "788", "739", None],
                "number_diagnoses": [9, 16, 10, 1],
                "race": ["Caucasian"] * 4,  # a column the block must not touch
            },
            index=pd.Index([201, 202, 203, 204], name="encounter_id"),
        )

    def test_group_icd9_known_codes(self):
        """Each code lands in its Strack et al. group, edges included."""
        for code, group in self.CASES.items():
            with self.subTest(code=code):
                self.assertEqual(group_icd9(code), group)

    def test_group_icd9_missing(self):
        """A missing code (None or NaN) becomes "Missing"."""
        self.assertEqual(group_icd9(None), "Missing")
        self.assertEqual(group_icd9(float("nan")), "Missing")

    def test_group_columns_are_right(self):
        """The 3 _group columns match the hand-computed groups."""
        out = preprocess_diagnoses(self.df)
        self.assertEqual(
            out["diag_1_group"].tolist(), ["Diabetes", "Circulatory", "Other", "Injury"]
        )
        self.assertEqual(
            out["diag_2_group"].tolist(), ["Respiratory", "Missing", "Neoplasms", "Other"]
        )
        self.assertEqual(
            out["diag_3_group"].tolist(), ["Other", "Genitourinary", "Musculoskeletal", "Missing"]
        )

    def test_number_diagnoses_capped_at_10(self):
        """9 stays 9, 16 becomes 10, 10 stays 10, 1 stays 1."""
        out = preprocess_diagnoses(self.df)
        self.assertEqual(out["number_diagnoses"].tolist(), [9, 10, 10, 1])

    def test_raw_diag_columns_are_gone(self):
        """diag_1, diag_2, diag_3 are dropped; the other columns stay."""
        out = preprocess_diagnoses(self.df)
        for col in ["diag_1", "diag_2", "diag_3"]:
            self.assertNotIn(col, out.columns)
        self.assertEqual(out["race"].tolist(), ["Caucasian"] * 4)

    def test_index_unchanged(self):
        """The index (encounter_id) is the same as the input's."""
        out = preprocess_diagnoses(self.df)
        self.assertEqual(out.index.tolist(), [201, 202, 203, 204])
        self.assertEqual(out.index.name, "encounter_id")

    def test_input_is_not_modified(self):
        """The DataFrame passed in is left exactly as it was."""
        before = self.df.copy()
        preprocess_diagnoses(self.df)
        pd.testing.assert_frame_equal(self.df, before)


class TestPreprocessLabs(unittest.TestCase):
    """Checks on `preprocess_labs`."""

    def setUp(self):
        """Build, before each test, a tiny table made by hand: five encounters.

        Covers every kind of value: two results, "Norm", "None" (test not
        performed) and a missing value.
        """
        self.df = pd.DataFrame(
            {
                "max_glu_serum": [">200", ">300", "Norm", "None", None],
                "A1Cresult": [">7", ">8", "Norm", "None", None],
                "age": [1, 2, 3, 4, 5],  # a column the block must not touch
            },
            index=pd.Index([301, 302, 303, 304, 305], name="encounter_id"),
        )

    def test_values_are_right(self):
        """"None" and missing become "Not measured"; results and "Norm" stay."""
        out = preprocess_labs(self.df)
        self.assertEqual(
            out["max_glu_serum"].tolist(),
            [">200", ">300", "Norm", "Not measured", "Not measured"],
        )
        self.assertEqual(
            out["A1Cresult"].tolist(), [">7", ">8", "Norm", "Not measured", "Not measured"]
        )

    def test_other_columns_and_index_unchanged(self):
        """The index and the other columns are the same as the input's."""
        out = preprocess_labs(self.df)
        self.assertEqual(out.index.tolist(), [301, 302, 303, 304, 305])
        self.assertEqual(out.index.name, "encounter_id")
        self.assertEqual(out["age"].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(list(out.columns), list(self.df.columns))

    def test_input_is_not_modified(self):
        """The DataFrame passed in is left exactly as it was."""
        before = self.df.copy()
        preprocess_labs(self.df)
        pd.testing.assert_frame_equal(self.df, before)


class TestPreprocessMedicationSummary(unittest.TestCase):
    """Checks on `preprocess_medication_summary`."""

    def setUp(self):
        """Build, before each test, a tiny table made by hand: four encounters."""
        self.df = pd.DataFrame(
            {
                "change": ["Ch", "No", "Ch", "No"],
                "diabetesMed": ["Yes", "No", "Yes", "Yes"],
                "insulin": ["Up", "No", "Steady", "No"],  # must not be touched
            },
            index=pd.Index([401, 402, 403, 404], name="encounter_id"),
        )

    def test_values_are_right(self):
        """Ch/Yes -> 1 and No -> 0, as whole numbers."""
        out = preprocess_medication_summary(self.df)
        self.assertEqual(out["change"].tolist(), [1, 0, 1, 0])
        self.assertEqual(out["diabetesMed"].tolist(), [1, 0, 1, 1])
        self.assertTrue(pd.api.types.is_integer_dtype(out["change"]))
        self.assertTrue(pd.api.types.is_integer_dtype(out["diabetesMed"]))

    def test_unexpected_value_raises_valueerror(self):
        """A value other than Ch/No or Yes/No raises a ValueError that names it."""
        self.df.loc[402, "diabetesMed"] = "Maybe"
        with self.assertRaises(ValueError) as ctx:
            preprocess_medication_summary(self.df)
        self.assertIn("diabetesMed", str(ctx.exception))
        self.assertIn("Maybe", str(ctx.exception))

    def test_missing_value_raises_valueerror(self):
        """A missing value raises too, instead of staying NaN."""
        self.df.loc[403, "change"] = None
        with self.assertRaises(ValueError):
            preprocess_medication_summary(self.df)

    def test_index_and_other_columns_unchanged(self):
        """The index and the other columns are the same as the input's."""
        out = preprocess_medication_summary(self.df)
        self.assertEqual(out.index.tolist(), [401, 402, 403, 404])
        self.assertEqual(out.index.name, "encounter_id")
        self.assertEqual(out["insulin"].tolist(), ["Up", "No", "Steady", "No"])

    def test_input_is_not_modified(self):
        """The DataFrame passed in is left exactly as it was."""
        before = self.df.copy()
        preprocess_medication_summary(self.df)
        pd.testing.assert_frame_equal(self.df, before)


if __name__ == "__main__":
    unittest.main()
