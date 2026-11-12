"""Tests for src/preprocessing.py. Run them from the project folder with:

    python -m unittest

or only this file with `python -m tests.test_preprocessing`.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.preprocessing import (
    AGE_ORDER,
    DRUG_COLS,
    MED_DROP,
    MED_KEEP,
    add_utilization_features,
    build_med_features,
    group_icd9,
    load_id_maps,
    preprocess_admin,
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


class TestLoadIdMaps(unittest.TestCase):
    """Checks on `load_id_maps`, with a small file in the same 3-table layout."""

    CSV = (
        "admission_type_id,description\n"
        "1,Emergency\n"
        "6,NULL\n"
        "8,Not Mapped\n"
        ",\n"
        "discharge_disposition_id,description\n"
        "1,Discharged to home\n"
        '19,"Expired at home. Medicaid only, hospice."\n'
        ",\n"
        "admission_source_id,description\n"
        "1, Physician Referral\n"
        "26,Transfer from Hospice\n"
    )

    def setUp(self):
        """Write the small CSV to a temporary folder and read it."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "IDS_mapping.csv"
            path.write_text(self.CSV)
            self.maps = load_id_maps(path)

    def test_three_tables(self):
        """One map per id column, with all its rows."""
        self.assertEqual(
            sorted(self.maps),
            ["admission_source_id", "admission_type_id", "discharge_disposition_id"],
        )
        self.assertEqual(self.maps["admission_type_id"].index.tolist(), [1, 6, 8])
        self.assertEqual(self.maps["discharge_disposition_id"].index.tolist(), [1, 19])

    def test_last_row_of_last_table_is_kept(self):
        """Admission source 26 (the file's last row) is kept: the notebook lost it."""
        self.assertEqual(self.maps["admission_source_id"].index.tolist(), [1, 26])
        self.assertEqual(self.maps["admission_source_id"][26], "Transfer from Hospice")

    def test_descriptions_stripped_and_null_missing(self):
        """Spaces around descriptions are removed; "NULL" becomes missing."""
        self.assertEqual(self.maps["admission_source_id"][1], "Physician Referral")
        self.assertTrue(pd.isna(self.maps["admission_type_id"][6]))
        self.assertEqual(
            self.maps["discharge_disposition_id"][19],
            "Expired at home. Medicaid only, hospice.",
        )


class TestPreprocessAdmin(unittest.TestCase):
    """Checks on `preprocess_admin`, with a tiny handmade id_maps."""

    def setUp(self):
        """Build, before each test, tiny maps and five encounters.

        Ids 6 (type) and 18 (discharge) have a missing description, like "NULL"
        in the real file; id 99 is not in the maps at all.
        """
        self.id_maps = {
            "admission_type_id": pd.Series(
                {1: "Emergency", 5: "Not Available", 6: None, 8: "Not Mapped"}
            ),
            "admission_source_id": pd.Series(
                {
                    1: "Physician Referral",
                    4: "Transfer from a hospital",
                    7: "Emergency Room",
                    17: None,
                    20: "Not Mapped",
                    26: "Transfer from Hospice",
                }
            ),
            "discharge_disposition_id": pd.Series(
                {
                    1: "Discharged to home",
                    11: "Expired",
                    13: "Hospice / home",
                    18: None,
                    25: "Not Mapped",
                }
            ),
        }
        self.df = pd.DataFrame(
            {
                "admission_type_id": [1, 5, 6, 8, 99],
                "admission_source_id": [1, 4, 26, 17, 20],
                "discharge_disposition_id": [1, 11, 13, 18, 25],
                "medical_specialty": ["Cardiology", None, "Surgery-General", None, "Nephrology"],
                "payer_code": [None, "MC", None, "BC", "SP"],
                "race": ["Caucasian"] * 5,  # a column the block must not touch
            },
            index=pd.Index([501, 502, 503, 504, 505], name="encounter_id"),
        )

    def test_admission_type(self):
        """Mapped names stay; Not Available, NULL, Not Mapped and unknown ids -> Unknown."""
        out = preprocess_admin(self.df, self.id_maps)
        self.assertEqual(
            out["admission_type_name"].tolist(),
            ["Emergency", "Unknown", "Unknown", "Unknown", "Unknown"],
        )

    def test_admission_source(self):
        """Transfer sources grouped (26 included); NULL and Not Mapped -> Unknown."""
        out = preprocess_admin(self.df, self.id_maps)
        self.assertEqual(
            out["admission_source_name"].tolist(),
            [
                "Physician Referral",
                "Transfer from a healthcare facility",
                "Transfer from a healthcare facility",
                "Unknown",
                "Unknown",
            ],
        )

    def test_discharge(self):
        """Expired and Hospice grouped; NULL and Not Mapped -> Unknown."""
        out = preprocess_admin(self.df, self.id_maps)
        self.assertEqual(
            out["discharge_disposition_name"].tolist(),
            ["Discharged to home", "Expired/Hospice", "Expired/Hospice", "Unknown", "Unknown"],
        )

    def test_specialty_and_payer_missing_to_unknown(self):
        """Missing -> Unknown; other values unchanged (no "<1% -> Other" here)."""
        out = preprocess_admin(self.df, self.id_maps)
        self.assertEqual(
            out["medical_specialty"].tolist(),
            ["Cardiology", "Unknown", "Surgery-General", "Unknown", "Nephrology"],
        )
        self.assertEqual(out["payer_code"].tolist(), ["Unknown", "MC", "Unknown", "BC", "SP"])

    def test_index_is_kept(self):
        """The encounter_id index is kept (the notebook's .merge() lost it)."""
        out = preprocess_admin(self.df, self.id_maps)
        self.assertEqual(out.index.tolist(), [501, 502, 503, 504, 505])
        self.assertEqual(out.index.name, "encounter_id")

    def test_id_columns_replaced_by_names(self):
        """The 3 _id columns are gone, the 3 _name columns are there, others kept."""
        out = preprocess_admin(self.df, self.id_maps)
        for col in ["admission_type_id", "admission_source_id", "discharge_disposition_id"]:
            self.assertNotIn(col, out.columns)
        for col in ["admission_type_name", "admission_source_name", "discharge_disposition_name"]:
            self.assertIn(col, out.columns)
        self.assertEqual(out["race"].tolist(), ["Caucasian"] * 5)

    def test_input_is_not_modified(self):
        """The DataFrame passed in is left exactly as it was."""
        before = self.df.copy()
        preprocess_admin(self.df, self.id_maps)
        pd.testing.assert_frame_equal(self.df, before)


class TestAddUtilizationFeatures(unittest.TestCase):
    """Checks on `add_utilization_features`."""

    def setUp(self):
        """Build, before each test, a tiny table made by hand: four encounters."""
        self.df = pd.DataFrame(
            {
                "number_outpatient": [0, 2, 1, 0],
                "number_emergency": [0, 1, 0, 3],
                "number_inpatient": [0, 3, 1, 0],
                "num_lab_procedures": [10, 45, 7, 60],
                "time_in_hospital": [1, 3, 2, 4],
                "race": ["Caucasian"] * 4,  # a column the block must not touch
            },
            index=pd.Index([601, 602, 603, 604], name="encounter_id"),
        )

    def test_total_prior_visits(self):
        """outpatient + emergency + inpatient, as whole numbers."""
        out = add_utilization_features(self.df)
        self.assertEqual(out["total_prior_visits"].tolist(), [0, 6, 2, 3])
        self.assertTrue(pd.api.types.is_integer_dtype(out["total_prior_visits"]))

    def test_lab_procedures_per_day(self):
        """num_lab_procedures / time_in_hospital, as decimals."""
        out = add_utilization_features(self.df)
        self.assertEqual(out["lab_procedures_per_day"].tolist(), [10.0, 15.0, 3.5, 15.0])
        self.assertTrue(pd.api.types.is_float_dtype(out["lab_procedures_per_day"]))

    def test_is_prior_inpatient(self):
        """0 when number_inpatient is 0, 1 when it is 1 or more."""
        out = add_utilization_features(self.df)
        self.assertEqual(out["is_prior_inpatient"].tolist(), [0, 1, 1, 0])
        self.assertTrue(pd.api.types.is_integer_dtype(out["is_prior_inpatient"]))

    def test_zero_days_raises_valueerror(self):
        """time_in_hospital = 0 raises instead of dividing by zero."""
        self.df.loc[602, "time_in_hospital"] = 0
        with self.assertRaises(ValueError):
            add_utilization_features(self.df)

    def test_missing_days_raises_valueerror(self):
        """A missing time_in_hospital raises too."""
        self.df["time_in_hospital"] = [1, None, 2, 4]
        with self.assertRaises(ValueError):
            add_utilization_features(self.df)

    def test_missing_column_raises_keyerror(self):
        """A missing input column raises a KeyError that names it."""
        df = self.df.drop(columns=["number_emergency"])
        with self.assertRaises(KeyError) as ctx:
            add_utilization_features(df)
        self.assertIn("number_emergency", str(ctx.exception))

    def test_index_and_other_columns_kept(self):
        """The index and the other columns are the same as the input's."""
        out = add_utilization_features(self.df)
        self.assertEqual(out.index.tolist(), [601, 602, 603, 604])
        self.assertEqual(out.index.name, "encounter_id")
        self.assertEqual(out["race"].tolist(), ["Caucasian"] * 4)
        self.assertEqual(out["number_inpatient"].tolist(), [0, 3, 1, 0])

    def test_input_is_not_modified(self):
        """The DataFrame passed in is left exactly as it was."""
        before = self.df.copy()
        add_utilization_features(self.df)
        pd.testing.assert_frame_equal(self.df, before)


if __name__ == "__main__":
    unittest.main()
