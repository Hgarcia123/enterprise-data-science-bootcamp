"""Preprocessing steps for the Diabetes 130-US hospitals readmission project.

This module holds the medication block: it turns the 23 raw drug columns into
4 engineered count/flag features and drops the drugs that are too rare to model
on their own. The other preprocessing blocks (demographics, diagnoses, admission
info, ...) will be added here by the team.

The constants and the feature logic come from the Medications section of
`notebooks/diabetes_predictive_model.ipynb`.
"""

import pandas as pd

# The 23 drug columns, in the order they appear in the raw file
# (metformin -> metformin-pioglitazone). Each holds "No", "Steady", "Up" or "Down".
DRUG_COLS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Drugs taken (not "No") by at least 1% of encounters: kept as their own columns.
# Ordered from most to least used.
MED_KEEP = [
    "insulin",
    "metformin",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "glimepiride",
    "repaglinide",
]

# Drugs taken by less than 1% of encounters (examide and citoglipton are "No"
# everywhere): dropped as columns, but still counted in n_meds_taken,
# n_dose_changes and n_other_meds. Ordered from most to least used.
MED_DROP = [
    "glyburide-metformin",
    "nateglinide",
    "acarbose",
    "chlorpropamide",
    "tolazamide",
    "miglitol",
    "tolbutamide",
    "glipizide-metformin",
    "troglitazone",
    "metformin-rosiglitazone",
    "acetohexamide",
    "glimepiride-pioglitazone",
    "metformin-pioglitazone",
    "examide",
    "citoglipton",
]


def build_med_features(df, drug_cols=DRUG_COLS, med_drop=MED_DROP):
    """Build engineered medication features from the drug columns.

    Args:
        df: DataFrame holding every column in `drug_cols`, with values
            "No", "Steady", "Up" or "Down". It is not modified.
        drug_cols: the drug columns to read (default: the 23 in DRUG_COLS).
            Must include "insulin".
        med_drop: the rare drugs dropped by the 1% rule (default: MED_DROP).

    Returns:
        A new DataFrame with the same index as df and 4 columns:
        - n_meds_taken: how many drug columns are not "No" (Steady, Up or Down).
        - n_dose_changes: how many drug columns are "Up" or "Down"
          (dose changed this encounter).
        - n_other_meds: how many of the `med_drop` drugs are not "No"; keeps the
          rare drugs dropped by the 1% rule in the data as a count.
        - on_insulin_only: True when insulin is not "No" and no other drug
          column is taken.
    """
    taking = df[drug_cols] != "No"
    feats = pd.DataFrame(index=df.index)
    feats["n_meds_taken"] = taking.sum(axis=1)
    feats["n_dose_changes"] = df[drug_cols].isin(["Up", "Down"]).sum(axis=1)
    feats["n_other_meds"] = taking[med_drop].sum(axis=1)
    feats["on_insulin_only"] = taking["insulin"] & (
        taking.drop(columns="insulin").sum(axis=1) == 0
    )
    return feats


def preprocess_medications(df):
    """Apply the medication preprocessing block.

    Steps:
    1. Check that all 23 DRUG_COLS are present; a KeyError naming the missing
       columns is raised otherwise.
    2. Add the 4 engineered features from `build_med_features` as ints
       (n_meds_taken, n_dose_changes, n_other_meds, on_insulin_only). They are
       built from ALL 23 drug columns, before anything is dropped, so the
       information in the rare drugs is not lost.
    3. Drop the 15 MED_DROP columns. These drugs are taken by fewer than 1% of
       encounters (two are "No" everywhere): too few users to estimate a reliable
       readmission rate per level, so as separate columns they mostly add noise.
       Their use still counts in the features from step 2.
    4. Keep the 8 MED_KEEP drug columns unchanged, as the raw strings
       "No"/"Steady"/"Up"/"Down". Encoding happens later, in the model pipeline.

    Args:
        df: DataFrame with the raw drug columns. It is not modified.

    Returns:
        A copy of df with the 4 features added and the MED_DROP columns removed.

    Raises:
        KeyError: if any of the DRUG_COLS columns is missing from df.
    """
    missing = [col for col in DRUG_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"preprocess_medications: missing drug columns {missing}")

    out = df.copy()
    feats = build_med_features(out)
    for col in feats.columns:
        out[col] = feats[col].astype(int)
    return out.drop(columns=MED_DROP)
