"""Preprocessing steps for the Diabetes 130-US hospitals readmission project.

This module holds, so far:
- the medication block: turns the 23 raw drug columns into 4 engineered
  count/flag features and drops the drugs that are too rare to model on their own;
- the sociodemographic block: race, gender, age and weight;
- the diagnoses block: ICD-9 codes grouped, number_diagnoses capped;
- the lab results block: tests not performed labelled "Not measured";
- the medication summary block: change and diabetesMed recoded to 1/0.
The other preprocessing blocks (admin, utilization) will be added here by the team.

The constants and the logic come from `notebooks/diabetes_predictive_model.ipynb`
(Medications section, and the Data Preprocessing part for the other blocks).
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

# The 10 age bands of the raw `age` column, youngest first; position = ordinal code 0-9.
AGE_ORDER = [f"[{i}-{i + 10})" for i in range(0, 100, 10)]

# The 3 diagnosis columns (ICD-9 codes): primary, secondary, additional secondary.
DIAG_COLS = ["diag_1", "diag_2", "diag_3"]

# The 2 lab result columns; "None" (or missing) means the test was not performed.
LAB_COLS = ["max_glu_serum", "A1Cresult"]

# The 2 medication summary columns and their 1/0 coding.
MED_SUMMARY_CODES = {
    "change": {"Ch": 1, "No": 0},
    "diabetesMed": {"Yes": 1, "No": 0},
}


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


def preprocess_demographics(df):
    """Apply the sociodemographic preprocessing block (race, gender, age, weight).

    Steps:
    1. race: missing -> "Unknown". Labelled rather than imputed, to avoid
       guessing a sensitive attribute (2.2% missing).
    2. gender: "Unknown/Invalid" -> "Unknown". The 3 rows are kept and gender is
       not inferred.
    3. age: band label -> ordinal code 0-9 (AGE_ORDER position). Age is only
       available in 10-year bands, and the code keeps their natural order.
       Done once, from the raw labels: a value that is not a band label (for
       example an age that was already encoded) raises an error instead of
       silently becoming -1, which is how the notebook's double encoding went
       unnoticed.
    4. weight: replaced by weight_recorded (1 = recorded, 0 = missing), then
       dropped. Weight is about 97% missing, so whether it was recorded is the
       only usable information.

    Args:
        df: DataFrame with the raw race, gender, age and weight columns.
            It is not modified.

    Returns:
        A copy of df (same index) with race, gender and age cleaned,
        weight_recorded added and weight removed.

    Raises:
        ValueError: if any age value is not one of the AGE_ORDER band labels;
            the message names the values.
    """
    out = df.copy()
    out["race"] = out["race"].fillna("Unknown")
    out["gender"] = out["gender"].replace("Unknown/Invalid", "Unknown")

    codes = out["age"].map({band: code for code, band in enumerate(AGE_ORDER)})
    if codes.isna().any():
        bad = sorted(map(str, out.loc[codes.isna(), "age"].unique()))
        raise ValueError(
            f"preprocess_demographics: age values that are not band labels {bad}; "
            f"expected one of {AGE_ORDER}"
        )
    out["age"] = codes.astype("int8")  # int8, as the notebook's .cat.codes

    out["weight_recorded"] = out["weight"].notna().astype(int)
    return out.drop(columns="weight")


def group_icd9(code):
    """Map one ICD-9 code to one of the diagnosis groups of Strack et al. (2014).

    The groups follow the Project Guide: Circulatory (390-459, 785), Respiratory
    (460-519, 786), Digestive (520-579, 787), Diabetes (250.xx), Injury (800-999),
    Musculoskeletal (710-739), Genitourinary (580-629, 788), Neoplasms (140-239),
    and Other for everything else. Only the integer part of the code is used
    (250.83 -> 250).

    Args:
        code: an ICD-9 code as a string (e.g. "428", "250.83", "V57") or a
            missing value.

    Returns:
        The group name. "Missing" for a missing code; "Other" for supplementary
        codes starting with V or E, which are not numeric.
    """
    # Missing code
    if pd.isna(code):
        return "Missing"
    code = str(code)
    # Supplementary codes (V, E) cannot be converted to a number
    if code.startswith(("V", "E")):
        return "Other"
    # Match on the integer part of the code (e.g. 250.83 -> 250)
    match int(float(code)):
        case 250:
            return "Diabetes"
        case n if 390 <= n < 460 or n == 785:
            return "Circulatory"
        case n if 460 <= n < 520 or n == 786:
            return "Respiratory"
        case n if 520 <= n < 580 or n == 787:
            return "Digestive"
        case n if 800 <= n < 1000:
            return "Injury"
        case n if 710 <= n < 740:
            return "Musculoskeletal"
        case n if 580 <= n < 630 or n == 788:
            return "Genitourinary"
        case n if 140 <= n < 240:
            return "Neoplasms"
        case _:
            return "Other"


def preprocess_diagnoses(df):
    """Apply the diagnoses preprocessing block (diag_1-3, number_diagnoses).

    Steps:
    1. diag_1, diag_2, diag_3 -> diag_1_group, diag_2_group, diag_3_group with
       `group_icd9`. Each column has 700+ distinct ICD-9 codes, too many to use
       directly; the 9 clinical groups of Strack et al. (2014) keep the meaning.
       Missing codes become their own group, "Missing".
    2. Drop the raw diag_1, diag_2, diag_3 (after grouping): they are replaced
       by the _group columns.
    3. number_diagnoses capped at 10 (values 10-16 become 10). The range is
       1-16 with a strong spike at 9 (about 49%) and very few encounters above 9,
       probably a recording limit in older systems; 10 then means "10 or more",
       kept separate from 9.

    Args:
        df: DataFrame with the raw diag_1, diag_2, diag_3 and number_diagnoses
            columns. It is not modified.

    Returns:
        A copy of df (same index) with the 3 _group columns added, the 3 raw
        diagnosis columns removed and number_diagnoses capped at 10.
    """
    out = df.copy()
    for col in DIAG_COLS:
        out[f"{col}_group"] = out[col].apply(group_icd9)
    out = out.drop(columns=DIAG_COLS)
    out["number_diagnoses"] = out["number_diagnoses"].clip(upper=10)
    return out


def preprocess_labs(df):
    """Apply the lab results preprocessing block (max_glu_serum, A1Cresult).

    Step: in both columns, "None" and missing values -> "Not measured".
    "None" means the test was not performed (about 95% of encounters for
    max_glu_serum and 83% for A1Cresult). It is labelled rather than imputed,
    because whether a test was ordered may itself be informative. Both columns
    stay text categories (">200", ">300", "Norm", "Not measured" and ">7", ">8",
    "Norm", "Not measured"); they are encoded later, in the modelling pipeline.

    Missing values (NaN) are also mapped, as in the notebook, although with
    `keep_default_na=False` the raw file has none: the absent tests are the
    string "None".

    Args:
        df: DataFrame with the raw max_glu_serum and A1Cresult columns.
            It is not modified.

    Returns:
        A copy of df (same index) with "None"/missing replaced by "Not measured"
        in the 2 lab columns.
    """
    out = df.copy()
    for col in LAB_COLS:
        out[col] = out[col].replace("None", "Not measured").fillna("Not measured")
    return out


def preprocess_medication_summary(df):
    """Apply the medication summary preprocessing block (change, diabetesMed).

    Step: recode both binary columns to 1/0 so the models can use them directly:
    change "Ch" -> 1, "No" -> 0 (diabetes medication changed during the
    encounter); diabetesMed "Yes" -> 1, "No" -> 0 (any diabetes medication
    prescribed). Both have no missing values in the raw data.

    Any other value raises an error. The notebook's `.map()` would silently turn
    it into NaN (and needed a separate check cell); here it fails loudly.

    diabetesMed is KEPT for now, although it is identical to n_meds_taken > 0
    (see the Medications EDA): dropping it is a team decision still pending.

    Args:
        df: DataFrame with the raw change and diabetesMed columns.
            It is not modified.

    Returns:
        A copy of df (same index) with change and diabetesMed as int 1/0.

    Raises:
        ValueError: if change or diabetesMed holds a value other than the
            expected ones; the message names the column and the values.
    """
    out = df.copy()
    for col, codes in MED_SUMMARY_CODES.items():
        unexpected = ~out[col].isin(list(codes))
        if unexpected.any():
            bad = sorted(map(str, out.loc[unexpected, col].unique()))
            raise ValueError(
                f"preprocess_medication_summary: unexpected values in {col}: {bad}; "
                f"expected one of {list(codes)}"
            )
        out[col] = out[col].map(codes).astype(int)
    return out
