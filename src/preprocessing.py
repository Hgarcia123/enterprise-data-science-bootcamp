"""Preprocessing steps for the Diabetes 130-US hospitals readmission project.

This module holds, so far:
- the medication block: turns the 23 raw drug columns into 4 engineered
  count/flag features and drops the drugs that are too rare to model on their own;
- the sociodemographic block: race, gender, age and weight;
- the diagnoses block: ICD-9 codes grouped, number_diagnoses capped;
- the lab results block: tests not performed labelled "Not measured";
- the medication summary block: change and diabetesMed recoded to 1/0;
- the administrative block: admission/discharge ids decoded with IDS_mapping.csv
  and grouped (fixed rules only; the learned groupings come later);
- the utilization block: 3 features from the visit counts and the stay;
- build_clean_table: the fixed-rule blocks in order, giving one table X and y.
- make_target and split_by_patient: the 0/1 target and the train / validation /
  test split grouped by patient;
- LearnedPreprocessor: the learned steps (caps, "<1% -> Other", discharge
  grouping), fitted on the training set only, then the utilization features.

The constants and the logic come from `notebooks/diabetes_predictive_model.ipynb`
(Medications section, and the Data Preprocessing part for the other blocks).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.validation import check_is_fitted

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

# The 3 id columns decoded with IDS_mapping.csv -> the name column that replaces each.
ADMIN_ID_COLS = {
    "admission_type_id": "admission_type_name",
    "admission_source_id": "admission_source_name",
    "discharge_disposition_id": "discharge_disposition_name",
}

# Raw discharge_disposition_id values for patients who died (11, 19, 20, 21: "Expired...")
# or went to hospice (13, 14). They cannot be readmitted, so keeping them would add
# encounters that are "not readmitted" for a reason unrelated to the care given and
# bias the model. Removing them is the default (team decision pending).
EXPIRED_HOSPICE_IDS = (11, 13, 14, 19, 20, 21)

# The target column of the raw file: "NO", ">30" or "<30".
TARGET_COL = "readmitted"

# The readmitted value that is the positive class (early readmission).
POSITIVE_CLASS = "<30"

# The team seed (restructure guide); default random_state of the split.
SEED = 81


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


def load_id_maps(path):
    """Read IDS_mapping.csv into one id -> description map per admin id column.

    The file holds 3 tables one after the other (admission_type_id,
    discharge_disposition_id, admission_source_id). Each starts with its own
    header row ("<name>,description") and is separated from the next by a blank
    row (","). The file is read as the notebook does (no header, 2 columns,
    pandas' default missing values), so "NULL" descriptions become missing, as
    today.

    Unlike the notebook, the last row of every table is kept: the notebook
    dropped the last row of each table to remove the blank separator, but the
    last table has no separator after it, so it lost admission source 26
    ("Transfer from Hospice"). Here the blank rows are removed instead.

    Args:
        path: path to IDS_mapping.csv.

    Returns:
        A dict with keys "admission_type_id", "discharge_disposition_id" and
        "admission_source_id". Each value is a Series indexed by the integer id,
        holding the description with surrounding spaces removed (missing for
        "NULL").
    """
    raw = pd.read_csv(path, header=None, names=range(2), delimiter=",")
    raw = raw[raw[0].notna()]  # drop the blank separator rows
    is_header = raw[0].isin(ADMIN_ID_COLS)
    maps = {}
    for _, table in raw.groupby(is_header.cumsum()):
        name = table.iloc[0, 0]
        rows = table.iloc[1:]
        maps[name] = pd.Series(
            rows[1].str.strip().values,
            index=rows[0].astype(int).values,
            name="description",
        ).rename_axis(name)
    return maps


def preprocess_admin(df, id_maps):
    """Apply the administrative preprocessing block (fixed rules only).

    Steps:
    1. admission_type_id, admission_source_id, discharge_disposition_id ->
       admission_type_name, admission_source_name, discharge_disposition_name,
       looked up in `id_maps`; the 3 _id columns are dropped. The ids are codes
       with no order, so the names are kept as categories (one-hot encoded in the
       modelling pipeline). Done with .map(), so the encounter_id index is kept
       (the notebook's .merge() reset it to 0..n-1).
    2. Admission type and source: "Not Mapped", "Not Available" and missing
       (ids with a NULL description) -> "Unknown"; these labels carry no
       information, so they are treated as missing.
    3. Admission source: every name containing "Transfer from" -> "Transfer from
       a healthcare facility", to reduce cardinality (the transfer sources are
       small and similar).
    4. Discharge: "Not Mapped" and missing -> "Unknown"; every name containing
       "expired" or "hospice" -> "Expired/Hospice". These patients cannot be
       readmitted and can bias the model; they are only grouped here (removing
       the rows is a separate, optional step).
    5. medical_specialty and payer_code: missing -> "Unknown". Inferring them is
       not worth it (weak correlation with the other variables, risk of bias).

    Not done here: grouping discharge names with fewer than 3,000 rows, and
    medical_specialty / payer_code values under 1% -> "Other". Those thresholds
    are learned from the data, so they are fitted on the training set later.

    Args:
        df: DataFrame with the 3 raw _id columns, medical_specialty and
            payer_code. It is not modified.
        id_maps: the dict returned by `load_id_maps`.

    Returns:
        A copy of df (same index) with the 3 _name columns added, the 3 _id
        columns removed and the rules above applied.
    """
    out = df.copy()
    for id_col, name_col in ADMIN_ID_COLS.items():
        out[name_col] = out[id_col].map(id_maps[id_col])
    out = out.drop(columns=list(ADMIN_ID_COLS))

    not_informative = {"Not Mapped": "Unknown", "Not Available": "Unknown"}
    for col in ["admission_type_name", "admission_source_name"]:
        out[col] = out[col].replace(not_informative).fillna("Unknown")
    is_transfer = out["admission_source_name"].str.contains("Transfer from", case=False)
    out.loc[is_transfer, "admission_source_name"] = "Transfer from a healthcare facility"

    col = "discharge_disposition_name"
    out[col] = out[col].replace("Not Mapped", "Unknown").fillna("Unknown")
    is_expired = out[col].str.contains("expired|hospice", case=False)
    out.loc[is_expired, col] = "Expired/Hospice"

    for col in ["medical_specialty", "payer_code"]:
        out[col] = out[col].fillna("Unknown")
    return out


# Inputs of the utilization features.
UTILIZATION_INPUT_COLS = [
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "num_lab_procedures",
    "time_in_hospital",
]


def add_utilization_features(df):
    """Add the 3 utilization features (utilization history and current stay).

    Features:
    - total_prior_visits = number_outpatient + number_emergency + number_inpatient
      (int): the patient's whole utilization history in one number.
    - lab_procedures_per_day = num_lab_procedures / time_in_hospital (float):
      how many lab procedures per day of stay, a sign of how severe the
      patient's condition is.
    - is_prior_inpatient = 1 if number_inpatient > 0, else 0 (int): whether the
      patient was admitted to hospital before, again a sign of severity.

    Order in the pipeline: as in the notebook, these features are built AFTER
    the 99th-percentile caps on number_outpatient, number_emergency,
    number_inpatient, num_lab_procedures and num_medications. So
    total_prior_visits sums capped counts and lab_procedures_per_day uses the
    capped num_lab_procedures. The caps themselves are a learned step (fitted on
    the training set) and are not applied here; neither is scaling.

    Args:
        df: DataFrame with the columns in UTILIZATION_INPUT_COLS. It is not
            modified.

    Returns:
        A copy of df (same index) with the 3 features added.

    Raises:
        KeyError: if any input column is missing; the message names them.
        ValueError: if time_in_hospital has a 0 or a missing value (no silent
            division by zero).
    """
    missing = [col for col in UTILIZATION_INPUT_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"add_utilization_features: missing columns {missing}")
    days = df["time_in_hospital"]
    bad_days = days.isna() | (days == 0)
    if bad_days.any():
        raise ValueError(
            f"add_utilization_features: time_in_hospital is 0 or missing in "
            f"{int(bad_days.sum())} rows (index {df.index[bad_days][:5].tolist()} ...)"
        )

    out = df.copy()
    visits = ["number_outpatient", "number_emergency", "number_inpatient"]
    out["total_prior_visits"] = out[visits].sum(axis=1).astype(int)
    out["lab_procedures_per_day"] = out["num_lab_procedures"] / out["time_in_hospital"]
    out["is_prior_inpatient"] = (out["number_inpatient"] > 0).astype(int)
    return out


def build_clean_table(df_raw, id_maps, drop_expired=True):
    """Apply every fixed-rule block and return one table X plus the target y.

    Order:
    1. Split off the target: y = the raw readmitted column, unchanged (3 classes
       "NO", ">30", "<30", as in the notebook; not binarised here). It is removed
       from X.
    2. If drop_expired: remove the encounters whose RAW discharge_disposition_id
       is in EXPIRED_HOSPICE_IDS, from X and y together. Done first, on the raw
       ids (not the decoded names), so that every later step, and every learned
       step fitted afterwards, only sees patients who could be readmitted.
    3. The block functions, in this order: preprocess_demographics,
       preprocess_diagnoses, preprocess_labs, preprocess_medication_summary,
       preprocess_medications, preprocess_admin.

    Not done here (all Step A5, fitted on the training set only): the
    99th-percentile caps, "<1% -> Other" for medical_specialty and payer_code,
    the discharge grouping of names with fewer than 3,000 rows, and scaling.
    The utilization features (`add_utilization_features`) are not added either,
    because in the notebook they are built AFTER the caps.

    patient_nbr stays in X: it is NOT a feature, it is kept only to split the
    data by patient (Step A5/B3) so that one patient is never in two sets. It
    must be removed before modelling.

    Args:
        df_raw: the raw diabetic_data table as loaded in the notebook
            (na_values="?", keep_default_na=False), indexed by encounter_id.
            It is not modified.
        id_maps: the dict returned by `load_id_maps`.
        drop_expired: remove the expired/hospice encounters (default True).

    Returns:
        (X, y): X is one DataFrame indexed by encounter_id with every block
        applied; y is the readmitted Series with the same index.

    Raises:
        ValueError: if the index of df_raw has duplicates, or if X and y end up
            with different indexes.
    """
    if not df_raw.index.is_unique:
        dups = df_raw.index[df_raw.index.duplicated()].unique()[:5].tolist()
        raise ValueError(f"build_clean_table: duplicated index values, e.g. {dups}")

    X = df_raw.drop(columns=TARGET_COL)
    y = df_raw[TARGET_COL].copy()

    if drop_expired:
        keep = ~X["discharge_disposition_id"].isin(EXPIRED_HOSPICE_IDS)
        X, y = X[keep], y[keep]

    X = preprocess_demographics(X)
    X = preprocess_diagnoses(X)
    X = preprocess_labs(X)
    X = preprocess_medication_summary(X)
    X = preprocess_medications(X)
    X = preprocess_admin(X, id_maps)

    if not X.index.equals(y.index):
        raise ValueError("build_clean_table: X and y do not have the same index")
    return X, y


def make_target(y):
    """Turn the 3-class readmitted column into the binary target: 1 = "<30", else 0.

    The business question (problem statement) is readmission within 30 days, so
    "<30" is the positive class (1). ">30" (readmitted later) and "NO" (not
    readmitted) both count as 0: a readmission after 30 days is not what the
    hospital wants to prevent here.

    Args:
        y: Series with the raw readmitted values "NO", ">30", "<30". It is not
            modified.

    Returns:
        A Series of int 0/1 with the same index and name as y.

    Raises:
        ValueError: if y holds a missing value or a value other than the three
            above; the message names the values.
    """
    expected = ["NO", ">30", POSITIVE_CLASS]
    unexpected = ~y.isin(expected)
    if unexpected.any():
        bad = sorted(map(str, y[unexpected].unique()))
        raise ValueError(f"make_target: unexpected values {bad}; expected one of {expected}")
    return (y == POSITIVE_CLASS).astype(int)


def split_by_patient(X, y, groups, val_size=0.2, test_size=0.2, random_state=SEED):
    """Split X and y into train / validation / test, grouped by patient.

    Why grouped: the data has 71,518 patients for 101,766 encounters, so many
    patients have several encounters. If the same patient were in train and in
    test, the model could learn that patient's history and look better on test
    than it would on new patients (leakage). So every patient's encounters go to
    exactly one set.

    How: scikit-learn's GroupShuffleSplit, in two passes. First the test set is
    taken off (test_size of the patients); then the validation set is taken from
    the rest, with its share adjusted to val_size / (1 - test_size), so the final
    proportions are about (1 - val_size - test_size) / val_size / test_size. The
    shares are of patients, so the shares of rows are close but not exact.

    The proportions are a team decision still pending; the default is 60/20/20.

    Args:
        X: feature table (index encounter_id). It is not modified.
        y: target Series with the same index as X.
        groups: the patient of each row (e.g. X["patient_nbr"]), a Series with
            the same index as X.
        val_size: share of patients in the validation set (default 0.2).
        test_size: share of patients in the test set (default 0.2).
        random_state: seed for both passes (default SEED = 81, the team seed); the
            same seed gives the same split.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: rows of X and y, each
        keeping its index, in the order of X.

    Raises:
        ValueError: if y or groups do not have the same index as X, or the sizes
            are not between 0 and 1 with val_size + test_size < 1.
    """
    if not (y.index.equals(X.index) and groups.index.equals(X.index)):
        raise ValueError("split_by_patient: X, y and groups must have the same index")
    if not (0 < val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
        raise ValueError(
            f"split_by_patient: need 0 < val_size, test_size and val_size + test_size < 1, "
            f"got {val_size}, {test_size}"
        )

    rows = np.arange(len(X))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    rest, test = next(outer.split(rows, groups=groups))
    inner = GroupShuffleSplit(
        n_splits=1, test_size=val_size / (1 - test_size), random_state=random_state
    )
    train, val = next(inner.split(rest, groups=groups.iloc[rest]))
    train, val = rest[train], rest[val]
    train, val, test = np.sort(train), np.sort(val), np.sort(test)

    return (
        X.iloc[train], X.iloc[val], X.iloc[test],
        y.iloc[train], y.iloc[val], y.iloc[test],
    )


# The 5 columns the notebook caps at the 99th percentile (Henrique's outlier cell).
CAP_COLS = (
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "num_lab_procedures",
    "num_medications",
)

# Discharge names grouped when rare, and the group they go to (notebook's wording).
DISCHARGE_GROUP_PATTERN = r"discharged|transferred"
DISCHARGE_GROUP_NAME = "Discharged to another health institution"


class LearnedPreprocessor(BaseEstimator, TransformerMixin):
    """The learned preprocessing steps, fitted on the training set only.

    Steps (Henrique's, from the notebook's Data Preprocessing part):
    1. Caps: each column in cap_cols is clipped at its cap_quantile (99th
       percentile). A few encounters have very large counts (e.g. 42 outpatient
       visits); the cap limits their weight without removing rows.
    2. Rare categories: in each rare_cols column, categories with a share of rows
       below rare_min_share (1%) become "Other". medical_specialty has 70+
       values, most very small; the few frequent ones are kept. As in the
       notebook, the share is a percentage rounded to 2 decimals before the
       comparison. "Unknown" is a category like any other here.
    3. Discharge grouping: discharge names that contain "discharged" or
       "transferred" and have a share below discharge_min_share become
       "Discharged to another health institution". The notebook's rule is
       "fewer than 3,000 rows" on all 101,766 rows; as a count it would change
       meaning on a smaller set (the training set has about 60% of the rows), so
       it is kept as the same share, 3,000 / 101,766 (about 2.95%).
    4. The utilization features (`add_utilization_features`), built AFTER the
       caps, as in the notebook.

    Why learned, and why train only: a percentile, a list of frequent
    categories and a list of small discharge names are all computed from the
    data. If they were computed on all rows (as the notebook does, before any
    split), the validation and test rows would help decide them, and the scores
    on those sets would be slightly optimistic. Here `fit` sees only the
    training rows; `transform` applies the same values to any set. A category
    seen only outside the training set is treated as rare: "Other" for the
    rare_cols, and grouped for a discharge name matching the pattern.

    Not here: scaling. Only some models need it (distance-based or
    regularised), so it goes inside the model pipelines (Phase B), fitted with
    the model on the training data. The notebook's `scale_columns` was never
    called.

    Parameters:
        cap_cols: columns to cap (default CAP_COLS, the notebook's 5).
        cap_quantile: the cap's quantile (default 0.99).
        rare_cols: columns where rare categories become "Other".
        rare_min_share: smallest share of rows a category needs to be kept
            (default 0.01).
        discharge_min_share: smallest share of rows a matching discharge name
            needs to stay on its own (default 3000 / 101766).

    Learned attributes (after fit):
        caps_: dict column -> cap value.
        kept_categories_: dict column -> sorted list of the categories kept.
        discharge_kept_: sorted list of the matching discharge names kept on
            their own (every other matching name is grouped).
        discharge_grouped_: sorted list of the names grouped in the fit data
            (for reporting only).
    """

    def __init__(
        self,
        cap_cols=CAP_COLS,
        cap_quantile=0.99,
        rare_cols=("medical_specialty", "payer_code"),
        rare_min_share=0.01,
        discharge_min_share=3000 / 101766,
    ):
        self.cap_cols = cap_cols
        self.cap_quantile = cap_quantile
        self.rare_cols = rare_cols
        self.rare_min_share = rare_min_share
        self.discharge_min_share = discharge_min_share

    def fit(self, X, y=None):
        """Learn the caps, the kept categories and the discharge grouping from X.

        Args:
            X: the training table (output of `build_clean_table`). Not modified.
            y: ignored (present for the scikit-learn API).

        Returns:
            self.
        """
        self.caps_ = {col: X[col].quantile(self.cap_quantile) for col in self.cap_cols}

        self.kept_categories_ = {}
        for col in self.rare_cols:
            pct = (X[col].value_counts(dropna=False) / len(X) * 100).round(2)
            self.kept_categories_[col] = sorted(pct[pct >= self.rare_min_share * 100].index)

        share = X["discharge_disposition_name"].value_counts() / len(X)
        matching = share[share.index.str.contains(DISCHARGE_GROUP_PATTERN, case=False)]
        self.discharge_kept_ = sorted(matching[matching >= self.discharge_min_share].index)
        self.discharge_grouped_ = sorted(matching[matching < self.discharge_min_share].index)
        return self

    def transform(self, X):
        """Apply the learned steps, then add the utilization features.

        Args:
            X: a table with the same columns as the fit data (train, validation
                or test). Not modified.

        Returns:
            A copy of X (same index) with the caps, "Other" and the discharge
            grouping applied and the 3 utilization features added.

        Raises:
            NotFittedError: if called before fit.
        """
        check_is_fitted(self, "caps_")
        out = X.copy()
        for col, cap in self.caps_.items():
            out[col] = out[col].clip(upper=cap)
        for col, kept in self.kept_categories_.items():
            out.loc[~out[col].isin(kept), col] = "Other"

        col = "discharge_disposition_name"
        rare_name = out[col].str.contains(DISCHARGE_GROUP_PATTERN, case=False) & ~out[col].isin(
            self.discharge_kept_
        )
        out.loc[rare_name, col] = DISCHARGE_GROUP_NAME
        return add_utilization_features(out)
