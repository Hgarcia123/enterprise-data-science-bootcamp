# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: .venv (3.13.15)
#     language: python
#     name: python3
# ---

# %% [markdown] id="1750205f"
# # Introduction (TBD)

# %% [markdown] id="a819a16a"
# # Libraries and Global Variables

# %% id="4d7953e8"
#Data Analysis and manupalation libraries
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from pathlib import Path

#Sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy import stats
#Seaborn
import seaborn as sns

#Matplotlib
import matplotlib.pyplot as plt

# %% id="f9c25391"
#Seed for ML Algorithms
SEED = 81

#Paths
PROJECT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "raw").exists()
)
GENERATED_REPORTS = (PROJECT / "reports" / "generated")

DIABETES_CSV = PROJECT / "data" / "raw" / "diabetic_data.csv"
IDS_MAPING_CSV = PROJECT / "data" / "raw" / "IDS_mapping.csv"


# %% [markdown] id="1823464b"
# ## Diabetes main fact table

# %% id="98dd9163"
# Read diabetes CSV
df_diabetes = pd.read_csv(
    DIABETES_CSV, delimiter=",", na_values="?", keep_default_na=False, low_memory=False
)
df_diabetes.head()


#CATEGORIZE VARIABLES

## NUMERIC
NUMERIC = [
    "encounter_id",
    "patient_nbr",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]

CATEGORICAL = [
    "race",
    "gender",
    "age",
    "weight",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
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
    "change",
    "diabetesMed",
    "readmitted"
]


# %% [markdown] id="67b140e8"
# ## IDS Mapping Tables
# TODO: Add small explanation behind logic of generator

# %% id="30113aee"
def process_ids_tables():
    #Read IDS mapping CSV
    df_ids_map = pd.read_csv(IDS_MAPING_CSV, header=None, names=range(2), delimiter=',')

    #Define table names
    table_names = ['admission_type', 'discharge_disposition_id', 'admission_source_id']

    groups = df_ids_map[0].isin(table_names).cumsum()

    tables = {g.iloc[0,0]: g.iloc[0:-1] for k,g in df_ids_map.groupby(groups)}

    for name, table in tables.items():
        #Create dataframe
        df = pd.DataFrame(table)

        #Promote first line to header
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

        yield df

ids_mapping_dataframes = process_ids_tables()

#Use generator to create our dataframes for each table

#Admission Types
df_admission_type = next(ids_mapping_dataframes)

#Cast admission_type_id to int for consistency with df_diabetes
df_admission_type['admission_type_id'] = df_admission_type['admission_type_id'].astype(int)

#Discharge Disposition
df_discharge_disposition = next(ids_mapping_dataframes)

#Cast discharge_disposition_id to int for consistency with df_diabetes
df_discharge_disposition['discharge_disposition_id'] = df_discharge_disposition['discharge_disposition_id'].astype(int)

#Admission Source
df_admission_source = next(ids_mapping_dataframes)

#Cast admission_source_id to int for consistency with df_diabetes
df_admission_source['admission_source_id'] = df_admission_source['admission_source_id'].astype(int)

# %% [markdown] id="13882095"
# ### Admission Type mapping

# %% id="c9bc3fbc"
print(f"Min Id: {df_admission_type['admission_type_id'].min()}")
print(f"Max Id: {df_admission_type['admission_type_id'].max()}")

n_rows = df_admission_type.shape[0]
df_admission_type[:n_rows]

# %% [markdown] id="cf390b2e"
# ### Discharge Disposition mapping

# %% id="a1ce02f2"
print(f"Min Id: {df_admission_type['admission_type_id'].min()}")
print(f"Max Id: {df_admission_type['admission_type_id'].max()}")

n_rows = df_admission_type.shape[0]
df_admission_type[:n_rows]

# %% [markdown] id="a5c04b50"
# ### Admission Source mapping

# %% id="3e495939"
print(f"Min Id: {df_admission_source['admission_source_id'].min()}")
print(f"Max Id: {df_admission_source['admission_source_id'].max()}")

n_rows = df_admission_source.shape[0]
df_admission_source[:n_rows]


# %% [markdown] id="da917467"
# # EDA

# %% [markdown] id="6209c1b0"
# Just for starters, we'll create a Profile Report for the diabetes main fact table

# %% id="d2fbf7bb"
def create_profile_report(df, title, file_name):

    profile = ProfileReport(
        df, title = title
    )
    profile.config.html.inline = True

    path_to_html = Path(PROJECT / "reports" / "generated" / file_name)

    if not path_to_html.exists():
        profile.to_file(GENERATED_REPORTS / file_name)

#Create profile report for this dataframe
create_profile_report(df_diabetes, "Diabetes Data Profile Report", "profile-report-diabetes.html")

# %% [markdown] id="a64d6557"
# ## Assigned Variables

# %% [markdown] id="eb051257"
# | Group | Variables |
# |---|---|
# | Identifiers | `encounter_id`, `patient_nbr` |
# | Sociodemographic | `race`, `gender`, `age`, `weight` |
# | Diagnoses | `diag_1`, `diag_2`, `diag_3`, `number_diagnoses` |
# | Lab results | `max_glu_serum`, `A1Cresult` |
# | Medication summary | `change`, `diabetesMed` |

# %% [markdown] id="a1693501"
# ### Identifiers
#
# | Variable | Description |
# |---|---|
# | `encounter_id` | Unique identifier of a hospital encounter |
# | `patient_nbr` | Patient identifier; a patient may have multiple encounters |

# %% id="2eeec45b"
# Check for fully duplicated rows
df_diabetes.duplicated().sum()

# %% id="7295ad23"
# Check that each encounter_id appears only once
df_diabetes["encounter_id"].is_unique

# %% id="8bb36185"
# Count unique patients
df_diabetes["patient_nbr"].nunique()

# %% id="c3491382"
# Count patients with more than one encounter
(df_diabetes["patient_nbr"].value_counts() > 1).sum()

# %% [markdown] id="b43ef7b4"
# **Identifiers decision:** `encounter_id` is unique per row and `patient_nbr` repeats across encounters. Neither is used as a predictor. `patient_nbr` is kept to split train/test by patient, so the same patient never appears in both sets (avoids leakage).

# %% [markdown] id="c235f819"
# ### Sociodemographic Variables
#
# | Variable | Description |
# |---|---|
# | `race` | Patient race as recorded in the source data |
# | `gender` | Patient gender (Female, Male, Unknown/Invalid) |
# | `age` | Age band in 10-year intervals, e.g. [60-70) |
# | `weight` | Weight band in pounds, e.g. [75-100) |

# %% id="ab067cc6"
# Preview the raw sociodemographic values (first 10 rows)
df_diabetes[["race", "gender", "age", "weight"]].head(10)

# %% id="56bb7bdd"
# Check race categories, including missing values
df_diabetes["race"].value_counts(dropna=False)

# %% id="eb44b91e"
# Replace missing race with 'Unknown'
df_diabetes["race"] = df_diabetes["race"].fillna("Unknown")
df_diabetes["race"].value_counts()

# %% id="152f5000"
# Check gender categories (the 3 'Unknown' rows are kept, not inferred)
df_diabetes["gender"].value_counts(dropna=False)

# Rename 'Unknown/Invalid' to 'Unknown' for consistency
df_diabetes["gender"] = df_diabetes["gender"].replace("Unknown/Invalid", "Unknown")
df_diabetes["gender"].value_counts()

# %% id="feb5469d"
# Define the correct order of the age bands
AGE_ORDER = [f"[{i}-{i+10})" for i in range(0, 100, 10)]

# %% id="836b0390"
# Convert age into an ordered categorical variable
df_diabetes["age"] = pd.Categorical(
    df_diabetes["age"], categories=AGE_ORDER, ordered=True
)

# %% id="3fdbf2bd"
# Summary table of age encoding and distribution
age_summary = pd.DataFrame(
    {
        "Age band": AGE_ORDER,
        "Code": range(len(AGE_ORDER)),
        "Count": df_diabetes["age"].value_counts().sort_index().values,
    }
)
age_summary["%"] = (age_summary["Count"] / age_summary["Count"].sum() * 100).round(2)
age_summary.style.hide(axis="index").format({"%": "{:.2f}"})

# %% id="88fdbbdf"
# Convert age bands to numbers (0–9)
df_diabetes["age"] = df_diabetes["age"].cat.codes

# %% id="48b076ba"
# Verify that all age values matched one of the 10 age bands
df_diabetes["age"].unique()

# %% id="622f2248"
# Create a flag indicating whether weight was recorded (96.86% missing)
df_diabetes["weight_recorded"] = df_diabetes["weight"].notna().astype(int)
df_diabetes["weight_recorded"].value_counts()

# %% id="5e861db1"
# Summary table of the weight_recorded flag
weight_summary = (
    df_diabetes["weight_recorded"]
    .value_counts()
    .rename(index={0: "Not recorded (0)", 1: "Recorded (1)"})
    .rename_axis("Weight")
    .reset_index(name="Count")
)
weight_summary["%"] = weight_summary["Count"] / weight_summary["Count"].sum() * 100
weight_summary.style.hide(axis="index").format({"%": "{:.0f}"})

# %% id="89f8b9d7"
# Drop the original weight column (replaced by weight_recorded flag)
df_diabetes.drop(columns="weight", inplace=True)

# %% [markdown] id="48237fef"
# **Sociodemographic decisions:**
# - **race:** missing values (2.2%) labelled `Unknown`, avoiding imputation of a sensitive attribute.
# - **gender:** 3 `Unknown/Invalid` rows renamed `Unknown` and kept; gender is not inferred.
# - **age:** available only in 10-year bands, encoded as ordinal 0–9 to keep the natural order.
# - **weight:** 97% missing, replaced by a binary `weight_recorded` flag; original column dropped.

# %% [markdown] id="5ac13034"
# ### Diagnoses
#
# | Variable | Description |
# |---|---|
# | `diag_1` | Primary diagnosis (ICD-9 code) |
# | `diag_2` | Secondary diagnosis (ICD-9 code) |
# | `diag_3` | Additional secondary diagnosis (ICD-9 code) |
# | `number_diagnoses` | Number of diagnoses entered for the encounter |

# %% id="ca1c45fa"
# Preview the raw diagnosis values (first 10 rows)
df_diabetes[["diag_1", "diag_2", "diag_3", "number_diagnoses"]].head(10)

# %% id="a3e129ef"
# Check the number of distinct ICD-9 codes and missing values per diagnosis column
diag_cols = ["diag_1", "diag_2", "diag_3"]
pd.DataFrame(
    {
        "Distinct codes": df_diabetes[diag_cols].nunique(),
        "Missing": df_diabetes[diag_cols].isna().sum(),
    }
)


# %% id="139077b0"
# Group ICD-9 codes into 9 diagnosis groups from the Project Guide (Strack et al., 2014)
def group_icd9(code):
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


# %% id="07ab5c27"
# Create grouped versions of the 3 diagnosis columns (raw codes are kept)
for col in diag_cols:
    df_diabetes[f"{col}_group"] = df_diabetes[col].apply(group_icd9)

# %% id="af06d0e0"
# Summary table of diagnosis groups (% of encounters)
diag_summary = pd.DataFrame(
    {
        col: df_diabetes[f"{col}_group"].value_counts(normalize=True) * 100
        for col in diag_cols
    }
)
diag_summary.sort_values("diag_1", ascending=False).style.format("{:.1f}")

# %% id="d17624fa"
# Check the range of number_diagnoses
df_diabetes["number_diagnoses"].describe()

# %% id="19f31d66"
# Check the distribution of number_diagnoses
df_diabetes["number_diagnoses"].value_counts().sort_index().to_frame().T

# %% [markdown] id="d95cc007"
# **Diagnoses decisions:**
# - **diag_1, diag_2, diag_3:** 700+ distinct ICD-9 codes each, grouped into 9 clinical categories (Strack et al., 2014) in new `_group` columns; raw codes kept; missing values labelled `Missing`.
# - **number_diagnoses:** range 1–16, no missing values; strong spike at 9 (approximately 49%) and values 10–16 very rare (115 encounters), likely a recording limit in older systems. Kept as is.

# %% [markdown] id="d6f1d9ae"
# ### Lab Results
#
# | Variable | Description |
# |---|---|
# | `max_glu_serum` | Serum glucose test result (>200, >300, Norm, or not measured) |
# | `A1Cresult` | HbA1c test result (>8, >7, Norm, or not measured) |

# %% id="c4266e30"
# Preview the raw lab result values (first 10 rows)
df_diabetes[["max_glu_serum", "A1Cresult"]].head(10).T

# %% id="8a6153b8"
# Check lab result categories, including missing values ('None' = test not performed)
df_diabetes[["max_glu_serum", "A1Cresult"]].apply(
    lambda s: s.value_counts(dropna=False)
).fillna(0).astype(int)

# %% id="f73a850e"
# Replace 'None' / missing lab results with 'Not measured' (the test was not performed)
for col in ["max_glu_serum", "A1Cresult"]:
    df_diabetes[col] = (
        df_diabetes[col].replace("None", "Not measured").fillna("Not measured")
    )

# %% id="a3bf5716"
# Check lab result categories after replacing missing values with 'Not measured'
df_diabetes[["max_glu_serum", "A1Cresult"]].apply(
    lambda s: s.value_counts(dropna=False)
).fillna(0).astype(int)

# %% id="70f50299"
# Percentage of encounters where each lab test was not performed
((df_diabetes[["max_glu_serum", "A1Cresult"]] == "Not measured").mean() * 100).round(1)

# %% [markdown] id="dd9b3ff1"
# **Lab results decisions:**
# - **max_glu_serum, A1Cresult:** missing values indicate that the test was not performed (approximately 95% and 83% of encounters, respectively). These were recoded as `Not measured` rather than imputed, since the absence of a test may itself be informative. Both variables were kept as categorical and will be encoded in the modelling pipeline.

# %% [markdown] id="dbc384ec"
# ### Medication Summary
#
# | Variable | Description |
# |---|---|
# | `change` | Whether diabetes medication changed during the encounter (Ch / No) |
# | `diabetesMed` | Whether any diabetes medication was prescribed (Yes / No) |

# %% id="5fc99f55"
# Preview the raw medication summary values (first 10 rows)
df_diabetes[["change", "diabetesMed"]].head(10).T

# %% id="b914afd0"
# Check medication summary categories, including missing values
df_diabetes[["change", "diabetesMed"]].apply(
    lambda s: s.value_counts(dropna=False)
).fillna(0).astype(int)

# %% id="d823d0c8"
# Convert change and diabetesMed into binary variables (1 = Ch/Yes, 0 = No)
df_diabetes["change"] = df_diabetes["change"].map({"Ch": 1, "No": 0})
df_diabetes["diabetesMed"] = df_diabetes["diabetesMed"].map({"Yes": 1, "No": 0})

# %% id="7791bd69"
# Check that no unexpected values were turned into NaN
df_diabetes[["change", "diabetesMed"]].apply(
    lambda s: s.value_counts(dropna=False)
).fillna(0).astype(int)

# %% [markdown] id="6224e2ae"
# **Medication summary decisions:**
# - **change, diabetesMed:** both are binary variables with no missing values, recoded as 1/0 (Ch/Yes = 1, No = 0) so they can be used directly in the models.

# %% [markdown] id="0b6f28fd"
# ### Distribution

# %% [markdown] id="d8009533"
# **Conclusion:**
# - **race:** mostly Caucasian (approximately 75%), followed by AfricanAmerican (approximately 19%); the other groups are small.
# - **gender:** fairly balanced (approximately 54% Female, 46% Male); only 3 `Unknown`.
# - **age:** concentrated in the older bands (codes 5–8, i.e. 50–90 years).
# - **weight_recorded:** almost always 0, confirming weight was rarely recorded.
# - **diagnosis groups:** Circulatory is the most frequent clinical group in all three diagnosis columns.
# - **number_diagnoses:** strong spike at 9, as noted in the diagnoses decisions.
# - **max_glu_serum, A1Cresult:** dominated by `Not measured`.
# - **change, diabetesMed:** approximately 46% of encounters had a medication change and 77% had diabetes medication prescribed.

# %% id="14cce84a"
# Create a list with my assigned columns (after cleaning)
my_cols = [
    "race",
    "gender",
    "age",
    "weight_recorded",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
]

# Create a dataframe with only these columns
df_my_data = df_diabetes[my_cols].copy()

# Plot distribution in bar charts
fig, axes = plt.subplots(3, 4, figsize=(18, 13))

for ax, col in zip(axes.flatten(), df_my_data.columns):
    # Numeric columns in natural order, text columns from most to least frequent
    if pd.api.types.is_numeric_dtype(df_my_data[col]):
        order = sorted(df_my_data[col].unique())
    else:
        order = df_my_data[col].value_counts().index
    sns.countplot(df_my_data, x=col, ax=ax, order=order)
    ax.set_title(f"Distribution of {col}", fontsize=10)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown] id="a6e9JIquHZqB"
# ### Correlation

# %% id="bf08d0f7"
# Encode text columns to int so they can enter the correlation matrix
df_my_data_encod = df_my_data.copy()
text_cols = [
    "race",
    "gender",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "max_glu_serum",
    "A1Cresult",
]
le = LabelEncoder()
for col in text_cols:
    df_my_data_encod[col] = le.fit_transform(df_my_data_encod[col])

# Add the target as 0/1 to see each variable's link with early readmission
df_my_data_encod["early_readmit"] = (df_diabetes["readmitted"] == "<30").astype(int)

# Plot heatmap using spearman correlation
plt.figure(figsize=(10, 8))
sns.heatmap(
    df_my_data_encod.corr("spearman"),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    square=True,
    linewidth=0.5,
    vmin=-1,
    vmax=1,
)
plt.title("Correlation Analysis - Assigned Variables", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown] id="2a61961b"
# ## Administrative Data
# Here we'll explore data regarding administrative data of the patient. This includes the following columns:
# - admission_type_id
# - admission_source_id
# - discharge_disposition_id
# - payer_code
# - medical_specialty
#

# %% [markdown] id="3872205a"
# ### Preview, describe, nulls

# %% id="73ef6bb4"
#Create a list with columns of administrative data
admin_data_cols = [
    "admission_type_id",
    "admission_source_id",
    "discharge_disposition_id",
    "payer_code",
    "medical_specialty"
]

#Create a daframe with only these columns
df_admin_data = df_diabetes[admin_data_cols].copy()

#check dtypes
print(df_admin_data.dtypes)

df_admin_data.tail(10)

# %% id="d6f3bd0e"
#Create profile report for this dataframe
create_profile_report(df_admin_data, "Diabetes - Administrative Data Profile Report", "profile-report-admin-data.html")

# %% id="23c9ead0"
#Describe numerical columns
df_admin_data.describe()

# %% [markdown] id="8f0a32ec"
# **Note to self:** Last id 29 for discharge disposition seems like it's not being used

# %% id="079715ad"
#Describe categorical columns
df_admin_data.describe(include='object')

# %% id="3e0ef972"
#Percentage of nan values across dataframe
round(df_admin_data.replace('?', np.nan).isna().sum() / df_admin_data.shape[0] * 100, 2)

# %% [markdown] id="dfe2477c"
# **Key Takeaway:** Columns payer_code and medical_specialty have high null values. 39% and 49% respectively

# %% id="3bf4e588"
#Check distinct values of payer code for context
pd.DataFrame(df_admin_data['payer_code'].unique())

# %% [markdown] id="6ccd7889"
# Not much to conclude from these values. Looks like country codes but verifying some of them like "OG" they do not correspond to any country.
# There is no DIM table for these column. Just keep values as is (mentioned in project guide) and later decide what to do with null values.

# %% [markdown] id="65bbd80c"
# ### Distribution

# %% id="05496a25"
#Plot distribution in bar charts
fig, axes = plt.subplots(2, 3, figsize=(12, 10))

for ax, col in zip(axes.flatten(), df_admin_data.columns):
    sns.countplot(df_admin_data, x=col, ax=ax)
    ax.set_title(f"Distribution of {col}", fontsize=10)
    ax.set_xlabel('')
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown] id="cd533719"
# Note: Check why the first value of these columns is almost always the value that occurs more frequently
#
# Looking at the dimension tables displayed above we see:
# - Admission Type: 1 -> Emergency
# - Admission Source: 7 -> Emergency Room
# - Discharge Disposition: 1 -> Discharge to home
# - Payer code: ? -> Null
# - Medical Specialty: ? -> Null
#
# We can conclude that the majority of the occurences are emergency related and that the patient is discharged to home
#

# %% [markdown] id="71ff7094"
# ### Correlation

# %% id="0538e0f6"
#This dataset has int and string values, so I'll encode the str values to int for an improved correlation

df_admin_data_encod = df_admin_data.copy()
#Columns: payer_code, medical_specialty
le = LabelEncoder()
df_admin_data_encod['payer_code'] = le.fit_transform(df_admin_data_encod['payer_code'])
df_admin_data_encod['medical_specialty'] = le.fit_transform(df_admin_data_encod['medical_specialty'])

print(df_admin_data_encod.dtypes)

df_admin_data_encod[['payer_code', 'medical_specialty']].tail(10)

# %% id="5c32cd17"
# Build correlation matrix
# matrix = df_admin_data_encod.corr()

# Build plot
plt.figure(figsize=(6, 5))

#Scale values using min-max scaling to improve correlation visualization
scaler = MinMaxScaler()
df_corr_analysis = pd.DataFrame(
    scaler.fit_transform(df_admin_data_encod),
    columns=df_admin_data_encod.columns,
    index=df_admin_data_encod.index,
)

# Plot heatmap using spearman correlation
sns.heatmap(
    df_admin_data_encod.corr("spearman"),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    square=True,
    linewidth=0.5,
    vmin=-1,
    vmax=1,
)
plt.title("Correlation Analysis - Administrative Data", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown] id="90d79e30"
# **Conclusion:**
# - `change` and `diabetesMed` are the most correlated pair (approximately 0.51): a medication change implies a diabetes medication was prescribed. Worth checking for redundancy before modelling.
# - `age` and `number_diagnoses` are moderately correlated (approximately 0.20): older patients tend to have more diagnoses recorded.
# - All variables show very weak linear correlation with early readmission (|ρ| ≤ 0.05). This does not make them useless: tree-based models can capture non-linear effects and interactions.

# %% [markdown] id="4946495e"
# Let's check for correlation between columns that have similar meaning using qui-square:  
# - admission_type_id <-> admission_source_id
#

# %% [markdown] id="178f19e6"
# ## Utilization history/current stay data

# %% [markdown] id="a5fc151e"
# ### Preview, describe, nulls

# %% id="909b9f76"
util_stay_cols = [
    "num_lab_procedures",
    "num_procedures",
    "number_diagnoses",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "time_in_hospital"
]

df_util_stay_data = df_diabetes[util_stay_cols].copy()

#dtypes
print(df_util_stay_data.dtypes)

df_util_stay_data.head(10)

# %% id="1f633db9"
#Describe
df_util_stay_data.describe()

# %% [markdown] id="bf6d5072"
# ### Distribution
# Check distribution of these numerical variables. The skewness of the distribution is important to determine what method of correlation should be used in the next step

# %% id="ca8d9326"
df_util_stay_data.columns

# %% id="102cdf27"
#Statistics of data
for col in df_util_stay_data.columns:
    data = df_util_stay_data[col]

    mean_val = np.mean(data)
    std_val = np.std(data)
    iqr_val = stats.iqr(data)

    print(f"{col}: \tMean Val: {mean_val} \tStandard Deviation: {std_val} \tIQR Val: {iqr_val}")

#Plot distribution in bar charts
fig, axes = plt.subplots(2, 4, figsize=(16, 10))

#Create bins dict
bins_dict = {
    "num_lab_procedures": 15,
    "num_procedures": 20,
    "num_medications": 20,
    "number_diagnoses": 10,
    "number_outpatient": 5,
    "number_emergency": 5,
    "number_inpatient": 50,
    "time_in_hospital": 10,
}

for ax, col in zip(axes.flatten(), df_util_stay_data.columns):
    sns.histplot(df_util_stay_data, x=col, ax=ax, bins=bins_dict[col], kde=True)
    ax.set_title(f"Distribution of {col}", fontsize=10)
    ax.set_xlabel('')

plt.tight_layout()
plt.show()

# %% [markdown] id="98e546df"
# **Conclusion:** Looking at the distribution plots (TBD)

# %% [markdown] id="2233e7e8"
# Let's plot the distribution of number_outpatient, number_emergency and number_inpatient in plots more tightly adjusted to better view the distributions, although they are most certainly right skewed

# %% id="99988672"
#Plot distribution in bar charts
fig, ax = plt.subplots(1, 3, figsize=(12, 6))
cols = ["number_outpatient", "number_emergency", "number_inpatient"]

for ax, col in zip(ax, cols):
    sns.histplot(df_util_stay_data, x=col, discrete=True, kde=False, ax=ax)
    ax.set_title(f"Distribution of {col}", fontsize=10)
    ax.set_xlabel('')

plt.tight_layout()
plt.show()

# %% [markdown] id="f83d1a45"
# **Conclusion** We can definitely conclude that all three variables are highly right skewed and presenting a high likelihood of having outliers. This also verifiable by the values of UIQR presented above where the difference between the first and thrid quartiles are very slim, which tells us the values are concentraded around the mean

# %% [markdown] id="2dd154cf"
# ### Correlation (with outliers)

# %% id="f14fce01"
# Build correlation matrix
# matrix = df_util_stay_da.corr()

# Build plot
plt.figure(figsize=(6, 5))

#Scale values using min-max scaling to improve correlation visualization
scaler = MinMaxScaler()
df_corr_analysis = pd.DataFrame(
    scaler.fit_transform(df_util_stay_data),
    columns=df_util_stay_data.columns,
    index=df_util_stay_data.index,
)

# Plot heatmap using spearman correlation
sns.heatmap(
    df_corr_analysis.corr(method='pearson'),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    square=True,
    linewidth=0.5,
    vmin=-1,
    vmax=1,
)

plt.title("Correlation Analysis - Utilization and Current Stay Data", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown] id="85903b34"
# ### Correlation (without outliers)
# Remove outliers from dataframe and quickly check if correlation is more relevant

# %% id="e99bcc0d"
#Upper trimm data in order to remove outliers
df_util_outliers = df_util_stay_data.copy()

cutoffs = {}

for c in df_util_outliers.columns:
    q1, q3 = df_util_outliers[c].quantile([0.25, 0.75])
    cutoffs[c] = df_util_outliers[c].quantile(0.90)

mask = pd.concat([df_util_outliers[c] <= cutoffs[c] for c in df_util_outliers.columns], axis=1).all(axis=1)
clean = df_util_outliers.loc[mask, df_util_outliers.columns]

corr = clean.corr(method = 'spearman')
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
plt.title("Correlation matrix (upper outliers removed)")
plt.show()

# %% [markdown] id="57d0ce48"
# **Conclusion:** Very similar correlation heatmap compared to the one with outliers, being the only difference where the correlation of medications and lab procedures dropped by about in half
#
# Ultimately, we can conclude that, in general, these columns aren't very correlated or similar in anyway.
#
# There is some correlation regarding the number of medications, where the higher the lab procedures, normal procedures and time spent in hospital are, the higher the number of medication is prescribed, which makes total sense.
# Also, and although not highly correlated, the higher the time spent in hospital, the more medication is prescribed.
#
# **We should also make sure to remove outliers in pre-processing before feeding the data to our model for atleast these columns:**
# - number_diagnoses
# - number_outpatient
# - number_emergency
# - number_inpatient

# %% [markdown]
# ## Medications
# Here we explore the 23 drug columns (`metformin` → `metformin-pioglitazone`) and how they relate to early readmission (<30 days).

# %%
# Imports used only in this section
from scipy.stats import chi2_contingency

# ydata-profiling switches matplotlib to the non-interactive 'agg' backend; switch back so charts display.
# Close figures left open under 'agg' first, so they don't appear in this section's chart cells.
plt.close("all")
# %matplotlib inline

# %% [markdown]
# ### Drug columns
#
# | Variable | Description |
# |---|---|
# | `metformin`, `repaglinide`, `nateglinide`, `chlorpropamide`, `glimepiride`, `acetohexamide`, `glipizide`, `glyburide`, `tolbutamide`, `pioglitazone`, `rosiglitazone`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`, `examide`, `citoglipton`, `insulin`, `glyburide-metformin`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone` | One column per diabetes drug (23 columns) |
# | Values | `No` = not prescribed, `Steady` = dose unchanged, `Up` / `Down` = dose increased / decreased during the encounter |

# %%
# Define the 23 drug columns (contiguous block in the file)
drug_cols = df_diabetes.loc[:, "metformin":"metformin-pioglitazone"].columns.tolist()

# Preview the raw drug values (first 10 rows)
df_diabetes[drug_cols].head(10).T

# %%
# Check that the drug columns only use No / Steady / Up / Down and have no missing values
print("Number of drug columns:", len(drug_cols))
print("Values found:", sorted(pd.unique(df_diabetes[drug_cols].values.ravel())))
print("Missing values:", df_diabetes[drug_cols].isna().sum().sum())

# %%
# Usage per drug and the drop rule: keep a drug if at least 1% of encounters take it
THRESHOLD = 0.01

taking = df_diabetes[drug_cols] != "No"
drug_usage = pd.DataFrame(
    {
        "n_taking": taking.sum(),
        "% taking": taking.mean() * 100,
        "n_levels": df_diabetes[drug_cols].nunique(),
    }
).sort_values("n_taking", ascending=False)
drug_usage["decision"] = np.select(
    [drug_usage["n_levels"] == 1, drug_usage["% taking"] < THRESHOLD * 100],
    ["drop_constant", "drop_rare"],
    default="keep",
)

MED_KEEP = drug_usage.index[drug_usage["decision"] == "keep"].tolist()
MED_DROP = drug_usage.index[drug_usage["decision"] != "keep"].tolist()
print(f"MED_KEEP ({len(MED_KEEP)}): {MED_KEEP}")
print(f"MED_DROP ({len(MED_DROP)}): {MED_DROP}")

drug_usage.drop(columns="n_levels").style.format({"% taking": "{:.2f}"})

# %%
# Profile report for the medication block (23 drug columns + target only;
# a report on all 50 columns is too heavy to read for this block)
create_profile_report(
    df_diabetes[drug_cols + ["readmitted"]],
    "Diabetes - Medications Profile Report",
    "profile-report-diabetes-medications.html",
)
# Profiling switches matplotlib to a backend that can't display charts; switch back
plt.close("all")
# %matplotlib inline

# %% [markdown]
# **Profiling result:** The medications profile report (`profile-report-diabetes-medications.html`) flags `examide` and `citoglipton` as constant, and 20 of the 23 drugs as highly imbalanced (all except `insulin` and the 2 constant columns). The imbalance flag alone can't be the drop rule, since it also covers widely used drugs like `metformin`, hence the 1% usage threshold.

# %% [markdown]
# **Drug decisions:**
# - **Drop rule:** a drug is kept if at least 1% of encounters take it (about 1,018 encounters). Below that there are too few users to estimate a reliable readmission rate per level.
# - **Result:** 8 drugs kept (`MED_KEEP`) and 15 dropped (`MED_DROP`). The cut-off falls in a clear gap: the smallest kept drug, `repaglinide`, has 1,539 users and the next one, `glyburide-metformin`, has 706.
# - **Constant columns:** `examide` and `citoglipton` are `No` for every encounter, so they carry no information.
# - **Not applied yet:** no columns are dropped from `df_diabetes` here; the drop is applied in preprocessing. The dropped drugs still count in the engineered features below.

# %% [markdown]
# ### Medication vs early readmission
# <30-day readmission rate by level for each kept drug. One chi-square test per drug; a drug gets a chart only if p < 0.05 / 8 (Bonferroni, since 8 drugs are tested).

# %%
# Early readmission flag (readmitted is not modified)
early_readmit = (df_diabetes["readmitted"] == "<30").astype(int)
BASE_RATE = early_readmit.mean()
ALPHA = 0.05
LEVELS = ["No", "Steady", "Up", "Down"]


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n; returns (low, high)."""
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


# <30 rate per drug and level, with group sizes and 95% CIs
rows = []
for drug in MED_KEEP:
    g = early_readmit.groupby(df_diabetes[drug]).agg(n="size", n_lt30="sum").reindex(LEVELS)
    for level, (n, k) in g.iterrows():
        low, high = wilson_ci(k, n)
        rows.append((drug, level, n, k / n, low, high))
drug_rates = pd.DataFrame(rows, columns=["drug", "level", "n", "rate", "ci_low", "ci_high"])

# Summary table: rate (%) by level, chi-square p-value and whether the drug gets a chart
drug_summary = drug_rates.pivot(index="drug", columns="level", values="rate").loc[MED_KEEP, LEVELS] * 100
drug_summary["p_value"] = [chi2_contingency(pd.crosstab(df_diabetes[d], early_readmit))[1] for d in MED_KEEP]
drug_summary["chart"] = drug_summary["p_value"] < ALPHA / len(MED_KEEP)
print(f"Base <30 rate: {BASE_RATE:.2%} | Bonferroni cut-off: p < {ALPHA / len(MED_KEEP):.4f}")
drug_summary.style.format({**{lvl: "{:.1f}" for lvl in LEVELS}, "p_value": "{:.1e}"})

# %% [markdown]
# **Note:** `repaglinide` (p = 0.007) and `glipizide` (p = 0.009) narrowly miss the cut-off; their striking rates (`repaglinide` `Up` 18.2%, `glipizide` `Down` 15.2%) sit on groups of fewer than 800 encounters with wide intervals.
#
# **Caveat:** the tests treat the 101,766 encounters as independent, but they come from 71,518 patients, so the p-values are optimistic. The ranking (insulin, then metformin, then the rest) is what matters here.

# %%
BAR, CONTEXT, INK, MUTED = "#2a78d6", "#b5b4ae", "#0b0b0b", "#52514e"


def plot_rates(d, title, ylabel=None):
    """Horizontal bars of the <30 rate per group with 95% CIs and the base rate.

    d is indexed by group (top to bottom) with columns rate, ci_low, ci_high, n;
    the first group is the grey reference.
    """
    reference = d.index[0]
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(d) + 1.2))
    colors = [CONTEXT if g == reference else BAR for g in d.index]
    ax.barh(d.index, d["rate"] * 100, height=0.55, color=colors)
    ax.errorbar(d["rate"] * 100, d.index,
                xerr=[(d["rate"] - d["ci_low"]) * 100, (d["ci_high"] - d["rate"]) * 100],
                fmt="none", ecolor=MUTED, elinewidth=1, capsize=3)
    for g, r in d.iterrows():
        ax.text(r["ci_high"] * 100 + 0.3, g, f"{r['rate']:.1%}  (n={r['n']:,.0f})",
                va="center", fontsize=9, color=INK)
    ax.axvline(BASE_RATE * 100, color=MUTED, linestyle="--", linewidth=1)
    ax.set_ylim(-0.5, len(d) - 0.1)  # headroom above the top bar for the base-rate label
    ax.text(BASE_RATE * 100 + 0.2, len(d) - 0.35, f"all encounters {BASE_RATE:.1%}",
            ha="left", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(0, max(d["ci_high"]) * 100 + 6)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.set_xlabel("<30-day readmission rate (%)", color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.tick_params(colors=MUTED, length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(CONTEXT)
    plt.tight_layout()
    plt.show()


DRUG_TITLES = {
    "insulin": "Insulin: dose changes are linked to higher <30 readmission",
    "metformin": "Metformin: Steady and Up are readmitted less often than No",
}
for drug in drug_summary.index[drug_summary["chart"]]:
    plot_rates(drug_rates[drug_rates["drug"] == drug].set_index("level"), DRUG_TITLES[drug])

# %%
# Sensitivity check: drop expired/hospice discharges (they cannot be readmitted) and recompute.
# Works on a filtered copy; df_diabetes, early_readmit and drug_rates are left unchanged.
EXPIRED_HOSPICE_IDS = [11, 13, 14, 19, 20, 21]  # discharge_disposition_id
expired = df_diabetes["discharge_disposition_id"].isin(EXPIRED_HOSPICE_IDS)
df_alive = df_diabetes.loc[~expired].copy()
early_readmit_alive = early_readmit[~expired]

print(f"Removed: {expired.sum():,} encounters ({expired.mean():.1%}), <30 rate among them: {early_readmit[expired].mean():.2%}")
print(f"Base <30 rate: {BASE_RATE:.2%} original -> {early_readmit_alive.mean():.2%} without")

sens = []
for drug in ["insulin", "metformin"]:
    original = drug_rates[drug_rates["drug"] == drug].set_index("level")["rate"]
    without = early_readmit_alive.groupby(df_alive[drug]).mean()
    for level in LEVELS:
        sens.append((drug, level, original[level] * 100, without[level] * 100))
pd.DataFrame(sens, columns=["drug", "level", "original %", "without %"]).set_index("drug").round(2)


# %% [markdown]
# **Sensitivity check:** expired/hospice discharges are not removed yet; re-checked without them, the patterns hold. Insulin rates move up ~0.2–0.3 pp, metformin users barely change (≤0.07 pp), so metformin's gap to `No` widens slightly.

# %% [markdown]
# ### Engineered features
#
# | Variable | Description |
# |---|---|
# | `n_meds_taken` | Number of the 23 drug columns not `No` |
# | `n_dose_changes` | Number of the 23 drug columns at `Up` or `Down` |
# | `n_other_meds` | Number of `MED_DROP` drugs taken, so the dropped drugs still count |
# | `on_insulin_only` | 1 if insulin is taken and no other diabetes drug, else 0 |

# %%
def build_med_features(df):
    """Build engineered medication features from the 23 drug columns.

    Returns a new DataFrame with the same index as df (df itself is not modified):
    - n_meds_taken: how many of the 23 drug columns are not "No" (any use: Steady, Up or Down).
    - n_dose_changes: how many of the 23 drug columns are "Up" or "Down" (dose changed this encounter).
    - n_other_meds: how many of the MED_DROP drugs are not "No"; keeps the rare drugs
      dropped by the 1% rule in the data as a count.
    - on_insulin_only: True when insulin is not "No" and no other drug column is taken.

    Uses the globals drug_cols and MED_DROP defined above.
    """
    taking = df[drug_cols] != "No"
    feats = pd.DataFrame(index=df.index)
    feats["n_meds_taken"] = taking.sum(axis=1)
    feats["n_dose_changes"] = df[drug_cols].isin(["Up", "Down"]).sum(axis=1)
    feats["n_other_meds"] = taking[MED_DROP].sum(axis=1)
    feats["on_insulin_only"] = taking["insulin"] & (taking.drop(columns="insulin").sum(axis=1) == 0)
    return feats


# %%
# Add the 4 engineered features to df_diabetes (the raw drug columns are kept)
med_feats = build_med_features(df_diabetes)
for col in med_feats.columns:
    df_diabetes[col] = med_feats[col].astype(int)

MED_FEATURES = list(med_feats.columns)
df_diabetes[MED_FEATURES].describe().loc[["mean", "min", "max"]].round(2)

# %%
# <30 rate per feature value (counts binned as 0 / 1 / 2 / 3+), with 95% CI and chi-square p-value
rows = []
for col in MED_FEATURES:
    if col == "on_insulin_only":
        bins = df_diabetes[col].astype(str)
    else:
        bins = df_diabetes[col].clip(upper=3).astype(str).replace("3", "3+")
    p_value = chi2_contingency(pd.crosstab(bins, early_readmit))[1]
    g = early_readmit.groupby(bins).agg(n="size", n_lt30="sum")
    for b, (n, k) in g.iterrows():
        low, high = wilson_ci(k, n)
        rows.append((col, b, n, k / n * 100, low * 100, high * 100, p_value))

feature_rates = pd.DataFrame(
    rows, columns=["feature", "value", "n", "rate %", "ci_low %", "ci_high %", "p_value"]
).set_index(["feature", "value"])
print(f"Base <30 rate: {BASE_RATE:.2%} | Bonferroni cut-off: p < {ALPHA / len(MED_FEATURES):.4f}")
feature_rates.style.format(
    {"n": "{:,}", "rate %": "{:.1f}", "ci_low %": "{:.1f}", "ci_high %": "{:.1f}", "p_value": "{:.1e}"}
)

# %%
# Chart the features that pass p < 0.05 / 4 (the first value is the grey reference)
FEATURE_TITLES = {
    "n_meds_taken": "<30 readmission rate by number of diabetes drugs taken",
    "n_dose_changes": "<30 readmission rate by number of dose changes",
    "n_other_meds": "<30 readmission rate by number of rare (dropped) drugs taken",
    "on_insulin_only": "<30 readmission rate: insulin as the only diabetes drug vs not",
}
for col in MED_FEATURES:
    d = feature_rates.loc[col]
    if d["p_value"].iloc[0] >= ALPHA / len(MED_FEATURES):
        continue
    d = pd.DataFrame(
        {"rate": d["rate %"] / 100, "ci_low": d["ci_low %"] / 100, "ci_high": d["ci_high %"] / 100, "n": d["n"]}
    )
    plot_rates(d, FEATURE_TITLES[col], ylabel=col)

# %%
# Check the shares quoted in the note below
dose_changed = df_diabetes["n_dose_changes"] > 0
insulin_changed = df_diabetes["insulin"].isin(["Up", "Down"])
one_drug = df_diabetes["n_meds_taken"] == 1
print(f"Dose-change encounters that are insulin Up/Down: {insulin_changed[dose_changed].mean():.1%} "
      f"({insulin_changed.sum():,} of {dose_changed.sum():,})")
print(f"One-drug encounters that are insulin only: {df_diabetes.loc[one_drug, 'on_insulin_only'].mean():.1%} "
      f"({df_diabetes['on_insulin_only'].sum():,} of {one_drug.sum():,})")
print(f"Encounters taking none of the dropped drugs: {(df_diabetes['n_other_meds'] == 0).mean():.1%}")

# %% [markdown]
# **Engineered features note:**
# - **n_dose_changes:** 85% of encounters with a dose change are insulin `Up` / `Down`, so this feature overlaps heavily with the raw `insulin` column. The useful contrast is any change vs none; the 2 and 3+ groups are small.
# - **n_meds_taken:** peaks at 1 drug because about two-thirds of one-drug encounters (65%) are insulin-only patients, matching `on_insulin_only` (12.9% vs 10.4%). Which drug is taken matters more than how many.
# - **n_other_meds:** shows no difference (98% of encounters take none of the dropped drugs), which supports the 1% drop rule.

# %% [markdown]
# **Overlap with the medication summary:**
# - `diabetesMed == 1` is identical to `n_meds_taken > 0` (100% agreement): `diabetesMed` is fully explained by the drug columns.
# - `n_dose_changes > 0` is a subset of `change == 1` (81% agreement): every encounter with an `Up` / `Down` has `change == 1`, but `change` also covers switching drugs, which the dose levels can't show.
# - This answers the open point in the correlation conclusion that flags `change` and `diabetesMed` (approximately 0.51) as worth checking for redundancy before modelling.

# %%
# Verify the overlap against the 1/0 medication summary columns
any_med = (df_diabetes["n_meds_taken"] > 0).astype(int)
any_dose_change = (df_diabetes["n_dose_changes"] > 0).astype(int)

print(f"diabetesMed vs n_meds_taken > 0: {(any_med == df_diabetes['diabetesMed']).mean():.1%} agreement")
print(f"change vs n_dose_changes > 0:    {(any_dose_change == df_diabetes['change']).mean():.1%} agreement")
print("Encounters with an Up/Down but change == 0:", ((any_dose_change == 1) & (df_diabetes["change"] == 0)).sum())

pd.crosstab(any_dose_change.rename("n_dose_changes > 0"), df_diabetes["change"], margins=True)

# %% [markdown]
# **Conclusion:**
# - **Drug columns:** 8 of the 23 drugs are used by at least 1% of encounters and are kept; 15 are dropped in preprocessing (13 rare, 2 constant).
# - **Insulin** has the strongest link with early readmission: the <30 rate rises from 10.0% (`No`) to 13.9% (`Down`). **Metformin** users are readmitted less often, probably because metformin is withheld from sicker patients (confounding), not because it protects. The other 6 kept drugs show no clear difference after the Bonferroni correction. Removing expired/hospice discharges doesn't change these patterns.
# - **Engineered features:** `on_insulin_only` and `n_dose_changes` show the clearest differences, but both lean heavily on insulin (85% of dose changes are insulin `Up` / `Down`); `n_meds_taken` peaks at 1 drug (mostly insulin alone), and `n_other_meds` shows no difference.
# - **Redundancy:** `diabetesMed` duplicates `n_meds_taken > 0` and is a candidate to drop in preprocessing; `change` holds information the drug columns don't.
