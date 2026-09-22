# %%
"""Explore the Lending Club data: load it, check one claim of its documentation, profile a sample.

A notebook written as plain Python (jupytext, percent format). Turn it into a real
notebook with:  jupytext --to ipynb notebooks/explore.py
It writes `reports/generated/profile.html` and `data/processed/lending-club-sample.csv`.

It is run on its own, not imported, so it finds the project folder by itself and
does not use `src.config`.
"""

# %% [markdown]
# # Explore the Lending Club data
# This file is a notebook written as plain Python. Every `# %%` starts a cell.

# %%
from pathlib import Path

import pandas as pd

# Find the project folder: the first folder above us that has data/raw inside.
PROJECT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "raw").exists()
)
CSV = PROJECT / "data" / "raw" / "lending-club.csv"
print(CSV)

# %% [markdown]
# ## 1. Load and look

# %%
df = pd.read_csv(CSV)
df.shape

# %%
df.head()

# %% [markdown]
# ## 2. Check the documentation
# The data dictionary, `references/lending-club-consumer-credit-risk.pdf`, says that
# `home_ownership` has five values and is empty in 826 rows. Is it right?

# %%
df["home_ownership"].value_counts(dropna=False)

# %% [markdown]
# ## 3. Profile a sample
# 5,000 rows are enough to see the problems, and the report is ready in under a minute.

# %%
from ydata_profiling import ProfileReport

sample = df.sample(5000, random_state=42)
report = ProfileReport(sample, minimal=True, title="Lending Club, 5,000 rows")
report.to_file(PROJECT / "reports" / "generated" / "profile.html")

# %% [markdown]
# ## 4. Save the sample
# Raw data is never changed. Anything we make goes to `data/processed/`.

# %%
sample.to_csv(PROJECT / "data" / "processed" / "lending-club-sample.csv", index=False)
print("Saved", len(sample), "rows")

# %% [markdown]
# ## 5. YOUR TURN
# The document was right about `home_ownership`. No document tells you everything.
# Open `reports/generated/profile.html` and find three things the data dictionary does NOT
# tell you. A hint: look at the minimum and the maximum of the number columns.
# Write one line of code below that proves one of them.

# %%
