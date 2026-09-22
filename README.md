# EDSB practice project: Lending Club

A small, complete data science project used in class 3 of the Enterprise Data
Science Bootcamp (NOVA IMS, 2026-27). It shows how a project is laid out, how
the environment is rebuilt, what goes into Git and what stays out, and how code
moves from a notebook into tested Python files.

**Project scaffold:** Henrique Carreiro and João Azambuja, Enterprise Data Science
Bootcamp, NOVA IMS, 2026-27

**Maintainer:** write your name here

## What is in this folder

```
.
|-- README.md                  this file: what the project is and how to run it
|-- class_week03.py            the class, step by step
|-- requirements.txt           the packages the project asks for
|-- requirements-installed.txt the packages really installed, with versions (pip freeze)
|-- .gitignore                 what Git must never track
|-- .env.example               the settings the project needs, without values (in Git)
|-- .env                       your own values for those settings (never in Git)
|-- data
|   |-- raw                    the data as received. Never changed
|   |-- interim                half-way files made while cleaning
|   |-- processed              clean tables, made by the code
|   `-- external               data from other sources
|-- notebooks                  exploration. Stored as .py, opened as .ipynb
|-- src                        the project's own code
|   |-- __init__.py            makes src a package (see below)
|   |-- check_data.py          a first look at the raw data. Run it with the Run button
|   |-- config.py              every path and setting, in one place
|   |-- dataset.py             raw data in, clean data out
|   |-- features.py            the table becomes X and y
|   |-- modeling
|   |   |-- __init__.py        makes modeling a package inside src
|   |   |-- train.py           trains and saves a model
|   |   `-- predict.py         uses the saved model
|   `-- plots.py               charts saved as files
|-- tests                      code that checks the code in src
|   |-- __init__.py            lets Python find the tests by itself
|   `-- test_data.py           the tests
|-- models                     trained models, made by the code
|-- reports                    the reports you write. Tracked by Git
|   `-- generated              reports and charts made by the code. Ignored by Git
`-- references                 guides, data dictionaries, papers. The data dictionary is here
```

Folders that would be empty hold a `.gitkeep` file, because Git does not track
empty folders.

## If you received this as a zip

Unzip it, then open the folder in VS Code (File > Open Folder). Some files have
names that start with a dot: `.gitignore`, `.env`, `.env.example`, `.gitkeep`.
Windows and macOS hide them in their file windows, but VS Code shows them. If you
copy the contents into another folder, make sure the dot files come along (on a
Mac, press Cmd+Shift+. in Finder to see them).

## Set up

You need Python 3.13, Git and Visual Studio Code, as in the course installation
guide. The same instructions are written at the top of `requirements.txt`.

1. Build the environment from the package list. In VS Code: Ctrl+Shift+P,
   then Python: Create Environment, Venv, Python 3.13, and tick
   `requirements.txt`. In a terminal it is:

   ```
   python -m venv .venv
   .venv\Scripts\activate            (Windows)
   source .venv/bin/activate         (Mac and Linux)
   python -m pip install -r requirements.txt
   ```

2. Check that the data is in place: `data/raw/lending-club.csv`. The class zip
   already has it. If you got the project from GitHub instead, the file is not
   there on purpose (it is about 60 MB): copy it from the course Moodle page.

3. Check your settings file, `.env`. See the next section.

4. Save what was really installed, with exact versions:

   ```
   python -m pip freeze > requirements-installed.txt
   ```

If the environment ever breaks, delete the `.venv` folder and repeat step 1.
Nothing is lost: `.venv` holds none of your work.

## The .env file

`.env` holds two kinds of things: secrets (an API key, a password) and settings
that differ from one computer to another. The code reads it when it starts
(`load_env` in `src/config.py`), so none of these values is ever typed in the
code.

| File | In Git? | What it holds |
|---|---|---|
| `.env.example` | yes | the names of the settings, with no secret values |
| `.env` | never | your own values |

This project uses two settings: `API_KEY`, a secret, empty because today we
have none, and `RAW_DATA_FILE`, the name of the file in `data/raw`.

The `.env` in the class zip is there so that everything works at once, and it
holds no secret. In a real project, nobody sends you a `.env`: you copy
`.env.example`, name the copy `.env`, and fill in your own values. If a secret
ever reaches GitHub, deleting the file is not enough, because the history keeps
it. Change the secret.

## Run

Run all commands from the project folder, with the environment active.

| What | Command |
|---|---|
| A first look at the data | `python src/check_data.py` |
| Clean the data | `python -m src.dataset` |
| Train the model | `python -m src.modeling.train` |
| Estimate the risk of default for five loans | `python -m src.modeling.predict` |
| Draw the chart | `python -m src.plots` |
| Run the tests | `python -m unittest` |
| Open the notebook | `jupytext --to ipynb notebooks/explore.py` |

Run them in that order the first time: training needs the clean data, and
predicting needs the trained model.

## The `__init__.py` files

A file called `__init__.py` makes a folder a regular Python package. It can be
empty: it only has to exist. Python can also import a folder without one, as a
"namespace package", so the code in `src` would still run. This project keeps
three of them, because being explicit avoids two real problems.

| File | Why it is there |
|---|---|
| `src/__init__.py` | Makes `src` a regular package: `src` always means this folder, and is never merged with another folder called `src` on Python's import path |
| `src/modeling/__init__.py` | Makes `modeling` a package inside `src`, so its files are `src.modeling.train` and `src.modeling.predict` |
| `tests/__init__.py` | Lets `python -m unittest` find the tests. Without it that command reports "Ran 0 tests", which is easy to mistake for success |

`notebooks` has none, because its files are run, never imported.

## The data

Lending Club loans, one row per loan, 331,865 rows and 28 columns, in one table.
The `split` column says which rows are for training (loans from January 2014 to
June 2016) and which are for testing (later loans). The data comes from the
public Lending Club data set on Figshare [4], released under CC0.

The column to predict is `loan_status`: **0 = charged off or default, 1 = fully
paid**. About 16% of the loans default. It is empty in 1,412 test rows, which
`src/dataset.py` drops.

The data dictionary is `references/lending-club-consumer-credit-risk.pdf`: what
each column means, and what to watch for. Look before you trust any
documentation. `notebooks/explore.py` checks one of its claims (`home_ownership`
has five values and 826 empty rows, which is right) and then profiles a sample,
which shows things the document does not mention.

## The example model

`src/modeling` trains a logistic regression on five columns and saves it.
`predict` returns the estimated **probability of default** for each loan: a
number between 0 and 1, where higher means riskier. It needs only the five input
columns, so it works on a new application whose outcome nobody knows yet. On the
test rows, the model reaches a ROC AUC of about 0.68.

The five columns (`annual_inc`, `dti`, `fico_range_low`, `revol_util`,
`open_acc`) come from the borrower or from the credit file, so they are known
before the lender decides. Columns the lender produces while deciding, such as
`int_rate`, `sub_grade` and `installment`, are left out on purpose: they already
contain the lender's own judgment of the risk, and a model that uses them is
not answering the question.

**This model is a class example of how a project is organized. It is not an
acceptable baseline for the course project.** It has no validation slice, no
treatment of the imbalance between the two outcomes, no decision threshold and
no justification of its features. Each group chooses its own features and
defends them in its own data audit.

## Rules of this project

- `data/raw` is read-only. `data/interim`, `data/processed`, `models`, and
  `reports/generated` are made by the code so that they can be deleted and
  rebuilt, and Git ignores them. `data/external` is ignored too, but the code
  cannot rebuild it: write down in `references` where each file came from.
- The reports you write (`reports/problem_statement.md`, `reports/data_audit.md`,
  the AI usage log) are your work. They live in `reports` and are committed.
- Secrets and settings of one computer go in `.env`, never in the code and
  never in Git.
- Notebooks are committed as `.py` files (jupytext), not as `.ipynb`: plain text
  is easy to read on GitHub and to merge, and no tables or charts end up in the
  history.
- `requirements.txt` says what we ask for. `requirements-installed.txt` says
  what we got. Both are committed.
- Every function has a docstring: what it does, what goes in, what comes out.
- Each commit does one thing, with a message that says what it does.

## Acknowledgments and references

The table shows what this project takes from elsewhere; the numbers refer to
the reference list below. Everything else was written for this class.

| Component | Source | Reference | License |
|---|---|---|---|
| Project structure: the `data` subfolders, `models`, `notebooks`, `references`, `reports`, the module layout of `src` (`config`, `dataset`, `features`, `modeling`, `plots`), and the name and unittest skeleton of `tests/test_data.py` | Cookiecutter Data Science | [1] | MIT |
| General section of `.gitignore`, copied unchanged | GitHub Python template | [2] | CC0-1.0 |
| Package list in `requirements.txt` | Course installation guide | [3] | Course material |
| Dataset, `data/raw/lending-club.csv`, described in `references/lending-club-consumer-credit-risk.pdf` | Lending Club loan data (Figshare) | [4] | CC0 |

### References

1. DrivenData. *Cookiecutter Data Science*. https://cookiecutter-data-science.drivendata.org/
2. GitHub. *A collection of .gitignore templates: Python.gitignore*. https://github.com/github/gitignore/blob/main/Python.gitignore
3. NOVA IMS. *Enterprise Data Science Bootcamp, 2026-27: software installation guide*. Course Moodle page.
4. Deepchecks Data (2023). *Lending Club* (version 4) [data set]. Figshare. https://doi.org/10.6084/m9.figshare.22121477.v4 Derived from the public Lending Club loan data, 2007 to 2018, on Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club

### Tools

The code uses pandas, scikit-learn, ydata-profiling, Jupytext, Black and the
Python standard library (`unittest`, `venv`). They are listed with their
versions in `requirements-installed.txt`.
