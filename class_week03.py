# %%
"""Class 3 of the Enterprise Data Science Bootcamp, as one file you follow top to bottom.

It is a notebook written as plain Python: every `# %%` starts a cell, and the cells
marked `[markdown]` are text. Run a cell with Shift+Enter. The file covers the Python
environment, Git, GitHub, notebooks and kernels, data profiling, and jupytext.

Functions defined here:
    monthly_payment: a small, untidy function used to show what Black does.
    git: runs a Git command and prints the answer.
"""

# %% [markdown]
# # Class 3. VS Code, Git, GitHub, notebooks
# Enterprise Data Science Bootcamp, NOVA IMS, 2026-27
#
# Follow this file from top to bottom. Every `# %%` starts a cell.
#
# - **DO** = something you click or type in VS Code. The Git command is shown beside it.
# - **RUN** = click inside the cell and press Shift+Enter.
# - **YOUR TURN** = a small exercise.
#
# On a Mac, use Cmd where we say Ctrl.

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 0. GET READY
# ----------------------------------------------------------------------
# - DO 0.1  Copy everything inside the class folder into your `edsb` folder, the one
#   that already has `.venv`. If asked to replace `requirements.txt`, say yes.
# - DO 0.2  In VS Code: File > Open Folder > `edsb`. Open `class_week03.py`. The status
#   bar (bottom right) must show `.venv`. If not: Ctrl+Shift+P > Python: Select Interpreter.
# - RUN 0.3  The cell below. You want `True` three times and `OK` three times.

# %%
import sys
from importlib.util import find_spec
from pathlib import Path

print("Python 3.13    :", sys.version_info[:2] == (3, 13))
print("Running .venv  :", ".venv" in sys.executable)
print("Data in place  :", Path("data/raw/lending-club.csv").exists())
for package in ["pandas", "ydata_profiling", "jupytext"]:
    print(f"{package:<15}:", "OK" if find_spec(package) else "MISSING")

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 1. PYTHON AND THE ENVIRONMENT
# ----------------------------------------------------------------------
# The project folders:
#
# - `data/raw` the data as we received it. We never change it.
# - `data/interim`, `data/processed` files our code creates. We can always rebuild them.
# - `data/external` data from other sources.
# - `notebooks` exploration.
# - `reports` what you write (tracked by Git). `reports/generated` what the code
#   writes (ignored).
# - `src` the project's own code, once it grows out of the notebook. `models` trained
#   models.
# - `tests` code that checks our code.
# - `references` guides, data dictionaries and papers.
#
# - DO 1.1  Open `requirements.txt`. It lists the packages this project needs, and the
#   lines starting with # explain how to build `.venv` from it. You did that at home.
# - DO 1.2  Terminal > New Terminal (the line starts with `(.venv)`), then type:
#   `python -m pip freeze > requirements-installed.txt`
# - RUN 1.3  The cell below. We asked for 10 packages. How many did we get?

# %%
for name in ["requirements.txt", "requirements-installed.txt"]:
    file = Path(name)
    lines = file.read_text(errors="ignore").splitlines() if file.exists() else []
    packages = [line for line in lines if line.strip() and not line.startswith("#")]
    print(f"{name:<27}: {len(packages)} packages")

# %% [markdown]
# - DO 1.4  Open `src/check_data.py`. Click Run Python File (the triangle, top right).
#   The result appears in the terminal.
# - RUN 1.5  Come back here and run the cell below. The result appears in the
#   Interactive Window. Click Variables there to see `df`.

# %%
import pandas as pd

df = pd.read_csv("data/raw/lending-club.csv")
df.shape

# %% [markdown]
# - YOUR TURN 1.6  The code below works, but it is untidy. Press Ctrl+S and watch
#   Black tidy it for you. Then run it.

# %%
def monthly_payment(amount,rate,months) :
    """Give the fixed monthly payment of a loan.

    Args:
        amount (float): The money borrowed.
        rate (float): The yearly interest rate, 0.12 for 12%. Zero is allowed.
        months (int): How many monthly payments.

    Returns:
        float: What is paid each month.
    """
    if rate==0 :
        return amount/months
    r=rate/12
    return amount*r/(1-(1+r)**-months)
print( monthly_payment(10000,0.12,36) )

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 2. GIT ON YOUR LAPTOP
# ----------------------------------------------------------------------
# - DO 2.1  Open `.gitignore`, scroll to the end and read the part called THIS PROJECT.
# - DO 2.2  Click the Source Control icon (Ctrl+Shift+G) > Initialize Repository.
#   Git: `git init`
#   Read the Changes list: `.env`, `.venv` and `lending-club.csv` are not there.
#   That is `.gitignore` working.
# - DO 2.3  Hover over Changes, click + (Stage All Changes), type the message
#   `Add class 3 project structure` and click Commit.
#   Git: `git add .` and `git commit -m "Add class 3 project structure"`
# - YOUR TURN 2.4  Open `README.md`. On the Maintainer line, replace `write your name
#   here` with your name and save. Click the file in Changes to read the diff: red is
#   what left, green is what came in. Stage it. Commit it with your own message.
# - RUN 2.5  The cell below shows your history.
#   Git: `git log --oneline`

# %%
import subprocess


def git(*args):
    """Run a Git command in this folder and print its output.

    Args:
        *args (str): The words of the command, one by one.
            `git("log", "--oneline")` runs `git log --oneline`.
    """
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    print(result.stdout or result.stderr)


git("log", "--oneline")

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 3. GITHUB
# ----------------------------------------------------------------------
# - DO 3.1  Source Control > Publish Branch > Publish to GitHub **private** repository.
#   Sign in to GitHub in the browser when VS Code asks, then come back.
#   Git: `git remote add origin <address>` and `git push -u origin main`
# - DO 3.2  Click Open on GitHub. Look inside `data/raw`: only `.gitkeep` is there.
#   The 60 MB file stayed on your laptop.
# - DO 3.3  Add one line to `README.md`, stage it, commit it. Then click Sync Changes.
#   Git: `git pull` and then `git push`
# - RUN 3.4  The cell below shows where your repository lives and if anything is waiting.

# %%
git("remote", "-v")
git("status", "-sb")

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 4. NOTEBOOK, KERNEL, PROFILING
# ----------------------------------------------------------------------
# - DO 4.1  In the terminal, type:
#   `jupytext --to ipynb notebooks/explore.py`
#   (if it is not found: `python -m jupytext --to ipynb notebooks/explore.py`)
# - DO 4.2  Open `notebooks/explore.ipynb`. Top right: Select Kernel > Python
#   Environments > `.venv`. Run the cells one by one with Shift+Enter, down to the end
#   of section 4. The profiling cell takes under a minute. Ignore its warning about a
#   new package name.
# - DO 4.3  In the Explorer, right-click `reports/generated/profile.html` > Reveal in
#   File Explorer (Mac: Reveal in Finder). Double-click it to open it in your browser.
# - YOUR TURN 4.4  Find three things in the report that the data dictionary in
#   `references` does not tell you. In the last cell of the notebook, write one line of
#   code that proves one of them. Save with Ctrl+S.

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 5. JUPYTEXT, THERE AND BACK
# ----------------------------------------------------------------------
# - DO 5.1  Source Control: click `notebooks/explore.ipynb` in Changes. VS Code shows a
#   tidy notebook view, but the file itself is one long JSON text with the results inside.
#   That is what GitHub shows, what Git must merge when two people edit it, and what stays
#   in the history, tables and charts included.
# - DO 5.2  Pair the notebook with the script. In the terminal:
#   `jupytext --set-formats ipynb,py:percent notebooks/explore.ipynb`
#   Then click `notebooks/explore.py` in Changes. The diff is short and readable: a small
#   header that jupytext adds at the top, and your line from 4.4 at the end.
# - DO 5.3  Open `.gitignore`. At the very end add the line `*.ipynb` and save. The
#   notebook leaves the Changes list: we commit the script, not the notebook. Stage all,
#   commit with the message `Add exploration notebook as a script`, then Sync Changes.

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 6. FINISH
# ----------------------------------------------------------------------
# - RUN 6.1  The cell below. The first part should be empty (nothing left to commit).
#   The second part is your history. If you get a NameError, run the cell in 2.5 first.

# %%
git("status", "--short")
git("log", "--oneline")

# %% [markdown]
# You can now:
#
# - build an environment from `requirements.txt`, and export what is installed with
#   `pip freeze`;
# - run Python in VS Code as a script, as a cell and in a notebook, always from `.venv`;
# - keep data, the environment and generated files out of Git with `.gitignore`, keep
#   the reports you write IN Git, and keep empty folders with `.gitkeep`;
# - stage, commit, read a diff, publish to GitHub and sync;
# - choose a kernel and profile a table before you trust what its documentation says;
# - keep a notebook reviewable with jupytext.
#
# The `src` and `tests` folders are explained in `README.md`, with the commands to run them.
#
# Before next class: read the pages marked with * in the Learning resources document.
# With your group: build the group repository the same way (one member publishes it as
# private, then adds the others and both lecturers as collaborators).
