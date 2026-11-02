# EDSB26_G9 — Diabetes Readmission Project Guide

Working guide for Paulo. Pulls together the course rules (Notion, Week 1 Extended Project Details), the official project guide (EDSB 2026 PDF), Sofia's preprocessing roadmap, and the three team meetings (EDSA Project Meeting Notes). Last updated: 18 Oct 2026.

---

## 1. The problem in one paragraph

At the moment of hospital discharge, estimate the risk that a diabetic patient is readmitted as an inpatient in **less than 30 days**, and use that risk to prioritise limited follow-up resources. Stakeholder: Director of Care Quality of a hospital network. The model predicts an observed outcome; it does **not** prove that follow-up reduces readmissions (say this explicitly in the presentation).

- Data: 101,766 encounters, 71,518 patients, 50 columns, US hospitals 1999–2008 (UCI Diabetes 130-US).
- Target: `readmitted == "<30"` → 1, `>30` and `NO` → 0. Positive rate is **11.2%** (checked on the file).

## 2. What the course expects (grading = 50% project / 50% exam)

**Deliverables**
- GitHub repo: `README.md`, docstrings on every function/class, pinned `requirements.txt`, `/src` and `/docs` folders, `.gitignore` (no venv, no `.env`, no large data), descriptive English commits.
- ZIP `EDSB26_9.zip` (confirm group number) with `EDSB26_9.csv` metadata (`group_letter,student_number,student_name`), `EDSB26_9_Presentation.pdf` (max **30 slides** incl. appendix), `/src`, `requirements.txt`. Every slide deck/PDF lists all names + student numbers.
- **Live Gradio demo** in the presentation (mandatory).
- Visuals follow **IBCS** (no pie charts, honest bar scales). Translate metrics into business terms (cost, capacity, risk).
- Responsible-AI disclosure: be ready to defend every split, feature and hyperparameter yourself — "the AI did it" is not accepted.

**Timeline**
| When | What |
|---|---|
| Week 3 | Status report 1 ✅ |
| Week 6 | Status report 2 ✅ (submitted at the Sunday 19:00 sync) |
| Week 9 | Status report 3 — status, challenges, learnings, **individual contributions**, AI usage, next steps (one form per group) |
| Weeks 9–10 | Course covers GitHub Actions CI/CD, Hugging Face Spaces, Gradio |
| Weeks 11–12 | **Final submission on Moodle** |
| Weeks 13–14 | Defence: 10 min executive presentation + 10 min technical Q&A |
| 27 Nov | Deep Learning midterm — team goal: preprocessing finished before this |

## 3. Team setup (from the meetings)

- **Henrique** — tech lead, repo admin. **Sofia** — demographics + diagnoses. **Paulo** — medications.
- Git: never commit to `main`; branch `<Name>_<Scope>` (e.g. `Paulo_EDA`), **Clear All Outputs** before committing notebooks, PR → 1 approval → merge, `git fetch --all && git pull` before starting.
- Env: `python -m venv .venv` → `.venv\Scripts\activate` (use `cmd` if PowerShell blocks it) → `pip install -r requirements.txt` → pick the `.venv` kernel.
- Storage: GitHub for code, Google Drive `EDS_Project` for raw data and HTML reports, Notion for notes, WhatsApp for quick syncs.
- EDA tooling: `ydata-profiling`, **one report per person on their own columns only** (full report is too heavy), saved to `reports/generated/`.
- Architecture idea (carried over from the MovieLens/Big Data project): local cleaning with DuckDB/Polars → Parquet → S3 + Athena. Optional for this project; don't let it block EDA.

## 4. Your scope: the medication block

**Confirmed by the team's column sheet (EDSB - Diabetes Data Organization.xlsx):** the **23 drug columns only**, `metformin` → `metformin-pioglitazone`. This supersedes the meeting notes.

- `change`, `diabetesMed`, `A1Cresult`, `max_glu_serum` are coloured as **Sofia's** in the sheet (the meeting notes had them under you or Henrique).
- `medical_specialty` → `number_inpatient` (7 columns) use a colour that isn't in the legend — worth checking with Henrique that someone owns them.

**First look at the data (run 18 Oct):**
| Group | Columns | Note |
|---|---|---|
| Constant — drop | `examide`, `citoglipton` | 100% `No` |
| Near-constant (<50 users) — drop or fold into "other" | `acetohexamide` (1), `troglitazone` (3), `glimepiride-pioglitazone` (1), `metformin-rosiglitazone` (2), `metformin-pioglitazone` (1), `glipizide-metformin` (13), `tolbutamide` (23), `miglitol` (38), `tolazamide` (39), `chlorpropamide` (86) | Too rare to learn from |
| Rare but usable | `acarbose` (308), `nateglinide` (703), `glyburide-metformin` (706), `repaglinide` (1,539) | Maybe keep as "taking yes/no" |
| Main drugs | `metformin` (~20k), `glipizide` (~13k), `glyburide` (~11k), `pioglitazone` (~7k), `rosiglitazone` (~6k), `glimepiride` (~5k) | Keep all 4 levels |
| Most important | `insulin` | 53% take it; Up/Down/Steady all large |

**Two loading traps (important):**
1. The file is **comma**-separated. Sofia's roadmap says `sep=";"` — that's wrong for the current file; the meeting fix (`delimiter=","`) is right.
2. `na_values="?"` alone turns the text **"None"** in `A1Cresult` / `max_glu_serum` into NaN, because pandas treats "None" as missing by default. But "None" means *test not done*, which is informative. Doesn't affect your drug columns, but **tell Sofia** — these are hers. Safe load for everyone:
   ```python
   df = pd.read_csv("data/raw/diabetic_data.csv", na_values="?", keep_default_na=False)
   ```

## 5. EDA plan — one small step at a time

Do one step per session. Each step ends with a short written note (2–3 bullets) in your notebook, so the Week 9 report and the slides write themselves.

- [x] **Step 1 — Load & sanity check.** Done 18 Oct — 101,766 × 50, no missing values in the 23 drug columns.
- [x] **Step 2 — Profile your block.** Done 18 Oct — 2 constant columns; imbalance flagged on every drug except insulin (too broad to act on); 1.2% "duplicates" are an artefact of profiling a column subset.
- [x] **Step 3 — Drop list.** Done 19 Oct — rule: keep if ≥1% of encounters take it (≥1,018).
  - `MED_KEEP` (8): insulin, metformin, glipizide, glyburide, pioglitazone, rosiglitazone, glimepiride, repaglinide.
  - `MED_DROP` (15): 13 rare + `examide`, `citoglipton` (constant). Rare drugs can still feed a "number of other diabetes drugs" count feature in Step 5.
  - Defence: natural gap between repaglinide (1,539) and the next drug (706); dropped drugs cover <2% of encounters combined.
- [x] **Step 4 — Medication vs target.** Done 1 Nov — chi-square per drug, Bonferroni p < 0.0063, Wilson 95% CIs.
  - Insulin strongest (p = 4e-41): <30 rate 10.0% No → 11.1% Steady → 13.0% Up → 13.9% Down.
  - Metformin lower when taken (Steady 9.7%, Up 8.2% vs No 11.5%) — likely confounding (withheld from sicker patients), not protective.
  - Other 6 drugs not significant after Bonferroni; kept as columns, no chart.
  - Sensitivity: without expired/hospice discharges (2,423 rows), patterns hold; insulin rates +0.2–0.3 pp, metformin users barely move.
- [ ] **Step 5 — Engineered features (draft).** `n_meds_taken`, `n_dose_changes` (Up+Down), `insulin_status`, maybe `any_oral_agent`. Check their <30 rates too.
- [ ] **Step 6 — Hand-off.** Reusable function(s) in `/src` (e.g. `src/features/medications.py`) with docstrings, PR for review.

Only after the team merges everyone's EDA: target definition, removing expired/hospice discharges (IDs 11, 13, 14, 19, 20, 21), grouped split by `patient_nbr`, modelling.

## 6. Later phases (parked — don't open until EDA is done)

1. **Clean rows & target** — remove death/hospice discharges; handle repeated patients (first encounter only vs grouped split). Keep the 3 `Unknown/Invalid` gender rows as unknown (official guide), not dropped as the roadmap suggests — team to agree.
2. **Feature engineering** — ICD-9 grouping (Strack et al., 9 groups), age bins → ordinal, ID collapsing via mapping, total prior visits.
3. **Pipeline** — `ColumnTransformer` fitted on train only; `StratifiedGroupKFold` by `patient_nbr`; 20% hold-out.
4. **Models** — logistic regression baseline → RF / XGBoost / LightGBM; `class_weight="balanced"` first, SMOTE only inside training folds.
5. **Evaluation** — ROC-AUC, PR-AUC, recall/precision at a tuned threshold; frame as "of the top X% riskiest discharges, we catch Y% of readmissions".
6. **Interpretation** — SHAP; sanity-check that prior inpatient visits and discharge disposition rank high.
7. **Delivery** — Gradio app, GitHub Actions, HF Space, 30-slide deck.

## 7. Where things live

- Notion: *Enterprise Data Science Bootcamp → Week 1 → Extended Project Details* (course rules); *EDSA Project Meeting Notes → Meetings 1–3*.
- Project files here: official guide PDF, Sofia's roadmap PDF, `diabetic_data.csv`, `IDS_mapping.csv`.
