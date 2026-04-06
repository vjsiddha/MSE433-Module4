# MSE433-Module4

Prototype workflow for **AF ablation surgical variability**: transcript-style process segments, mock multi-clinic cases, dashboards, and statistical analyses on real timing data.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `apps/` | Runnable applications (Streamlit UI). |
| `data/` | **Inputs only** — representative procedure transcript; place course Excel here (see below). |
| `notebooks/` | Jupyter analyses (`same_doctor_same_abl_analysis.ipynb`, `explaining_variation.ipynb`). |
| `outputs/` | Generated CSV, JSON, and HTML (mock variability tables, ANOVA outputs, dashboard build). |
| `scripts/` | Python modules and command-line pipelines. |

---

## Setup

From the **repository root** (`MSE433-Module4-1/`):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data you must provide

- **`data/MSE433_M4_Data.xlsx`** — cath-lab timing spreadsheet used by the doctor / `#ABL` analyses and the “explaining variation” notebook.  
  `scripts/data_utils.py` loads this path by default. Add your own copy under `data/` (it is not committed here).

- **`data/Procedure_Transcript.txt`** — representative narrative transcript (included in the repo).

---

## Run the analysis scripts

All commands assume the repository root as the current directory.

### Mock variability dataset + static HTML dashboard

Regenerates mock tables, `variability_schema.json`, dashboard JSON payload, and standalone HTML:

```bash
python scripts/build_variability_dashboard.py
```

```
open outputs/variability_dashboard.html
```

Open `outputs/variability_dashboard.html` in a browser.

### Mock data only (no dashboard)

```bash
python scripts/variability_data.py
```

### Transcript phase summaries (from `data/Procedure_Transcript.txt`)

```bash
python scripts/transcript_macro_analysis.py
```

Writes `outputs/transcript_phase_summary.csv` and `outputs/transcript_micro_step_coding.csv`.

### Real data: same-doctor, same–`#ABL` style stats

Requires `data/MSE433_M4_Data.xlsx`.

**Baseline ANOVA / within-cell summaries:**

```bash
python scripts/doctor_abl_variation_analysis.py
```

**Extended models** (additional ANCOVA tables, e.g. extended and pre-map controls):

```bash
python scripts/doctor_abl_variation_analysis_anova.py
```

Outputs include `outputs/phase_effect_anova.csv`, `outputs/case_time_ancova.csv`, `outputs/doctor_abl_cell_summary.csv`, `outputs/within_cell_phase_variation.csv`, and related CSVs documented in each script’s `main` printout.

### Regenerate the large same-doctor notebook (optional)

Requires `nbformat`:

```bash
python scripts/build_same_doctor_same_abl_notebook.py
```

Overwrites `notebooks/same_doctor_same_abl_analysis.ipynb`.

---

## Run the Jupyter notebooks

```bash
source .venv/bin/activate
jupyter lab
```

Open notebooks from **`notebooks/`**. The first cells resolve the repo root whether Jupyter’s working directory is the repo root or `notebooks/`.

### `notebooks/same_doctor_same_abl_analysis.ipynb`

**Data:** `data/MSE433_M4_Data.xlsx` plus functions from `scripts/` (e.g. `data_utils`, `doctor_abl_variation_analysis`, `transcript_macro_analysis`).

| Kind of work | What this notebook does |
|--------------|-------------------------|
| **Framing** | States the hypothesis, assumptions (representative transcript vs timed cases), and how the transcript was coded. |
| **EDA / cohort definition (Step 1)** | Loads cath-lab timing data, defines the analysis sample, and summarizes the comparison set. |
| **ANOVA / ANCOVA (Step 2)** | Fits models to see which factors matter for phase-level outcomes overall (omnibus-style inference). |
| **Core solution design (Step 3)** | Holds **physician** and **`#ABL`** fixed and quantifies **residual (within-cell) variation** across procedure phases. |
| **Derived metrics / EDA (Step 4)** | Examines **hidden-time proxies** (e.g. non-energy ablation time, post-ablation LA dwell) as direct outcomes of interest. |
| **Transcript integration (Step 5)** | Merges **transcript phase summaries** and **micro-step coding** with measured variability; compares transcript complexity to residual variation by phase. |
| **Synthesis** | Interprets category mix, complexity vs. variability, and a short **final conclusion** tying timing patterns to the process narrative. |

### `notebooks/explaining_variation.ipynb`

**Data:** `data/MSE433_M4_Data.xlsx`. The notebook **re-fits** ANCOVA-style models via `scripts/doctor_abl_variation_analysis_anova.py` helpers (same logic as the CSVs written to `outputs/`). Narrative in the notebook points to files such as `outputs/case_time_ancova.csv` and `outputs/within_cell_phase_variation.csv` for consistency with the batch scripts.

| Kind of work | What this notebook does |
|--------------|-------------------------|
| **Setup** | Repo path resolution and imports. |
| **EDA (Section 1)** | Describes **what the dataset contains** (columns, structure) and includes a **macro-phase visualization** (schematic timeline view). |
| **ANCOVA / inference (Section 2)** | **Case-time ANCOVA**: what explains **overall case time** (Type II ANOVA, effect summaries). Includes **extended predictor** variants (e.g. extra terms) as noted in the notebook. |
| **Residual / solution logic (Section 3)** | **Same doctor, same `#ABL`**: characterizes **residual variation** after fixing those factors (tables and focused comparisons, e.g. ablation window vs pulse-on time). |
| **EDA / comparison (Section 4)** | **Case time spread by physician** — distributional comparison across doctors. |

---

## What the mock layer represents

The mock pipeline encodes a **transcript-to-process** layer (stages, microsteps, times, actors, delay drivers) and a **cross-case variability** layer (doctors, clinics, benchmarks). Generated artifacts include:

- `variability_cases.csv`, `variability_stage_summary.csv`, `variability_transcript_annotations.csv`
- `variability_process_benchmarks.csv`, `variability_step_rankings.csv`, `variability_schema.json`

This is **synthetic**, not production transcription; the structure matches how a real multi-clinic implementation could store annotations and drive benchmarking.

---

## Assumption

The solution assumes every procedure can be segmented into timed transcript annotations joined with patient, provider, and clinic context. The mock generator illustrates that design; clinical validation is separate from this repository layout.
