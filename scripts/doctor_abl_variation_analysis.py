from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.data_utils import DATA_PATH, load_case_data

OUTPUT_DIR = ROOT / "outputs"

CASE_TIME = "CASE TIME (Cath In-Out)"
SKIN_SKIN = "SKIN-SKIN (Access to Cath-Out)"
PT_PREP = "PT PREP/INTUBATION Pt-In-Access"
ACCESS = "ACCESSS (Min)"
TSP = "TSP (Min)"
PRE_MAP = "PRE-MAP (Min)"
ABL_DURATION = "ABL DURATION (Abl Start-End)"
ABL_TIME = "ABL TIME (Min)"
POST_CARE = "POST CARE/EXTUBATION (Cath-Out to Pt-Out)"
ABL_COUNT = "#ABL"

PHASE_OUTCOMES = [
    CASE_TIME,
    SKIN_SKIN,
    PT_PREP,
    ACCESS,
    TSP,
    PRE_MAP,
    ABL_DURATION,
    POST_CARE,
]


def slug(text: str) -> str:
    clean = (
        text.lower()
        .replace("#", "num_")
        .replace("/", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace(",", "_")
    )
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")


def build_phase_effect_anova(standard_cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outcome in PHASE_OUTCOMES:
        data = standard_cases.dropna(subset=[outcome, "PHYSICIAN", ABL_COUNT]).copy()
        model = smf.ols(f'Q("{outcome}") ~ C(PHYSICIAN) + Q("{ABL_COUNT}")', data=data).fit()
        table = anova_lm(model, typ=2)
        total_sum_sq = table["sum_sq"].sum()

        for effect in ["C(PHYSICIAN)", f'Q("{ABL_COUNT}")']:
            rows.append(
                {
                    "outcome": outcome,
                    "effect": effect,
                    "n": len(data),
                    "r_squared": round(model.rsquared, 4),
                    "sum_sq": round(float(table.loc[effect, "sum_sq"]), 4),
                    "df": round(float(table.loc[effect, "df"]), 4),
                    "F": round(float(table.loc[effect, "F"]), 4),
                    "p_value": round(float(table.loc[effect, "PR(>F)"]), 6),
                    "eta_sq_total": round(float(table.loc[effect, "sum_sq"] / total_sum_sq), 4),
                }
            )

    return pd.DataFrame(rows)


def build_case_time_ancova(standard_cases: pd.DataFrame) -> pd.DataFrame:
    data = standard_cases.dropna(subset=[CASE_TIME, "PHYSICIAN", ABL_COUNT, ABL_TIME]).copy()
    model = smf.ols(
        f'Q("{CASE_TIME}") ~ C(PHYSICIAN) + Q("{ABL_COUNT}") + Q("{ABL_TIME}")',
        data=data,
    ).fit()
    table = anova_lm(model, typ=2)
    total_sum_sq = table["sum_sq"].sum()

    rows: list[dict[str, object]] = []
    for effect in ["C(PHYSICIAN)", f'Q("{ABL_COUNT}")', f'Q("{ABL_TIME}")']:
        rows.append(
            {
                "effect": effect,
                "n": len(data),
                "r_squared": round(model.rsquared, 4),
                "sum_sq": round(float(table.loc[effect, "sum_sq"]), 4),
                "df": round(float(table.loc[effect, "df"]), 4),
                "F": round(float(table.loc[effect, "F"]), 4),
                "p_value": round(float(table.loc[effect, "PR(>F)"]), 6),
                "eta_sq_total": round(float(table.loc[effect, "sum_sq"] / total_sum_sq), 4),
            }
        )

    return pd.DataFrame(rows)


def build_doctor_abl_cell_summary(standard_cases: pd.DataFrame, min_cell_size: int = 2) -> pd.DataFrame:
    metrics = [
        CASE_TIME,
        SKIN_SKIN,
        PT_PREP,
        ACCESS,
        TSP,
        PRE_MAP,
        ABL_DURATION,
        ABL_TIME,
        POST_CARE,
    ]

    rows: list[dict[str, object]] = []
    for (physician, abl_count), group in standard_cases.groupby(["PHYSICIAN", ABL_COUNT]):
        if len(group) < min_cell_size:
            continue

        row: dict[str, object] = {
            "physician": physician,
            "num_abl": int(abl_count),
            "cell_n": int(len(group)),
            "case_numbers": ",".join(str(int(case_no)) for case_no in group["CASE #"].tolist()),
        }

        for metric in metrics:
            values = group[metric].dropna()
            metric_slug = slug(metric)
            row[f"{metric_slug}_mean"] = round(float(values.mean()), 2)
            row[f"{metric_slug}_sd"] = round(float(values.std(ddof=1)), 2)
            row[f"{metric_slug}_range"] = round(float(values.max() - values.min()), 2)

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["physician", "num_abl"]).reset_index(drop=True)


def build_within_cell_variation_summary(cell_summary: pd.DataFrame) -> pd.DataFrame:
    metric_pairs = [
        (CASE_TIME, slug(CASE_TIME)),
        (SKIN_SKIN, slug(SKIN_SKIN)),
        (PT_PREP, slug(PT_PREP)),
        (ACCESS, slug(ACCESS)),
        (TSP, slug(TSP)),
        (PRE_MAP, slug(PRE_MAP)),
        (ABL_DURATION, slug(ABL_DURATION)),
        (ABL_TIME, slug(ABL_TIME)),
        (POST_CARE, slug(POST_CARE)),
    ]

    rows: list[dict[str, object]] = []
    for label, metric_slug in metric_pairs:
        rows.append(
            {
                "phase": label,
                "mean_within_cell_sd": round(float(cell_summary[f"{metric_slug}_sd"].mean()), 2),
                "median_within_cell_sd": round(float(cell_summary[f"{metric_slug}_sd"].median()), 2),
                "mean_within_cell_range": round(float(cell_summary[f"{metric_slug}_range"].mean()), 2),
                "median_within_cell_range": round(float(cell_summary[f"{metric_slug}_range"].median()), 2),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_within_cell_sd", ascending=False).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_cases = load_case_data(DATA_PATH)
    standard_cases = all_cases.loc[~all_cases["complex_case"]].copy()

    phase_effects = build_phase_effect_anova(standard_cases)
    case_time_ancova = build_case_time_ancova(standard_cases)
    cell_summary = build_doctor_abl_cell_summary(standard_cases, min_cell_size=2)
    within_cell_summary = build_within_cell_variation_summary(cell_summary)

    phase_effects.to_csv(OUTPUT_DIR / "phase_effect_anova.csv", index=False)
    case_time_ancova.to_csv(OUTPUT_DIR / "case_time_ancova.csv", index=False)
    cell_summary.to_csv(OUTPUT_DIR / "doctor_abl_cell_summary.csv", index=False)
    within_cell_summary.to_csv(OUTPUT_DIR / "within_cell_phase_variation.csv", index=False)

    print("Saved:")
    print("-", OUTPUT_DIR / "phase_effect_anova.csv")
    print("-", OUTPUT_DIR / "case_time_ancova.csv")
    print("-", OUTPUT_DIR / "doctor_abl_cell_summary.csv")
    print("-", OUTPUT_DIR / "within_cell_phase_variation.csv")
    print()
    print("Case-time ANCOVA")
    print(case_time_ancova.to_string(index=False))
    print()
    print("Top within-cell variation summary")
    print(within_cell_summary.to_string(index=False))


if __name__ == "__main__":
    main()
