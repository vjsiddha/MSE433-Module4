from __future__ import annotations

from html import escape
from pathlib import Path
import json
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.variability_data import OUTPUT_DIR, STAGES, benchmark_filter_mask, write_mock_outputs


PROCESS_LABELS = {
    "transfer_lift_used": "Transfer lift and extra hands",
    "dedicated_runner": "Dedicated runner",
    "ultrasound_access": "Ultrasound-first access",
    "standardized_sheath_pack": "Standardized sheath pack",
    "mapping_tech_support": "Mapping tech support",
    "closure_checklist": "Closure checklist",
}


def round_half(value: float) -> float:
    return round(float(value), 2)


def pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def reliability_index(mean_duration: float, sd_duration: float) -> int:
    if not mean_duration:
        return 0
    index = 100 - ((sd_duration / mean_duration) * 130)
    return int(max(52, min(98, round(index))))


def add_case_types(cases_df: pd.DataFrame) -> pd.DataFrame:
    cases = cases_df.copy()

    def classify(row: pd.Series) -> tuple[str, str]:
        if bool(row["prior_ablation"]):
            return "Redo / scar burden", "Repeat ablation cases with scar or prior work to navigate."
        if int(row["septum_thickness"]) >= 3:
            return "Thick septum crossing", "Cases where TSP crossing is usually the main source of delay."
        if bool(row["mobility_limited"]) or int(row["obesity_class"]) >= 2:
            return "Transfer-risk obesity", "Patient transfer and wake-up stages are more exposed to delay."
        if bool(row["persistent_af"]) or int(row["planned_lesions"]) >= 60:
            return "Persistent / high lesion", "Longer lesion sets and more mapping or verification work."
        return "Standard anatomy", "Lower-complexity cases with fewer obvious delay drivers."

    classified = cases.apply(classify, axis=1, result_type="expand")
    cases["case_type"] = classified[0]
    cases["case_type_note"] = classified[1]
    return cases


def top_driver_rows(series: pd.Series, limit: int = 3) -> list[dict[str, object]]:
    counts = series.loc[series != "Routine variation"].value_counts().head(limit)
    total = int(counts.sum())
    rows = []
    for label, count in counts.items():
        rows.append(
            {
                "label": str(label),
                "count": int(count),
                "share": pct(count / total) if total else "0%",
            }
        )
    return rows


def driver_text(series: pd.Series, limit: int = 3) -> str:
    counts = series.loc[series != "Routine variation"].value_counts().head(limit)
    if counts.empty:
        return "Routine variation"
    return ", ".join(f"{label} ({count})" for label, count in counts.items())


def stage_summary_rows(stages_df: pd.DataFrame) -> list[dict[str, object]]:
    stage_catalog = pd.DataFrame(STAGES)[["stage_code", "stage_name", "stage_order", "color"]]
    group = (
        stages_df.groupby(["stage_code", "stage_name", "stage_order"], as_index=False)["stage_duration_min"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_duration_min", "std": "sd_duration_min"})
        .merge(stage_catalog, on=["stage_code", "stage_name", "stage_order"], how="left")
        .sort_values("stage_order")
    )
    rows = []
    for _, row in group.iterrows():
        rows.append(
            {
                "stage_code": row["stage_code"],
                "stage_name": row["stage_name"],
                "stage_order": int(row["stage_order"]),
                "color": row["color"],
                "mean_duration_min": round_half(float(row["mean_duration_min"])),
                "sd_duration_min": round_half(float(row["sd_duration_min"])),
            }
        )
    return rows


def build_recommendations(cases_df: pd.DataFrame, benchmarks_df: pd.DataFrame) -> list[dict[str, object]]:
    recommendations: list[dict[str, object]] = []

    for clinic_name, case_group in cases_df.groupby("clinic_name", sort=True):
        for _, row in benchmarks_df.iterrows():
            process_flag = str(row["process_flag"])
            eligible_cases = case_group.loc[benchmark_filter_mask(case_group, str(row["filter_name"]))]
            if eligible_cases.empty:
                continue

            adoption = float(eligible_cases[process_flag].mean())
            if float(row["mean_minutes_saved"]) < 0.8 or float(row["sd_reduction_min"]) < 0 or adoption >= 0.82:
                continue

            impact_score = float(row["mean_minutes_saved"]) * (1.0 - adoption) + float(row["sd_reduction_min"])
            recommendations.append(
                {
                    "clinic_name": clinic_name,
                    "stage_name": str(row["stage_name"]),
                    "process_label": str(row["process_label"]),
                    "adoption_rate": pct(adoption),
                    "mean_minutes_saved": round_half(float(row["mean_minutes_saved"])),
                    "sd_reduction_min": round_half(float(row["sd_reduction_min"])),
                    "impact_score": round_half(impact_score),
                    "recommendation_text": (
                        f"Standardize {row['process_label']} in {row['stage_name']}: "
                        f"{float(row['mean_minutes_saved']):.1f} min faster with "
                        f"{float(row['sd_reduction_min']):.1f} min less spread. Current eligible-case adoption is {pct(adoption)}."
                    ),
                }
            )

    recommendations.sort(key=lambda item: item["impact_score"], reverse=True)
    return recommendations[:6]


def build_summary_cards(cases_df: pd.DataFrame, stages_df: pd.DataFrame, case_types_df: pd.DataFrame) -> list[dict[str, str]]:
    clinic_group = cases_df.groupby("clinic_name")["total_case_duration_min"].mean().sort_values()
    stage_variability = stages_df.groupby("stage_name")["stage_duration_min"].std(ddof=1).sort_values(ascending=False)
    case_type_group = case_types_df.groupby("case_type")["total_case_duration_min"].mean().sort_values(ascending=False)

    return [
        {
            "label": "Total mock cases",
            "value": str(int(cases_df["case_id"].nunique())),
            "detail": "Every case has stage and microstep timing.",
        },
        {
            "label": "Fastest clinic",
            "value": clinic_group.index[0],
            "detail": f"{clinic_group.iloc[0]:.1f} min average case time.",
        },
        {
            "label": "Most variable stage",
            "value": stage_variability.index[0],
            "detail": f"{stage_variability.iloc[0]:.1f} min SD across all cases.",
        },
        {
            "label": "Slowest case type",
            "value": case_type_group.index[0],
            "detail": f"{case_type_group.iloc[0]:.1f} min average case time.",
        },
    ]


def build_clinic_profiles(cases_df: pd.DataFrame, stages_df: pd.DataFrame, recommendations: list[dict[str, object]]) -> list[dict[str, object]]:
    profiles: list[dict[str, object]] = []

    for rank, (clinic_name, case_group) in enumerate(cases_df.groupby("clinic_name", sort=True), start=1):
        clinic_stages = stages_df.loc[stages_df["clinic_name"] == clinic_name]
        stage_rows = stage_summary_rows(clinic_stages)
        mean_case = float(case_group["total_case_duration_min"].mean())
        sd_case = float(case_group["total_case_duration_min"].std(ddof=1))

        stage_sd = clinic_stages.groupby("stage_name")["stage_duration_min"].std(ddof=1).sort_values(ascending=False)
        top_doctor = (
            case_group.groupby("doctor_name")["total_case_duration_min"].mean().sort_values().index[0]
        )
        top_case_type = (
            case_group.groupby("case_type")["total_case_duration_min"].mean().sort_values(ascending=False).index[0]
        )

        clinic_reco = next((item for item in recommendations if item["clinic_name"] == clinic_name), None)
        top_process = max(
            PROCESS_LABELS.items(),
            key=lambda item: float(case_group[item[0]].mean()),
        )

        profiles.append(
            {
                "rank": rank,
                "clinic_name": clinic_name,
                "city": str(case_group["clinic_city"].iloc[0]),
                "cases": int(case_group["case_id"].nunique()),
                "mean_case_duration_min": round_half(mean_case),
                "sd_case_duration_min": round_half(sd_case),
                "reliability_index": reliability_index(mean_case, sd_case),
                "most_variable_stage": stage_sd.index[0],
                "top_doctor": str(top_doctor),
                "heaviest_case_type": str(top_case_type),
                "top_process": f"{top_process[1]} ({pct(float(case_group[top_process[0]].mean()))})",
                "top_opportunity": clinic_reco["recommendation_text"] if clinic_reco else "No major modeled gap on current benchmarks.",
                "stage_rows": stage_rows,
            }
        )

    profiles.sort(key=lambda item: item["mean_case_duration_min"])
    for index, profile in enumerate(profiles, start=1):
        profile["rank"] = index
    return profiles


def case_flags(case_row: pd.Series) -> list[str]:
    flags: list[str] = []
    if int(case_row["obesity_class"]) >= 2:
        flags.append("Obese")
    if bool(case_row["mobility_limited"]):
        flags.append("Mobility limited")
    if bool(case_row["prior_ablation"]):
        flags.append("Prior ablation")
    if int(case_row["septum_thickness"]) >= 3:
        flags.append("Thick septum")
    if bool(case_row["persistent_af"]):
        flags.append("Persistent AF")
    return flags or ["Standard"]


def flagged_events_from_timeline(timeline: pd.DataFrame, limit: int = 4) -> list[dict[str, object]]:
    flagged = (
        timeline.loc[timeline["time_deviation_min"] > 0.75]
        .sort_values("time_deviation_min", ascending=False)
        .head(limit)
    )
    return [
        {
            "timestamp": str(row["timestamp_label"]),
            "action": str(row["microstep_label"]),
            "category": str(row["category"]),
            "driver": str(row["primary_delay_driver"]),
            "deviation_min": round_half(float(row["time_deviation_min"])),
        }
        for _, row in flagged.iterrows()
    ]


def build_doctor_profiles(cases_df: pd.DataFrame, stages_df: pd.DataFrame, steps_df: pd.DataFrame) -> list[dict[str, object]]:
    cohort_stage_means = stages_df.groupby("stage_code")["stage_duration_min"].mean().to_dict()
    expected_df = build_expected_step_benchmarks(cases_df, steps_df)
    profiles: list[dict[str, object]] = []

    for doctor_name, case_group in cases_df.groupby("doctor_name", sort=True):
        doctor_stages = stages_df.loc[stages_df["doctor_name"] == doctor_name]
        stage_rows = stage_summary_rows(doctor_stages)
        for row in stage_rows:
            row["delta_vs_network_min"] = round_half(float(row["mean_duration_min"]) - float(cohort_stage_means[row["stage_code"]]))

        mean_case = float(case_group["total_case_duration_min"].mean())
        sd_case = float(case_group["total_case_duration_min"].std(ddof=1))
        stage_sd = doctor_stages.groupby("stage_name")["stage_duration_min"].std(ddof=1).sort_values(ascending=False)
        case_mix = case_group.groupby("case_type")["case_id"].nunique().sort_values(ascending=False)
        case_rows: list[dict[str, object]] = []

        for _, case_row in case_group.sort_values("total_case_duration_min", ascending=False).iterrows():
            timeline = build_case_timeline(case_row, cases_df, steps_df, expected_df)
            stage_breakdown = (
                stages_df.loc[stages_df["case_id"] == case_row["case_id"], ["stage_name", "stage_duration_min"]]
                .sort_values("stage_duration_min", ascending=False)
            )
            case_rows.append(
                {
                    "case_id": str(case_row["case_id"]),
                    "case_type": str(case_row["case_type"]),
                    "total_case_duration_min": round_half(float(case_row["total_case_duration_min"])),
                    "primary_delay_driver": str(case_row["primary_delay_driver"]),
                    "flags": case_flags(case_row),
                    "flagged_events": flagged_events_from_timeline(timeline),
                    "stage_breakdown": [
                        {
                            "stage_name": str(row["stage_name"]),
                            "stage_duration_min": round_half(float(row["stage_duration_min"])),
                        }
                        for _, row in stage_breakdown.iterrows()
                    ],
                    "timeline_rows": [
                        {
                            "timestamp": str(row["timestamp_label"]),
                            "phase": str(row["stage_name"]),
                            "action": str(row["microstep_label"]),
                            "category": str(row["category"]),
                            "actual_min": round_half(float(row["actual_time_min"])),
                            "deviation_min": round_half(float(row["time_deviation_min"])),
                            "driver": str(row["primary_delay_driver"]),
                        }
                        for _, row in timeline.iterrows()
                    ],
                }
            )

        profiles.append(
            {
                "doctor_name": doctor_name,
                "doctor_slug": doctor_name.lower().replace(" ", "-").replace(".", ""),
                "clinic_name": str(case_group["clinic_name"].iloc[0]),
                "clinic_slug": str(case_group["clinic_name"].iloc[0]).lower().replace(" ", "-"),
                "cases": int(case_group["case_id"].nunique()),
                "mean_case_duration_min": round_half(mean_case),
                "sd_case_duration_min": round_half(sd_case),
                "reliability_index": reliability_index(mean_case, sd_case),
                "most_variable_stage": stage_sd.index[0],
                "top_delay_drivers": driver_text(case_group["primary_delay_driver"]),
                "largest_case_type": f"{case_mix.index[0]} ({int(case_mix.iloc[0])} cases)",
                "stage_rows": stage_rows,
                "case_rows": case_rows,
            }
        )

    profiles.sort(key=lambda item: (item["clinic_name"], item["mean_case_duration_min"]))
    return profiles


def build_case_profiles(cases_df: pd.DataFrame, stages_df: pd.DataFrame, steps_df: pd.DataFrame) -> list[dict[str, object]]:
    expected_df = build_expected_step_benchmarks(cases_df, steps_df)
    rows: list[dict[str, object]] = []
    for _, case_row in cases_df.sort_values("total_case_duration_min", ascending=False).iterrows():
        timeline = build_case_timeline(case_row, cases_df, steps_df, expected_df)
        stage_breakdown = (
            stages_df.loc[stages_df["case_id"] == case_row["case_id"], ["stage_name", "stage_duration_min"]]
            .sort_values("stage_duration_min", ascending=False)
        )
        rows.append(
            {
                "case_id": str(case_row["case_id"]),
                "clinic_name": str(case_row["clinic_name"]),
                "clinic_slug": str(case_row["clinic_name"]).lower().replace(" ", "-"),
                "doctor_name": str(case_row["doctor_name"]),
                "doctor_slug": str(case_row["doctor_name"]).lower().replace(" ", "-").replace(".", ""),
                "case_type": str(case_row["case_type"]),
                "total_case_duration_min": round_half(float(case_row["total_case_duration_min"])),
                "primary_delay_driver": str(case_row["primary_delay_driver"]),
                "flags": case_flags(case_row),
                "flagged_events": flagged_events_from_timeline(timeline),
                "stage_breakdown": [
                    {
                        "stage_name": str(row["stage_name"]),
                        "stage_duration_min": round_half(float(row["stage_duration_min"])),
                    }
                    for _, row in stage_breakdown.iterrows()
                ],
                "timeline_rows": [
                    {
                        "timestamp": str(row["timestamp_label"]),
                        "phase": str(row["stage_name"]),
                        "action": str(row["microstep_label"]),
                        "category": str(row["category"]),
                        "actual_min": round_half(float(row["actual_time_min"])),
                        "deviation_min": round_half(float(row["time_deviation_min"])),
                        "driver": str(row["primary_delay_driver"]),
                    }
                    for _, row in timeline.iterrows()
                ],
            }
        )
    return rows


def build_stage_variance_profiles(cases_df: pd.DataFrame, stages_df: pd.DataFrame, steps_df: pd.DataFrame) -> list[dict[str, object]]:
    profiles: list[dict[str, object]] = []

    for stage in STAGES:
        stage_code = str(stage["stage_code"])
        stage_group = stages_df.loc[stages_df["stage_code"] == stage_code].copy()
        step_group = steps_df.loc[steps_df["stage_code"] == stage_code].copy()
        stage_with_type = stage_group.copy()

        microsteps = (
            step_group.groupby(["microstep_label"], as_index=False)["duration_min"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "mean_duration_min", "std": "sd_duration_min"})
            .sort_values("sd_duration_min", ascending=False)
            .head(3)
        )

        top_case_types = (
            stage_with_type.groupby("case_type")["stage_duration_min"]
            .mean()
            .sort_values(ascending=False)
            .head(2)
        )

        profiles.append(
            {
                "stage_code": stage_code,
                "stage_name": str(stage["stage_name"]),
                "color": str(stage["color"]),
                "mean_duration_min": round_half(float(stage_group["stage_duration_min"].mean())),
                "sd_duration_min": round_half(float(stage_group["stage_duration_min"].std(ddof=1))),
                "p90_duration_min": round_half(float(stage_group["stage_duration_min"].quantile(0.9))),
                "common_drivers": top_driver_rows(stage_group["primary_delay_driver"]),
                "top_case_types": [
                    {"label": str(label), "value": round_half(float(value))}
                    for label, value in top_case_types.items()
                ],
                "top_microsteps": [
                    {
                        "label": str(row["microstep_label"]),
                        "mean_duration_min": round_half(float(row["mean_duration_min"])),
                        "sd_duration_min": round_half(float(row["sd_duration_min"])),
                    }
                    for _, row in microsteps.iterrows()
                ],
            }
        )

    return profiles


def build_expected_step_benchmarks(cases_df: pd.DataFrame, steps_df: pd.DataFrame) -> pd.DataFrame:
    case_cutoffs = (
        cases_df.groupby("case_type")["total_case_duration_min"]
        .quantile(0.25)
        .rename("ideal_cutoff")
        .reset_index()
    )

    with_cutoffs = steps_df.merge(case_cutoffs, on="case_type", how="left")
    case_times = cases_df[["case_id", "total_case_duration_min"]]
    with_cutoffs = with_cutoffs.merge(case_times, on="case_id", how="left")

    ideal_subset = with_cutoffs.loc[with_cutoffs["total_case_duration_min"] <= with_cutoffs["ideal_cutoff"]].copy()
    fallback_cutoff = float(cases_df["total_case_duration_min"].quantile(0.25))
    fallback_cases = set(cases_df.loc[cases_df["total_case_duration_min"] <= fallback_cutoff, "case_id"].tolist())
    fallback_subset = with_cutoffs.loc[with_cutoffs["case_id"].isin(fallback_cases)].copy()

    expected_case_type = (
        ideal_subset.groupby(["case_type", "microstep_code"], as_index=False)["duration_min"]
        .mean()
        .rename(columns={"duration_min": "expected_time_min"})
    )
    fallback_expected = (
        fallback_subset.groupby("microstep_code", as_index=False)["duration_min"]
        .mean()
        .rename(columns={"duration_min": "fallback_expected_time_min"})
    )
    mean_times = (
        steps_df.groupby("microstep_code", as_index=False)["duration_min"]
        .mean()
        .rename(columns={"duration_min": "mean_time_min"})
    )

    return expected_case_type.merge(fallback_expected, on="microstep_code", how="outer").merge(mean_times, on="microstep_code", how="left")


def build_case_timeline(case_row: pd.Series, cases_df: pd.DataFrame, steps_df: pd.DataFrame, expected_df: pd.DataFrame) -> pd.DataFrame:
    case_steps = steps_df.loc[steps_df["case_id"] == case_row["case_id"]].copy()
    case_steps = case_steps.merge(expected_df, on=["case_type", "microstep_code"], how="left")
    case_steps["expected_time_min"] = case_steps["expected_time_min"].fillna(case_steps["fallback_expected_time_min"])
    case_steps["actual_time_min"] = case_steps["duration_min"]
    case_steps["time_deviation_min"] = case_steps["actual_time_min"] - case_steps["expected_time_min"]
    case_steps["timestamp_label"] = case_steps["transcript_start_min"].map(lambda value: f"{float(value):05.1f}m")
    return case_steps.sort_values(["stage_order", "microstep_order"]).reset_index(drop=True)


def build_diagnostic_case(cases_df: pd.DataFrame, steps_df: pd.DataFrame) -> dict[str, object]:
    expected_df = build_expected_step_benchmarks(cases_df, steps_df)
    best_case_id = None
    best_delay = -1.0
    best_timeline: pd.DataFrame | None = None

    for _, case_row in cases_df.iterrows():
        timeline = build_case_timeline(case_row, cases_df, steps_df, expected_df)
        total_delay = float(timeline["time_deviation_min"].clip(lower=0).sum())
        if total_delay > best_delay:
            best_delay = total_delay
            best_case_id = str(case_row["case_id"])
            best_timeline = timeline

    case_row = cases_df.loc[cases_df["case_id"] == best_case_id].iloc[0]
    timeline_df = best_timeline if best_timeline is not None else build_case_timeline(case_row, cases_df, steps_df, expected_df)
    delayed_steps = timeline_df.loc[timeline_df["time_deviation_min"] > 0].copy()

    driver_rows = (
        delayed_steps.groupby("category", as_index=False)["time_deviation_min"]
        .sum()
        .sort_values("time_deviation_min", ascending=False)
        .head(4)
    )
    total_delay = float(driver_rows["time_deviation_min"].sum()) if not driver_rows.empty else 0.0
    driver_list = [
        {
            "label": str(row["category"]),
            "delay_min": round_half(float(row["time_deviation_min"])),
            "share": pct(float(row["time_deviation_min"]) / total_delay) if total_delay else "0%",
        }
        for _, row in driver_rows.iterrows()
    ]

    root_stage = (
        timeline_df.groupby("stage_name")["time_deviation_min"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )
    root_stage_rows = timeline_df.loc[timeline_df["stage_name"] == root_stage].copy()
    root_microsteps = (
        root_stage_rows.loc[root_stage_rows["time_deviation_min"] > 0]
        .sort_values("time_deviation_min", ascending=False)
        .head(4)
    )
    root_rows = [
        {
            "microstep": str(row["microstep_label"]),
            "category": str(row["category"]),
            "deviation_min": round_half(float(row["time_deviation_min"])),
            "cause": str(row["primary_delay_driver"]),
        }
        for _, row in root_microsteps.iterrows()
    ]

    peer_cases = cases_df.loc[cases_df["case_type"] == case_row["case_type"]].copy()
    peer_cutoff = float(peer_cases["total_case_duration_min"].quantile(0.25))
    ideal_case_ids = peer_cases.loc[peer_cases["total_case_duration_min"] <= peer_cutoff, "case_id"].tolist()
    ideal_steps = steps_df.loc[steps_df["case_id"].isin(ideal_case_ids)].copy()
    ideal_map = (
        ideal_steps.groupby(["stage_name", "microstep_order", "microstep_label"], as_index=False)["duration_min"]
        .mean()
        .rename(columns={"duration_min": "ideal_time_min"})
    )
    compare = timeline_df.merge(
        ideal_map,
        on=["stage_name", "microstep_order", "microstep_label"],
        how="left",
    )
    compare["ideal_gap_min"] = compare["actual_time_min"] - compare["ideal_time_min"]
    phase_compare = (
        compare.groupby("stage_name", as_index=False)
        .agg(
            ideal_time_min=("ideal_time_min", "sum"),
            actual_time_min=("actual_time_min", "sum"),
            ideal_gap_min=("ideal_gap_min", "sum"),
        )
        .sort_values("ideal_gap_min", ascending=False)
    )
    ideal_rows = [
        {
            "phase": str(row["stage_name"]),
            "ideal_min": round_half(float(row["ideal_time_min"])),
            "actual_min": round_half(float(row["actual_time_min"])),
            "gap_min": round_half(float(row["ideal_gap_min"])),
        }
        for _, row in phase_compare.iterrows()
    ]

    top_opportunities = (
        compare.loc[compare["ideal_gap_min"] > 1.0]
        .sort_values("ideal_gap_min", ascending=False)
        .head(4)
    )
    opportunity_rows = [
        {
            "microstep": str(row["microstep_label"]),
            "gap_min": round_half(float(row["ideal_gap_min"])),
            "suggestion": str(row["primary_delay_driver"]),
        }
        for _, row in top_opportunities.iterrows()
    ]

    timeline_rows = [
        {
            "timestamp": str(row["timestamp_label"]),
            "phase": str(row["stage_name"]),
            "action": str(row["microstep_label"]),
            "category": str(row["category"]),
            "expected_min": round_half(float(row["expected_time_min"])),
            "mean_min": round_half(float(row["mean_time_min"])),
            "actual_min": round_half(float(row["actual_time_min"])),
            "deviation_min": round_half(float(row["time_deviation_min"])),
            "driver": str(row["primary_delay_driver"]),
        }
        for _, row in timeline_df.head(8).iterrows()
    ]

    return {
        "case_id": str(case_row["case_id"]),
        "clinic_name": str(case_row["clinic_name"]),
        "doctor_name": str(case_row["doctor_name"]),
        "case_type": str(case_row["case_type"]),
        "total_delay_min": round_half(float(timeline_df["time_deviation_min"].clip(lower=0).sum())),
        "root_stage": str(root_stage),
        "timeline_rows": timeline_rows,
        "driver_rows": driver_list,
        "root_rows": root_rows,
        "ideal_rows": ideal_rows,
        "opportunity_rows": opportunity_rows,
    }


def build_payload(dataset: dict[str, object]) -> dict[str, object]:
    cases_df = add_case_types(dataset["cases"])
    stages_df = dataset["stages"].merge(
        cases_df[["case_id", "case_type"]],
        on="case_id",
        how="left",
    )
    steps_df = dataset["steps"].merge(
        cases_df[["case_id", "case_type"]],
        on="case_id",
        how="left",
    )
    recommendations = build_recommendations(cases_df, dataset["benchmarks"])

    return {
        "summary_cards": build_summary_cards(cases_df, stages_df, cases_df),
        "clinic_profiles": build_clinic_profiles(cases_df, stages_df, recommendations),
        "doctor_profiles": build_doctor_profiles(cases_df, stages_df, steps_df),
        "case_profiles": build_case_profiles(cases_df, stages_df, steps_df),
        "stage_variance_profiles": build_stage_variance_profiles(cases_df, stages_df, steps_df),
        "diagnostic_case": build_diagnostic_case(cases_df, steps_df),
        "recommendations": recommendations,
    }


def render_summary_cards(cards: list[dict[str, str]]) -> str:
    return "\n".join(
        f"""
        <article class="metric-card">
          <div class="metric-label">{escape(card["label"])}</div>
          <div class="metric-value">{escape(card["value"])}</div>
          <div class="metric-note">{escape(card["detail"])}</div>
        </article>
        """
        for card in cards
    )


def render_hierarchy() -> str:
    levels = [
        ("1", "Clinic", "Site"),
        ("2", "Doctor", "Operator"),
        ("3", "Case Type", "Mix"),
    ]
    return "\n".join(
        f"""
        <article class="hierarchy-card">
          <div class="hierarchy-step">{step}</div>
          <div class="hierarchy-title">{escape(title)}</div>
          <p>{escape(body)}</p>
        </article>
        """
        for step, title, body in levels
    )


def render_stage_profile_bars(stage_rows: list[dict[str, object]], scale: float) -> str:
    bars = []
    for row in stage_rows:
        width = (float(row["mean_duration_min"]) / scale) * 100 if scale else 0
        bars.append(
            f'<span class="mini-bar" style="width:{width:.2f}%; background:{row["color"]};" title="{escape(row["stage_name"])} {row["mean_duration_min"]:.1f} min"></span>'
        )
    return "".join(bars)


def render_stage_profile_labeled(stage_rows: list[dict[str, object]], scale: float) -> str:
    segments = []
    for row in stage_rows:
        width = (float(row["mean_duration_min"]) / scale) * 100 if scale else 0
        segments.append(
            f"""
            <div class="mini-stage-segment" style="width:{width:.2f}%;">
              <div class="mini-stage-label">{escape(row["stage_name"])} {row["mean_duration_min"]:.1f}m</div>
              <span class="mini-bar" style="width:100%; background:{row["color"]};" title="{escape(row["stage_name"])} {row["mean_duration_min"]:.1f} min"></span>
            </div>
            """
        )
    return "".join(segments)


def render_clinic_table(profiles: list[dict[str, object]]) -> str:
    scale = max(profile["mean_case_duration_min"] for profile in profiles)
    rows = []
    for profile in profiles:
        total_width = (float(profile["mean_case_duration_min"]) / scale) * 100 if scale else 0
        rows.append(
            f"""
            <tr>
              <td>{profile['rank']}</td>
              <td>
                <div class="entity-name">{escape(profile['clinic_name'])}</div>
                <div class="entity-sub">{escape(profile['city'])} · {profile['cases']} cases</div>
              </td>
              <td>
                <div class="score-main">{profile['mean_case_duration_min']:.1f} min</div>
                <div class="score-sub">{profile['sd_case_duration_min']:.1f} min SD</div>
              </td>
              <td>{profile['reliability_index']}</td>
              <td>{escape(profile['most_variable_stage'])}</td>
              <td>{escape(profile['top_doctor'])}</td>
              <td>{escape(profile['heaviest_case_type'])}</td>
              <td>{escape(profile['top_process'])}</td>
              <td>
                <div class="row-bar-track">
                  <div class="row-bar-fill" style="width:{total_width:.2f}%;"></div>
                </div>
              </td>
            </tr>
            """
        )
    return "\n".join(rows)


def render_clinic_explorer(profiles: list[dict[str, object]]) -> str:
    scale = max(
        stage["mean_duration_min"]
        for profile in profiles
        for stage in profile["stage_rows"]
    )
    blocks = []
    for profile in profiles:
        stage_rows = "".join(
            f"""
            <div class="compact-stage-row">
              <div>{escape(row['stage_name'])}</div>
              <div class="compact-stage-track">
                <div class="compact-stage-fill" style="width:{(float(row['mean_duration_min']) / scale) * 100:.2f}%; background:{row['color']};"></div>
              </div>
              <div>{row['mean_duration_min']:.1f}m</div>
              <div>{row['sd_duration_min']:.1f} SD</div>
              <div></div>
            </div>
            """
            for row in profile["stage_rows"]
        )
        blocks.append(
            f"""
            <details class="dashboard-card filterable-clinic aggregate-card" data-aggregate="clinic" data-clinic="{escape(profile['clinic_name'].lower().replace(' ', '-'))}" data-doctor="all">
              <summary>
                <div class="card-head">
                  <div>
                    <div class="entity-name">{escape(profile['clinic_name'])}</div>
                    <div class="entity-sub">{escape(profile['city'])} · {profile['cases']} cases · rel {profile['reliability_index']}</div>
                  </div>
                  <div class="summary-metrics">
                    <span>{profile['mean_case_duration_min']:.1f} min mean</span>
                    <span>{profile['sd_case_duration_min']:.1f} min SD</span>
                  </div>
                </div>
                <div class="mini-profile-bars">{render_stage_profile_bars(profile['stage_rows'], scale)}</div>
              </summary>
              <div class="card-body">
                <div class="note-grid">
                  <div><strong>Var stage</strong><span>{escape(profile['most_variable_stage'])}</span></div>
                  <div><strong>Fastest doctor</strong><span>{escape(profile['top_doctor'])}</span></div>
                  <div><strong>Top process</strong><span>{escape(profile['top_process'])}</span></div>
                </div>
                <div class="compact-stage-table">{stage_rows}</div>
              </div>
            </details>
            """
        )
    return "\n".join(blocks)


def render_doctor_explorer(profiles: list[dict[str, object]]) -> str:
    scale = max(
        stage["mean_duration_min"]
        for profile in profiles
        for stage in profile["stage_rows"]
    )
    blocks = []
    for profile in profiles:
        stage_rows = []
        for row in profile["stage_rows"]:
            delta = float(row["delta_vs_network_min"])
            stage_rows.append(
                f"""
                <div class="compact-stage-row">
                  <div>{escape(row['stage_name'])}</div>
                  <div class="compact-stage-track">
                    <div class="compact-stage-fill" style="width:{(float(row['mean_duration_min']) / scale) * 100:.2f}%; background:{row['color']};"></div>
                  </div>
                  <div>{row['mean_duration_min']:.1f}m</div>
                  <div>{row['sd_duration_min']:.1f} SD</div>
                  <div>{delta:+.1f}m</div>
                </div>
                """
            )

        case_blocks = []
        for case in profile["case_rows"]:
            stage_pills = "".join(
                f'<span class="pill">{escape(item["stage_name"])} {item["stage_duration_min"]:.1f}m</span>'
                for item in case["stage_breakdown"][:4]
            )
            flag_pills = "".join(f'<span class="pill">{escape(flag)}</span>' for flag in case["flags"])
            event_rows = "".join(
                f'<li class="event-flag"><div><span class="event-symbol">!</span>{escape(item["timestamp"])} · {escape(item["action"])}</div><span>{item["deviation_min"]:+.1f}m · {escape(item["driver"])}</span></li>'
                for item in case["flagged_events"]
            ) or '<li><div><span class="event-symbol event-symbol-neutral">•</span>No flagged events</div><span>Low deviation</span></li>'
            timeline_rows = "".join(
                f"""
                <tr>
                  <td>{escape(item['timestamp'])}</td>
                  <td>{escape(item['phase'])}</td>
                  <td>{escape(item['action'])}</td>
                  <td>{escape(item['category'])}</td>
                  <td>{item['actual_min']:.1f}</td>
                  <td>{item['deviation_min']:+.1f}</td>
                  <td>{escape(item['driver'])}</td>
                </tr>
                """
                for item in case["timeline_rows"]
            )
            case_blocks.append(
                f"""
                <details class="case-panel">
                  <summary>
                    <div class="case-head">
                      <div>
                        <div class="entity-name">{escape(case['case_id'])}</div>
                        <div class="entity-sub">{escape(case['case_type'])} · {case['total_case_duration_min']:.1f} min · {escape(case['primary_delay_driver'])}</div>
                      </div>
                      <div class="pill-row">{flag_pills}</div>
                    </div>
                  </summary>
                  <div class="card-body">
                    <div class="pill-row">{stage_pills}</div>
                    <div class="mini-title section-gap-sm">Flagged Events</div>
                    <ul class="stack-list">{event_rows}</ul>
                    <details class="transcript-drawer">
                      <summary><span class="event-symbol">▶</span> Event-by-event transcript</summary>
                      <table class="mini-table">
                        <thead>
                          <tr>
                            <th>Time</th>
                            <th>Phase</th>
                            <th>Action</th>
                            <th>Cat</th>
                            <th>Act</th>
                            <th>Dev</th>
                            <th>Cause</th>
                          </tr>
                        </thead>
                        <tbody>
                          {timeline_rows}
                        </tbody>
                      </table>
                    </details>
                  </div>
                </details>
                """
            )

        blocks.append(
            f"""
            <details class="dashboard-card doctor-card filterable-doctor aggregate-card" data-aggregate="doctor" data-clinic="{escape(profile['clinic_slug'])}" data-doctor="{escape(profile['doctor_slug'])}">
              <summary>
                <div class="card-head">
                  <div>
                    <div class="entity-name">{escape(profile['doctor_name'])}</div>
                    <div class="entity-sub">{escape(profile['clinic_name'])} · {profile['cases']} cases · rel {profile['reliability_index']}</div>
                  </div>
                  <div class="summary-metrics">
                    <span>{profile['mean_case_duration_min']:.1f} min mean</span>
                    <span>{profile['sd_case_duration_min']:.1f} min SD</span>
                  </div>
                </div>
                <div class="mini-profile-bars labeled-profile-bars">{render_stage_profile_labeled(profile['stage_rows'], scale)}</div>
              </summary>
              <div class="card-body">
                <div class="note-grid">
                  <div><strong>Most variable stage</strong><span>{escape(profile['most_variable_stage'])}</span></div>
                  <div><strong>Largest case mix</strong><span>{escape(profile['largest_case_type'])}</span></div>
                  <div><strong>Common reasons</strong><span>{escape(profile['top_delay_drivers'])}</span></div>
                </div>
                <div class="compact-stage-table">
                  {''.join(stage_rows)}
                </div>
                <div class="mini-title section-gap">Cases</div>
                <div class="case-stack">
                  {''.join(case_blocks)}
                </div>
              </div>
            </details>
            """
        )
    return "\n".join(blocks)


def render_case_explorer(rows: list[dict[str, object]]) -> str:
    blocks = []
    for case in rows:
        flag_pills = "".join(f'<span class="pill">{escape(flag)}</span>' for flag in case["flags"])
        stage_pills = "".join(
            f'<span class="pill">{escape(item["stage_name"])} {item["stage_duration_min"]:.1f}m</span>'
            for item in case["stage_breakdown"][:4]
        )
        event_rows = "".join(
            f'<li class="event-flag"><div><span class="event-symbol">!</span>{escape(item["timestamp"])} · {escape(item["action"])}</div><span>{item["deviation_min"]:+.1f}m · {escape(item["driver"])}</span></li>'
            for item in case["flagged_events"]
        ) or '<li><div><span class="event-symbol event-symbol-neutral">•</span>No flagged events</div><span>Low deviation</span></li>'
        timeline_rows = "".join(
            f"""
            <tr>
              <td>{escape(item['timestamp'])}</td>
              <td>{escape(item['phase'])}</td>
              <td>{escape(item['action'])}</td>
              <td>{escape(item['category'])}</td>
              <td>{item['actual_min']:.1f}</td>
              <td>{item['deviation_min']:+.1f}</td>
              <td>{escape(item['driver'])}</td>
            </tr>
            """
            for item in case["timeline_rows"]
        )
        blocks.append(
            f"""
            <details class="dashboard-card case-panel-global filterable-case aggregate-card" data-aggregate="case" data-clinic="{escape(case['clinic_slug'])}" data-doctor="{escape(case['doctor_slug'])}">
              <summary>
                <div class="case-head">
                  <div>
                    <div class="entity-name">{escape(case['case_id'])}</div>
                    <div class="entity-sub">{escape(case['clinic_name'])} · {escape(case['doctor_name'])} · {case['total_case_duration_min']:.1f} min · {escape(case['primary_delay_driver'])}</div>
                  </div>
                  <div class="pill-row">{flag_pills}</div>
                </div>
              </summary>
              <div class="card-body">
                <div class="pill-row">{stage_pills}</div>
                <div class="mini-title section-gap-sm">Flagged Events</div>
                <ul class="stack-list">{event_rows}</ul>
                <details class="transcript-drawer">
                  <summary><span class="event-symbol">▶</span> Event-by-event transcript</summary>
                  <table class="mini-table">
                    <thead>
                      <tr>
                        <th>Time</th>
                        <th>Phase</th>
                        <th>Action</th>
                        <th>Cat</th>
                        <th>Act</th>
                        <th>Dev</th>
                        <th>Cause</th>
                      </tr>
                    </thead>
                    <tbody>
                      {timeline_rows}
                    </tbody>
                  </table>
                </details>
              </div>
            </details>
            """
        )
    return "\n".join(blocks)


def render_stage_variance(profiles: list[dict[str, object]]) -> str:
    items = []
    for index, profile in enumerate(profiles):
        common_drivers = "".join(
            f'<li>{escape(driver["label"])} <span>{driver["count"]} cases · {driver["share"]}</span></li>'
            for driver in profile["common_drivers"]
        ) or "<li>Routine variation <span>No dominant driver</span></li>"

        top_case_types = "".join(
            f'<li>{escape(item["label"])} <span>{item["value"]:.1f} min mean</span></li>'
            for item in profile["top_case_types"]
        )
        top_microsteps = "".join(
            f"""
            <tr>
              <td>{escape(step['label'])}</td>
              <td>{step['mean_duration_min']:.1f} min</td>
              <td>{step['sd_duration_min']:.1f} min</td>
            </tr>
            """
            for step in profile["top_microsteps"]
        )

        open_attr = " open" if index == 0 else ""
        items.append(
            f"""
            <details class="dashboard-card stage-card"{open_attr}>
              <summary>
                <div class="card-head">
                  <div>
                    <div class="entity-name">{escape(profile['stage_name'])}</div>
                    <div class="entity-sub">Expand</div>
                  </div>
                  <div class="summary-metrics">
                    <span>{profile['mean_duration_min']:.1f} min mean</span>
                    <span>{profile['sd_duration_min']:.1f} min SD</span>
                    <span>P90 {profile['p90_duration_min']:.1f} min</span>
                  </div>
                </div>
              </summary>
              <div class="card-body">
                <div class="stage-split">
                  <div>
                    <div class="mini-title">Most common reasons</div>
                    <ul class="stack-list">{common_drivers}</ul>
                  </div>
                  <div>
                    <div class="mini-title">Case types most exposed</div>
                    <ul class="stack-list">{top_case_types}</ul>
                  </div>
                </div>
                <div class="mini-title">Most variable microsteps inside this stage</div>
                <table>
                  <thead>
                    <tr>
                      <th>Microstep</th>
                      <th>Mean</th>
                      <th>SD</th>
                    </tr>
                  </thead>
                  <tbody>
                    {top_microsteps}
                  </tbody>
                </table>
              </div>
            </details>
            """
        )
    return "\n".join(items)


def render_recommendations(recommendations: list[dict[str, object]]) -> str:
    return "\n".join(
        f"""
        <li>
          <strong>{escape(item['clinic_name'])}</strong>
          <span>{escape(item['process_label'])} · {escape(item['stage_name'])} · {item['mean_minutes_saved']:.1f}m faster · {item['sd_reduction_min']:.1f}m less spread</span>
        </li>
        """
        for item in recommendations
    )


def render_timeline_rows(rows: list[dict[str, object]]) -> str:
    return "\n".join(
        f"""
        <tr>
          <td>{escape(row['timestamp'])}</td>
          <td>{escape(row['phase'])}</td>
          <td>{escape(row['action'])}</td>
          <td>{escape(row['category'])}</td>
          <td>{row['expected_min']:.1f}</td>
          <td>{row['actual_min']:.1f}</td>
          <td>{row['deviation_min']:+.1f}</td>
        </tr>
        """
        for row in rows
    )


def render_driver_rows(rows: list[dict[str, object]]) -> str:
    return "\n".join(
        f'<li>{escape(row["label"])} <span>{row["delay_min"]:.1f}m · {escape(row["share"])}</span></li>'
        for row in rows
    )


def render_root_rows(rows: list[dict[str, object]]) -> str:
    return "\n".join(
        f"""
        <tr>
          <td>{escape(row['microstep'])}</td>
          <td>{escape(row['category'])}</td>
          <td>{row['deviation_min']:+.1f}</td>
          <td>{escape(row['cause'])}</td>
        </tr>
        """
        for row in rows
    )


def render_ideal_rows(rows: list[dict[str, object]]) -> str:
    return "\n".join(
        f"""
        <tr>
          <td>{escape(row['phase'])}</td>
          <td>{row['ideal_min']:.1f}</td>
          <td>{row['actual_min']:.1f}</td>
          <td>{row['gap_min']:+.1f}</td>
        </tr>
        """
        for row in rows
    )


def render_opportunity_rows(rows: list[dict[str, object]]) -> str:
    return "\n".join(
        f'<li>{escape(row["microstep"])} <span>{row["gap_min"]:+.1f}m · {escape(row["suggestion"])}</span></li>'
        for row in rows
    )


def build_dashboard_html(payload: dict[str, object]) -> str:
    diagnostic = payload["diagnostic_case"]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Surgical Variability Dashboard</title>
  <style>
    :root {{
      --bg: #eef3f7;
      --panel: #ffffff;
      --panel-soft: #f7f9fc;
      --ink: #172033;
      --muted: #64748b;
      --line: #d9e2ec;
      --accent: #2563eb;
      --accent-soft: rgba(37, 99, 235, 0.12);
      --shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
      --sans: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      --mono: "SFMono-Regular", Consolas, monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(37, 99, 235, 0.10), transparent 28%),
        linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
    }}
    .shell {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px 24px 40px;
    }}
    header {{
      background: linear-gradient(135deg, #0f172a 0%, #172554 100%);
      color: #fff;
      border-radius: 20px;
      padding: 24px 28px;
      box-shadow: var(--shadow);
      margin-bottom: 16px;
    }}
    .eyebrow {{
      font: 700 11px/1.2 var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: rgba(255, 255, 255, 0.72);
      margin-bottom: 10px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      line-height: 1.06;
      font-weight: 700;
    }}
    .lede {{
      margin: 0;
      max-width: 760px;
      font-size: 14px;
      line-height: 1.35;
      color: rgba(255, 255, 255, 0.82);
    }}
    .subnav {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}
    .subnav span,
    .summary-metrics span,
    .pill {{
      background: var(--panel-soft);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }}
    header .subnav span {{
      background: rgba(255, 255, 255, 0.10);
      border-color: rgba(255, 255, 255, 0.16);
      color: rgba(255, 255, 255, 0.88);
    }}
    section {{ margin-top: 24px; }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      margin-bottom: 14px;
    }}
    .section-head h2 {{
      margin: 0;
      font-size: 20px;
      line-height: 1.1;
    }}
    .section-head p {{
      margin: 0;
      max-width: 480px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.3;
      text-align: right;
    }}
    .metric-grid,
    .card-grid,
    .case-grid {{
      display: grid;
      gap: 14px;
    }}
    .metric-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .card-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .case-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .two-col {{
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 14px;
      align-items: start;
    }}
    .metric-card,
    .dashboard-card,
    .table-shell {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }}
    .metric-card,
    .dashboard-card {{
      padding: 18px;
    }}
    .table-shell {{
      overflow: hidden;
    }}
    .metric-label,
    .mini-title,
    th {{
      font: 700 11px/1.2 var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.10em;
      color: var(--muted);
    }}
    .metric-label {{ margin-bottom: 8px; }}
    .metric-value {{
      font-size: 28px;
      font-weight: 700;
      line-height: 1.05;
      margin-bottom: 4px;
    }}
    .metric-note,
    .entity-sub,
    .score-sub,
    .card-copy,
    footer {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .entity-name {{
      font-size: 16px;
      font-weight: 700;
      line-height: 1.2;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th {{
      text-align: left;
      padding: 11px 14px;
      background: var(--panel-soft);
      border-bottom: 1px solid var(--line);
    }}
    td {{
      padding: 12px 14px;
      border-bottom: 1px solid #edf2f7;
      vertical-align: middle;
      font-size: 13px;
      line-height: 1.3;
    }}
    tr:last-child td {{ border-bottom: none; }}
    .score-main {{ font-weight: 700; }}
    .row-bar-track,
    .compact-stage-track {{
      width: 100%;
      height: 10px;
      background: #e8eef6;
      border-radius: 999px;
      overflow: hidden;
    }}
    .row-bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, #1d4ed8, #60a5fa);
      border-radius: 999px;
    }}
    .filter-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-bottom: 18px;
      padding: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: var(--shadow);
    }}
    .filter-group {{
      display: grid;
      gap: 6px;
    }}
    .filter-bar label {{
      font: 700 11px/1.2 var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.10em;
      color: var(--muted);
    }}
    .select-wrap {{
      position: relative;
      min-width: 180px;
    }}
    .select-wrap::after {{
      content: "▼";
      position: absolute;
      right: 12px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--muted);
      font-size: 10px;
      pointer-events: none;
    }}
    .filter-bar select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 34px 8px 10px;
      background: var(--panel);
      color: var(--ink);
      font: 500 13px/1.2 var(--sans);
      appearance: none;
      -webkit-appearance: none;
      -moz-appearance: none;
    }}
    details.dashboard-card {{ padding: 0; }}
    details summary {{
      list-style: none;
      cursor: pointer;
      padding: 18px;
    }}
    details summary::-webkit-details-marker {{ display: none; }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }}
    .summary-metrics {{
      display: flex;
      flex-wrap: wrap;
      justify-content: end;
      gap: 8px;
    }}
    .mini-profile-bars {{
      display: flex;
      gap: 6px;
      margin-top: 14px;
    }}
    .labeled-profile-bars {{
      align-items: flex-end;
    }}
    .mini-stage-segment {{
      display: grid;
      gap: 6px;
      align-items: end;
    }}
    .mini-stage-label {{
      font-size: 10px;
      line-height: 1.2;
      color: var(--ink);
      text-align: center;
    }}
    .mini-bar {{
      display: block;
      height: 12px;
      border-radius: 999px;
      min-width: 14px;
    }}
    .card-body {{
      padding: 0 18px 18px;
    }}
    .section-gap {{
      margin-top: 18px;
    }}
    .section-gap-sm {{
      margin-top: 14px;
    }}
    .note-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .case-card .note-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 12px;
      margin-bottom: 0;
    }}
    .note-grid div {{
      background: var(--panel-soft);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
    }}
    .note-grid strong {{
      display: block;
      font-size: 11px;
      margin-bottom: 5px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .note-grid span {{
      font-size: 13px;
      line-height: 1.35;
    }}
    .compact-stage-table {{
      display: grid;
      gap: 10px;
    }}
    .compact-stage-row {{
      display: grid;
      grid-template-columns: 170px 1fr 56px 56px 60px;
      gap: 10px;
      align-items: center;
      font-size: 12px;
    }}
    .compact-stage-fill {{
      height: 100%;
      border-radius: 999px;
    }}
    .case-stack {{
      display: grid;
      gap: 14px;
      margin-top: 10px;
    }}
    .case-panel {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel-soft);
    }}
    .case-panel-global {{
      padding: 0;
    }}
    .aggregate-card {{
      display: block;
    }}
    .case-panel summary {{
      list-style: none;
      cursor: pointer;
      padding: 14px;
    }}
    .case-panel summary::-webkit-details-marker {{ display: none; }}
    .case-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
    }}
    .stage-stack {{
      display: grid;
      gap: 14px;
    }}
    .stage-split {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 12px;
    }}
    .stack-list {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 10px;
    }}
    .stack-list li {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      background: #f8fbff;
      border: 1px solid #d9e7fb;
      border-radius: 12px;
      padding: 12px 14px;
      font-size: 12px;
      color: var(--ink);
    }}
    .stack-list li div {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 600;
    }}
    .stack-list span {{
      color: var(--muted);
      text-align: right;
    }}
    .event-flag {{
      background: linear-gradient(180deg, #fff5f5 0%, #fffaf0 100%);
      border-color: #f7c9c9;
    }}
    .event-symbol {{
      display: inline-flex;
      width: 18px;
      height: 18px;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      background: #dc2626;
      color: #fff;
      font-size: 10px;
      line-height: 1;
      flex: 0 0 auto;
    }}
    .event-symbol-neutral {{
      background: #64748b;
    }}
    .transcript-drawer {{
      margin-top: 14px;
      border: 1px solid #d7e3f4;
      border-radius: 14px;
      background: #fcfdff;
      overflow: hidden;
    }}
    .transcript-drawer summary {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 12px 14px;
      font: 700 12px/1.2 var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #0f172a;
      background: #eef5ff;
      border-bottom: 1px solid #d7e3f4;
    }}
    .mini-table th,
    .mini-table td {{
      padding: 8px 10px;
      font-size: 12px;
    }}
    .insight-grid {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 16px;
      align-items: start;
    }}
    .ideal-layout {{
      display: grid;
      gap: 16px;
    }}
    .recommendation-list {{
      margin: 0;
      padding-left: 18px;
      display: grid;
      gap: 8px;
      color: var(--muted);
    }}
    .recommendation-list strong {{ color: var(--ink); }}
    .recommendation-list span {{
      display: block;
      margin-top: 3px;
      line-height: 1.35;
    }}
    footer {{ margin-top: 16px; }}
    @media (max-width: 1100px) {{
      .metric-grid,
      .card-grid,
      .case-grid,
      .stage-split,
      .note-grid,
      .two-col,
      .insight-grid {{
        grid-template-columns: 1fr;
      }}
      .card-head,
      .section-head,
      .compact-stage-row,
      .case-head {{
        display: block;
      }}
      .summary-metrics {{
        justify-content: start;
        margin-top: 10px;
      }}
      .section-head p {{
        text-align: left;
        margin-top: 6px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="eyebrow">Surgical Variability Dashboard</div>
      <h1>Clinic. Doctor. Case. Root Cause.</h1>
      <p class="lede">Transcript-linked variability dashboard.</p>
      <div class="subnav">
        <span>Clinic</span>
        <span>Doctor</span>
        <span>Case</span>
        <span>Ideal</span>
      </div>
    </header>

    <section>
      <div class="metric-grid">
        {render_summary_cards(payload["summary_cards"])}
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Clinic Comparison</h2>
        <p>Site level.</p>
      </div>
      <div class="table-shell">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Clinic</th>
              <th>Case Time</th>
              <th>Reliability</th>
              <th>Var Stage</th>
              <th>Fastest Doctor</th>
              <th>Heaviest Case Type</th>
              <th>Top Process</th>
              <th>Relative Time</th>
            </tr>
          </thead>
          <tbody>
            {render_clinic_table(payload["clinic_profiles"])}
          </tbody>
        </table>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Explorer</h2>
        <p>Filter first. Aggregate by clinic, doctor, or case.</p>
      </div>
      <div class="filter-bar">
        <div class="filter-group">
          <label for="clinicFilter">Clinic</label>
          <div class="select-wrap">
            <select id="clinicFilter">
              <option value="all">All clinics</option>
              <option value="northside-heart-center">Northside Heart Center</option>
              <option value="lakeshore-ep-lab">Lakeshore EP Lab</option>
              <option value="metro-rhythm-institute">Metro Rhythm Institute</option>
            </select>
          </div>
        </div>
        <div class="filter-group">
          <label for="doctorFilter">Doctor</label>
          <div class="select-wrap">
            <select id="doctorFilter">
              <option value="all">All doctors</option>
              <option value="dr-a">Dr. A</option>
              <option value="dr-b">Dr. B</option>
              <option value="dr-c">Dr. C</option>
              <option value="dr-d">Dr. D</option>
              <option value="dr-e">Dr. E</option>
              <option value="dr-f">Dr. F</option>
            </select>
          </div>
        </div>
        <div class="filter-group">
          <label for="aggregateFilter">Aggregate by</label>
          <div class="select-wrap">
            <select id="aggregateFilter">
              <option value="doctor">Doctor</option>
              <option value="clinic">Clinic</option>
              <option value="case">Case</option>
            </select>
          </div>
        </div>
      </div>
      <div class="aggregate-explorer">
        {render_clinic_explorer(payload["clinic_profiles"])}
        {render_doctor_explorer(payload["doctor_profiles"])}
        {render_case_explorer(payload["case_profiles"])}
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Root Cause Explorer</h2>
        <p>{escape(diagnostic["case_id"])} · {escape(diagnostic["doctor_name"])} · {escape(diagnostic["root_stage"])}</p>
      </div>
      <div class="insight-grid">
        <article class="dashboard-card ideal-layout">
          <div class="card-head">
            <div>
              <div class="entity-name">Delay Drivers</div>
              <div class="entity-sub">{diagnostic["total_delay_min"]:.1f}m total delay</div>
            </div>
            <div class="summary-metrics">
              <span>{escape(diagnostic["clinic_name"])}</span>
              <span>{escape(diagnostic["case_type"])}</span>
            </div>
          </div>
          <ul class="stack-list section-gap-sm">
            {render_driver_rows(diagnostic["driver_rows"])}
          </ul>
        </article>

        <article class="dashboard-card ideal-layout">
          <div class="card-head">
            <div>
              <div class="entity-name">Root Stage Breakdown</div>
              <div class="entity-sub">{escape(diagnostic["root_stage"])}</div>
            </div>
          </div>
          <table class="mini-table section-gap-sm">
            <thead>
              <tr>
                <th>Microstep</th>
                <th>Cat</th>
                <th>Dev</th>
                <th>Cause</th>
              </tr>
            </thead>
            <tbody>
              {render_root_rows(diagnostic["root_rows"])}
            </tbody>
          </table>
        </article>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Ideal Procedure Map</h2>
        <p>Actual vs ideal.</p>
      </div>
      <div class="insight-grid">
        <article class="dashboard-card ideal-layout">
          <div class="card-head">
            <div>
              <div class="entity-name">Ideal vs Actual</div>
              <div class="entity-sub">{escape(diagnostic["case_id"])}</div>
            </div>
          </div>
          <table class="mini-table">
            <thead>
              <tr>
                <th>Phase</th>
                <th>Ideal</th>
                <th>Actual</th>
                <th>Gap</th>
              </tr>
            </thead>
            <tbody>
              {render_ideal_rows(diagnostic["ideal_rows"])}
            </tbody>
          </table>
          <ul class="stack-list">
            {render_opportunity_rows(diagnostic["opportunity_rows"])}
          </ul>
        </article>

        <article class="dashboard-card ideal-layout">
          <div class="card-head">
            <div>
              <div class="entity-name">Recommended Moves</div>
              <div class="entity-sub">Clinic patterns worth standardizing</div>
            </div>
          </div>
          <ol class="recommendation-list">
            {render_recommendations(payload["recommendations"])}
          </ol>
        </article>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Stage Variance</h2>
        <p>Open for detail.</p>
      </div>
      <div class="stage-stack">
        {render_stage_variance(payload["stage_variance_profiles"])}
      </div>
    </section>

    <footer>Synthetic proposal demo.</footer>
  </div>
  <script>
    const clinicFilter = document.getElementById("clinicFilter");
    const doctorFilter = document.getElementById("doctorFilter");
    const aggregateFilter = document.getElementById("aggregateFilter");
    const aggregateCards = Array.from(document.querySelectorAll(".aggregate-card"));

    function applyExplorerFilters() {{
      const clinicValue = clinicFilter.value;
      const doctorValue = doctorFilter.value;
      const aggregateValue = aggregateFilter.value;

      aggregateCards.forEach((card) => {{
        const aggregateMatch = card.dataset.aggregate === aggregateValue;
        const clinicMatch = clinicValue === "all" || card.dataset.clinic === clinicValue;
        const doctorMatch = doctorValue === "all" || card.dataset.doctor === doctorValue || aggregateValue === "clinic";
        card.style.display = aggregateMatch && clinicMatch && doctorMatch ? "block" : "none";
      }});
    }}

    clinicFilter.addEventListener("change", applyExplorerFilters);
    doctorFilter.addEventListener("change", applyExplorerFilters);
    aggregateFilter.addEventListener("change", applyExplorerFilters);
    applyExplorerFilters();
  </script>
</body>
</html>
"""


def main() -> None:
    dataset = write_mock_outputs(output_dir=OUTPUT_DIR)
    payload = build_payload(dataset)

    (OUTPUT_DIR / "variability_dashboard_data.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "variability_dashboard.html").write_text(
        build_dashboard_html(payload),
        encoding="utf-8",
    )

    print("Saved dashboard assets:")
    print("-", OUTPUT_DIR / "variability_dashboard.html")
    print("-", OUTPUT_DIR / "variability_dashboard_data.json")


if __name__ == "__main__":
    main()
