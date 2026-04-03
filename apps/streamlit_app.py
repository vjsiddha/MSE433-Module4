from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.variability_data import PROCESS_LABELS, STAGES, write_mock_outputs


st.set_page_config(
    page_title="Surgical Variability Dashboard",
    page_icon="",
    layout="wide",
)


def add_case_types(cases_df: pd.DataFrame) -> pd.DataFrame:
    cases = cases_df.copy()

    def classify(row: pd.Series) -> tuple[str, str]:
        if bool(row["prior_ablation"]):
            return "Redo / scar burden", "Scar-heavy repeat cases."
        if int(row["septum_thickness"]) >= 3:
            return "Thick septum crossing", "TSP-heavy difficulty."
        if bool(row["mobility_limited"]) or int(row["obesity_class"]) >= 2:
            return "Transfer-risk obesity", "Transfer and recovery friction."
        if bool(row["persistent_af"]) or int(row["planned_lesions"]) >= 60:
            return "Persistent / high lesion", "Higher mapping and lesion load."
        return "Standard anatomy", "Lower-complexity baseline."

    classified = cases.apply(classify, axis=1, result_type="expand")
    cases["case_type"] = classified[0]
    cases["case_type_note"] = classified[1]
    return cases


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    dataset = write_mock_outputs()
    cases = add_case_types(dataset["cases"])
    stages = dataset["stages"].merge(cases[["case_id", "case_type"]], on="case_id", how="left")
    steps = dataset["steps"].merge(cases[["case_id", "case_type"]], on="case_id", how="left")
    return {
        "cases": cases,
        "stages": stages,
        "steps": steps,
        "benchmarks": dataset["benchmarks"],
    }


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
    fallback_cases = set(
        cases_df.loc[cases_df["total_case_duration_min"] <= fallback_cutoff, "case_id"].tolist()
    )
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


def build_timeline(case_id: str, cases_df: pd.DataFrame, steps_df: pd.DataFrame, expected_df: pd.DataFrame) -> pd.DataFrame:
    case_row = cases_df.loc[cases_df["case_id"] == case_id].iloc[0]
    case_steps = steps_df.loc[steps_df["case_id"] == case_id].copy()
    case_steps = case_steps.merge(
        expected_df,
        on=["case_type", "microstep_code"],
        how="left",
    )
    case_steps["expected_time_min"] = case_steps["expected_time_min"].fillna(case_steps["fallback_expected_time_min"])
    case_steps["actual_time_min"] = case_steps["duration_min"]
    case_steps["time_deviation_min"] = case_steps["actual_time_min"] - case_steps["expected_time_min"]
    case_steps["timestamp_label"] = case_steps["transcript_start_min"].map(lambda value: f"{float(value):05.1f}m")
    case_steps["case_type"] = case_row["case_type"]
    return case_steps.sort_values(["stage_order", "microstep_order"]).reset_index(drop=True)


def reliability_index(mean_duration: float, sd_duration: float) -> int:
    if not mean_duration:
        return 0
    return int(max(52, min(98, round(100 - ((sd_duration / mean_duration) * 130)))))


def top_driver_text(series: pd.Series, limit: int = 3) -> str:
    counts = series.loc[series != "Routine variation"].value_counts().head(limit)
    return ", ".join(f"{label} ({count})" for label, count in counts.items()) or "Routine variation"


def render_header() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f7f9fc;
            --panel-bg: #ffffff;
            --panel-soft: #f8fafc;
            --line: #d9e2ec;
            --text: #172033;
            --muted: #64748b;
            --accent: #2563eb;
        }
        .stApp {
            background: var(--app-bg);
            color: var(--text);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        html, body, [class*="css"] {
            color: var(--text);
        }
        .stApp, .stApp p, .stApp span, .stApp label, .stApp div, .stApp li {
            color: var(--text);
        }
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption, .stCaption p {
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: var(--panel-bg);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * {
            color: var(--text);
        }
        div[data-testid="stMetric"] {
            background: var(--panel-bg);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 14px 16px;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--text) !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 16px;
            overflow: hidden;
            background: var(--panel-bg);
        }
        div[data-testid="stDataFrame"] * {
            color: var(--text) !important;
        }
        [data-testid="stTabs"] button {
            color: var(--muted) !important;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--accent) !important;
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {
            color: var(--text) !important;
        }
        [data-testid="stSelectbox"] label,
        [data-testid="stSelectbox"] div,
        [data-testid="stSelectbox"] input,
        [data-testid="stSelectbox"] span {
            color: var(--text) !important;
        }
        [data-baseweb="select"] > div {
            background: var(--panel-bg) !important;
            border-color: var(--line) !important;
        }
        [data-baseweb="select"] * {
            color: var(--text) !important;
        }
        [role="listbox"] * {
            color: var(--text) !important;
            background: var(--panel-bg) !important;
        }
        .stSelectbox svg, .stMultiSelect svg {
            fill: var(--text) !important;
        }
        .stAlert, .stInfo, .stSuccess, .stWarning {
            color: var(--text) !important;
        }
        .panel {
            background: var(--panel-bg);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .panel h3 {
            margin: 0 0 8px 0;
            font-size: 18px;
            color: var(--text);
        }
        .panel p {
            margin: 0;
            color: var(--muted);
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Surgical Variability Streamlit Dashboard")
    st.caption("Compact view: clinic, doctor, case type, timeline replay, drivers, root causes, and ideal workflow.")


def show_overview(cases_df: pd.DataFrame, stages_df: pd.DataFrame) -> None:
    st.subheader("Overview")
    clinic_tab, doctor_tab, case_type_tab = st.tabs(["Clinic vs clinic", "Doctor vs doctor", "Case type vs case type"])

    with clinic_tab:
        clinic_view = (
            cases_df.groupby("clinic_name", as_index=False)
            .agg(
                cases=("case_id", "nunique"),
                mean_case_time_min=("total_case_duration_min", "mean"),
                sd_case_time_min=("total_case_duration_min", "std"),
            )
        )
        clinic_view["reliability"] = clinic_view.apply(
            lambda row: reliability_index(float(row["mean_case_time_min"]), float(row["sd_case_time_min"])),
            axis=1,
        )
        clinic_view["most_variable_stage"] = clinic_view["clinic_name"].map(
            lambda clinic: stages_df.loc[stages_df["clinic_name"] == clinic].groupby("stage_name")["stage_duration_min"].std(ddof=1).sort_values(ascending=False).index[0]
        )
        clinic_view = clinic_view.sort_values("mean_case_time_min").reset_index(drop=True)
        st.dataframe(
            clinic_view.rename(
                columns={
                    "clinic_name": "Clinic",
                    "cases": "Cases",
                    "mean_case_time_min": "Mean case time (min)",
                    "sd_case_time_min": "SD (min)",
                    "reliability": "Reliability",
                    "most_variable_stage": "Most variable stage",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with doctor_tab:
        doctor_view = (
            cases_df.groupby(["clinic_name", "doctor_name"], as_index=False)
            .agg(
                cases=("case_id", "nunique"),
                mean_case_time_min=("total_case_duration_min", "mean"),
                sd_case_time_min=("total_case_duration_min", "std"),
            )
        )
        doctor_view["reliability"] = doctor_view.apply(
            lambda row: reliability_index(float(row["mean_case_time_min"]), float(row["sd_case_time_min"])),
            axis=1,
        )
        doctor_view["most_variable_stage"] = doctor_view["doctor_name"].map(
            lambda doctor: stages_df.loc[stages_df["doctor_name"] == doctor].groupby("stage_name")["stage_duration_min"].std(ddof=1).sort_values(ascending=False).index[0]
        )
        doctor_view = doctor_view.sort_values(["clinic_name", "mean_case_time_min"]).reset_index(drop=True)
        st.dataframe(
            doctor_view.rename(
                columns={
                    "clinic_name": "Clinic",
                    "doctor_name": "Doctor",
                    "cases": "Cases",
                    "mean_case_time_min": "Mean case time (min)",
                    "sd_case_time_min": "SD (min)",
                    "reliability": "Reliability",
                    "most_variable_stage": "Most variable stage",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with case_type_tab:
        case_type_view = (
            cases_df.groupby("case_type", as_index=False)
            .agg(
                cases=("case_id", "nunique"),
                mean_case_time_min=("total_case_duration_min", "mean"),
                sd_case_time_min=("total_case_duration_min", "std"),
            )
            .sort_values("mean_case_time_min", ascending=False)
        )
        case_type_view["top_driver"] = case_type_view["case_type"].map(
            lambda case_type: top_driver_text(cases_df.loc[cases_df["case_type"] == case_type, "primary_delay_driver"], limit=2)
        )
        st.dataframe(
            case_type_view.rename(
                columns={
                    "case_type": "Case type",
                    "cases": "Cases",
                    "mean_case_time_min": "Mean case time (min)",
                    "sd_case_time_min": "SD (min)",
                    "top_driver": "Top drivers",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def show_timeline_replay(case_row: pd.Series, timeline_df: pd.DataFrame) -> None:
    st.subheader("1. Procedure Timeline Replay")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Case", case_row["case_id"])
    c2.metric("Clinic", case_row["clinic_name"])
    c3.metric("Doctor", case_row["doctor_name"])
    c4.metric("Case type", case_row["case_type"])

    timeline_view = timeline_df[
        [
            "timestamp_label",
            "stage_name",
            "microstep_label",
            "category",
            "expected_time_min",
            "mean_time_min",
            "actual_time_min",
            "time_deviation_min",
        ]
    ].rename(
        columns={
            "timestamp_label": "Timestamp",
            "stage_name": "Phase",
            "microstep_label": "Action",
            "category": "Category",
            "expected_time_min": "Expected",
            "mean_time_min": "Mean",
            "actual_time_min": "Actual",
            "time_deviation_min": "Deviation",
        }
    )
    st.dataframe(timeline_view.round(2), use_container_width=True, hide_index=True)

    for _, row in timeline_df.iterrows():
        deviation_label = f"{float(row['time_deviation_min']):+.1f} min"
        with st.expander(f"{row['timestamp_label']} · {row['microstep_label']} · {deviation_label}"):
            d1, d2 = st.columns(2)
            d1.write(f"Phase: {row['stage_name']}")
            d1.write(f"Category: {row['category']}")
            d1.write(f"Expected / mean / actual: {row['expected_time_min']:.1f} / {row['mean_time_min']:.1f} / {row['actual_time_min']:.1f} min")
            d2.write(f"Delay driver: {row['primary_delay_driver']}")
            d2.write(f"Processes: {row['relevant_processes']}")
            d2.write(f"Actor: {row['actor']}")
            st.caption(row["transcript_snippet"])
            st.caption(f"Inferred video event: {row['inferred_video_event']}")


def show_variability_drivers(timeline_df: pd.DataFrame) -> None:
    st.subheader("2. Variability Driver Detection")
    delayed_steps = timeline_df.loc[timeline_df["time_deviation_min"] > 0].copy()
    if delayed_steps.empty:
        st.info("Selected case is at or below expected time on all benchmarked steps.")
        return

    category_view = (
        delayed_steps.groupby("category", as_index=False)["time_deviation_min"]
        .sum()
        .sort_values("time_deviation_min", ascending=False)
    )
    total_delay = float(category_view["time_deviation_min"].sum())
    category_view["share"] = category_view["time_deviation_min"] / total_delay
    category_view["top_support"] = category_view["category"].map(
        lambda category: top_driver_text(delayed_steps.loc[delayed_steps["category"] == category, "primary_delay_driver"], limit=2)
    )
    category_view["example_video_evidence"] = category_view["category"].map(
        lambda category: delayed_steps.loc[delayed_steps["category"] == category, "inferred_video_event"].iloc[0]
    )

    left, right = st.columns([1.2, 1.8])
    with left:
        st.metric("Total delayed time", f"{total_delay:.1f} min")
        st.bar_chart(category_view.set_index("category")["time_deviation_min"])
    with right:
        st.dataframe(
            category_view.rename(
                columns={
                    "category": "Driver bucket",
                    "time_deviation_min": "Delay (min)",
                    "share": "Share",
                    "top_support": "Top causes",
                    "example_video_evidence": "Example evidence",
                }
            ).assign(Share=lambda df: (df["Share"] * 100).round(0).astype(int).astype(str) + "%"),
            use_container_width=True,
            hide_index=True,
        )

    root_driver_view = (
        delayed_steps.groupby("primary_delay_driver", as_index=False)["time_deviation_min"]
        .sum()
        .sort_values("time_deviation_min", ascending=False)
        .head(6)
    )
    st.dataframe(
        root_driver_view.rename(
            columns={
                "primary_delay_driver": "Specific cause",
                "time_deviation_min": "Delay impact (min)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def show_root_cause_explorer(timeline_df: pd.DataFrame) -> None:
    st.subheader("3. Root Cause Explorer")
    selected_phase = st.selectbox("Phase", options=timeline_df["stage_name"].unique().tolist(), key="phase_select")
    phase_df = timeline_df.loc[timeline_df["stage_name"] == selected_phase].copy()
    phase_delay = float(phase_df["time_deviation_min"].clip(lower=0).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Phase delay vs expected", f"{phase_delay:.1f} min")
    c2.metric("Phase actual", f"{phase_df['actual_time_min'].sum():.1f} min")
    c3.metric("Phase expected", f"{phase_df['expected_time_min'].sum():.1f} min")

    category_breakdown = (
        phase_df.groupby("category", as_index=False)["time_deviation_min"]
        .sum()
        .sort_values("time_deviation_min", ascending=False)
    )
    repeated_patterns = phase_df.loc[
        (phase_df["time_deviation_min"] > 1.0)
        & (phase_df["category"].isin(["Verification", "Equipment handling", "Waiting / transfer"]))
    ].copy()
    repeated_patterns["inferred_cluster"] = repeated_patterns["primary_delay_driver"].map(
        lambda label: f"Repeat loop around {label.lower()}" if label != "Routine variation" else "No clear loop"
    )

    left, right = st.columns(2)
    with left:
        st.dataframe(
            category_breakdown.rename(
                columns={
                    "category": "Category",
                    "time_deviation_min": "Delay (min)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        if repeated_patterns.empty:
            st.info("No obvious repeated-loop signal in this phase.")
        else:
            st.dataframe(
                repeated_patterns[
                    ["microstep_label", "time_deviation_min", "primary_delay_driver", "inferred_cluster"]
                ].rename(
                    columns={
                        "microstep_label": "Microstep",
                        "time_deviation_min": "Deviation",
                        "primary_delay_driver": "Cause",
                        "inferred_cluster": "Sequence deviation",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.dataframe(
        phase_df[
            [
                "microstep_label",
                "category",
                "expected_time_min",
                "actual_time_min",
                "time_deviation_min",
                "primary_delay_driver",
            ]
        ].rename(
            columns={
                "microstep_label": "Microstep",
                "category": "Category",
                "expected_time_min": "Expected",
                "actual_time_min": "Actual",
                "time_deviation_min": "Deviation",
                "primary_delay_driver": "Cause",
            }
        ).round(2),
        use_container_width=True,
        hide_index=True,
    )


def show_ideal_procedure_map(case_row: pd.Series, timeline_df: pd.DataFrame, cases_df: pd.DataFrame, steps_df: pd.DataFrame) -> None:
    st.subheader("4. Ideal Procedure Map")
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

    opportunity_rows = compare.loc[compare["ideal_gap_min"] > 1.0].copy()
    opportunity_rows["suggestion"] = opportunity_rows["primary_delay_driver"].map(
        {
            "Obesity / transfer risk": "Use transfer support workflow earlier.",
            "Mobility limitation": "Pre-stage extra hands and transfer equipment.",
            "Equipment issue": "Stage backup equipment before case start.",
            "Thick septum": "Standardize TSP escalation pathway.",
            "Prior ablation scar": "Flag redo workflow and mapping support in advance.",
            "Anatomy complexity": "Pre-brief anatomy complexity before access.",
        }
    ).fillna("Standardize the faster sequence from the ideal cohort.")

    left, right = st.columns([1.2, 1.8])
    with left:
        st.metric("Peer case type", case_row["case_type"])
        st.metric("Ideal cohort size", len(ideal_case_ids))
        st.metric("Gap to ideal", f"{phase_compare['ideal_gap_min'].sum():.1f} min")
    with right:
        st.dataframe(
            phase_compare.rename(
                columns={
                    "stage_name": "Phase",
                    "ideal_time_min": "Ideal",
                    "actual_time_min": "Actual",
                    "ideal_gap_min": "Gap",
                }
            ).round(2),
            use_container_width=True,
            hide_index=True,
        )

    st.dataframe(
        opportunity_rows[
            ["microstep_label", "ideal_time_min", "actual_time_min", "ideal_gap_min", "suggestion"]
        ].rename(
            columns={
                "microstep_label": "Microstep",
                "ideal_time_min": "Ideal",
                "actual_time_min": "Actual",
                "ideal_gap_min": "Gap",
                "suggestion": "Recommended improvement",
            }
        ).round(2),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    render_header()
    data = load_data()
    cases_df = data["cases"]
    stages_df = data["stages"]
    steps_df = data["steps"]

    with st.sidebar:
        st.header("Filters")
        clinic = st.selectbox("Clinic", ["All"] + sorted(cases_df["clinic_name"].unique().tolist()))
        clinic_filtered = cases_df if clinic == "All" else cases_df.loc[cases_df["clinic_name"] == clinic]

        doctor = st.selectbox("Doctor", ["All"] + sorted(clinic_filtered["doctor_name"].unique().tolist()))
        doctor_filtered = clinic_filtered if doctor == "All" else clinic_filtered.loc[clinic_filtered["doctor_name"] == doctor]

        case_type = st.selectbox("Case type", ["All"] + sorted(doctor_filtered["case_type"].unique().tolist()))
        filtered_cases = doctor_filtered if case_type == "All" else doctor_filtered.loc[doctor_filtered["case_type"] == case_type]
        selected_case = st.selectbox("Current case", filtered_cases["case_id"].tolist(), index=0)
        st.caption("The replay and diagnostics update for the selected case.")

    expected_df = build_expected_step_benchmarks(cases_df, steps_df)
    case_row = filtered_cases.loc[filtered_cases["case_id"] == selected_case].iloc[0]
    timeline_df = build_timeline(selected_case, filtered_cases, steps_df, expected_df)

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Case time", f"{case_row['total_case_duration_min']:.1f} min")
    top2.metric("Primary driver", str(case_row["primary_delay_driver"]))
    top3.metric("Clinic", str(case_row["clinic_name"]))
    top4.metric("Doctor", str(case_row["doctor_name"]))

    show_overview(cases_df, stages_df)
    st.divider()
    show_timeline_replay(case_row, timeline_df)
    st.divider()
    show_variability_drivers(timeline_df)
    st.divider()
    show_root_cause_explorer(timeline_df)
    st.divider()
    show_ideal_procedure_map(case_row, timeline_df, cases_df, steps_df)


if __name__ == "__main__":
    main()
