from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
RNG_SEED = 433


STAGES: list[dict[str, object]] = [
    {
        "stage_code": "pre_op",
        "stage_order": 1,
        "stage_name": "Pre-op and positioning",
        "color": "#0f766e",
        "description": "Patient transfer, anesthesia setup, safety imaging, and room preparation.",
    },
    {
        "stage_code": "access",
        "stage_order": 2,
        "stage_name": "Access and sheath setup",
        "color": "#2563eb",
        "description": "Groin access, sheath placement, cabling, and irrigation checks.",
    },
    {
        "stage_code": "tsp",
        "stage_order": 3,
        "stage_name": "Transseptal puncture",
        "color": "#b45309",
        "description": "ICE-guided septal crossing, sheath exchange, and left atrial entry.",
    },
    {
        "stage_code": "mapping",
        "stage_order": 4,
        "stage_name": "Mapping and baseline validation",
        "color": "#7c3aed",
        "description": "Geometry build, map registration, and baseline voltage review.",
    },
    {
        "stage_code": "ablation",
        "stage_order": 5,
        "stage_name": "Ablation and verification",
        "color": "#dc2626",
        "description": "Lesion delivery, touch-up maneuvering, and endpoint confirmation.",
    },
    {
        "stage_code": "close",
        "stage_order": 6,
        "stage_name": "Close and transfer",
        "color": "#475569",
        "description": "Sheath removal, wake-up, extubation, and transfer off table.",
    },
]


MICROSTEP_LIBRARY: dict[str, list[dict[str, object]]] = {
    "pre_op": [
        {
            "microstep_code": "patient_transfer_positioning",
            "microstep_label": "Patient transfer and positioning",
            "actor": "Nursing + anesthesia",
            "base_min": 8.0,
            "noise_sd": 1.0,
        },
        {
            "microstep_code": "anesthesia_induction_airway",
            "microstep_label": "Anesthesia induction and airway",
            "actor": "Anesthesia",
            "base_min": 9.5,
            "noise_sd": 1.1,
        },
        {
            "microstep_code": "tee_safety_check",
            "microstep_label": "TEE safety and clot check",
            "actor": "Physician + anesthesia",
            "base_min": 4.8,
            "noise_sd": 0.6,
        },
        {
            "microstep_code": "sterile_prep_monitor_hookup",
            "microstep_label": "Sterile prep, drape, and monitor hookup",
            "actor": "Nursing + technologist",
            "base_min": 6.5,
            "noise_sd": 0.8,
        },
    ],
    "access": [
        {
            "microstep_code": "ultrasound_groin_scan",
            "microstep_label": "Ultrasound groin scan",
            "actor": "Physician",
            "base_min": 3.0,
            "noise_sd": 0.6,
        },
        {
            "microstep_code": "venous_access_sheath_insertion",
            "microstep_label": "Venous access and sheath insertion",
            "actor": "Physician + technologist",
            "base_min": 8.5,
            "noise_sd": 1.1,
        },
        {
            "microstep_code": "catheter_cable_console_setup",
            "microstep_label": "Catheter cable and console setup",
            "actor": "Technologist + circulator",
            "base_min": 6.0,
            "noise_sd": 0.9,
        },
        {
            "microstep_code": "anticoagulation_irrigation_check",
            "microstep_label": "Anticoagulation and irrigation check",
            "actor": "Nursing + physician",
            "base_min": 3.2,
            "noise_sd": 0.5,
        },
    ],
    "tsp": [
        {
            "microstep_code": "ice_septal_survey",
            "microstep_label": "ICE survey and septal visualization",
            "actor": "Physician",
            "base_min": 5.8,
            "noise_sd": 0.8,
        },
        {
            "microstep_code": "first_transseptal_crossing",
            "microstep_label": "First transseptal crossing",
            "actor": "Physician",
            "base_min": 6.4,
            "noise_sd": 1.2,
        },
        {
            "microstep_code": "second_crossing_sheath_exchange",
            "microstep_label": "Second crossing and sheath exchange",
            "actor": "Physician + technologist",
            "base_min": 5.2,
            "noise_sd": 1.0,
        },
        {
            "microstep_code": "left_atrial_confirmation",
            "microstep_label": "Left atrial confirmation",
            "actor": "Physician + mapping tech",
            "base_min": 3.4,
            "noise_sd": 0.6,
        },
    ],
    "mapping": [
        {
            "microstep_code": "left_atrial_geometry_build",
            "microstep_label": "Left atrial geometry build",
            "actor": "Physician + mapping tech",
            "base_min": 7.5,
            "noise_sd": 1.1,
        },
        {
            "microstep_code": "ct_map_registration",
            "microstep_label": "CT and map registration",
            "actor": "Mapping tech",
            "base_min": 4.2,
            "noise_sd": 0.7,
        },
        {
            "microstep_code": "baseline_voltage_review",
            "microstep_label": "Baseline voltage review",
            "actor": "Physician + mapping tech",
            "base_min": 4.0,
            "noise_sd": 0.7,
        },
    ],
    "ablation": [
        {
            "microstep_code": "lesion_set_delivery",
            "microstep_label": "Lesion set delivery",
            "actor": "Physician",
            "base_min": 12.0,
            "noise_sd": 1.6,
        },
        {
            "microstep_code": "catheter_reposition_touchup",
            "microstep_label": "Catheter repositioning and touch-up lesions",
            "actor": "Physician + technologist",
            "base_min": 6.8,
            "noise_sd": 1.0,
        },
        {
            "microstep_code": "endpoint_verification",
            "microstep_label": "Endpoint verification",
            "actor": "Physician + mapping tech",
            "base_min": 5.0,
            "noise_sd": 0.7,
        },
    ],
    "close": [
        {
            "microstep_code": "sheath_removal_hemostasis",
            "microstep_label": "Sheath removal and hemostasis",
            "actor": "Physician + nursing",
            "base_min": 6.2,
            "noise_sd": 0.9,
        },
        {
            "microstep_code": "wake_up_extubation",
            "microstep_label": "Wake-up and extubation",
            "actor": "Anesthesia",
            "base_min": 7.8,
            "noise_sd": 1.0,
        },
        {
            "microstep_code": "transfer_off_table",
            "microstep_label": "Transfer off table",
            "actor": "Nursing + anesthesia",
            "base_min": 4.5,
            "noise_sd": 0.8,
        },
    ],
}


MICROSTEP_METADATA: dict[str, dict[str, str]] = {
    "patient_transfer_positioning": {
        "category": "Waiting / transfer",
        "transcript_template": "Team coordinates patient transfer, positioning, and safety handoff before sterile work begins.",
        "video_event": "Multiple staff around bed, transfer motion, brief idle gap before final table alignment.",
    },
    "anesthesia_induction_airway": {
        "category": "Verification / anesthesia",
        "transcript_template": "Anesthesia team confirms induction status, airway readiness, and table-side monitoring.",
        "video_event": "Head-of-bed activity, airway setup, and reduced surgical motion while induction completes.",
    },
    "tee_safety_check": {
        "category": "Verification",
        "transcript_template": "Physician reviews TEE findings and confirms there is no clot before proceeding.",
        "video_event": "Probe manipulation with imaging console attention and low room movement.",
    },
    "sterile_prep_monitor_hookup": {
        "category": "Equipment handling",
        "transcript_template": "Staff complete sterile prep, drape placement, and monitor or patch hookup.",
        "video_event": "Cable setup, tray interaction, and repeated circulator movement around the field.",
    },
    "ultrasound_groin_scan": {
        "category": "Verification",
        "transcript_template": "Ultrasound is used to assess the groin and identify the safest access path.",
        "video_event": "Ultrasound probe use, focused physician posture, and brief table-side adjustment.",
    },
    "venous_access_sheath_insertion": {
        "category": "Procedural manipulation",
        "transcript_template": "Venous access is obtained and sheaths are advanced while staff confirm readiness.",
        "video_event": "Needle or sheath insertion motion with coordinated handoff between physician and scrub tech.",
    },
    "catheter_cable_console_setup": {
        "category": "Equipment handling",
        "transcript_template": "Catheters and cables are connected to the console and validated before use.",
        "video_event": "Console interaction, cable routing, and circulator movement toward storage or backup equipment.",
    },
    "anticoagulation_irrigation_check": {
        "category": "Verification",
        "transcript_template": "The team verifies irrigation flow, anticoagulation status, and line readiness.",
        "video_event": "Pump or line inspection with short pauses while settings are confirmed.",
    },
    "ice_septal_survey": {
        "category": "Verification",
        "transcript_template": "ICE is used to survey the septum and confirm the crossing plan.",
        "video_event": "ICE catheter manipulation and sustained attention to the imaging monitor.",
    },
    "first_transseptal_crossing": {
        "category": "Procedural manipulation",
        "transcript_template": "The physician performs the first transseptal crossing under imaging guidance.",
        "video_event": "Slow deliberate catheter movement with minimal room motion during the crossing attempt.",
    },
    "second_crossing_sheath_exchange": {
        "category": "Equipment handling",
        "transcript_template": "A second crossing or sheath exchange is completed to finalize left atrial access.",
        "video_event": "Repeated tool exchange, scrub handoff, and brief delays while alternate equipment is prepared.",
    },
    "left_atrial_confirmation": {
        "category": "Verification",
        "transcript_template": "The team confirms final left atrial position and readiness for mapping.",
        "video_event": "Short pause at the console with attention shifting between map and echo displays.",
    },
    "left_atrial_geometry_build": {
        "category": "Mapping",
        "transcript_template": "Mapping staff and physician build the left atrial geometry and align landmarks.",
        "video_event": "Sustained console interaction and controlled catheter motion around the atrium.",
    },
    "ct_map_registration": {
        "category": "Mapping",
        "transcript_template": "CT registration is refined to align anatomy and mapping geometry.",
        "video_event": "Console-heavy work with low physical motion and intermittent calibration adjustments.",
    },
    "baseline_voltage_review": {
        "category": "Verification",
        "transcript_template": "Baseline voltage is reviewed to confirm lesion targets and pre-ablation state.",
        "video_event": "Multi-screen review and brief discussion before the next catheter maneuver.",
    },
    "lesion_set_delivery": {
        "category": "Procedural manipulation",
        "transcript_template": "The ablation lesion set is delivered while the team tracks progression and stability.",
        "video_event": "Repeated catheter repositioning with stable room layout and periodic console checks.",
    },
    "catheter_reposition_touchup": {
        "category": "Procedural manipulation",
        "transcript_template": "Additional catheter repositioning or touch-up lesions are performed to close gaps.",
        "video_event": "Back-and-forth catheter motion with occasional pauses to assess contact or signal quality.",
    },
    "endpoint_verification": {
        "category": "Verification",
        "transcript_template": "Endpoints are verified and the team confirms durable isolation before closing.",
        "video_event": "Reduced motion, console focus, and short confirmation loops around the endpoint check.",
    },
    "sheath_removal_hemostasis": {
        "category": "Procedural manipulation",
        "transcript_template": "Sheaths are removed and hemostasis is achieved before recovery begins.",
        "video_event": "Table-side manual work with limited room movement and focused nursing support.",
    },
    "wake_up_extubation": {
        "category": "Waiting / transfer",
        "transcript_template": "Anesthesia leads wake-up and extubation while the rest of the team prepares transfer.",
        "video_event": "Low procedural motion, airway attention, and a short reset before patient transfer.",
    },
    "transfer_off_table": {
        "category": "Waiting / transfer",
        "transcript_template": "The patient is transferred off the table with nursing and anesthesia coordination.",
        "video_event": "Multi-person transfer movement with a brief congestion point near the table edge.",
    },
}


CLINICS: list[dict[str, object]] = [
    {
        "clinic_id": "CL1",
        "clinic_name": "Northside Heart Center",
        "clinic_city": "Toronto",
        "clinic_color": "#0f766e",
        "age_mean": 65,
        "bmi_mean": 30.5,
        "mobility_rate": 0.22,
        "prior_ablation_rate": 0.18,
        "persistent_af_rate": 0.32,
        "anatomy_probs": [0.50, 0.35, 0.15],
        "septum_probs": [0.56, 0.31, 0.13],
        "planned_lesion_mean": 53,
        "equipment_issue_rate": 0.18,
        "noise_multiplier": 0.85,
        "stage_bias": {"pre_op": -0.7, "access": -0.8, "tsp": -0.6, "close": -0.5},
        "process_rates": {
            "transfer_lift_used": 0.90,
            "dedicated_runner": 0.92,
            "ultrasound_access": 0.96,
            "standardized_sheath_pack": 0.88,
            "ice_first_tsp": 0.90,
            "mapping_tech_support": 0.84,
            "closure_checklist": 0.94,
        },
    },
    {
        "clinic_id": "CL2",
        "clinic_name": "Lakeshore EP Lab",
        "clinic_city": "Mississauga",
        "clinic_color": "#b45309",
        "age_mean": 69,
        "bmi_mean": 33.5,
        "mobility_rate": 0.31,
        "prior_ablation_rate": 0.25,
        "persistent_af_rate": 0.42,
        "anatomy_probs": [0.36, 0.42, 0.22],
        "septum_probs": [0.42, 0.36, 0.22],
        "planned_lesion_mean": 57,
        "equipment_issue_rate": 0.31,
        "noise_multiplier": 1.08,
        "stage_bias": {"pre_op": 1.1, "tsp": 1.0, "close": 0.9},
        "process_rates": {
            "transfer_lift_used": 0.52,
            "dedicated_runner": 0.54,
            "ultrasound_access": 0.72,
            "standardized_sheath_pack": 0.48,
            "ice_first_tsp": 0.63,
            "mapping_tech_support": 0.44,
            "closure_checklist": 0.58,
        },
    },
    {
        "clinic_id": "CL3",
        "clinic_name": "Metro Rhythm Institute",
        "clinic_city": "Hamilton",
        "clinic_color": "#2563eb",
        "age_mean": 63,
        "bmi_mean": 31.0,
        "mobility_rate": 0.20,
        "prior_ablation_rate": 0.28,
        "persistent_af_rate": 0.48,
        "anatomy_probs": [0.39, 0.40, 0.21],
        "septum_probs": [0.46, 0.34, 0.20],
        "planned_lesion_mean": 60,
        "equipment_issue_rate": 0.24,
        "noise_multiplier": 0.95,
        "stage_bias": {"mapping": -0.7, "ablation": -1.0},
        "process_rates": {
            "transfer_lift_used": 0.58,
            "dedicated_runner": 0.67,
            "ultrasound_access": 0.88,
            "standardized_sheath_pack": 0.62,
            "ice_first_tsp": 0.82,
            "mapping_tech_support": 0.91,
            "closure_checklist": 0.73,
        },
    },
]


DOCTORS: list[dict[str, object]] = [
    {
        "doctor_id": "DR_A",
        "doctor_name": "Dr. A",
        "clinic_id": "CL1",
        "global_bias": -2.6,
        "variability_scale": 0.82,
        "teaching_prob": 0.10,
        "custom_sheath_pref": 0.10,
        "stage_bias": {"pre_op": -0.4, "tsp": -1.0},
    },
    {
        "doctor_id": "DR_B",
        "doctor_name": "Dr. B",
        "clinic_id": "CL1",
        "global_bias": 0.8,
        "variability_scale": 0.94,
        "teaching_prob": 0.24,
        "custom_sheath_pref": 0.22,
        "stage_bias": {"pre_op": 0.7, "mapping": 0.5},
    },
    {
        "doctor_id": "DR_C",
        "doctor_name": "Dr. C",
        "clinic_id": "CL2",
        "global_bias": 1.8,
        "variability_scale": 1.06,
        "teaching_prob": 0.12,
        "custom_sheath_pref": 0.18,
        "stage_bias": {"pre_op": 0.9, "close": 0.6},
    },
    {
        "doctor_id": "DR_D",
        "doctor_name": "Dr. D",
        "clinic_id": "CL2",
        "global_bias": 1.1,
        "variability_scale": 1.24,
        "teaching_prob": 0.18,
        "custom_sheath_pref": 0.55,
        "stage_bias": {"access": 0.8, "tsp": 1.3},
    },
    {
        "doctor_id": "DR_E",
        "doctor_name": "Dr. E",
        "clinic_id": "CL3",
        "global_bias": -1.4,
        "variability_scale": 0.90,
        "teaching_prob": 0.08,
        "custom_sheath_pref": 0.14,
        "stage_bias": {"mapping": -0.5, "ablation": -1.2},
    },
    {
        "doctor_id": "DR_F",
        "doctor_name": "Dr. F",
        "clinic_id": "CL3",
        "global_bias": 1.5,
        "variability_scale": 1.18,
        "teaching_prob": 0.20,
        "custom_sheath_pref": 0.64,
        "stage_bias": {"access": 0.9, "tsp": 0.8, "close": 0.8},
    },
]


PROCESS_LABELS = {
    "transfer_lift_used": "Transfer lift and extra hands",
    "dedicated_runner": "Dedicated runner",
    "ultrasound_access": "Ultrasound-first access",
    "standardized_sheath_pack": "Standardized sheath pack",
    "ice_first_tsp": "ICE-first TSP",
    "mapping_tech_support": "Mapping tech support",
    "closure_checklist": "Closure checklist",
}


STAGE_PROCESS_MAP = {
    "pre_op": ["transfer_lift_used", "dedicated_runner"],
    "access": ["ultrasound_access", "standardized_sheath_pack", "dedicated_runner"],
    "tsp": ["ice_first_tsp", "standardized_sheath_pack"],
    "mapping": ["mapping_tech_support"],
    "ablation": ["mapping_tech_support"],
    "close": ["closure_checklist", "transfer_lift_used", "dedicated_runner"],
}


PROCESS_BENCHMARK_CONFIG = [
    {
        "process_flag": "transfer_lift_used",
        "process_label": PROCESS_LABELS["transfer_lift_used"],
        "stage_code": "pre_op",
        "stage_name": "Pre-op and positioning",
        "filter_name": "high_transfer_risk",
    },
    {
        "process_flag": "ultrasound_access",
        "process_label": PROCESS_LABELS["ultrasound_access"],
        "stage_code": "access",
        "stage_name": "Access and sheath setup",
        "filter_name": "all",
    },
    {
        "process_flag": "standardized_sheath_pack",
        "process_label": PROCESS_LABELS["standardized_sheath_pack"],
        "stage_code": "tsp",
        "stage_name": "Transseptal puncture",
        "filter_name": "all",
    },
    {
        "process_flag": "mapping_tech_support",
        "process_label": PROCESS_LABELS["mapping_tech_support"],
        "stage_code": "mapping",
        "stage_name": "Mapping and baseline validation",
        "filter_name": "complex_mapping",
    },
    {
        "process_flag": "closure_checklist",
        "process_label": PROCESS_LABELS["closure_checklist"],
        "stage_code": "close",
        "stage_name": "Close and transfer",
        "filter_name": "all",
    },
]


def stage_lookup() -> dict[str, dict[str, object]]:
    return {stage["stage_code"]: stage for stage in STAGES}


def clinic_lookup() -> dict[str, dict[str, object]]:
    return {clinic["clinic_id"]: clinic for clinic in CLINICS}


def slugify(text: str) -> str:
    clean = text.lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")


def obesity_class_from_bmi(bmi: float) -> int:
    if bmi >= 40:
        return 3
    if bmi >= 35:
        return 2
    if bmi >= 30:
        return 1
    return 0


def choose_weighted(rng: np.random.Generator, choices: list[int], probabilities: list[float]) -> int:
    return int(rng.choice(choices, p=probabilities))


def round_half(value: float) -> float:
    return round(float(value), 2)


def driver_summary(contributions: dict[str, float], *, positive_only: bool = False, limit: int = 3) -> str:
    items = []
    for label, minutes in contributions.items():
        if label == "Routine variation":
            continue
        if positive_only and minutes <= 0:
            continue
        items.append((label, minutes))

    if not items:
        if positive_only:
            return "No single delay driver dominated"
        return "Routine flow"

    items.sort(key=lambda item: abs(item[1]), reverse=True)
    parts = [f"{label} ({minutes:+.1f}m)" for label, minutes in items[:limit]]
    return ", ".join(parts)


def dominant_driver(contributions: dict[str, float], *, positive_only: bool = False) -> str:
    items = []
    for label, minutes in contributions.items():
        if label == "Routine variation":
            continue
        if positive_only and minutes <= 0:
            continue
        items.append((label, minutes))

    if not items:
        return "Routine variation"

    items.sort(key=lambda item: abs(item[1]), reverse=True)
    return items[0][0]


def relevant_process_labels(case_flags: dict[str, object], stage_code: str) -> str:
    labels = [
        PROCESS_LABELS[flag]
        for flag in STAGE_PROCESS_MAP[stage_code]
        if bool(case_flags.get(flag))
    ]
    return ", ".join(labels) if labels else "Baseline workflow"


def process_signature(case_flags: dict[str, object]) -> str:
    labels = [label for flag, label in PROCESS_LABELS.items() if bool(case_flags.get(flag))]
    return " | ".join(labels) if labels else "No enhanced workflow flags"


def benchmark_filter_mask(df: pd.DataFrame, filter_name: str) -> pd.Series:
    if filter_name == "high_transfer_risk":
        return (df["obesity_class"] >= 2) | (df["mobility_limited"])
    if filter_name == "complex_mapping":
        return (df["anatomy_complexity"] >= 2) | df["prior_ablation"]
    return pd.Series(True, index=df.index)


def build_schema_definition() -> dict[str, object]:
    return {
        "assumption": (
            "Every case has a full transcript annotation stream that is segmented into stages, "
            "mapped to microsteps, timestamped, and joined with patient, provider, and clinic context."
        ),
        "tables": [
            {
                "table_name": "variability_cases.csv",
                "grain": "One row per surgery case",
                "primary_key": "case_id",
                "purpose": "Patient, physician, clinic, and process context for each case.",
                "key_fields": [
                    "case_id",
                    "case_date",
                    "clinic_name",
                    "doctor_name",
                    "age",
                    "bmi",
                    "mobility_limited",
                    "prior_ablation",
                    "anatomy_complexity",
                    "septum_thickness",
                    "planned_lesions",
                    "equipment_issue_count",
                    "total_case_duration_min",
                    "primary_delay_driver",
                ],
            },
            {
                "table_name": "variability_stage_summary.csv",
                "grain": "One row per case-stage",
                "primary_key": "case_id + stage_code",
                "purpose": "Duration and driver summary for each high-level stage in the procedure.",
                "key_fields": [
                    "case_id",
                    "stage_code",
                    "stage_name",
                    "stage_order",
                    "stage_start_min",
                    "stage_end_min",
                    "stage_duration_min",
                    "relevant_processes",
                    "primary_delay_driver",
                    "delay_driver_summary",
                ],
            },
            {
                "table_name": "variability_transcript_annotations.csv",
                "grain": "One row per transcript microstep segment",
                "primary_key": "case_id + stage_code + microstep_code + transcript_start_min",
                "purpose": "Transcript-derived microstep labels, durations, actors, and attributed variability factors.",
                "key_fields": [
                    "case_id",
                    "stage_code",
                    "stage_name",
                    "microstep_code",
                    "microstep_label",
                    "category",
                    "actor",
                    "transcript_start_min",
                    "transcript_end_min",
                    "duration_min",
                    "transcript_snippet",
                    "inferred_video_event",
                    "primary_delay_driver",
                    "delay_driver_summary",
                ],
            },
            {
                "table_name": "variability_process_benchmarks.csv",
                "grain": "One row per process-stage benchmark",
                "primary_key": "process_flag + stage_code",
                "purpose": "Estimated timing and reliability benefit of clinic workflow choices.",
                "key_fields": [
                    "process_flag",
                    "process_label",
                    "stage_code",
                    "stage_name",
                    "filter_name",
                    "cases_with_process",
                    "cases_without_process",
                    "mean_minutes_saved",
                    "sd_reduction_min",
                ],
            },
            {
                "table_name": "variability_step_rankings.csv",
                "grain": "One row per microstep type",
                "primary_key": "microstep_code",
                "purpose": "Cross-case ranking of the most variable microsteps and their leading causes.",
                "key_fields": [
                    "microstep_code",
                "microstep_label",
                "stage_name",
                "category",
                "cases",
                "mean_duration_min",
                "sd_duration_min",
                    "cv",
                    "top_delay_drivers",
                ],
            },
        ],
    }


def sample_case_context(
    rng: np.random.Generator,
    clinic: dict[str, object],
    doctor: dict[str, object],
    case_number: int,
) -> dict[str, object]:
    age = float(np.clip(rng.normal(clinic["age_mean"], 8), 45, 86))
    bmi = float(np.clip(rng.normal(clinic["bmi_mean"], 4.8), 22, 47))
    obesity_class = obesity_class_from_bmi(bmi)

    mobility_rate = float(clinic["mobility_rate"])
    mobility_rate += 0.06 if age >= 75 else 0.0
    mobility_rate += 0.10 if obesity_class >= 2 else 0.0
    mobility_limited = bool(rng.random() < min(mobility_rate, 0.72))

    persistent_af = bool(rng.random() < float(clinic["persistent_af_rate"]))
    prior_ablation = bool(rng.random() < float(clinic["prior_ablation_rate"]))
    anatomy_complexity = choose_weighted(rng, [1, 2, 3], list(clinic["anatomy_probs"]))
    septum_thickness = choose_weighted(rng, [1, 2, 3], list(clinic["septum_probs"]))

    if prior_ablation and anatomy_complexity < 3 and rng.random() < 0.35:
        anatomy_complexity += 1

    asa_base = 2
    asa_base += 1 if age >= 72 else 0
    asa_base += 1 if obesity_class >= 2 else 0
    asa_class = int(min(4, max(2, asa_base + rng.choice([-1, 0, 0, 1]))))

    planned_lesions = int(
        np.clip(
            rng.normal(clinic["planned_lesion_mean"] + (5 if persistent_af else 0) + (4 if prior_ablation else 0), 5.0),
            44,
            76,
        )
    )

    equipment_issue_rate = float(clinic["equipment_issue_rate"]) + (0.05 if rng.random() < 0.15 else 0.0)
    equipment_issue_count = int(min(rng.poisson(equipment_issue_rate), 2))
    teaching_case = bool(rng.random() < float(doctor["teaching_prob"]))

    flags = {}
    for flag, probability in dict(clinic["process_rates"]).items():
        adjusted_probability = float(probability)
        if flag == "transfer_lift_used":
            adjusted_probability *= 1.0 if mobility_limited or obesity_class >= 2 else 0.28
        if flag == "standardized_sheath_pack":
            adjusted_probability *= 0.75 if rng.random() < float(doctor["custom_sheath_pref"]) else 1.0
        flags[flag] = bool(rng.random() < min(max(adjusted_probability, 0.02), 0.98))

    complexity_score = int(
        20
        + obesity_class * 6
        + anatomy_complexity * 9
        + septum_thickness * 6
        + (5 if prior_ablation else 0)
        + (4 if persistent_af else 0)
    )

    return {
        "case_number": case_number,
        "age": int(round(age)),
        "bmi": round_half(bmi),
        "obesity_class": obesity_class,
        "mobility_limited": mobility_limited,
        "persistent_af": persistent_af,
        "prior_ablation": prior_ablation,
        "anatomy_complexity": anatomy_complexity,
        "septum_thickness": septum_thickness,
        "asa_class": asa_class,
        "planned_lesions": planned_lesions,
        "equipment_issue_count": equipment_issue_count,
        "teaching_case": teaching_case,
        "complexity_score": complexity_score,
        **flags,
    }


def compute_microstep_duration(
    step: dict[str, object],
    stage_code: str,
    context: dict[str, object],
    clinic: dict[str, object],
    doctor: dict[str, object],
    rng: np.random.Generator,
) -> tuple[float, dict[str, float]]:
    contributions: defaultdict[str, float] = defaultdict(float)
    base_min = float(step["base_min"])
    duration = base_min

    def add(label: str, minutes: float) -> None:
        nonlocal duration
        if minutes == 0:
            return
        duration += minutes
        contributions[label] += minutes

    steps_in_stage = len(MICROSTEP_LIBRARY[stage_code])
    add("Doctor technique", float(doctor["global_bias"]) / 18.0)
    add("Doctor technique", float(dict(doctor["stage_bias"]).get(stage_code, 0.0)) / steps_in_stage)
    add("Clinic workflow", float(dict(clinic["stage_bias"]).get(stage_code, 0.0)) / steps_in_stage)

    obesity_class = int(context["obesity_class"])
    anatomy_complexity = int(context["anatomy_complexity"])
    septum_thickness = int(context["septum_thickness"])
    equipment_issue_count = int(context["equipment_issue_count"])
    planned_lesions = int(context["planned_lesions"])

    if step["microstep_code"] == "patient_transfer_positioning":
        add("Mobility limitation", 2.6 if bool(context["mobility_limited"]) else 0.0)
        add("Obesity / transfer risk", [0.0, 1.4, 3.0, 4.8][obesity_class])
        add("Older patient mobility", 1.0 if int(context["age"]) >= 75 else 0.0)
        add(PROCESS_LABELS["transfer_lift_used"], -2.3 if bool(context["transfer_lift_used"]) else 0.0)
        add(PROCESS_LABELS["dedicated_runner"], -0.5 if bool(context["dedicated_runner"]) else 0.0)
    elif step["microstep_code"] == "anesthesia_induction_airway":
        add("Higher anesthesia risk", 1.2 if int(context["asa_class"]) >= 3 else 0.0)
        add("Obesity / airway complexity", 1.8 if obesity_class >= 2 else 0.0)
        add("Older patient mobility", 0.7 if int(context["age"]) >= 75 else 0.0)
    elif step["microstep_code"] == "tee_safety_check":
        add("Anatomy complexity", 0.5 * max(anatomy_complexity - 1, 0))
        add("Prior ablation scar", 0.6 if bool(context["prior_ablation"]) else 0.0)
    elif step["microstep_code"] == "sterile_prep_monitor_hookup":
        add("Equipment issue", 0.8 * equipment_issue_count)
        add(PROCESS_LABELS["dedicated_runner"], -0.7 if bool(context["dedicated_runner"]) else 0.0)
    elif step["microstep_code"] == "ultrasound_groin_scan":
        if bool(context["ultrasound_access"]):
            add(PROCESS_LABELS["ultrasound_access"], -0.5)
            add("Obesity / access difficulty", 0.4 * obesity_class)
        else:
            add("Obesity / access difficulty", 1.1 * obesity_class)
            add("Anatomy complexity", 0.6 * max(anatomy_complexity - 1, 0))
    elif step["microstep_code"] == "venous_access_sheath_insertion":
        add("Anatomy complexity", 0.9 * max(anatomy_complexity - 1, 0))
        add("Obesity / access difficulty", 0.7 * obesity_class)
        add(PROCESS_LABELS["standardized_sheath_pack"], -1.0 if bool(context["standardized_sheath_pack"]) else 0.0)
        add("Custom sheath choice", 1.0 if rng.random() < float(doctor["custom_sheath_pref"]) else 0.0)
    elif step["microstep_code"] == "catheter_cable_console_setup":
        add("Equipment issue", 1.4 * equipment_issue_count)
        add(PROCESS_LABELS["dedicated_runner"], -0.6 if bool(context["dedicated_runner"]) else 0.0)
        add(PROCESS_LABELS["mapping_tech_support"], -0.4 if bool(context["mapping_tech_support"]) else 0.0)
    elif step["microstep_code"] == "anticoagulation_irrigation_check":
        add("Equipment issue", 0.4 * equipment_issue_count)
        add(PROCESS_LABELS["standardized_sheath_pack"], -0.3 if bool(context["standardized_sheath_pack"]) else 0.0)
    elif step["microstep_code"] == "ice_septal_survey":
        add("Anatomy complexity", 0.7 * max(anatomy_complexity - 1, 0))
        add(PROCESS_LABELS["ice_first_tsp"], -0.7 if bool(context["ice_first_tsp"]) else 0.0)
    elif step["microstep_code"] == "first_transseptal_crossing":
        add("Thick septum", [0.0, 0.6, 2.2, 4.6][septum_thickness])
        add("Anatomy complexity", 0.5 * max(anatomy_complexity - 1, 0))
    elif step["microstep_code"] == "second_crossing_sheath_exchange":
        add("Thick septum", [0.0, 0.4, 1.4, 2.8][septum_thickness])
        add(PROCESS_LABELS["standardized_sheath_pack"], -0.6 if bool(context["standardized_sheath_pack"]) else 0.0)
        add("Custom sheath choice", 0.8 if rng.random() < float(doctor["custom_sheath_pref"]) else 0.0)
    elif step["microstep_code"] == "left_atrial_confirmation":
        add("Anatomy complexity", 0.4 * max(anatomy_complexity - 1, 0))
        add(PROCESS_LABELS["mapping_tech_support"], -0.3 if bool(context["mapping_tech_support"]) else 0.0)
    elif step["microstep_code"] == "left_atrial_geometry_build":
        add("Anatomy complexity", 1.0 * max(anatomy_complexity - 1, 0))
        add("Prior ablation scar", 1.0 if bool(context["prior_ablation"]) else 0.0)
        add(PROCESS_LABELS["mapping_tech_support"], -1.2 if bool(context["mapping_tech_support"]) else 0.0)
    elif step["microstep_code"] == "ct_map_registration":
        add("Anatomy complexity", 0.6 * max(anatomy_complexity - 1, 0))
        add(PROCESS_LABELS["mapping_tech_support"], -0.6 if bool(context["mapping_tech_support"]) else 0.0)
    elif step["microstep_code"] == "baseline_voltage_review":
        add("Prior ablation scar", 1.4 if bool(context["prior_ablation"]) else 0.0)
        add("Persistent AF burden", 0.9 if bool(context["persistent_af"]) else 0.0)
        add("Teaching moments", 0.8 if bool(context["teaching_case"]) else 0.0)
    elif step["microstep_code"] == "lesion_set_delivery":
        add("Planned lesion count", max(planned_lesions - 52, 0) * 0.18)
        add("Persistent AF burden", 2.0 if bool(context["persistent_af"]) else 0.0)
        add("Prior ablation scar", 1.5 if bool(context["prior_ablation"]) else 0.0)
    elif step["microstep_code"] == "catheter_reposition_touchup":
        add("Anatomy complexity", 0.9 * max(anatomy_complexity - 1, 0))
        add("Equipment issue", 0.7 * equipment_issue_count)
        add("Prior ablation scar", 1.0 if bool(context["prior_ablation"]) else 0.0)
    elif step["microstep_code"] == "endpoint_verification":
        add("Prior ablation scar", 0.8 if bool(context["prior_ablation"]) else 0.0)
        add("Anatomy complexity", 0.4 * max(anatomy_complexity - 1, 0))
        add(PROCESS_LABELS["mapping_tech_support"], -0.4 if bool(context["mapping_tech_support"]) else 0.0)
    elif step["microstep_code"] == "sheath_removal_hemostasis":
        add("Obesity / transfer risk", 0.8 * obesity_class)
        add(PROCESS_LABELS["closure_checklist"], -0.7 if bool(context["closure_checklist"]) else 0.0)
    elif step["microstep_code"] == "wake_up_extubation":
        add("Obesity / airway complexity", 1.2 * obesity_class)
        add("Older patient mobility", 1.0 if int(context["age"]) >= 75 else 0.0)
        add("Higher anesthesia risk", 1.2 if int(context["asa_class"]) >= 3 else 0.0)
    elif step["microstep_code"] == "transfer_off_table":
        add("Mobility limitation", 2.0 if bool(context["mobility_limited"]) else 0.0)
        add("Obesity / transfer risk", 1.3 * obesity_class)
        add(PROCESS_LABELS["transfer_lift_used"], -1.8 if bool(context["transfer_lift_used"]) else 0.0)
        add(PROCESS_LABELS["dedicated_runner"], -0.4 if bool(context["dedicated_runner"]) else 0.0)

    noise_sd = float(step["noise_sd"]) * float(doctor["variability_scale"]) * float(clinic["noise_multiplier"])
    if stage_code == "access" and bool(context["ultrasound_access"]):
        noise_sd *= 0.85
    if stage_code == "tsp" and bool(context["ice_first_tsp"]):
        noise_sd *= 0.85
    if stage_code in {"mapping", "ablation"} and bool(context["mapping_tech_support"]):
        noise_sd *= 0.82
    if stage_code == "close" and bool(context["closure_checklist"]):
        noise_sd *= 0.84
    if stage_code in {"pre_op", "close"} and bool(context["transfer_lift_used"]):
        noise_sd *= 0.83

    noise = float(rng.normal(0.0, noise_sd))
    add("Routine variation", noise)
    duration = max(duration, 1.0)

    return round_half(duration), dict(contributions)


def build_step_rankings(steps_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (microstep_code, microstep_label, stage_name, category), group in steps_df.groupby(
        ["microstep_code", "microstep_label", "stage_name", "category"],
        sort=False,
    ):
        driver_counts = (
            group.loc[group["primary_delay_driver"] != "Routine variation", "primary_delay_driver"]
            .value_counts()
            .head(3)
        )
        top_drivers = ", ".join(
            f"{driver} ({count} cases)"
            for driver, count in driver_counts.items()
        ) or "Routine variation"

        mean_duration = float(group["duration_min"].mean())
        sd_duration = float(group["duration_min"].std(ddof=1))
        rows.append(
            {
                "microstep_code": microstep_code,
                "microstep_label": microstep_label,
                "stage_name": stage_name,
                "category": category,
                "cases": int(group["case_id"].nunique()),
                "mean_duration_min": round_half(mean_duration),
                "sd_duration_min": round_half(sd_duration),
                "p90_duration_min": round_half(float(group["duration_min"].quantile(0.9))),
                "cv": round_half(sd_duration / mean_duration) if mean_duration else 0.0,
                "top_delay_drivers": top_drivers,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["sd_duration_min", "mean_duration_min"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_process_benchmarks(cases_df: pd.DataFrame, stages_df: pd.DataFrame) -> pd.DataFrame:
    merged = stages_df.merge(
        cases_df[
            [
                "case_id",
                "clinic_name",
                "doctor_name",
                "obesity_class",
                "mobility_limited",
                "prior_ablation",
                "anatomy_complexity",
                "transfer_lift_used",
                "ultrasound_access",
                "standardized_sheath_pack",
                "mapping_tech_support",
                "closure_checklist",
            ]
        ],
        on="case_id",
        how="left",
    )

    rows: list[dict[str, object]] = []
    for config in PROCESS_BENCHMARK_CONFIG:
        subset = merged.loc[merged["stage_code"] == config["stage_code"]].copy()
        subset = subset.loc[benchmark_filter_mask(subset, str(config["filter_name"]))]

        with_process = subset.loc[subset[str(config["process_flag"])]]
        without_process = subset.loc[~subset[str(config["process_flag"])]]
        if with_process.empty or without_process.empty:
            continue

        mean_with = float(with_process["stage_duration_min"].mean())
        mean_without = float(without_process["stage_duration_min"].mean())
        sd_with = float(with_process["stage_duration_min"].std(ddof=1))
        sd_without = float(without_process["stage_duration_min"].std(ddof=1))

        rows.append(
            {
                "process_flag": config["process_flag"],
                "process_label": config["process_label"],
                "stage_code": config["stage_code"],
                "stage_name": config["stage_name"],
                "filter_name": config["filter_name"],
                "cases_with_process": int(with_process["case_id"].nunique()),
                "cases_without_process": int(without_process["case_id"].nunique()),
                "mean_duration_with_process_min": round_half(mean_with),
                "mean_duration_without_process_min": round_half(mean_without),
                "mean_minutes_saved": round_half(mean_without - mean_with),
                "sd_with_process_min": round_half(sd_with),
                "sd_without_process_min": round_half(sd_without),
                "sd_reduction_min": round_half(sd_without - sd_with),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["mean_minutes_saved", "sd_reduction_min"], ascending=False)
        .reset_index(drop=True)
    )


def generate_mock_dataset(seed: int = RNG_SEED, cases_per_doctor: int = 30) -> dict[str, object]:
    rng = np.random.default_rng(seed)

    clinic_by_id = clinic_lookup()
    stage_by_code = stage_lookup()

    case_rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []

    base_date = pd.Timestamp("2025-01-06")
    case_counter = 1

    for doctor in DOCTORS:
        clinic = clinic_by_id[str(doctor["clinic_id"])]
        for local_index in range(cases_per_doctor):
            case_id = f"CASE_{case_counter:03d}"
            case_date = base_date + pd.Timedelta(days=case_counter - 1)
            context = sample_case_context(rng, clinic, doctor, case_counter)

            timeline = 0.0
            case_contributions: defaultdict[str, float] = defaultdict(float)

            for stage in STAGES:
                stage_code = str(stage["stage_code"])
                stage_start = timeline
                stage_contributions: defaultdict[str, float] = defaultdict(float)
                relevant_processes = relevant_process_labels(context, stage_code)

                for step_order, step in enumerate(MICROSTEP_LIBRARY[stage_code], start=1):
                    duration, contributions = compute_microstep_duration(step, stage_code, context, clinic, doctor, rng)
                    start_min = timeline
                    end_min = timeline + duration
                    timeline = end_min
                    metadata = MICROSTEP_METADATA[step["microstep_code"]]

                    for label, minutes in contributions.items():
                        stage_contributions[label] += minutes
                        case_contributions[label] += minutes

                    step_rows.append(
                        {
                            "case_id": case_id,
                            "case_date": case_date.date().isoformat(),
                            "clinic_id": clinic["clinic_id"],
                            "clinic_name": clinic["clinic_name"],
                            "doctor_id": doctor["doctor_id"],
                            "doctor_name": doctor["doctor_name"],
                            "stage_code": stage_code,
                            "stage_name": stage["stage_name"],
                            "stage_order": int(stage["stage_order"]),
                            "microstep_code": step["microstep_code"],
                            "microstep_label": step["microstep_label"],
                            "category": metadata["category"],
                            "microstep_order": step_order,
                            "actor": step["actor"],
                            "transcript_start_min": round_half(start_min),
                            "transcript_end_min": round_half(end_min),
                            "duration_min": duration,
                            "transcript_snippet": metadata["transcript_template"],
                            "inferred_video_event": metadata["video_event"],
                            "relevant_processes": relevant_processes,
                            "primary_delay_driver": dominant_driver(contributions, positive_only=True),
                            "primary_efficiency_driver": dominant_driver(
                                {label: minutes for label, minutes in contributions.items() if minutes < 0}
                            ),
                            "delay_driver_summary": driver_summary(contributions, positive_only=True),
                            "contribution_snapshot": driver_summary(contributions, positive_only=False),
                        }
                    )

                stage_duration = timeline - stage_start
                stage_rows.append(
                    {
                        "case_id": case_id,
                        "case_date": case_date.date().isoformat(),
                        "clinic_id": clinic["clinic_id"],
                        "clinic_name": clinic["clinic_name"],
                        "doctor_id": doctor["doctor_id"],
                        "doctor_name": doctor["doctor_name"],
                        "stage_code": stage_code,
                        "stage_name": stage["stage_name"],
                        "stage_order": int(stage["stage_order"]),
                        "stage_start_min": round_half(stage_start),
                        "stage_end_min": round_half(timeline),
                        "stage_duration_min": round_half(stage_duration),
                        "relevant_processes": relevant_processes,
                        "primary_delay_driver": dominant_driver(stage_contributions, positive_only=True),
                        "primary_efficiency_driver": dominant_driver(
                            {label: minutes for label, minutes in stage_contributions.items() if minutes < 0}
                        ),
                        "delay_driver_summary": driver_summary(stage_contributions, positive_only=True),
                        "contribution_snapshot": driver_summary(stage_contributions, positive_only=False),
                    }
                )

            case_rows.append(
                {
                    "case_id": case_id,
                    "case_number": int(context["case_number"]),
                    "case_date": case_date.date().isoformat(),
                    "clinic_id": clinic["clinic_id"],
                    "clinic_name": clinic["clinic_name"],
                    "clinic_city": clinic["clinic_city"],
                    "doctor_id": doctor["doctor_id"],
                    "doctor_name": doctor["doctor_name"],
                    "age": int(context["age"]),
                    "bmi": float(context["bmi"]),
                    "obesity_class": int(context["obesity_class"]),
                    "mobility_limited": bool(context["mobility_limited"]),
                    "persistent_af": bool(context["persistent_af"]),
                    "prior_ablation": bool(context["prior_ablation"]),
                    "anatomy_complexity": int(context["anatomy_complexity"]),
                    "septum_thickness": int(context["septum_thickness"]),
                    "asa_class": int(context["asa_class"]),
                    "planned_lesions": int(context["planned_lesions"]),
                    "equipment_issue_count": int(context["equipment_issue_count"]),
                    "teaching_case": bool(context["teaching_case"]),
                    "complexity_score": int(context["complexity_score"]),
                    **{flag: bool(context[flag]) for flag in PROCESS_LABELS},
                    "process_signature": process_signature(context),
                    "total_case_duration_min": round_half(timeline),
                    "total_microsteps": int(sum(len(steps) for steps in MICROSTEP_LIBRARY.values())),
                    "primary_delay_driver": dominant_driver(case_contributions, positive_only=True),
                    "primary_efficiency_driver": dominant_driver(
                        {label: minutes for label, minutes in case_contributions.items() if minutes < 0}
                    ),
                    "delay_driver_summary": driver_summary(case_contributions, positive_only=True),
                    "contribution_snapshot": driver_summary(case_contributions, positive_only=False),
                }
            )
            case_counter += 1

    cases_df = pd.DataFrame(case_rows).sort_values("case_number").reset_index(drop=True)
    stages_df = pd.DataFrame(stage_rows).sort_values(["case_id", "stage_order"]).reset_index(drop=True)
    steps_df = pd.DataFrame(step_rows).sort_values(["case_id", "stage_order", "microstep_order"]).reset_index(drop=True)
    benchmarks_df = build_process_benchmarks(cases_df, stages_df)
    step_rankings_df = build_step_rankings(steps_df)

    return {
        "cases": cases_df,
        "stages": stages_df,
        "steps": steps_df,
        "benchmarks": benchmarks_df,
        "step_rankings": step_rankings_df,
        "schema": build_schema_definition(),
        "stage_catalog": pd.DataFrame(STAGES),
        "doctor_catalog": pd.DataFrame(DOCTORS),
        "clinic_catalog": pd.DataFrame(CLINICS),
    }


def write_mock_outputs(output_dir: Path = OUTPUT_DIR, seed: int = RNG_SEED, cases_per_doctor: int = 30) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_mock_dataset(seed=seed, cases_per_doctor=cases_per_doctor)

    dataset["cases"].to_csv(output_dir / "variability_cases.csv", index=False)
    dataset["stages"].to_csv(output_dir / "variability_stage_summary.csv", index=False)
    dataset["steps"].to_csv(output_dir / "variability_transcript_annotations.csv", index=False)
    dataset["benchmarks"].to_csv(output_dir / "variability_process_benchmarks.csv", index=False)
    dataset["step_rankings"].to_csv(output_dir / "variability_step_rankings.csv", index=False)
    dataset["stage_catalog"].to_csv(output_dir / "variability_stage_catalog.csv", index=False)
    dataset["doctor_catalog"].to_csv(output_dir / "variability_doctor_catalog.csv", index=False)
    dataset["clinic_catalog"].to_csv(output_dir / "variability_clinic_catalog.csv", index=False)
    (output_dir / "variability_schema.json").write_text(
        json.dumps(dataset["schema"], indent=2),
        encoding="utf-8",
    )

    return dataset


if __name__ == "__main__":
    data = write_mock_outputs()
    print("Saved variability mock data:")
    print("-", OUTPUT_DIR / "variability_cases.csv")
    print("-", OUTPUT_DIR / "variability_stage_summary.csv")
    print("-", OUTPUT_DIR / "variability_transcript_annotations.csv")
    print("-", OUTPUT_DIR / "variability_process_benchmarks.csv")
    print("-", OUTPUT_DIR / "variability_step_rankings.csv")
    print("-", OUTPUT_DIR / "variability_schema.json")
    print()
    print(
        f"Generated {len(data['cases'])} cases, {len(data['stages'])} stage rows, and {len(data['steps'])} transcript annotations."
    )
