from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

TRANSCRIPT_PATH = ROOT / "data" / "Procedure_Transcript.txt"
OUTPUT_DIR = ROOT / "outputs"


def seconds_to_timestamp(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


CATEGORY_ORDER = [
    "Communication patterns",
    "Personnel interactions",
    "Equipment handling",
    "Room setup",
    "Procedural manipulation",
    "Imaging and mapping",
    "Verification and safety",
    "Handoff or waiting",
]


PHASE_CODING: list[dict[str, object]] = [
    {
        "phase_group": "Prep/intubation",
        "linked_metric": "PT PREP/INTUBATION Pt-In-Access",
        "start_sec": 0,
        "end_sec": 65,
        "micro_steps": [
            {"micro_step": "patient positioned on table", "category": "Room setup"},
            {"micro_step": "nurse final instructions", "category": "Communication patterns"},
            {"micro_step": "checklist run-through", "category": "Communication patterns"},
            {"micro_step": "anesthesia readiness", "category": "Personnel interactions"},
            {"micro_step": "physician safety and goal briefing", "category": "Communication patterns"},
            {"micro_step": "mapping and monitoring patches connected", "category": "Room setup"},
            {"micro_step": "TEE probe insertion", "category": "Equipment handling"},
            {"micro_step": "heart and valve assessment", "category": "Imaging and mapping"},
            {"micro_step": "left atrial appendage clot check", "category": "Verification and safety"},
            {"micro_step": "Doppler flow confirmation", "category": "Verification and safety"},
            {"micro_step": "technical details completed before access", "category": "Handoff or waiting"},
        ],
        "examples": "nurse instructions, checklist, anesthesia prep, TEE clot check, Doppler confirmation",
    },
    {
        "phase_group": "Access/setup",
        "linked_metric": "ACCESSS (Min)",
        "start_sec": 65,
        "end_sec": 198,
        "micro_steps": [
            {"micro_step": "bilateral femoral venous access", "category": "Procedural manipulation"},
            {"micro_step": "ultrasound-guided entry", "category": "Imaging and mapping"},
            {"micro_step": "sterile field setup", "category": "Room setup"},
            {"micro_step": "catheter-to-console electrical planning", "category": "Equipment handling"},
            {"micro_step": "cable layout on table", "category": "Equipment handling"},
            {"micro_step": "ablation catheter removed and prepped", "category": "Equipment handling"},
            {"micro_step": "irrigation function check", "category": "Verification and safety"},
            {"micro_step": "mechanical function check", "category": "Verification and safety"},
            {"micro_step": "catheter defect check", "category": "Verification and safety"},
            {"micro_step": "deflection mechanism check", "category": "Verification and safety"},
            {"micro_step": "electrical connection attached", "category": "Equipment handling"},
            {"micro_step": "intracardiac echo catheter prepared", "category": "Equipment handling"},
            {"micro_step": "transseptal needle prepared", "category": "Equipment handling"},
            {"micro_step": "short sheath advanced", "category": "Procedural manipulation"},
            {"micro_step": "long transseptal sheaths prepared", "category": "Equipment handling"},
            {"micro_step": "coagulation profile monitored", "category": "Verification and safety"},
            {"micro_step": "irrigation reconfirmed", "category": "Verification and safety"},
            {"micro_step": "coronary sinus catheter advanced", "category": "Procedural manipulation"},
        ],
        "examples": "ultrasound access, cable setup, irrigation/mechanical checks, sheath prep, coagulation monitoring",
    },
    {
        "phase_group": "TSP",
        "linked_metric": "TSP (Min)",
        "start_sec": 198,
        "end_sec": 429,
        "micro_steps": [
            {"micro_step": "prepare for transseptal catheterization", "category": "Handoff or waiting"},
            {"micro_step": "remove sheath dilator", "category": "Equipment handling"},
            {"micro_step": "flush transseptal sheaths", "category": "Equipment handling"},
            {"micro_step": "maintain continuous saline through sheaths", "category": "Equipment handling"},
            {"micro_step": "recreate right-sided geometry", "category": "Imaging and mapping"},
            {"micro_step": "incorporate coronary sinus landmarks", "category": "Imaging and mapping"},
            {"micro_step": "position additional catheters using map", "category": "Procedural manipulation"},
            {"micro_step": "ICE survey of cardiac structures", "category": "Imaging and mapping"},
            {"micro_step": "confirm no baseline effusion", "category": "Verification and safety"},
            {"micro_step": "perform first transseptal puncture", "category": "Procedural manipulation"},
            {"micro_step": "visualize septal tenting", "category": "Imaging and mapping"},
            {"micro_step": "confirm flight path and orientation", "category": "Verification and safety"},
            {"micro_step": "advance needle across septum", "category": "Procedural manipulation"},
            {"micro_step": "advance sheath and dilator into LA", "category": "Procedural manipulation"},
            {"micro_step": "saline bubble confirmation", "category": "Verification and safety"},
            {"micro_step": "confirm unobstructed left atrial access", "category": "Verification and safety"},
            {"micro_step": "advance ablation catheter into LA", "category": "Procedural manipulation"},
            {"micro_step": "anchor catheter for second puncture", "category": "Procedural manipulation"},
            {"micro_step": "perform second transseptal", "category": "Procedural manipulation"},
            {"micro_step": "advance mapping catheter and confirm final position", "category": "Procedural manipulation"},
        ],
        "examples": "sheath flushes, ICE guidance, septal tenting, saline bubbles, second transseptal",
    },
    {
        "phase_group": "Pre-map",
        "linked_metric": "PRE-MAP (Min)",
        "start_sec": 429,
        "end_sec": 482,
        "micro_steps": [
            {"micro_step": "manipulate mapping catheter in LA", "category": "Procedural manipulation"},
            {"micro_step": "reproduce left atrial geometry", "category": "Imaging and mapping"},
            {"micro_step": "compare geometry to CT", "category": "Imaging and mapping"},
            {"micro_step": "create chamber and pulmonary vein model", "category": "Imaging and mapping"},
            {"micro_step": "mapping experts guide control-panel work", "category": "Personnel interactions"},
            {"micro_step": "staff guidance during setup completion", "category": "Personnel interactions"},
            {"micro_step": "anesthesia stabilizes patient for ablation start", "category": "Handoff or waiting"},
        ],
        "examples": "3D geometry, CT comparison, mapping experts, staff guidance, stable anesthesia state",
    },
    {
        "phase_group": "Ablation/non-energy work",
        "linked_metric": "ablation_non_energy",
        "start_sec": 482,
        "end_sec": 585,
        "micro_steps": [
            {"micro_step": "start ablation sequence", "category": "Handoff or waiting"},
            {"micro_step": "point-by-point lesions around pulmonary veins", "category": "Procedural manipulation"},
            {"micro_step": "continuous echo guidance", "category": "Imaging and mapping"},
            {"micro_step": "continuous electroanatomic map guidance", "category": "Imaging and mapping"},
            {"micro_step": "continuous live electrogram guidance", "category": "Imaging and mapping"},
            {"micro_step": "integrate modalities simultaneously", "category": "Communication patterns"},
            {"micro_step": "add inter-vein lesions when needed", "category": "Procedural manipulation"},
            {"micro_step": "review chamber voltage pattern", "category": "Verification and safety"},
            {"micro_step": "confirm durable pulmonary vein isolation endpoint", "category": "Verification and safety"},
        ],
        "examples": "point-by-point lesion work, multi-screen guidance, inter-vein lesions, endpoint confirmation",
    },
    {
        "phase_group": "Post-care/extubation",
        "linked_metric": "POST CARE/EXTUBATION (Cath-Out to Pt-Out)",
        "start_sec": None,
        "end_sec": None,
        "micro_steps": [],
        "examples": "not visible in the transcript excerpt",
    },
]


def parse_transcript_entries(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^(\d{2}:\d{2}:\d{2})\n", text)

    rows: list[dict[str, object]] = []
    for index in range(1, len(parts), 2):
        timestamp = parts[index]
        body = parts[index + 1].strip().replace("\n", " ")
        hours, minutes, seconds = map(int, timestamp.split(":"))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        rows.append({"timestamp": timestamp, "sec": total_seconds, "text": body})

    return pd.DataFrame(rows)


def category_slug(category: str) -> str:
    return (
        category.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def build_transcript_micro_step_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for phase in PHASE_CODING:
        for item in phase["micro_steps"]:
            rows.append(
                {
                    "phase_group": phase["phase_group"],
                    "linked_metric": phase["linked_metric"],
                    "micro_step": item["micro_step"],
                    "primary_category": item["category"],
                }
            )

    return pd.DataFrame(rows)


def build_transcript_phase_summary(transcript_path: Path = TRANSCRIPT_PATH) -> pd.DataFrame:
    entries = parse_transcript_entries(transcript_path)
    micro_step_table = build_transcript_micro_step_table()

    rows: list[dict[str, object]] = []
    for phase in PHASE_CODING:
        start_sec = phase["start_sec"]
        end_sec = phase["end_sec"]

        if start_sec is not None and end_sec is not None:
            phase_entries = entries.loc[(entries["sec"] >= start_sec) & (entries["sec"] < end_sec)].copy()
            duration_sec = end_sec - start_sec
            timestamp_blocks = int(len(phase_entries))
            start_ts = seconds_to_timestamp(start_sec)
            end_ts = seconds_to_timestamp(end_sec)
        else:
            phase_entries = entries.iloc[0:0].copy()
            duration_sec = None
            timestamp_blocks = 0
            start_ts = None
            end_ts = None

        phase_micro_steps = [item["micro_step"] for item in phase["micro_steps"]]
        phase_table = micro_step_table.loc[micro_step_table["phase_group"] == phase["phase_group"]].copy()
        category_counts = phase_table["primary_category"].value_counts()

        row = {
            "phase_group": phase["phase_group"],
            "linked_metric": phase["linked_metric"],
            "transcript_start": start_ts,
            "transcript_end": end_ts,
            "duration_sec": duration_sec,
            "duration_min": round(duration_sec / 60, 2) if duration_sec is not None else None,
            "timestamp_blocks": timestamp_blocks,
            "coded_micro_steps": len(phase_micro_steps),
            "coded_unique_categories": int(phase_table["primary_category"].nunique()) if not phase_table.empty else 0,
            "examples": phase["examples"],
            "micro_step_list": " | ".join(phase_micro_steps),
        }

        for category in CATEGORY_ORDER:
            row[f"micro_steps_{category_slug(category)}"] = int(category_counts.get(category, 0))

        if duration_sec:
            row["timestamp_blocks_per_min"] = round(timestamp_blocks / (duration_sec / 60), 2)
            row["coded_micro_steps_per_min"] = round(len(phase_micro_steps) / (duration_sec / 60), 2)
        else:
            row["timestamp_blocks_per_min"] = None
            row["coded_micro_steps_per_min"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_transcript_phase_summary()
    micro_step_table = build_transcript_micro_step_table()
    output_path = OUTPUT_DIR / "transcript_phase_summary.csv"
    micro_step_output_path = OUTPUT_DIR / "transcript_micro_step_coding.csv"
    summary.to_csv(output_path, index=False)
    micro_step_table.to_csv(micro_step_output_path, index=False)

    print("Saved:")
    print("-", output_path)
    print("-", micro_step_output_path)
    print()
    print(summary[[
        "phase_group",
        "transcript_start",
        "transcript_end",
        "duration_min",
        "timestamp_blocks",
        "coded_micro_steps",
        "coded_unique_categories",
        "micro_steps_communication_patterns",
        "micro_steps_personnel_interactions",
        "micro_steps_equipment_handling",
        "micro_steps_room_setup",
        "micro_steps_procedural_manipulation",
        "micro_steps_imaging_and_mapping",
        "micro_steps_verification_and_safety",
        "micro_steps_handoff_or_waiting",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
