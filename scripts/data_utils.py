from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "MSE433_M4_Data.xlsx"


def load_case_data(path: Path = DATA_PATH) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)

    hdr1 = raw.iloc[2].fillna("")
    hdr2 = raw.iloc[3].fillna("")

    headers: list[str] = []
    for top, bottom in zip(hdr1, hdr2):
        top = str(top).strip()
        bottom = str(bottom).strip()
        if top == "" and bottom == "":
            headers.append("")
        elif bottom != "":
            headers.append(f"{top} {bottom}".strip())
        else:
            headers.append(top)

    df = raw.iloc[4:].copy()
    df.columns = headers
    df = df.loc[:, df.columns != ""].copy()

    for column in df.columns:
        if column not in ["DATE", "PHYSICIAN", "PT OUT TIME", "Note"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["complex_case"] = df["Note"].notna() & (df["Note"].astype(str).str.strip() != "")
    df["ablation_non_energy"] = df["ABL DURATION (Abl Start-End)"] - df["ABL TIME (Min)"]
    df["post_ablation_la_time"] = df["LA DWELL TIME (Abl Start-Cath-Out)"] - df["ABL DURATION (Abl Start-End)"]
    df["non_procedural_room_time"] = df["PT IN-OUT (Min)"] - df["SKIN-SKIN (Access to Cath-Out)"]
    return df
