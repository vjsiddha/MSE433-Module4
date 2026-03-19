from __future__ import annotations

from pathlib import Path
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


NOTEBOOK_PATH = ROOT / "notebooks" / "same_doctor_same_abl_analysis.ipynb"


def build_notebook() -> None:
    nb = nbf.v4.new_notebook()
    cells: list = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# Same-Doctor, Same-#ABL Analysis

This notebook replaces the top-10% baseline as the **main analysis**.

The core question is:

**If we hold the doctor and the number of ablation lesions (`#ABL`) constant, how much variation is still left, where does it appear, and does that pattern fit what we know from the transcript?**

This is a stronger design for your case because it separates:

- **between-group effects**: different doctors and different lesion counts
- **within-group variation**: cases that should look similar on observable workload but still take different amounts of time

That remaining within-group variation is the part most likely to reflect hidden operational differences such as communication, verification, coordination, setup, and short waits between steps.
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Hypothesis

We will test three claims:

1. `doctor` and `#ABL` explain some variation in procedure time, but not all of it.
2. Even within the same doctor and the same `#ABL`, there is still meaningful variation across cases.
3. The remaining variation should be concentrated in coordination-heavy phases that line up with the transcript, especially prep, transseptal work, and non-energy parts of ablation.

If this pattern appears, then the transcript can be used as an interpretation layer for the hidden operational variability.
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Important Assumption

The transcript is **not** from one of the timed cases in the dataset. It is a representative description of the standard procedure.

That means we use the transcript as a **process map**, not as direct evidence for any specific case. The timed dataset tells us **where variation exists**. The transcript helps us describe **what kinds of coded micro-steps are embedded inside those macro stages and which hidden-variability categories those micro-steps belong to**.
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Coding Approach

To make the representative transcript usable in analysis, we **coded** it at the micro-step level.

- `Micro-step`: a discrete task or event needed for the procedure to move forward
- `Primary category`: the main hidden-variability category assigned to that micro-step

The primary categories used in this notebook are:

- `Communication patterns`
- `Personnel interactions`
- `Equipment handling`
- `Room setup`
- `Procedural manipulation`
- `Imaging and mapping`
- `Verification and safety`
- `Handoff or waiting`

So these labels are not raw fields from the timing dataset. They are the result of a structured LLM-assisted coding of the representative transcript.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

ROOT = Path.cwd().resolve().parents[0] if Path.cwd().name == "notebooks" else Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.data_utils import (
    DATA_PATH,
    load_case_data,
)
from scripts.doctor_abl_variation_analysis import (
    build_case_time_ancova,
    build_doctor_abl_cell_summary,
    build_phase_effect_anova,
    build_within_cell_variation_summary,
)
from scripts.transcript_macro_analysis import (
    build_transcript_micro_step_table,
    build_transcript_phase_summary,
)

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
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Step 1. Load the data and define the comparison set

We use only **standard cases** for the main analysis. Cases with notes such as `CTI`, `BOX`, `PST BOX`, `SVC`, and `TROUBLESHOOT` are excluded because they represent visibly different work.

We then focus on doctor-by-`#ABL` cells with at least **2 cases**, because those are the cells where within-group variation can actually be measured.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """all_cases = load_case_data(DATA_PATH)
standard_cases = all_cases.loc[~all_cases["complex_case"]].copy()

cell_counts = (
    standard_cases.groupby(["PHYSICIAN", ABL_COUNT])
    .size()
    .reset_index(name="n_cases")
    .sort_values(["PHYSICIAN", ABL_COUNT])
)

eligible_cells = cell_counts.loc[cell_counts["n_cases"] >= 2].copy()
cases_in_eligible_cells = int(eligible_cells["n_cases"].sum())

display(cell_counts)
print(f"Total cases in workbook: {len(all_cases)}")
print(f"Standard cases used in this analysis: {len(standard_cases)}")
print(f"Doctor-#ABL cells with at least 2 cases: {len(eligible_cells)}")
print(f"Standard cases covered by repeated doctor-#ABL cells: {cases_in_eligible_cells}")
print(f"Coverage of standard cases: {cases_in_eligible_cells / len(standard_cases):.1%}")
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """### Interpretation of Step 1

This is enough structure to support a same-doctor, same-`#ABL` analysis.

The key point is coverage: if most standard cases fall into repeated doctor-`#ABL` cells, then we can study within-cell variation without throwing away the dataset. That makes the grouped design a practical main analysis rather than a niche subset check.
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Step 2. Use ANOVA/ANCOVA to see which effects matter overall

Before looking inside the cells, we first ask:

- Does `doctor` matter?
- Does `#ABL` matter?
- Where do those effects show up across the procedure?

This does **not** replace the grouped analysis. It only tells us which observable effects are important at the overall dataset level.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """phase_effects = build_phase_effect_anova(standard_cases)
case_time_ancova = build_case_time_ancova(standard_cases)

display(case_time_ancova)

phase_view = (
    phase_effects[["outcome", "effect", "F", "p_value", "eta_sq_total"]]
    .sort_values(["outcome", "effect"])
    .reset_index(drop=True)
)
display(phase_view)
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            '''ancova = case_time_ancova.set_index("effect")

effect_text = f"""
### Interpretation of Step 2

The overall effect tests show that **doctor matters** for case time even after controlling for `#ABL` and `ABL TIME`.

- In the case-time ANCOVA, `doctor` is significant with **p = {ancova.loc['C(PHYSICIAN)', 'p_value']:.6f}**.
- In that same model, `#ABL` is weaker once `ABL TIME` is also included (**p = {ancova.loc['Q("#ABL")', 'p_value']:.6f}**).
- `ABL TIME` itself is not significant in that case-time model (**p = {ancova.loc['Q("ABL TIME (Min)")', 'p_value']:.6f}**).

At the phase level, the pattern is more informative:

- `doctor` matters for **case time, skin-skin, prep/intubation, TSP, PRE-MAP, ablation duration, and post-care**.
- `doctor` does **not** matter much for **access**.
- `#ABL` matters most for **ablation duration**, and it also shows up in **case time, PRE-MAP, and post-care**.

So ANOVA/ANCOVA gives a useful answer to “which effects matter where?”, but it does not answer the main operational question by itself. For that, we need to look at the variation that remains **inside** doctor-by-`#ABL` cells.
"""

display(Markdown(effect_text))
'''
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Step 3. Make same-doctor, same-#ABL the main comparison

Now we switch to the main design.

For each doctor and each `#ABL` value, we look at all repeated cases in that cell and measure how much they still vary. This tells us how unstable the process is **after holding doctor and lesion count fixed**.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """cell_summary = build_doctor_abl_cell_summary(standard_cases, min_cell_size=2)
within_cell_summary = build_within_cell_variation_summary(cell_summary)

display(cell_summary[[
    "physician",
    "num_abl",
    "cell_n",
    "case_time_cath_in_out_mean",
    "case_time_cath_in_out_sd",
    "case_time_cath_in_out_range",
    "tsp_min_sd",
    "pre_map_min_sd",
    "abl_duration_abl_start_end_sd",
    "abl_time_min_sd",
    "post_care_extubation_cath_out_to_pt_out_sd",
]].sort_values(["physician", "num_abl"]))

display(within_cell_summary)
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            '''w = within_cell_summary.set_index("phase")

grouped_text = f"""
### Interpretation of Step 3

This is the strongest evidence in the notebook.

Even after fixing **doctor** and **`#ABL`**, the process is still quite variable:

- Mean within-cell SD for **case time** = **{w.loc['CASE TIME (Cath In-Out)', 'mean_within_cell_sd']:.2f} min**
- Mean within-cell SD for **skin-skin** = **{w.loc['SKIN-SKIN (Access to Cath-Out)', 'mean_within_cell_sd']:.2f} min**
- Mean within-cell SD for **prep/intubation** = **{w.loc['PT PREP/INTUBATION Pt-In-Access', 'mean_within_cell_sd']:.2f} min**
- Mean within-cell SD for **ablation duration** = **{w.loc['ABL DURATION (Abl Start-End)', 'mean_within_cell_sd']:.2f} min**
- Mean within-cell SD for **TSP** = **{w.loc['TSP (Min)', 'mean_within_cell_sd']:.2f} min**
- Mean within-cell SD for **ABL TIME** = **{w.loc['ABL TIME (Min)', 'mean_within_cell_sd']:.2f} min**

The contrast is the key result:

- `ABL TIME` is almost fixed inside these cells.
- The larger variation shows up in the **surrounding process**, especially skin-skin, prep, ablation duration, and TSP.

That means the residual variation is not mainly “how much ablation was delivered.” It is much more about **how the case flowed**.
"""

display(Markdown(grouped_text))
'''
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Step 4. Look at the hidden-time proxies directly

To make the interpretation cleaner, we add three derived measures:

- `ablation_non_energy = ABL DURATION - ABL TIME`
- `post_ablation_la_time = LA DWELL TIME - ABL DURATION`
- `non_procedural_room_time = PT IN-OUT - SKIN-SKIN`

These help separate active treatment from the time around treatment.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """extended = standard_cases.copy()
extended["ablation_non_energy"] = extended[ABL_DURATION] - extended[ABL_TIME]
extended["post_ablation_la_time"] = extended["LA DWELL TIME (Abl Start-Cath-Out)"] - extended[ABL_DURATION]
extended["non_procedural_room_time"] = extended["PT IN-OUT (Min)"] - extended[SKIN_SKIN]

metrics = [
    CASE_TIME,
    SKIN_SKIN,
    PT_PREP,
    ACCESS,
    TSP,
    PRE_MAP,
    ABL_DURATION,
    ABL_TIME,
    "ablation_non_energy",
    "post_ablation_la_time",
    POST_CARE,
    "non_procedural_room_time",
]

rows = []
for (physician, abl_count), group in extended.groupby(["PHYSICIAN", ABL_COUNT]):
    if len(group) < 2:
        continue
    row = {"physician": physician, "num_abl": abl_count, "cell_n": len(group)}
    for metric in metrics:
        row[f"{metric}_sd"] = group[metric].dropna().std(ddof=1)
    rows.append(row)

extended_cell = pd.DataFrame(rows)
hidden_variation = pd.DataFrame(
    {
        "metric": metrics,
        "mean_within_cell_sd": [extended_cell[f"{m}_sd"].mean() for m in metrics],
        "median_within_cell_sd": [extended_cell[f"{m}_sd"].median() for m in metrics],
    }
).sort_values("mean_within_cell_sd", ascending=False)

display(hidden_variation.round(2))
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            '''hv = hidden_variation.set_index("metric")

hidden_text = f"""
### Interpretation of Step 4

The hidden-time proxies sharpen the story:

- Mean within-cell SD for **case time** = **{hv.loc['CASE TIME (Cath In-Out)', 'mean_within_cell_sd']:.2f} min**
- Mean within-cell SD for **skin-skin** = **{hv.loc['SKIN-SKIN (Access to Cath-Out)', 'mean_within_cell_sd']:.2f} min**
- Mean within-cell SD for **non-procedural room time** = **{hv.loc['non_procedural_room_time', 'mean_within_cell_sd']:.2f} min**
- Mean within-cell SD for **ablation non-energy time** = **{hv.loc['ablation_non_energy', 'mean_within_cell_sd']:.2f} min**
- Mean within-cell SD for **post-ablation LA time** = **{hv.loc['post_ablation_la_time', 'mean_within_cell_sd']:.2f} min**
- Mean within-cell SD for **ABL TIME** = **{hv.loc['ABL TIME (Min)', 'mean_within_cell_sd']:.2f} min**

The important contrast is that **ablation non-energy time varies a lot, while ABL TIME barely varies**. That is exactly the pattern we would expect if hidden variation comes from repositioning, confirmation, communication, setup, and short delays between actions rather than from energy delivery itself.
"""

display(Markdown(hidden_text))
'''
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Step 5. Connect the variation pattern to the transcript

Now that the timestamped transcript is available, we can code each macro phase directly.

For each visible phase, we code the transcript to record:

- transcript start and end time
- elapsed transcript duration
- number of timestamp blocks
- coded micro-steps
- coded category counts for the hidden-variability taxonomy

This gives us a transcript-side complexity summary that can be compared directly to the data-side residual variation.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
"""transcript_summary = build_transcript_phase_summary()

transcript_compare = transcript_summary.merge(
    hidden_variation[["metric", "mean_within_cell_sd", "median_within_cell_sd"]],
    left_on="linked_metric",
    right_on="metric",
    how="left",
).drop(columns=["metric"])

display(transcript_compare[[
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
    "mean_within_cell_sd",
    "examples",
]])
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
"""transcript_step_map = build_transcript_micro_step_table()

display(transcript_step_map)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """### Interpretation of the step map

This table shows the actual **micro-step to macro-step mapping** used in the notebook, along with the primary hidden-variability category assigned to each micro-step.

That matters because it makes the transcript contribution explicit:

- `Prep/intubation` includes communication-heavy and verification-heavy micro-steps.
- `Access/setup` is dominated by equipment handling, room setup, and safety checks.
- `TSP` is dominated by procedural manipulation, imaging, and verification.
- `Pre-map` is dominated by imaging/mapping plus personnel interaction.
- `Ablation/non-energy work` mixes procedural manipulation, imaging, and verification around the lesion-delivery sequence.

So when we later compare transcript complexity to measured variability, we are doing it with an explicit coded micro-step map and category taxonomy rather than a vague description.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """phase_variability_for_categories = transcript_compare.loc[
    transcript_compare["duration_min"].notna(),
    ["phase_group", "mean_within_cell_sd"],
].copy()

category_summary = (
    transcript_step_map.merge(
        phase_variability_for_categories,
        on="phase_group",
        how="left",
    )
    .groupby("primary_category")
    .agg(
        coded_micro_steps=("micro_step", "count"),
        phases_covered=("phase_group", "nunique"),
        avg_phase_variability=("mean_within_cell_sd", "mean"),
        weighted_variability_exposure=("mean_within_cell_sd", "sum"),
    )
    .sort_values("weighted_variability_exposure", ascending=False)
    .reset_index()
)

display(category_summary.round(2))
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """### Interpretation of the category analysis

This is the first place where the new coding scheme becomes analytically useful.

The category table should be read in two ways:

- `coded_micro_steps` tells us how common that category is in the representative procedure
- `avg_phase_variability` and `weighted_variability_exposure` tell us how much that category tends to sit inside macro phases that remain variable in the timed data

So this is not just a transcript count. It is a bridge between the transcript coding and the measured variability.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            '''observed_map = transcript_compare.loc[transcript_compare["duration_min"].notna()].copy()

transcript_text = f"""
### Interpretation of Step 5

The transcript coding fits the data pattern reasonably well.

- **Prep/intubation** lasts **{observed_map.loc[observed_map['phase_group'] == 'Prep/intubation', 'duration_min'].iloc[0]:.2f} transcript minutes** and contains **{int(observed_map.loc[observed_map['phase_group'] == 'Prep/intubation', 'coded_micro_steps'].iloc[0])} coded micro-steps**. Its coded mix is communication-heavy and verification-heavy, and in the timed data it still shows substantial within-cell variation (**{observed_map.loc[observed_map['phase_group'] == 'Prep/intubation', 'mean_within_cell_sd'].iloc[0]:.2f} min**).
- **TSP** is the densest transcript stage: **{observed_map.loc[observed_map['phase_group'] == 'TSP', 'duration_min'].iloc[0]:.2f} minutes**, **{int(observed_map.loc[observed_map['phase_group'] == 'TSP', 'timestamp_blocks'].iloc[0])} timestamp blocks**, and **{int(observed_map.loc[observed_map['phase_group'] == 'TSP', 'coded_micro_steps'].iloc[0])} coded micro-steps**. It is dominated by procedural manipulation, imaging/mapping, and verification/safety, and it also shows meaningful within-cell variation in the data (**{observed_map.loc[observed_map['phase_group'] == 'TSP', 'mean_within_cell_sd'].iloc[0]:.2f} min**).
- **Ablation/non-energy work** contains **{int(observed_map.loc[observed_map['phase_group'] == 'Ablation/non-energy work', 'coded_micro_steps'].iloc[0])} coded micro-steps** and remains highly variable in the data (**{observed_map.loc[observed_map['phase_group'] == 'Ablation/non-energy work', 'mean_within_cell_sd'].iloc[0]:.2f} min**). Its coded categories center on procedural manipulation, imaging/mapping, and verification, which fits the idea that the time around energy delivery is operationally dense.
- **Pre-map** shows real transcript complexity (**{int(observed_map.loc[observed_map['phase_group'] == 'Pre-map', 'coded_micro_steps'].iloc[0])} coded micro-steps**) but relatively low residual variation (**{observed_map.loc[observed_map['phase_group'] == 'Pre-map', 'mean_within_cell_sd'].iloc[0]:.2f} min**). That is a useful nuance: a phase can be complex but still standardized.
- **Access/setup** has many coded micro-steps, especially equipment handling and room setup, but the timed `ACCESS` column is comparatively stable (**{observed_map.loc[observed_map['phase_group'] == 'Access/setup', 'mean_within_cell_sd'].iloc[0]:.2f} min**). That suggests some setup burden is either routinized or absorbed into neighboring timing windows rather than showing up as large residual spread in the access column alone.

This is a good fit to the theory:

The phases with the clearest residual variation after controlling for doctor and `#ABL` are also the phases where the transcript shows dense clusters of coded micro-steps in categories like communication, procedural manipulation, imaging/mapping, and verification/safety. That does not prove second-by-second causality, but it gives a much stronger operational explanation for the remaining variance.
"""

display(Markdown(transcript_text))
'''
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            '''cat = category_summary.set_index("primary_category")

category_text = f"""
### What the category results suggest

- **Verification and safety** has the largest weighted variability exposure (**{cat.loc['Verification and safety', 'weighted_variability_exposure']:.2f}**) and one of the highest total coded counts (**{int(cat.loc['Verification and safety', 'coded_micro_steps'])} micro-steps**). That fits the idea that confirmation and safety checks are embedded in the stages where the process still varies.
- **Procedural manipulation** is similarly high (**{cat.loc['Procedural manipulation', 'weighted_variability_exposure']:.2f}**), which makes sense because TSP and ablation non-energy work both require repeated positioning, advancement, and site-to-site movement.
- **Imaging and mapping** is also prominent (**{cat.loc['Imaging and mapping', 'weighted_variability_exposure']:.2f}**), which supports the idea that visualization and interpretation work are tightly linked to variable procedural flow.
- **Communication patterns** has a smaller raw count (**{int(cat.loc['Communication patterns', 'coded_micro_steps'])} micro-steps**) but the **highest average phase variability** (**{cat.loc['Communication patterns', 'avg_phase_variability']:.2f} min**). That means communication-heavy micro-steps are concentrated in some of the most variable phases even if they are not the most numerous overall.
- **Equipment handling** is common (**{int(cat.loc['Equipment handling', 'coded_micro_steps'])} micro-steps**) but less strongly tied to high-variability phases than the categories above. That matches the earlier phase result that access/setup is complex but relatively stable.
- **Personnel interactions** has the lowest average variability exposure (**{cat.loc['Personnel interactions', 'avg_phase_variability']:.2f} min**), suggesting that not all social coordination creates large timing variation when the task is routinized.

So the strongest explanatory categories for variability are not just “more work” in general. They are the categories that combine repeated manipulation, imaging, and safety confirmation inside already sensitive phases like TSP and the non-energy portion of ablation.
"""

display(Markdown(category_text))
'''
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
"""complexity_view = observed_map[[
    "phase_group",
    "coded_micro_steps",
    "coded_unique_categories",
    "timestamp_blocks",
    "mean_within_cell_sd",
]].copy()

complexity_view["variation_rank"] = complexity_view["mean_within_cell_sd"].rank(ascending=False, method="min")
complexity_view["micro_step_rank"] = complexity_view["coded_micro_steps"].rank(ascending=False, method="min")
complexity_view["category_diversity_rank"] = complexity_view["coded_unique_categories"].rank(ascending=False, method="min")

display(complexity_view.sort_values("variation_rank"))
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """### Interpretation of the complexity-versus-variability comparison

This comparison gives a useful but nuanced result.

- The relationship is **not perfectly linear**.
- `TSP`, `Prep/intubation`, and `Ablation/non-energy work` support the theory well: they contain many coded micro-steps across multiple hidden-variability categories, and they also show meaningful residual variation.
- `Pre-map` is short and comparatively stable, which also makes sense.
- `Access/setup` is the main exception: it looks complex in the transcript, but the timed `ACCESS` metric itself is relatively stable. That suggests some setup work is either highly standardized or absorbed into neighboring timing windows rather than appearing as variability in the access column alone.

So the transcript-coding idea is a **solid start**, but it should be interpreted as a structured explanation of likely variability sources, not as a strict rule that “more coded micro-steps or more category diversity always means more timing variation.”
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Final conclusion

For this case, **same-doctor, same-`#ABL` should be the main analysis**.

Why:

- ANOVA/ANCOVA tells us that `doctor` and `#ABL` do matter, but they do not explain everything.
- The grouped analysis shows that even after holding those effects fixed, there is still large variation in case time and skin-skin time.
- That remaining variation is concentrated in prep, TSP, and non-energy ablation time, not in `ABL TIME`.
- The transcript-informed coding fits that pattern reasonably well: the phases with the densest micro-step structure and the richest mix of hidden-variability categories are often the same phases with the strongest residual variation, although the relationship is not perfectly one-to-one.

So the recommended storyline for the report is:

1. Use ANOVA/ANCOVA to establish which observable effects matter overall.
2. Use same-doctor, same-`#ABL` cells as the main analysis of residual variation.
3. Use the transcript to interpret that residual variation as hidden operational variability, especially around communication, confirmation, navigation, and setup.
"""
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13",
        },
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


def main() -> None:
    build_notebook()
    print(f"Saved notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
