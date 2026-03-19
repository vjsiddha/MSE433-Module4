# MSE433-Module4

This project analyzes variability in AF ablation procedure times using a same-doctor, same-`#ABL` design. The main idea is to hold visible workload as constant as possible, measure how much timing variation still remains, and then use a representative transcript of the procedure to explain where that residual variability is most likely coming from.

## Project Goal

We are trying to identify hidden sources of variability in the procedure that are not directly captured by the macro timing columns alone. Examples include communication patterns, personnel interactions, equipment handling, verification work, handoff friction, and waiting.

The main question is:

When the same doctor performs cases with the same lesion count, where does the remaining time variation live, and can the transcript help explain it?

## Main Analysis Approach

The project uses three layers of analysis:

1. Overall effect testing

We first test whether observable factors such as `doctor`, `#ABL`, and `ABL TIME` explain case duration at all. This is done with ANOVA and ANCOVA.

2. Same-doctor, same-`#ABL` comparison

This is the main analysis. Cases are grouped by doctor and lesion count so that we can look at within-group variation after holding those two factors fixed. This gives a cleaner view of hidden process variation.

3. Transcript-informed coding

We use a representative timestamped procedure transcript as a process map. The transcript is not from one of the surgeries in the dataset, so it is not used for direct timing matching. Instead, we code its micro-steps into hidden-variability categories and map them to the macro phases in the timing data.

## Transcript Coding Approach

Each timestamped action in the representative transcript was coded as a `micro-step` and assigned to a primary hidden-variability category. The coded categories are:

- `Communication patterns`
- `Personnel interactions`
- `Equipment handling`
- `Room setup`
- `Procedural manipulation`
- `Imaging and mapping`
- `Verification and safety`
- `Handoff or waiting`

These coded micro-steps were then mapped into the macro phases used in the case dataset:

- `PT PREP/INTUBATION`
- `ACCESS`
- `TSP`
- `PRE-MAP`
- `ABL DURATION / ABL TIME`
- `POST CARE / EXTUBATION`

This gives an interpretation layer for the timing results. It does not prove exact causal time allocation for any individual case, but it helps explain why some phases are more variable than others.

## Overall Process

The analysis workflow is:

1. Load and clean the case workbook.
2. Restrict to standard cases used for comparison.
3. Run ANOVA/ANCOVA to test whether `doctor`, `#ABL`, and `ABL TIME` explain overall case time.
4. Group cases by same doctor and same `#ABL`.
5. Measure within-cell variation for each macro phase.
6. Code the representative transcript into micro-steps and hidden-variability categories.
7. Map transcript-coded micro-steps to macro phases.
8. Compare transcript phase complexity to observed within-cell variability.

## Main Findings

### 1. Doctor matters overall

In the case-time ANCOVA, `doctor` remains statistically significant even after controlling for `#ABL` and `ABL TIME`:

- `C(PHYSICIAN)` p-value: `0.000319`

By contrast:

- `#ABL` p-value: `0.097038`
- `ABL TIME` p-value: `0.296102`

This suggests provider-level differences remain important even after accounting for visible workload.

### 2. Large residual variation remains even within same-doctor, same-`#ABL` groups

Across repeated doctor-`#ABL` cells, there is still substantial within-group variation:

- `CASE TIME` mean within-cell SD: `10.11` min
- `SKIN-SKIN` mean within-cell SD: `9.96` min
- `PT PREP/INTUBATION` mean within-cell SD: `4.19` min
- `ABL DURATION` mean within-cell SD: `4.14` min
- `TSP` mean within-cell SD: `2.99` min
- `POST CARE / EXTUBATION` mean within-cell SD: `2.95` min
- `ACCESS` mean within-cell SD: `1.71` min
- `PRE-MAP` mean within-cell SD: `0.60` min
- `ABL TIME` mean within-cell SD: `0.13` min

The most important pattern is that `ABL TIME` barely varies once doctor and lesion count are held fixed, while surrounding phases still vary a lot. That implies the main variability is not the active pulse-on treatment itself, but the process wrapped around it.

### 3. The transcript supports where that hidden variation likely lives

The coded transcript shows that the densest macro phases are:

- `TSP`: `20` coded micro-steps
- `ACCESS/setup`: `18` coded micro-steps
- `Prep/intubation`: `11` coded micro-steps
- `Ablation/non-energy work`: `9` coded micro-steps
- `Pre-map`: `7` coded micro-steps

Category totals across the full representative transcript are:

- `Verification and safety`: `14`
- `Procedural manipulation`: `14`
- `Imaging and mapping`: `12`
- `Equipment handling`: `11`
- `Communication patterns`: `4`
- `Handoff or waiting`: `4`
- `Room setup`: `3`
- `Personnel interactions`: `3`

This aligns reasonably well with the timing results:

- `TSP` is both transcript-dense and meaningfully variable.
- `Prep/intubation` also has substantial variability and contains multiple setup, communication, and verification micro-steps.
- `ABL DURATION` is variable even though `ABL TIME` is not, which supports the idea that repositioning, imaging, verification, and non-energy procedural work are important.
- `ACCESS` looks complex in the transcript, but it is less variable in the timed data, which suggests some of that work is routinized or absorbed into nearby phases.

## Interpretation

The main interpretation is:

Once `doctor` and `#ABL` are held fixed, a large amount of procedural variation still remains, and that remaining variation is concentrated in macro phases that contain many micro-steps tied to manipulation, imaging, verification, setup, and coordination.

In other words, the data supports a hidden-variability story. The largest differences across otherwise similar cases do not appear to come from the active ablation energy time itself. They appear to come from the surrounding process needed to prepare, position, verify, coordinate, and complete the procedure.

This is exactly where the transcript is useful. It helps explain what is operationally happening inside the macro phases that remain variable in the timing data.

## Repository Contents

- [same_doctor_same_abl_analysis.ipynb](/Users/benfogerty/Desktop/MSCI433Case4/MSE433-Module4/notebooks/same_doctor_same_abl_analysis.ipynb): main notebook with analysis, results, and interpretation
- [data_utils.py](/Users/benfogerty/Desktop/MSCI433Case4/MSE433-Module4/scripts/data_utils.py): shared data loading and cleaning helpers
- [doctor_abl_variation_analysis.py](/Users/benfogerty/Desktop/MSCI433Case4/MSE433-Module4/scripts/doctor_abl_variation_analysis.py): ANOVA/ANCOVA and same-doctor same-`#ABL` variation analysis
- [transcript_macro_analysis.py](/Users/benfogerty/Desktop/MSCI433Case4/MSE433-Module4/scripts/transcript_macro_analysis.py): transcript coding and macro-phase mapping
- [build_same_doctor_same_abl_notebook.py](/Users/benfogerty/Desktop/MSCI433Case4/MSE433-Module4/scripts/build_same_doctor_same_abl_notebook.py): notebook generation script

Key generated outputs:

- [case_time_ancova.csv](/Users/benfogerty/Desktop/MSCI433Case4/MSE433-Module4/outputs/case_time_ancova.csv)
- [phase_effect_anova.csv](/Users/benfogerty/Desktop/MSCI433Case4/MSE433-Module4/outputs/phase_effect_anova.csv)
- [doctor_abl_cell_summary.csv](/Users/benfogerty/Desktop/MSCI433Case4/MSE433-Module4/outputs/doctor_abl_cell_summary.csv)
- [within_cell_phase_variation.csv](/Users/benfogerty/Desktop/MSCI433Case4/MSE433-Module4/outputs/within_cell_phase_variation.csv)
- [transcript_micro_step_coding.csv](/Users/benfogerty/Desktop/MSCI433Case4/MSE433-Module4/outputs/transcript_micro_step_coding.csv)
- [transcript_phase_summary.csv](/Users/benfogerty/Desktop/MSCI433Case4/MSE433-Module4/outputs/transcript_phase_summary.csv)

## How To Reproduce

Run the scripts in this order from the repository root:

```bash
./.venv/bin/python scripts/transcript_macro_analysis.py
./.venv/bin/python scripts/doctor_abl_variation_analysis.py
./.venv/bin/python scripts/build_same_doctor_same_abl_notebook.py
./.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/same_doctor_same_abl_analysis.ipynb
```

## Important Limitation

The transcript is a representative process description, not one of the timed cases in the dataset. That means the transcript is used to interpret the structure of the work inside each macro phase, not to assign exact transcript seconds to exact case-level delays.
