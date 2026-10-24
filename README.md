# multi-day-multi-metric-flags-wearable-devices

## Getting the data

The pipeline is built and evaluated on the PhysioNet dataset **"Wearable Physiological Signals under Acute Stress and Exercise Conditions"** (Empatica E4 wristband), covering 36 stress, 30 aerobic and 31 anaerobic participants, described by Hongn et al. (2025). The session recordings are gitignored, so each user downloads the raw data themselves.

**Where it lives.** The canonical source is PhysioNet, DOI [10.13026/he0v-tf17](https://doi.org/10.13026/he0v-tf17); the dataset is also described in a [Nature *Scientific Data* paper](https://www.nature.com/articles/s41597-025-04845-9). It is released under the **ODC-By** license, which requires attribution.

**Downloading.** Open the dataset's PhysioNet project page and use either option:

- **ZIP:** click the "Download the ZIP file" button in the project's Files section.
- **Command line (`wget`):** follow the pattern below, but copy the EXACT command — including the real `<project-slug>` and `<version>` — from the "Files" section of the dataset's PhysioNet page. Do not trust a guessed slug.

  ```bash
  wget -r -N -c -np https://physionet.org/files/<project-slug>/<version>/
  ```

**Where to put it.** After downloading, place the session recordings so the loader and `scripts/demo.py` can find them:

```
data/sessions/Wearable_Dataset_sessions/Wearable_Dataset/<SESSION>/<participant>/
```

where `<SESSION>` is one of `STRESS`, `AEROBIC`, or `ANAEROBIC` and `<participant>` is an id like `S07` or `f13` (split recordings use `_a`/`_b` suffixes, e.g. `f14_a`, `f14_b`). The reference files — `data_constraints.txt`, `Data_Dictionary.csv`, `Stress_Level_v1.csv`, `Stress_Level_v2.csv`, `subject-info.csv` — already live in `data/external/`.

**Citation.** PhysioNet DOI 10.13026/he0v-tf17; Hongn et al., *Scientific Data* (2025); ODC-By license (attribution required).

*Content paraphrased for compliance; verify the exact download command on the PhysioNet project page.*

## Quickstart

**Install** (Python 3.11+):

```bash
pip install -e .
```

**Module layout** (one pipeline step per module):

- `data/` — load the Empatica E4 CSVs, apply the dataset's cleaning rules, window the signals, and define the shared per-window schema.
- `flag/` — rest baseline, per-signal z-scores and combined score, persistence gate and severity tiers, the movement check, and plain-language explanations.
- `eval/` — phase labels and the self-report join, detection metrics, leave-one-subject-out calibration, and the end-to-end evaluation harness.
- `app/` — the per-participant timeline view (signals, combined score, tier bands, phase shading, flag explanations).
- `export/` — stretch add-on: FHIR / CMS care-update export of a Tier 2–3 flag.

**Run the demo:**

```bash
python scripts/demo.py              # full pipeline on the real PhysioNet data
python scripts/demo.py --synthetic  # deterministic synthetic cohort (no data needed)
```

The demo loads a small hand-picked cohort, builds the combined per-window table, runs the LOSO evaluation, and renders a strong-responder "hero" timeline plus a calm contrast. It writes these artifacts to `output/`:

- `timeline_<hero>.png`, `timeline_<calm>.png` — per-participant timelines.
- `false_alarms_vs_delay.png` — headline figure 1 (false alarms per rest hour vs detection delay).
- `tier_vs_self_report.png` — headline figure 2 (assigned tier vs self-reported stress).
- `metrics_by_method_cohort.csv` — the per-method, per-cohort numbers behind the figures.

The FHIR / CMS care-update export under `export/` is a stretch add-on, not part of the core run.

## Pitch

We flag sustained, multi-metric physiological change from wrist-worn wearable signals, rate it by severity, and explain each flag in plain terms. We build and test the method on the PhysioNet wearable dataset, where we know exactly when each stressor starts and how hard it is.

**Problem.** WHOOP judges each metric on its own, one night at a time. When resting HR, HRV and skin temperature all shift slightly together and stay shifted, there is no combined signal, and no wearable maker publishes how often its flags are right or wrong.

**What we build.** A pipeline that turns wrist signals into: (1) a per-person baseline from their rest period, (2) a combined deviation score across heart rate, HRV and skin temperature, (3) a flag only when that score persists, (4) a severity tier, with movement checked as a benign cause, (5) a plain-language explanation, and (6) measured false alarms, detection delay and tier accuracy against the protocol's known events and self-reported stress. The same pipeline runs on daily WHOOP summaries, with nights in place of minutes.

## Gaps we address

WHOOP judges each metric one night at a time; we close four flagging gaps using data it already collects.

| Gap | What WHOOP does today | How we address it |
| --- | --- | --- |
| Change over time | Health Monitor marks each night green, amber or red against a rolling baseline; multi-week trends are graphs the member must read, and the AI coach mentions them only when asked | **Persistence rule:** flag only when the combined score stays high for k of n nights |
| Metrics in combination | Each metric gets its own indicator; a reviewer noted it "stops short of interpreting these stats" (Wareable) | **Combined score** across metrics, so several small shifts together can trigger a flag |
| Known patterns | The illness-like pattern (higher skin temp and resting HR, lower HRV) is something the member watches for themselves (community thread) | Each flag **names its top contributing metrics** in real units |
| Accuracy unknown | No published false-alarm or detection numbers for Health Monitor | We report **sensitivity, false alarms per person-month and detection delay** for every variant |

### Why it matters

Combined flagging is feasible, but its weak point is false alarms, and that is the part we measure.

| | How it flags | What we learn from it |
| --- | --- | --- |
| WHOOP Health Monitor | Per metric, per night, against a rolling baseline | Our comparison baseline (the WHOOP-style rule) |
| Oura Symptom Radar | Combined deviation across metrics, 7 of 14 nights, "no / minor / major signs" | Combined, multi-night flagging can ship in a consumer product |
| TemPredict study (research behind Oura) | Per-person z-scores vs a 21-day baseline, random forest | 82% sensitivity but 63% specificity: many healthy days flagged |
| WHOOP developer API | Exposes no flags, only daily `read:recovery` and `read:sleep` summaries | We rebuild flags from those summaries |

- **Massachusetts context:** in the hackathon's CMS readmissions file (55 hospitals, Jul 2021–Jun 2024), 30-day readmission is about 20% for heart failure and COPD, the kind of sustained change a combined flag targets.
- **Why we avoid diagnostic claims:** FDA sent WHOOP a warning letter in July 2025 over a color-coded blood pressure feature. Our flags say "sustained change," not "you may be sick."

## Data

PhysioNet is our core dataset: real wrist recordings with timed stressors and self-reported stress, which give us ground truth that daily wearable data never has.

| Source | What it gives us | Role |
| --- | --- | --- |
| PhysioNet wearable dataset (Empatica E4 wristband) | Healthy adults aged 18–30: stress tasks (36 people), stationary-bike aerobic test (30), sprint test (31). Heart rate, beat-to-beat intervals, skin temperature (4 Hz), accelerometer (32 Hz), protocol tags, self-reported stress 1–10 before and after each phase | **Core:** build, tune and evaluate the flag |
| Teammate WHOOP account via OAuth | Daily `read:recovery` + `read:sleep` summaries, one person | Optional demo: same pipeline on nightly data |
| CMS readmissions, CDC PLACES short sleep (kit files) | Massachusetts readmission rates; share of adults sleeping < 7 h by county | Motivation slide only |

### Rules for using PhysioNet

- **WHOOP-equivalent signals only:** the score uses heart rate (HR.csv, 1 Hz), HRV (RMSSD from IBI.csv) and skin temperature (TEMP.csv, 4 Hz); the accelerometer (ACC.csv, 32 Hz, units of 1/64 g) is used only for the movement check. Electrodermal activity is left out because WHOOP doesn't measure it.
- **Time unit:** several stress tasks last only 30 seconds, so scores are computed every 10 seconds on rolling 30-second windows (HRV on 120-second windows: the E4 only logs beats it is confident about, about 15–30% of the time in the first session we ran).
- **Phase labels:** tags.csv holds button-press times; the first row of every signal file is the session start (UTC) and the second the sampling rate. Map tags to phases with the protocol table below (the dataset notebook's `graph_multiple` function shows the stress mapping).
- **Two cohorts:** S IDs ran protocol V1 and are all male; f IDs ran V2 and are all female except f18. Cohort, protocol and sex are therefore tangled; keep cohort as a factor and don't claim sex differences.
- **Encoding:** `data_constraints.txt` and `Data_Dictionary.csv` are Windows-1252, not UTF-8; open with `encoding="cp1252"`.
- **Citation:** DOI 10.13026/he0v-tf17 and Hongn et al. (2025), ODC Attribution license.

### Protocol timing (from the dataset notebook's diagrams)

| Session | Version | Phases in order (duration) | Quiet baseline? |
| --- | --- | --- | --- |
| Stress | V1 (S01–S18) | Baseline 3:00 → Stroop 2:00 → Rest 5:00 → TMCT 3:00 → Rest 5:00 → Real opinion 0:30 → Opposite opinion 0:30 → Subtract 0:30 | Yes, 3 min |
| Stress | V2 (f01–f18) | Baseline → TMCT 3:00 → Rest 10:00 → Real opinion 0:30 → Opposite opinion 0:30 → Rest 10:00 → Subtract 0:30 | Yes |
| Aerobic | V1 | Warm-up 3:00 → 60 to 110 rpm in 10 stages (2–3 min each) → Cool-down 4:00 → Rest 2:00 | No |
| Aerobic | V2 | Baseline 4:30 → Warm-up 2:15 → 70, 75, 80 rpm (1:30 each) → 85 rpm 11:15 → 90/95 rpm 4:30 → Cool-down 3:00 → Rest 2:00 | Yes, 4.5 min |
| Anaerobic | V1 | Warm-up 3:00 → 3 × (Sprint 0:30 + Cool-down 4:00) → Rest 2:00 | No |
| Anaerobic | V2 | Baseline 4:30 → Warm-up 4:30 → 4 × (Sprint 0:45 + Cool-down ~3:45) → Rest 2:00 | Yes, 4.5 min |

V1 exercise sessions have no quiet baseline, so for those participants we reuse the baseline from their stress session.

### Known data issues (`data_constraints.txt`)

| Participant | Session | Issue | What we do |
| --- | --- | --- | --- |
| S02 | Stress | Signals duplicated partway through | Truncate at the duplicate start rows given in the file; check HR and IBI by hand |
| f07 | Stress | Dock left on, covering heart and temperature sensors | Exclude (only EDA and accelerometer valid) |
| f14 | Stress | Connection lost; baseline in f14_a, rest in f14_b | Join the two files |
| S11, S16 | Aerobic, anaerobic | Connection lost; recording split into _a and _b | Join the two files |
| S03, S07, S06 | Aerobic, anaerobic | Stopped early | Keep the phases they completed |
| S12 | Aerobic | Did not do it | None |
| S01 | Anaerobic | IBI file empty | HR only, no HRV |
| f14–f18 | All | No demographics; no exercise sessions | Stress analysis only |

## Method

Six steps, each one a module: rest baseline → per-signal z-scores → combined score → persistence gate → severity tier with a movement check → explained flag. Persistence decides whether a flag fires; severity alone decides its tier. On PhysioNet each step runs every 10 seconds on 30-second windows; on WHOOP data the same code runs on nights.

**In plain language:** we learn each person's resting normal for every signal, score how unusual each moment is in their own terms, and add those scores so several small shifts count as one bigger signal. The total has to stay up for most of the last half-minute, and its size sets the warning level.

Each box is one module in the repo; no flag fires until a combined shift persists, and its tier sets how strongly it is worded.

1. **Rest baseline (per person, per signal).** Each participant's opening rest segment (3 minutes in the stress protocol, the pre-exercise rest in the others), minus its first minute while heart rate settles, summarized by median and MAD.
   - Because a few minutes is a small sample, each person's spread is shrunk toward the cohort's pooled estimate (empirical Bayes, as in DESeq2/limma).
2. **Signed z-scores per window.** Heart rate up and HRV (RMSSD) down both count as positive. Skin temperature counts in either direction, since wrist temperature tends to fall under acute stress but rises with fever.
3. **Combined score.** Two versions to compare:
   - Directional sum of the z-scores (simple, explainable).
   - Mahalanobis distance with a shrinkage covariance (Ledoit–Wolf), since a short baseline can't estimate correlations alone.
4. **Persistence gate.** A flag can fire only if the combined score is above τ₁ in 2 of the last 3 windows (about 30 seconds). The gate is the same for every tier; it only screens out momentary spikes.
5. **Severity tier.** Once the gate is passed, the tier depends only on how large the shift is: the median combined score over the last 3 windows, so one spike can't set the tier (table below). How long it has lasted does not raise the tier.
   - Hysteresis: a tier steps down only after 2 windows below its threshold, so a flag doesn't flicker.
   - Rule to test: one signal alone can't push a flag above Watch.
6. **Movement check.** When the accelerometer shows sustained movement, or rhythmic wrist motion (pedalling barely moves the wrist but shows up as a steady beat at the cadence), the flag drops one tier and is labeled "explained by movement." Exercise sessions should mostly end up here; stress tasks, done sitting still, should not.

| Tier | Severity rule (median score, last 3 windows) | Wording to the user |
| --- | --- | --- |
| 0 · None | Gate not passed, or score below τ₁ | Nothing shown |
| 1 · Watch | τ₁ ≤ score < τ₂ | "Some of your signals have shifted from your resting normal" |
| 2 · Notable | τ₂ ≤ score < τ₃ | "Several signals are clearly off your resting normal" |
| 3 · Strong | score ≥ τ₃ | "A large shift across several signals" |

Thresholds τ₁ < τ₂ < τ₃ are tuned with leave-one-subject-out cross-validation, not hand-picked: τ₁ from a false-alarm budget on rest segments (for example, ≤ 1 flag per hour of rest), τ₂ and τ₃ so that tiers track self-reported stress and the bike test's rising intensity stages. Labels stay neutral, with no medical color-coding, given FDA's objection to color-coded ranges.

Output per flag: tier, start time, seconds persisted, combined score, the top contributing signals in real units (e.g. "heart rate +18 bpm vs your rest"), and whether movement explains it.

```latex
z_{m,t} = s_m \cdot \frac{x_{m,t} - \mathrm{median}_m}{1.4826 \cdot \mathrm{MAD}_m}, \qquad D_t = \sum_m z_{m,t}
```

## Evaluation

PhysioNet's protocol tags tell us when each stressor starts and ends, rest segments give us known-quiet time, and self-reported stress gives a severity label. We compare our flag with a WHOOP-style single-signal rule, using leave-one-subject-out cross-validation, and report every metric per tier.

| Metric | Definition | Why it matters |
| --- | --- | --- |
| Sensitivity | Share of stress-task phases that get a flag at any tier, reported separately for long tasks (Stroop, TMCT) and 30-second tasks | Missed change |
| False alarms per hour of rest | Flags during baseline and rest, skipping the first 2 minutes after each task (recovery tail, reported separately) | Alert fatigue |
| Detection delay | Seconds from phase start to first flag | How quickly it reacts |
| Tier agreement | Rank correlation between tier and self-reported stress; tier rising with bike-test stage | Whether severity means anything |
| Movement attribution | Share of exercise flags labeled "explained by movement," and share of stress flags wrongly labeled so | Whether the benign-cause check works |

Comparisons to run:

- **Baselines:** single-signal out-of-range rule (WHOOP-style) vs directional sum vs Mahalanobis.
- **Tier rules:** thresholds τ₁–τ₃ and the persistence gate (2 of 3 vs 3 of 4 windows), swept against the false-alarm budget and self-reported stress.
- **Baseline estimate:** each person's raw rest segment vs the shrunk estimate.
- **Cohorts:** results for S and f participants separately, since their protocols differ.

Headline figures: (1) false alarms per hour of rest against detection delay, one line per method, with each tier's operating point marked; (2) tier against self-reported stress. Self-reported stress is recorded only before and after each phase, so tier agreement is judged per phase, not per window. TMCT is the most stressful phase in the self-reports (mean 5.1 of 10 in V1 and 6.4 in V2, against 3.4 and 3.3 at baseline), so it is the natural target for Strong.

### Stretch: exporting a flag

If time allows, a Tier 2 or 3 flag can be exported in standard formats; this is a demo add-on, not the core.

- **FHIR (spec):** a `Bundle` with daily `Observation`s per metric (candidate LOINC codes to verify: resting HR 40443-4, respiratory rate 9279-1, skin temp 39106-0) and one summary `Observation` for the flag.
- **Care update (CMS template):** fill Clinical Measures (baseline vs most recent) and Other relevant clinical data (the flag explanation); leave Assessment and Plan to a clinician.

## Repo and roles

One Python repo, one module per pipeline step, so four people can work in parallel against a shared per-minute table.

```
wearable-flag/
  data/
    physionet_load.py  # read E4 CSVs (row 1 = start time, row 2 = Hz), tags, stress levels
    clean.py           # apply data_constraints.txt
    windows.py         # 10-s steps: 30-s HR, skin temp, movement; 60-s RMSSD from IBI
    schema.py          # person_id, cohort, protocol, phase, t_min, hr, rmssd, skin_temp, acc_mag
    whoop_client.py    # optional demo: OAuth, read:recovery + read:sleep
  flag/
    baseline.py        # rest-segment median/MAD, shrinkage toward cohort
    score.py           # signed z, directional sum, Mahalanobis (Ledoit-Wolf)
    tiers.py           # persistence gate, severity tiers, hysteresis
    context.py         # movement check from accelerometer
    explain.py         # top contributing signals in real units, tier wording
  eval/
    labels.py          # phases from tags, rest vs task, self-reported stress
    metrics.py         # per-tier sensitivity, false alarms, delay, tier agreement
    calibrate.py       # tune tau1-tau3 with leave-one-subject-out
    run_eval.py        # sweeps + headline figures
  app/
    demo.py            # per-person timeline: signals, score, tier bands, phase shading
  export/              # stretch: fhir.py, care_update.py
  notebooks/
```

| Workstream | Owns | First deliverable |
| --- | --- | --- |
| Data | `data/` | Per-minute table for every clean participant, with phase labels |
| Scoring | `flag/baseline.py`, `score.py` | Combined score per person-minute |
| Tiers + explanations | `flag/tiers.py`, `context.py`, `explain.py` | Tier per minute with explanation and movement label |
| Evaluation + demo | `eval/`, `app/` | Per-tier false alarms vs the WHOOP-style rule; timeline view |

## Build order, decisions, limits

Get one flag end to end on simulated data first; everything after that is improvement.

1. Download PhysioNet, apply `data_constraints.txt`, agree on the per-minute schema.
2. Window the signals (HR, RMSSD, skin temp, movement) and label phases from the tags.
3. Rest baseline + directional sum + a single flag, end to end on one participant.
4. Evaluation table across all participants: WHOOP-style rule vs ours.
5. Severity tiers; tune τ₁–τ₃ with leave-one-subject-out.
6. Movement check, explanations, Mahalanobis and shrinkage; rerun the sweeps.
7. Timeline demo; optionally the same pipeline on a teammate's nightly WHOOP data.
8. Stretch: FHIR / care-update export of a Tier 2–3 flag.
9. Slides: problem, method, tiers, headline figures, limits.

### Open decisions

- V1 exercise baseline: reuse each person's stress-session baseline (same wrist, different session), or use the first minute of warm-up?
- Main ground truth for tiers: self-reported stress, protocol phase, or both?
- Pitch framing: an acute-change flag (what PhysioNet tests) or a general sustained-change flag (what the method is meant for)?
- Keep the WHOOP nightly-data demo in scope, and whose account?

### Limitations to state on the slide

- **Healthy 18–30-year-olds in short lab sessions: results show the method on acute, induced change, not illness or multi-day drift.**
- Wrist HRV is unreliable during heavy movement, so exercise-phase HRV is noisy.
- Rest baselines are only minutes long, so each person's normal is uncertain (hence shrinkage); V1 exercise sessions have no rest baseline at all.
- Self-reported stress is coarse (one rating per phase), subjective, and not zero at baseline (mean about 3.3 of 10, likely anticipation).
- Cohort, protocol version and sex move together (V1 all male, V2 almost all female), so differences between cohorts can't be pinned on any one of them.
- Applying the method to WHOOP's nightly data is a transfer we argue for, not one we test.
- Not a medical device; a flag at any tier means "sustained change," not a diagnosis.
