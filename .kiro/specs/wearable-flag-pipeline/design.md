# Design Document

## Overview

The Wearable Flag Pipeline turns raw Empatica E4 wrist-signal recordings from the PhysioNet dataset into explained, severity-tiered flags of sustained, multi-metric physiological change. It learns each person's resting normal, scores how unusual each 10-second window is in that person's own terms, combines heart rate, heart-rate variability (RMSSD) and skin temperature into a single deviation score, fires a flag only when that score persists, rates the flag by size, checks movement as a benign cause, explains the flag in plain units, and measures detection performance against the protocol's known events and self-reported stress.

The system is a **simulation on PhysioNet session data**. Every window on PhysioNet stands in for a "night" in the nightly-wearable framing; that transfer is argued conceptually but never implemented or tested here. There is no live WHOOP data and no OAuth client. The single-signal `WHOOP_Rule` is implemented only as an evaluation comparison baseline.

The system is explicitly **not a medical device**. Flags describe "sustained change," never a diagnosis, and severity uses neutral tier labels rather than color-coded ranges.

### Design goals and non-goals

| Goal | Reflected in |
| --- | --- |
| Deterministic, per-module pipeline that four people can build in parallel against a shared table | Architecture, `Per_Window_Table` schema |
| Measured accuracy (sensitivity, false alarms, delay, tier agreement) rather than assumed | `eval/` components |
| Thresholds tuned by cross-validation, not hand-picked | `Calibrator`, LOSO harness |
| Plain-language, non-diagnostic output | `Explainer`, scope constraints |

Non-goals: live WHOOP ingestion, OAuth, real-time streaming, clinical interpretation, demographic/sex inference.

### Research notes informing the design

- **Robust baselines.** Short rest segments (a few minutes) give unstable per-person spread estimates, so the design uses the **median and MAD** (robust to outliers) rather than mean/SD, and shrinks each person's spread toward the cohort pooled estimate via **empirical Bayes**, the same family of idea used in genomics tooling (DESeq2/limma) for small-sample variance stabilization. Scaling MAD by `1.4826` makes it a consistent estimator of the standard deviation for normally distributed data.
- **Combined scoring.** A **directional sum** of signed z-scores is the simple, explainable baseline. A **Mahalanobis distance** accounts for correlation between signals, but a short baseline cannot estimate a full covariance reliably, so the design uses a **Ledoit–Wolf shrinkage covariance** estimate (available in `scikit-learn` as `sklearn.covariance.LedoitWolf`), which shrinks the sample covariance toward a structured target and is well-conditioned for small samples.
- **Persistence over spikes.** Consumer products (e.g. Oura Symptom Radar) require a signal to persist across many nights before flagging. The design mirrors this with a 2-of-3-window persistence gate and tier hysteresis to prevent flicker.
- **Encoding.** `data_constraints.txt` and `Data_Dictionary.csv` are Windows-1252, not UTF-8, and must be opened with `encoding="cp1252"` to avoid decode errors on non-ASCII bytes.

Sources: dataset DOI 10.13026/he0v-tf17 and Hongn et al. (2025), ODC Attribution license; project README. Content was rephrased for compliance with licensing restrictions.

## Architecture

The system is one Python repository with one module per pipeline step. Modules communicate through a shared, serializable `Per_Window_Table` so workstreams can develop in parallel. Data flows strictly left to right; no module reaches back upstream.

```mermaid
flowchart TD
    subgraph data["data/ (ingest + shape)"]
        PL[physionet_load.py\nData_Loader]
        CL[clean.py\nCleaner]
        WN[windows.py\nWindower]
        SC[schema.py\nPer_Window_Table]
    end
    subgraph flag["flag/ (score + flag)"]
        BL[baseline.py\nBaseline_Estimator]
        SR[score.py\nScorer]
        TR[tiers.py\nFlagger]
        CX[context.py\nMovement_Check]
        EX[explain.py\nExplainer]
    end
    subgraph eval["eval/ (measure + tune)"]
        LB[labels.py]
        MT[metrics.py\nEvaluator]
        CB[calibrate.py\nCalibrator]
        RE[run_eval.py]
    end
    subgraph app["app/ (demo)"]
        DM[demo.py\nDemo_App]
    end
    subgraph export["export/ (stretch)"]
        FH[fhir.py]
        CU[care_update.py]
    end

    RAW[(E4 CSVs, tags,\nstress levels)] --> PL
    PL --> CL --> WN
    SC -. defines .-> WN
    WN --> BL --> SR --> TR --> CX --> EX
    WN --> LB
    EX --> MT
    TR --> MT
    LB --> MT
    MT --> CB
    CB -. tau1..tau3 .-> TR
    RE --> MT
    EX --> DM
    WN --> DM
    EX --> FH
    EX --> CU
    MT --> OUT[(output/ figures,\ntables, exports)]
```

### Layout mapping to the existing repo

The repo already contains `data/`, `flag/`, `eval/`, `output/`, and `notebooks/` as Python packages (each with `__init__.py`). The design keeps these and adds `app/` and `export/` as new packages.

| Directory | Status | Modules |
| --- | --- | --- |
| `data/` | exists (`__init__.py`, `external/`, `raw/`, `sessions/`) | `physionet_load.py`, `clean.py`, `windows.py`, `schema.py` |
| `flag/` | exists (`__init__.py`) | `baseline.py`, `score.py`, `tiers.py`, `context.py`, `explain.py` |
| `eval/` | exists (`__init__.py`) | `labels.py`, `metrics.py`, `calibrate.py`, `run_eval.py` |
| `app/` | new | `demo.py` |
| `export/` | new | `fhir.py`, `care_update.py` |
| `output/` | exists (`__init__.py`) | generated figures, metric tables, and exported bundles |
| `notebooks/` | exists | exploration only, not part of the pipeline contract |

Raw PhysioNet recordings live under `data/sessions/Wearable_Dataset_sessions/Wearable_Dataset/<SESSION>/<PARTICIPANT>/`; dictionary and constraint files under `data/external/`.

### Dependencies

Python 3.11+, `numpy`, `pandas`, `scipy` (stats, rank correlation), `scikit-learn` (`LedoitWolf`), `matplotlib` (demo timeline), and a property-based testing library (`hypothesis`). No from-scratch statistical estimators where a vetted library exists.

## Components and Interfaces

Each component is a module with pure, testable functions over the `Per_Window_Table` and typed records. Side effects (file reads, figure writes) are isolated to the edges (`physionet_load`, `run_eval`, `demo`, `export`).

### Data_Loader — `data/physionet_load.py` (Requirement 1)

Reads E4 CSVs, tags, and self-reported stress into in-memory structures.

```python
@dataclass
class SignalSeries:
    name: str              # "HR" | "TEMP" | "ACC" | "IBI"
    start_utc: float       # row 1 of the CSV, epoch seconds UTC
    sample_rate_hz: float  # row 2 (not present for IBI/tags)
    values: np.ndarray     # remaining rows; ACC is (N,3)

@dataclass
class IBISeries:
    start_utc: float       # row 1
    offsets_s: np.ndarray  # successive-row inter-beat interval measurements

def load_signal(path: str, name: str) -> SignalSeries: ...   # AC 1
def load_ibi(path: str) -> IBISeries: ...                    # AC 2
def load_tags(path: str) -> list[float]: ...                 # AC 3 button-press timestamps
def cohort_of(participant_id: str) -> str: ...               # AC 4,5 "S*"->V1, "f*"->V2
def read_cp1252(path: str) -> str: ...                       # AC 6 data_constraints/Data_Dictionary
def load_participant(session_dir: str, pid: str) -> ParticipantSignals: ...  # AC 7 records absent signals, continues
```

Header contract: for HR/TEMP/ACC the first row is `start_utc`, the second row is `sample_rate_hz`; for IBI the first row is `start_utc` and subsequent rows are interval measurements. Missing files are recorded as absent on the `ParticipantSignals` record rather than raising (AC 7).

### Cleaner — `data/clean.py` (Requirement 2)

Applies the documented known-data-issue cases from `data_constraints.txt`. Implemented as a dispatch table keyed on `(participant, session)` so each rule is independently testable.

```python
def clean(signals: ParticipantSignals, constraints: Constraints) -> ParticipantSignals: ...
```

| Case | Rule | AC |
| --- | --- | --- |
| S02 stress | Truncate each signal at the duplicate-start row | 2.1 |
| f07 stress | Exclude from HR/RMSSD/Skin_Temperature analysis | 2.2 |
| f14 stress | Join baseline-file and rest-file segments into one recording | 2.3 |
| S11, S16 aerobic/anaerobic | Join split `_a`/`_b` segments | 2.4 |
| S03, S07, S06 aerobic/anaerobic | Keep only completed phases | 2.5 |
| S01 anaerobic | Produce HR, mark RMSSD unavailable (empty IBI) | 2.6 |
| f14–f18 | Restrict to stress session | 2.7 |
| missing demographics | Retain for signal analysis, mark demographics unavailable | 2.8 |

### Windower — `data/windows.py` (Requirements 3, 4)

Maps tags to phases and segments signals into the `Per_Window_Table`.

```python
def assign_phases(tags: list[float], protocol: str) -> list[Phase]: ...  # 3.1 protocol timing table
def window(signals: ParticipantSignals, phases: list[Phase]) -> PerWindowTable: ...
```

Windowing contract: the analysis window advances in **10-second steps** (AC 4.1). For each step time `t`, HR/Skin_Temperature/Accelerometer_Magnitude aggregate samples in the trailing **30 s** `(t-30, t]` (AC 4.2); RMSSD aggregates inter-beat intervals in the trailing **120 s** `(t-120, t]` (AC 4.3). A signal with too few samples in its interval is marked unavailable (`NaN`) for that window (AC 4.5). Phase for a window is taken from the inter-tag interval containing `t` (AC 3.1, 3.3). For a Cohort_V1 exercise session with no quiet baseline, the baseline phase is sourced from that participant's stress session (AC 3.2) — see Open Decision 1.

### schema.py — `Per_Window_Table` (Requirement 4.4)

Single source of truth for the shared table (see Data Models).

### Baseline_Estimator — `flag/baseline.py` (Requirement 5)

```python
@dataclass
class SignalBaseline:
    median: float
    mad: float          # scaled MAD used downstream
    shrunk_mad: float   # empirical-Bayes shrinkage toward cohort pooled spread

def person_baseline(table: PerWindowTable, pid: str) -> dict[str, SignalBaseline]: ...  # 5.1,5.2,5.4
def shrink_to_cohort(per_person: dict[str, dict[str, SignalBaseline]],
                     cohort: dict[str, str]) -> dict[str, dict[str, SignalBaseline]]: ...  # 5.3
```

Baseline uses the opening rest segment **excluding its first 60 s** while HR settles (AC 5.1), summarized by median and MAD (AC 5.2), shrunk toward the cohort pooled estimate (AC 5.3). Cohort_V1 exercise sessions without a rest baseline reuse the stress-session rest segment (AC 5.4).

### Scorer — `flag/score.py` (Requirement 6)

```python
def signed_z(value: float, b: SignalBaseline, signal: str, direction: int | None) -> float: ...  # 6.1-6.4
def directional_sum(z: dict[str, float]) -> float: ...                                            # 6.5
def mahalanobis_score(z_vec: np.ndarray, cov: LedoitWolf) -> float: ...                            # 6.6
def combined_score(window_row, baselines, method: str) -> float: ...                              # 6.7 skips unavailable
```

Sign convention: HR increase is positive (AC 6.1); RMSSD decrease is positive (AC 6.2); skin temperature is signed by direction of deviation (AC 6.3). Unavailable signals are dropped from the combination for that window (AC 6.7).

### Flagger — `flag/tiers.py` (Requirement 7) and Movement_Check — `flag/context.py` (Requirement 8)

```python
def persistence_gate(scores_last3: list[float], tau1: float) -> bool: ...           # 7.1
def assign_tier(median3: float, taus: Thresholds, single_signal: bool) -> int: ...  # 7.2-7.7
def apply_hysteresis(history: TierHistory, candidate: int, taus) -> int: ...        # 7.8
def movement_label(acc_window, prev_state) -> MovementState: ...                    # 8.1,8.2
def apply_movement(tier: int, mv: MovementState) -> tuple[int, bool]: ...           # 8.3 returns (tier, explained)
```

Gate: flag fires only when `Combined_Score > tau1` in at least 2 of the last 3 windows (AC 7.1). Tier is taken from the **median Combined_Score over the last 3 windows** (AC 7.2): `<tau1 or gate failed -> 0`; `[tau1,tau2) -> 1 Watch`; `[tau2,tau3) -> 2 Notable`; `>=tau3 -> 3 Strong` (AC 7.3-7.6). If only one signal contributes the deviation, the tier is capped at Watch (AC 7.7). A tier steps down only after 2 consecutive windows below its threshold (hysteresis, AC 7.8). Sustained or rhythmic/cadence movement drops the tier by one and labels the flag "explained by movement" (AC 8.1, 8.2); otherwise the tier is unchanged (AC 8.3).

### Explainer — `flag/explain.py` (Requirement 9)

```python
def explain(flag: Flag, contributions: dict[str, float], baselines) -> Explanation: ...
```

Produces text naming the top contributing signals ranked by z-score contribution (AC 9.1), each shift in that signal's real unit relative to rest (e.g. "heart rate +18 bpm vs your rest") (AC 9.2), the tier wording (AC 9.3), and the flag start time, seconds persisted, Combined_Score, and movement-explained boolean (AC 9.4). Output excludes diagnostic claims and color-coded labels (AC 9.5, 15.1, 15.2).

### Evaluator — `eval/metrics.py`, labels — `eval/labels.py` (Requirements 10, 11)

```python
def phase_labels(table, tags) -> PhaseLabels: ...                         # rest vs task, self-report join
def sensitivity(flags, labels) -> dict: ...                               # 10.1 long vs 30-s tasks
def false_alarms_per_rest_hour(flags, labels) -> float: ...               # 10.2 skip first 120 s post-task
def detection_delay(flags, labels) -> dict: ...                           # 10.3
def tier_agreement(flags, self_report) -> float: ...                      # 10.4 per-phase rank correlation
def movement_attribution(flags, labels) -> dict: ...                      # 10.5
def evaluate(method: str) -> MetricTable: ...                             # 10.6 per cohort
def compare_methods() -> ComparisonTable: ...                             # 11.1-11.4 WHOOP vs sum vs Mahalanobis
```

Every metric is reported separately for Cohort_V1 and Cohort_V2 (AC 10.6) and side by side across `WHOOP_Rule`, directional-sum, and Mahalanobis methods (AC 11.4).

### Calibrator — `eval/calibrate.py` (Requirement 12)

```python
def calibrate_loso(table, labels, budget: float) -> Thresholds: ...  # 12.1-12.4
```

Tunes `tau1, tau2, tau3` with leave-one-subject-out cross-validation (AC 12.1): `tau1` from the rest false-alarm budget (AC 12.2); `tau2, tau3` so tiers track self-reported stress and aerobic intensity stages (AC 12.3); output ordered `tau1 < tau2 < tau3` (AC 12.4).

### run_eval — `eval/run_eval.py`

Orchestrates sweeps and writes the headline figures (false alarms vs detection delay; tier vs self-reported stress) to `output/`.

### Demo_App — `app/demo.py` (Requirement 13)

Renders a selected participant's signals, Combined_Score, tier bands, and phase shading on a shared timeline (AC 13.1) and shows the flag explanation at a flagged window (AC 13.2).

### Exporter — `export/fhir.py`, `export/care_update.py` (Requirement 14, stretch)

For Tier 2/3 flags when export is enabled: emit a FHIR `Bundle` with one `Observation` per metric plus one summary `Observation` (AC 14.1); emit a CMS care-update populated with baseline and most-recent clinical measures and the explanation (AC 14.2), leaving Assessment and Plan unpopulated (AC 14.3).

## Data Models

### `Per_Window_Table` (schema.py)

One row per person per 10-second window. This is the contract shared across all workstreams.

| Column | Type | Notes |
| --- | --- | --- |
| `person_id` | str | e.g. `S01`, `f12` |
| `cohort` | str | `V1` or `V2` (AC 1.4, 1.5) |
| `protocol` | str | protocol version label |
| `session` | str | `STRESS` / `AEROBIC` / `ANAEROBIC` |
| `phase` | str | phase name from tags (AC 3.3); `None` if outside any phase |
| `t_win` | float | window end time, epoch seconds UTC |
| `hr` | float \| NaN | trailing-30 s HR aggregate; NaN if insufficient (AC 4.5) |
| `rmssd` | float \| NaN | trailing-120 s RMSSD; NaN if unavailable (AC 4.5, 2.6) |
| `skin_temp` | float \| NaN | trailing-30 s skin temperature |
| `acc_mag` | float \| NaN | trailing-30 s accelerometer magnitude |

### Derived records

```python
@dataclass
class Thresholds:
    tau1: float; tau2: float; tau3: float   # invariant: tau1 < tau2 < tau3

@dataclass
class Flag:
    person_id: str; start_t: float; persisted_s: float
    combined_score: float; tier: int          # 0..3
    top_signals: list[tuple[str, float]]       # (signal, real-unit delta)
    movement_explained: bool

@dataclass
class MovementState:
    sustained: bool; rhythmic: bool; cadence_hz: float | None
```

Tier wording (neutral, non-diagnostic, AC 9.3, 15.1, 15.2):

| Tier | Label | Wording |
| --- | --- | --- |
| 0 | None | nothing shown |
| 1 | Watch | "Some of your signals have shifted from your resting normal" |
| 2 | Notable | "Several signals are clearly off your resting normal" |
| 3 | Strong | "A large shift across several signals" |

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: E4 artifact round-trip

*For any* valid start time, sampling rate, and sample values, writing them as an E4-format artifact (HR/TEMP/ACC header, IBI series, tags list, or cp1252 text) and then loading it SHALL recover the original start time, rate, and values.

**Validates: Requirements 1.1, 1.2, 1.3, 1.6**

### Property 2: Cohort mapping

*For any* participant identifier, `cohort_of` SHALL return `V1` when the identifier begins with "S" and `V2` when it begins with "f".

**Validates: Requirements 1.4, 1.5**

### Property 3: Window phase labeling

*For any* set of non-overlapping labeled phase intervals and any window end time falling inside one interval, the window's recorded phase SHALL equal that interval's name.

**Validates: Requirements 3.1, 3.3**

### Property 4: Window step spacing

*For any* signal span, consecutive window end times in the Per_Window_Table SHALL differ by exactly 10 seconds and cover the span.

**Validates: Requirements 4.1**

### Property 5: Trailing-window membership

*For any* step time t, the samples aggregated for HR, Skin_Temperature, and Accelerometer_Magnitude SHALL be exactly those in the interval (t−30s, t], and the inter-beat intervals aggregated for RMSSD SHALL be exactly those in (t−120s, t].

**Validates: Requirements 4.2, 4.3**

### Property 6: Table shape

*For any* cleaned participant input, the Per_Window_Table SHALL contain the full column set (person_id, cohort, protocol, phase, window time, HR, RMSSD, Skin_Temperature, Accelerometer_Magnitude) and exactly one row per person per window step.

**Validates: Requirements 4.4**

### Property 7: Insufficient-sample cells are unavailable

*For any* window whose trailing interval contains fewer than the minimum required samples for a signal, that signal's cell SHALL be marked unavailable, and every other cell SHALL hold a finite value.

**Validates: Requirements 4.5**

### Property 8: Baseline warm-up exclusion

*For any* opening rest segment, no sample earlier than 60 seconds after the segment start SHALL contribute to the baseline computation.

**Validates: Requirements 5.1**

### Property 9: Baseline statistics match reference

*For any* baseline segment, the computed baseline center and spread SHALL equal the median and the scaled median absolute deviation of that segment.

**Validates: Requirements 5.2**

### Property 10: Shrinkage stays bounded

*For any* per-person spread and cohort pooled spread, the shrunk spread SHALL lie between the raw per-person spread and the cohort pooled spread and remain finite and positive.

**Validates: Requirements 5.3**

### Property 11: Signed z-score direction and magnitude

*For any* window value and baseline, the signed z-score SHALL be positive when HR rises above baseline, positive when RMSSD falls below baseline, signed by the direction of deviation for skin temperature, and SHALL have magnitude equal to the baseline-centered value divided by the scaled median absolute deviation.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4**

### Property 12: Directional sum is the permutation-invariant sum

*For any* set of signed z-scores, the directional-sum Combined_Score SHALL equal their sum and SHALL be unchanged under any reordering of the signals.

**Validates: Requirements 6.5**

### Property 13: Mahalanobis score is non-negative

*For any* z-score vector, the Mahalanobis Combined_Score SHALL be non-negative and SHALL be zero only at the baseline center.

**Validates: Requirements 6.6**

### Property 14: Combined score over available signals

*For any* window with one or more unavailable signals, the Combined_Score SHALL be computed from exactly the available signals and SHALL be finite.

**Validates: Requirements 6.7**

### Property 15: Persistence gate equals the 2-of-3 count

*For any* sequence of the last three Combined_Scores, the persistence gate SHALL pass if and only if at least two of those scores exceed tau1.

**Validates: Requirements 7.1**

### Property 16: Tier uses the median of the last three windows

*For any* sequence of three window scores, changing a single window's score while leaving the median of the three unchanged SHALL NOT change the assigned Severity_Tier.

**Validates: Requirements 7.2**

### Property 17: Tier is a monotone step of the median score

*For any* two median scores m₁ ≤ m₂ under the same ordered thresholds, the tier assigned to m₁ SHALL be less than or equal to the tier assigned to m₂, with tier 0 below tau1, Watch on [tau1,tau2), Notable on [tau2,tau3), and Strong at or above tau3.

**Validates: Requirements 7.3, 7.4, 7.5, 7.6**

### Property 18: Single-signal cap

*For any* window in which only one signal contributes the deviation, the assigned Severity_Tier SHALL be at most 1 Watch regardless of the Combined_Score.

**Validates: Requirements 7.7**

### Property 19: Hysteresis prevents premature down-steps

*For any* score history, the assigned tier SHALL NOT decrease unless the Combined_Score has been below the current tier's threshold for two consecutive windows.

**Validates: Requirements 7.8**

### Property 20: Movement check adjusts tier correctly

*For any* flagged window, the Flagger SHALL reduce the Severity_Tier by exactly one level (floored at 0) and set the "explained by movement" label when sustained or rhythmic/cadence movement is present, and SHALL leave the tier and label unchanged when no such movement is present.

**Validates: Requirements 8.1, 8.2, 8.3**

### Property 21: Explanation ranks contributors by contribution

*For any* flag, the signals named in the explanation SHALL appear in non-increasing order of their z-score contribution.

**Validates: Requirements 9.1**

### Property 22: Explanation reports real-unit shifts

*For any* flag, each named signal's reported shift SHALL equal the signal's value minus its rest baseline expressed in that signal's real unit.

**Validates: Requirements 9.2**

### Property 23: Explanation carries required fields and neutral wording

*For any* flag, the explanation SHALL include the tier wording mapped to the assigned tier, the flag start time, seconds persisted, Combined_Score, and movement-explained flag, and SHALL contain no diagnostic claim or color-coded severity term.

**Validates: Requirements 9.3, 9.4, 9.5, 15.1, 15.2**

### Property 24: Detection delay is bounded

*For any* flagged phase, the detection delay SHALL be non-negative and no greater than the duration of that phase.

**Validates: Requirements 10.3**

### Property 25: LOSO folds partition the subjects

*For any* set of subjects, the leave-one-subject-out folds SHALL leave each subject out exactly once, and each fold's training set SHALL exclude exactly the held-out subject.

**Validates: Requirements 12.1**

### Property 26: Calibrated thresholds are strictly ordered

*For any* calibration run, the produced thresholds SHALL satisfy tau1 < tau2 < tau3.

**Validates: Requirements 12.4**

### Property 27: FHIR export round-trip and completeness

*For any* Tier 2 or Tier 3 flag with export enabled, the emitted FHIR Bundle SHALL re-parse to an equivalent structure containing exactly one Observation per metric plus one summary Observation for the flag.

**Validates: Requirements 14.1**

### Property 28: Care-update leaves clinician sections empty

*For any* exported CMS care-update, the Assessment and Plan sections SHALL be unpopulated.

**Validates: Requirements 14.3**

## Error Handling

The pipeline is deterministic and batch-oriented. Errors are classified as **recoverable** (recorded and skipped so the run continues) or **fatal** (abort the run with a clear message).

| Condition | Component | Handling | Requirement |
| --- | --- | --- | --- |
| Required signal file missing for a participant/session | Data_Loader | Record signal as absent on `ParticipantSignals`, continue loading other signals (recoverable) | 1.7 |
| Empty IBI file (e.g. S01 anaerobic) | Data_Loader / Cleaner | Produce HR, mark RMSSD unavailable; RMSSD-dependent windows get NaN | 2.6 |
| Split recordings (`_a`/`_b`: f14, S11, S16) | Cleaner | Join segments in time order before windowing | 2.3, 2.4 |
| Duplicated segment (S02) | Cleaner | Truncate at documented duplicate-start row | 2.1 |
| Early-stopped sessions (S03, S07, S06, S12) | Cleaner | Keep completed phases; S12 aerobic contributes nothing | 2.5 |
| Dock-on recording (f07) | Cleaner | Exclude from HR/RMSSD/Skin_Temperature analysis | 2.2 |
| Missing demographics / no exercise session (f14–f18) | Cleaner | Retain for signal analysis, mark demographics unavailable, restrict to stress | 2.7, 2.8 |
| Insufficient samples in a window interval | Windower | Mark that signal's cell unavailable (NaN) rather than inventing a value | 4.5 |
| Some signals unavailable at a window | Scorer | Compute Combined_Score from the available signals only | 6.7 |
| V1 exercise session without a quiet baseline | Windower / Baseline_Estimator | Reuse the participant's stress-session rest baseline (Open Decision 1) | 3.2, 5.4 |
| cp1252-encoded file opened as UTF-8 | Data_Loader | Always decode constraints/dictionary with cp1252; a decode error is fatal with a guiding message | 1.6 |
| Thresholds returned out of order | Calibrator | Assertion failure (fatal); ordering is an invariant, never silently reordered | 12.4 |

Edge cases covered by generators in property tests: all-whitespace/empty inputs, single-sample windows, all-NaN windows (Combined_Score undefined → no flag), zero-MAD baselines (degenerate spread → guarded with a small floor so z-scores stay finite), and participants present in one cohort only.

## Testing Strategy

### Dual approach

- **Unit / example tests** cover the documented cleaning cases, the fixed protocol-timing mapping, the demo rendering, and the exact metric definitions — each a concrete, named scenario rather than a universal statement.
- **Property-based tests** cover the 28 correctness properties above, which express universal behavior over generated inputs (parsing round-trips, windowing boundaries, scoring sign/magnitude, gate/tier/hysteresis logic, movement adjustment, explanation structure, LOSO partitioning, export completeness).

Property-based testing applies here because the core of the pipeline is pure functions with clear input/output behavior (parsers, windowing, z-scoring, gating, tier assignment, serialization). It does **not** apply to the demo timeline (UI rendering → example/snapshot tests) or to the exact numeric metric definitions and most cleaning cases (fixed named scenarios → example tests).

### Property test configuration

- Library: **Hypothesis**.
- Each property test runs a **minimum of 100 iterations**.
- Each property test is implemented as a **single** property-based test and tagged with a comment referencing its design property, in the format:
  `# Feature: wearable-flag-pipeline, Property {number}: {property_text}`
- Generators produce valid E4-shaped artifacts (headers, IBI series, tag sequences), score sequences for the gate/tier/hysteresis logic, and flag records for the explainer and exporter, including the edge cases listed under Error Handling.

### Unit / example test coverage

| Area | Tests | Requirements |
| --- | --- | --- |
| Cleaning cases | One fixture per documented participant case | 2.1–2.5, 2.7 |
| Protocol phase mapping | Known tag sets → expected phase sequence per protocol version | 3.1, 3.2 |
| Baseline sourcing for V1 exercise | Fixture asserting stress-session reuse | 5.4 |
| Evaluation metrics | Known flags+labels → expected sensitivity split, false-alarm rate with 120 s tail excluded, delay, rank correlation, movement shares, per-cohort keys | 10.1, 10.2, 10.4, 10.5, 10.6 |
| Method comparison | Full metric set emitted per method, reported side by side | 11.1–11.4 |
| Calibration objectives | Rest false-alarm ≤ budget; tier rises with aerobic intensity on a fixture | 12.2, 12.3 |
| Demo timeline | Render fixture; assert signals, score, tier bands, phase shading, and flag explanation present | 13.1, 13.2 |
| Care-update export | Fixture flag → measures and explanation populated | 14.2 |

### Ground-truth-driven validation (LOSO evaluation harness)

The PhysioNet protocol tags give known event times, rest segments give known-quiet time, and self-reported stress gives a per-phase severity label. `eval/run_eval.py` uses these as the validation oracle:

1. Build the Per_Window_Table and phase labels for every clean participant.
2. For each method (`WHOOP_Rule`, directional-sum, Mahalanobis), run leave-one-subject-out: hold out one subject, calibrate tau1–tau3 on the rest, flag the held-out subject, accumulate metrics.
3. Aggregate per-cohort sensitivity, false alarms per rest-hour, detection delay, tier agreement, and movement attribution.
4. Emit the headline figures (false alarms vs detection delay with per-tier operating points; tier vs self-reported stress) to `output/`.

LOSO guarantees thresholds are never tuned on the subject they are evaluated on, so reported accuracy reflects generalization rather than fit.

## Open Design Decisions

The README raises decisions the design must resolve. Each is settled below with a recommendation and a rationale; the design treats these as configurable so the alternative can be swept in evaluation.

### Decision 1: V1 exercise-session baseline source

V1 exercise sessions have no quiet baseline phase.

- **Recommendation: reuse each participant's stress-session rest baseline** (same wrist, same person, a genuinely quiet segment), per AC 3.2 and 5.4.
- Rationale: warm-up is already elevated and would understate deviation; a true rest segment is the better "resting normal." The warm-up-first-minute alternative is retained as a config option and can be compared in the baseline-estimate sweep the README calls for.

### Decision 2: Ground truth driving tier calibration

- **Recommendation: use both, with distinct roles.** Calibrate `tau1` from the **rest false-alarm budget** (objective, protocol-independent), and calibrate `tau2`/`tau3` against **self-reported stress** as the primary tier oracle, cross-checked against the **aerobic intensity stages** as an ordinal secondary signal (AC 12.2, 12.3).
- Rationale: self-reported stress is the closest thing to a severity label but is coarse (one rating per phase) and nonzero at baseline; the bike-test's rising stages give an independent ordinal check that tiers track intensity. Using both reduces reliance on either weak signal alone.

### Decision 3 (framing, non-code)

Pitch framing (acute-change vs general sustained-change) and keeping a WHOOP nightly demo are product/presentation choices outside this system's scope; per the approved requirements, no WHOOP data or OAuth client is built. Noted here only to record that they were considered and deliberately excluded from scope.
