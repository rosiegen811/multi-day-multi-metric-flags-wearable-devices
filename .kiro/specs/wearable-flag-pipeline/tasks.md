# Implementation Plan: Wearable Flag Pipeline

## Overview

This plan builds the Wearable Flag Pipeline as one Python repository with one module per pipeline step, communicating through a shared `Per_Window_Table`. The build order follows data dependencies: scaffolding and the shared schema first, then data ingest/clean/window, then baseline/score/flag/context/explain, then evaluation/calibration/orchestration, then the demo, with FHIR/CMS export as stretch work last. Each task builds on earlier tasks and wires its output into the shared table or a downstream consumer so no code is left orphaned.

Implementation language is **Python 3.11+** (as specified in the design). Property-based tests use **Hypothesis** at a minimum of 100 iterations each and are tagged `# Feature: wearable-flag-pipeline, Property N: ...`. Test sub-tasks are marked optional with `*`.

## Tasks

- [x] 1. Set up project scaffolding, dependencies, and shared schema
  - [x] 1.1 Create package scaffolding and dependency manifest
    - Add `app/__init__.py` and `export/__init__.py` so the new packages described in the design exist alongside existing `data/`, `flag/`, `eval/`, `output/`
    - Create a dependency manifest (`pyproject.toml` or `requirements.txt`) pinning `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, and `hypothesis`, targeting Python 3.11+
    - Configure the test runner (pytest) and a test directory layout mirroring the package structure
    - _Requirements: all (shared foundation)_

  - [x] 1.2 Implement the `Per_Window_Table` schema and shared records in `data/schema.py`
    - Define the `Per_Window_Table` column set and dtypes: `person_id`, `cohort`, `protocol`, `session`, `phase`, `t_win`, `hr`, `rmssd`, `skin_temp`, `acc_mag` (NaN-capable float columns)
    - Define shared dataclasses used across workstreams: `Thresholds` (invariant `tau1 < tau2 < tau3`), `Flag`, `MovementState`, `SignalBaseline`, and the tier wording map (0 None / 1 Watch / 2 Notable / 3 Strong)
    - Provide a constructor/validator that enforces the full column set and one row per person per window step
    - _Requirements: 4.4, 9.3, 12.4, 15.1, 15.2_

  - [ ]* 1.3 Write property test for table shape
    - `# Feature: wearable-flag-pipeline, Property 6: Table shape`
    - **Property 6: Table shape** — full column set and exactly one row per person per window step
    - **Validates: Requirements 4.4**

- [x] 2. Implement the Data_Loader in `data/physionet_load.py`
  - [x] 2.1 Implement E4 signal, IBI, tags, cohort, and cp1252 loaders
    - `load_signal` reads row 1 as `start_utc`, row 2 as `sample_rate_hz`, remaining rows as values (ACC as (N,3)) into a `SignalSeries`
    - `load_ibi` reads row 1 as `start_utc` and subsequent rows as inter-beat interval measurements into an `IBISeries`
    - `load_tags` reads each entry as a button-press timestamp
    - `cohort_of` maps `S*` -> V1 and `f*` -> V2
    - `read_cp1252` decodes `data_constraints.txt` / `Data_Dictionary.csv` with `encoding="cp1252"`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x] 2.2 Implement participant assembly with missing-file tolerance
    - `load_participant` loads all signals for a session from `data/sessions/Wearable_Dataset_sessions/Wearable_Dataset/<SESSION>/<PID>/`, recording absent signals on `ParticipantSignals` and continuing rather than raising
    - _Requirements: 1.7_

  - [ ]* 2.3 Write property test for E4 artifact round-trip
    - `# Feature: wearable-flag-pipeline, Property 1: E4 artifact round-trip`
    - **Property 1: E4 artifact round-trip** — write-then-load recovers start time, rate, and values for HR/TEMP/ACC headers, IBI series, tags, and cp1252 text
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.6**

  - [ ]* 2.4 Write property test for cohort mapping
    - `# Feature: wearable-flag-pipeline, Property 2: Cohort mapping`
    - **Property 2: Cohort mapping** — `S`-prefixed -> V1, `f`-prefixed -> V2
    - **Validates: Requirements 1.4, 1.5**

- [x] 3. Implement the Cleaner in `data/clean.py`
  - [x] 3.1 Implement the documented data-cleaning dispatch table
    - Parse `data_constraints.txt` (cp1252) into a `Constraints` structure keyed on `(participant, session)`
    - Implement each documented case as an independently testable rule: S02 truncate at duplicate-start row (2.1); f07 exclude from HR/RMSSD/Skin_Temperature (2.2); f14 join baseline+rest segments (2.3); S11/S16 join split `_a`/`_b` segments (2.4); S03/S07/S06 keep completed phases (2.5); S01 anaerobic produce HR, mark RMSSD unavailable (2.6); f14–f18 restrict to stress session (2.7); missing demographics retained with demographics marked unavailable (2.8)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [ ]* 3.2 Write unit/example tests for cleaning cases
    - One fixture per documented participant case asserting the expected truncation/join/exclusion outcome
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.7_

- [x] 4. Implement phase mapping and windowing in `data/windows.py`
  - [x] 4.1 Implement protocol phase assignment from tags
    - `assign_phases` maps each inter-tag interval to a named phase using the protocol timing table for the session's protocol version
    - For a Cohort_V1 exercise session without a quiet baseline phase, source the baseline phase from that participant's stress session (Open Decision 1)
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 4.2 Implement trailing-window segmentation into the Per_Window_Table
    - `window` advances in 10 s steps; aggregates HR/Skin_Temperature/Accelerometer_Magnitude over trailing `(t-30, t]` and RMSSD over trailing `(t-120, t]`
    - Mark a signal's cell unavailable (NaN) when its interval has too few samples; set phase from the inter-tag interval containing `t`; write one row per person per window via the `data/schema.py` constructor
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 4.3 Write property test for window step spacing
    - `# Feature: wearable-flag-pipeline, Property 4: Window step spacing`
    - **Property 4: Window step spacing** — consecutive window end times differ by exactly 10 s and cover the span
    - **Validates: Requirements 4.1**

  - [ ]* 4.4 Write property test for trailing-window membership
    - `# Feature: wearable-flag-pipeline, Property 5: Trailing-window membership`
    - **Property 5: Trailing-window membership** — HR/Skin_Temperature/Accelerometer_Magnitude use exactly `(t-30, t]`; RMSSD uses exactly `(t-120, t]`
    - **Validates: Requirements 4.2, 4.3**

  - [ ]* 4.5 Write property test for insufficient-sample cells
    - `# Feature: wearable-flag-pipeline, Property 7: Insufficient-sample cells are unavailable`
    - **Property 7: Insufficient-sample cells are unavailable** — sparse-interval signal marked unavailable, other cells finite
    - **Validates: Requirements 4.5**

  - [ ]* 4.6 Write property test for window phase labeling
    - `# Feature: wearable-flag-pipeline, Property 3: Window phase labeling`
    - **Property 3: Window phase labeling** — a window inside a labeled interval records that interval's name
    - **Validates: Requirements 3.1, 3.3**

  - [ ]* 4.7 Write unit/example tests for protocol phase mapping and V1 baseline sourcing
    - Known tag sets -> expected phase sequence per protocol version; fixture asserting V1 exercise reuses the stress-session rest segment
    - _Requirements: 3.1, 3.2_

- [x] 5. Checkpoint - data layer
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement the Baseline_Estimator in `flag/baseline.py`
  - [x] 6.1 Implement per-person baseline with warm-up exclusion and cohort shrinkage
    - `person_baseline` uses the opening rest segment excluding its first 60 s, summarizing each signal as median and scaled MAD (`1.4826 * MAD`) into `SignalBaseline`
    - `shrink_to_cohort` shrinks each per-person spread toward the cohort pooled estimate via empirical-Bayes; V1 exercise sessions without a rest baseline reuse the stress-session rest segment; guard zero-MAD with a small positive floor
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ]* 6.2 Write property test for baseline warm-up exclusion
    - `# Feature: wearable-flag-pipeline, Property 8: Baseline warm-up exclusion`
    - **Property 8: Baseline warm-up exclusion** — no sample earlier than 60 s after segment start contributes
    - **Validates: Requirements 5.1**

  - [ ]* 6.3 Write property test for baseline statistics
    - `# Feature: wearable-flag-pipeline, Property 9: Baseline statistics match reference`
    - **Property 9: Baseline statistics match reference** — center and spread equal median and scaled MAD
    - **Validates: Requirements 5.2**

  - [ ]* 6.4 Write property test for shrinkage bounds
    - `# Feature: wearable-flag-pipeline, Property 10: Shrinkage stays bounded`
    - **Property 10: Shrinkage stays bounded** — shrunk spread lies between per-person and cohort pooled spread, finite and positive
    - **Validates: Requirements 5.3**

  - [ ]* 6.5 Write unit/example test for V1 exercise baseline sourcing
    - Fixture asserting stress-session rest reuse for a V1 exercise session
    - _Requirements: 5.4_

- [x] 7. Implement the Scorer in `flag/score.py`
  - [x] 7.1 Implement signed z-scores and combined scores
    - `signed_z` makes HR increase positive, RMSSD decrease positive, skin temperature signed by direction of deviation, magnitude = baseline-centered value / scaled MAD
    - `directional_sum` sums signed z-scores; `mahalanobis_score` uses a `sklearn.covariance.LedoitWolf` shrinkage covariance
    - `combined_score` dispatches by method and drops unavailable signals from the combination for that window, returning a finite value
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ]* 7.2 Write property test for signed z-score direction and magnitude
    - `# Feature: wearable-flag-pipeline, Property 11: Signed z-score direction and magnitude`
    - **Property 11: Signed z-score direction and magnitude**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4**

  - [ ]* 7.3 Write property test for directional sum
    - `# Feature: wearable-flag-pipeline, Property 12: Directional sum is the permutation-invariant sum`
    - **Property 12: Directional sum is the permutation-invariant sum**
    - **Validates: Requirements 6.5**

  - [ ]* 7.4 Write property test for Mahalanobis non-negativity
    - `# Feature: wearable-flag-pipeline, Property 13: Mahalanobis score is non-negative`
    - **Property 13: Mahalanobis score is non-negative** — non-negative, zero only at baseline center
    - **Validates: Requirements 6.6**

  - [ ]* 7.5 Write property test for combined score over available signals
    - `# Feature: wearable-flag-pipeline, Property 14: Combined score over available signals`
    - **Property 14: Combined score over available signals** — computed from available signals only, finite
    - **Validates: Requirements 6.7**

- [x] 8. Implement the Flagger tiers and hysteresis in `flag/tiers.py`
  - [x] 8.1 Implement persistence gate, tier assignment, single-signal cap, and hysteresis
    - `persistence_gate` passes iff at least 2 of the last 3 Combined_Scores exceed tau1
    - `assign_tier` maps the median of the last 3 scores: `<tau1 or gate failed -> 0`; `[tau1,tau2) -> 1`; `[tau2,tau3) -> 2`; `>=tau3 -> 3`; caps at Watch when only one signal contributes the deviation
    - `apply_hysteresis` steps a tier down only after 2 consecutive windows below its threshold
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8_

  - [ ]* 8.2 Write property test for persistence gate
    - `# Feature: wearable-flag-pipeline, Property 15: Persistence gate equals the 2-of-3 count`
    - **Property 15: Persistence gate equals the 2-of-3 count**
    - **Validates: Requirements 7.1**

  - [ ]* 8.3 Write property test for median-of-three tier
    - `# Feature: wearable-flag-pipeline, Property 16: Tier uses the median of the last three windows`
    - **Property 16: Tier uses the median of the last three windows**
    - **Validates: Requirements 7.2**

  - [ ]* 8.4 Write property test for monotone tier steps
    - `# Feature: wearable-flag-pipeline, Property 17: Tier is a monotone step of the median score`
    - **Property 17: Tier is a monotone step of the median score**
    - **Validates: Requirements 7.3, 7.4, 7.5, 7.6**

  - [ ]* 8.5 Write property test for single-signal cap
    - `# Feature: wearable-flag-pipeline, Property 18: Single-signal cap`
    - **Property 18: Single-signal cap** — tier at most Watch when one signal contributes
    - **Validates: Requirements 7.7**

  - [ ]* 8.6 Write property test for hysteresis
    - `# Feature: wearable-flag-pipeline, Property 19: Hysteresis prevents premature down-steps`
    - **Property 19: Hysteresis prevents premature down-steps**
    - **Validates: Requirements 7.8**

- [x] 9. Implement the Movement_Check in `flag/context.py`
  - [x] 9.1 Implement movement labeling and tier adjustment
    - `movement_label` detects sustained movement and rhythmic/cadence motion from the accelerometer window into a `MovementState`
    - `apply_movement` reduces the tier by exactly one (floored at 0) and sets "explained by movement" when such movement is present, and leaves tier and label unchanged otherwise
    - _Requirements: 8.1, 8.2, 8.3_

  - [ ]* 9.2 Write property test for movement check
    - `# Feature: wearable-flag-pipeline, Property 20: Movement check adjusts tier correctly`
    - **Property 20: Movement check adjusts tier correctly**
    - **Validates: Requirements 8.1, 8.2, 8.3**

- [x] 10. Implement the Explainer in `flag/explain.py`
  - [x] 10.1 Implement plain-language flag explanations
    - `explain` ranks named signals by z-score contribution (non-increasing), reports each signal's real-unit shift relative to rest, includes the tier wording for the assigned tier, and includes start time, seconds persisted, Combined_Score, and movement-explained flag
    - Exclude diagnostic claims and color-coded severity terms
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 15.1, 15.2_

  - [ ]* 10.2 Write property test for contributor ranking
    - `# Feature: wearable-flag-pipeline, Property 21: Explanation ranks contributors by contribution`
    - **Property 21: Explanation ranks contributors by contribution**
    - **Validates: Requirements 9.1**

  - [ ]* 10.3 Write property test for real-unit shifts
    - `# Feature: wearable-flag-pipeline, Property 22: Explanation reports real-unit shifts`
    - **Property 22: Explanation reports real-unit shifts**
    - **Validates: Requirements 9.2**

  - [ ]* 10.4 Write property test for required fields and neutral wording
    - `# Feature: wearable-flag-pipeline, Property 23: Explanation carries required fields and neutral wording`
    - **Property 23: Explanation carries required fields and neutral wording**
    - **Validates: Requirements 9.3, 9.4, 9.5, 15.1, 15.2**

- [x] 11. Checkpoint - scoring and flagging layer
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Implement evaluation labels and metrics in `eval/labels.py` and `eval/metrics.py`
  - [~] 12.1 Implement phase labels and the ground-truth join
    - `phase_labels` derives rest-vs-task labels from tags and joins per-phase self-reported stress
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [~] 12.2 Implement detection metrics per cohort
    - `sensitivity` (long vs 30 s tasks), `false_alarms_per_rest_hour` (skip first 120 s post-task), `detection_delay`, `tier_agreement` (per-phase rank correlation), `movement_attribution`, and `evaluate(method)` reporting each metric separately for Cohort_V1 and Cohort_V2
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [ ]* 12.3 Write property test for detection-delay bound
    - `# Feature: wearable-flag-pipeline, Property 24: Detection delay is bounded`
    - **Property 24: Detection delay is bounded** — non-negative and no greater than the phase duration
    - **Validates: Requirements 10.3**

  - [ ]* 12.4 Write unit/example tests for metric definitions
    - Known flags+labels -> expected sensitivity split, false-alarm rate with 120 s tail excluded, delay, rank correlation, movement shares, per-cohort keys
    - _Requirements: 10.1, 10.2, 10.4, 10.5, 10.6_

- [ ] 13. Implement the WHOOP_Rule baseline and method comparison in `eval/metrics.py`
  - [~] 13.1 Implement the WHOOP_Rule and side-by-side comparison
    - Implement the single-signal out-of-range `WHOOP_Rule` as an evaluation baseline and `compare_methods` computing the full metric set for WHOOP_Rule, directional-sum, and Mahalanobis side by side
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [ ]* 13.2 Write unit/example test for method comparison
    - Assert the full metric set is emitted per method and reported side by side
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [ ] 14. Implement the Calibrator in `eval/calibrate.py`
  - [~] 14.1 Implement LOSO threshold calibration
    - `calibrate_loso` tunes `tau1` from the rest false-alarm budget and `tau2`/`tau3` so tiers track self-reported stress and aerobic intensity stages, using leave-one-subject-out folds; assert output ordering `tau1 < tau2 < tau3`
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [ ]* 14.2 Write property test for LOSO partitioning
    - `# Feature: wearable-flag-pipeline, Property 25: LOSO folds partition the subjects`
    - **Property 25: LOSO folds partition the subjects**
    - **Validates: Requirements 12.1**

  - [ ]* 14.3 Write property test for ordered thresholds
    - `# Feature: wearable-flag-pipeline, Property 26: Calibrated thresholds are strictly ordered`
    - **Property 26: Calibrated thresholds are strictly ordered**
    - **Validates: Requirements 12.4**

  - [ ]* 14.4 Write unit/example test for calibration objectives
    - Rest false-alarm <= budget; tier rises with aerobic intensity on a fixture
    - _Requirements: 12.2, 12.3_

- [ ] 15. Implement evaluation orchestration in `eval/run_eval.py`
  - [~] 15.1 Wire the LOSO evaluation harness and headline figures
    - Build the Per_Window_Table and phase labels for every clean participant, run LOSO per method (calibrate on the rest, flag the held-out subject, accumulate metrics), aggregate per cohort, and write the headline figures (false alarms vs detection delay; tier vs self-reported stress) to `output/`
    - This integrates Data_Loader -> Cleaner -> Windower -> Baseline -> Scorer -> Flagger -> Movement -> Explainer -> Evaluator -> Calibrator end to end
    - _Requirements: 10.6, 11.4, 12.1_

- [~] 16. Checkpoint - evaluation layer
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 17. Implement the Demo_App in `app/demo.py`
  - [~] 17.1 Render the per-person timeline
    - Render a selected participant's signals, Combined_Score, Severity_Tier bands, and phase shading on a shared timeline, and show the flag explanation at a flagged window
    - _Requirements: 13.1, 13.2_

  - [ ]* 17.2 Write unit/example test for the demo timeline
    - Render fixture; assert signals, score, tier bands, phase shading, and flag explanation are present
    - _Requirements: 13.1, 13.2_

- [ ] 18. (Stretch / optional) Implement standard-format export in `export/fhir.py` and `export/care_update.py`
  - [~] 18.1 Implement FHIR and CMS care-update export for Tier 2/3 flags
    - When export is enabled and a flag is Tier 2 or Tier 3: emit a FHIR `Bundle` with one `Observation` per metric plus one summary `Observation`; emit a CMS care-update populated with baseline and most-recent clinical measures and the explanation, leaving Assessment and Plan unpopulated
    - _Requirements: 14.1, 14.2, 14.3_

  - [ ]* 18.2 Write property test for FHIR export round-trip
    - `# Feature: wearable-flag-pipeline, Property 27: FHIR export round-trip and completeness`
    - **Property 27: FHIR export round-trip and completeness**
    - **Validates: Requirements 14.1**

  - [ ]* 18.3 Write property test for empty clinician sections
    - `# Feature: wearable-flag-pipeline, Property 28: Care-update leaves clinician sections empty`
    - **Property 28: Care-update leaves clinician sections empty**
    - **Validates: Requirements 14.3**

  - [ ]* 18.4 Write unit/example test for care-update population
    - Fixture flag -> measures and explanation populated
    - _Requirements: 14.2_

- [~] 19. Final checkpoint - full pipeline
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster MVP.
- Task 18 and its sub-tasks are stretch/optional work; the core pipeline (tasks 1–17) is complete without them.
- Each task references specific requirement acceptance criteria for traceability, and property-test tasks reference the design's correctness property they validate.
- Checkpoints ensure incremental validation at layer boundaries.
- Property tests use Hypothesis at a minimum of 100 iterations and the required `# Feature: wearable-flag-pipeline, Property N: ...` tag; unit/example tests cover the fixed cleaning cases, metric definitions, demo rendering, and care-update population.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3", "2.1"] },
    { "id": 3, "tasks": ["2.2", "2.3", "2.4"] },
    { "id": 4, "tasks": ["3.1"] },
    { "id": 5, "tasks": ["3.2", "4.1"] },
    { "id": 6, "tasks": ["4.2"] },
    { "id": 7, "tasks": ["4.3", "4.4", "4.5", "4.6", "4.7", "6.1", "7.1"] },
    { "id": 8, "tasks": ["6.2", "6.3", "6.4", "6.5", "7.2", "7.3", "7.4", "7.5", "8.1"] },
    { "id": 9, "tasks": ["8.2", "8.3", "8.4", "8.5", "8.6", "9.1"] },
    { "id": 10, "tasks": ["9.2", "10.1"] },
    { "id": 11, "tasks": ["10.2", "10.3", "10.4", "12.1"] },
    { "id": 12, "tasks": ["12.2", "13.1"] },
    { "id": 13, "tasks": ["12.3", "12.4", "13.2", "14.1"] },
    { "id": 14, "tasks": ["14.2", "14.3", "14.4"] },
    { "id": 15, "tasks": ["15.1", "17.1"] },
    { "id": 16, "tasks": ["17.2", "18.1"] },
    { "id": 17, "tasks": ["18.2", "18.3", "18.4"] }
  ]
}
```
