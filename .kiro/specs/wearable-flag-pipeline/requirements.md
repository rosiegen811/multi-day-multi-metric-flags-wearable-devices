# Requirements Document

## Introduction

The Wearable Flag Pipeline detects sustained, multi-metric physiological change from wrist-worn wearable signals, rates each detected change by severity, and explains each flag in plain language. The pipeline is built and evaluated on the PhysioNet Empatica E4 wearable dataset, which provides timed stressor protocols and self-reported stress ratings as ground truth.

The pipeline transforms raw wrist signals into a per-window table, learns a per-person resting baseline, computes a combined deviation score across heart rate, heart-rate variability, and skin temperature, fires a flag only when that score persists, assigns a severity tier (with a movement check as a benign cause), produces a plain-language explanation, and measures detection performance against the protocol's known events. The method is designed so the same approach would transfer to daily WHOOP nightly summaries, with nights in place of 10-second windows; that transfer is argued for conceptually and is not implemented or tested in this system. The system is built and evaluated only on the PhysioNet session data, which serves as a simulation of what nightly flagging would do.

The system is organized as one Python repository with one module per pipeline step: data loading and cleaning (`data/`), scoring and flagging (`flag/`), evaluation and calibration (`eval/`), a demo timeline (`app/`), and optional export (`export/`). The system does not include a WHOOP OAuth client; live WHOOP nightly data is out of scope.

The system is explicitly not a medical device. Flags state "sustained change," not a diagnosis, and the system avoids color-coded or diagnostic claims.

## Glossary

- **Pipeline**: The complete Wearable Flag Pipeline system that ingests wearable signals and produces explained, tiered flags.
- **Data_Loader**: The component that reads Empatica E4 CSV files, tags, and self-reported stress levels into in-memory structures.
- **Cleaner**: The component that applies the documented known data issues from `data_constraints.txt`.
- **Windower**: The component that segments signals into fixed-length rolling windows and produces the per-window table.
- **Baseline_Estimator**: The component that computes a per-person, per-signal resting baseline using median and MAD with empirical-Bayes shrinkage toward the cohort estimate.
- **Scorer**: The component that computes signed z-scores and combined deviation scores per window.
- **Flagger**: The component that applies the persistence gate, assigns severity tiers with hysteresis, and applies the movement check.
- **Explainer**: The component that produces plain-language flag explanations naming top contributing signals in real units.
- **Evaluator**: The component that computes detection metrics against protocol tags and self-reported stress.
- **Calibrator**: The component that tunes thresholds using leave-one-subject-out cross-validation.
- **Demo_App**: The component that renders a per-person timeline of signals, score, tier bands, and phase shading.
- **Exporter**: The optional component that emits a flag in FHIR and CMS care-update formats.
- **Per_Window_Table**: A tabular structure with one row per person per window, with columns: person_id, cohort, protocol, phase, window time, heart rate, RMSSD, skin temperature, and accelerometer magnitude.
- **HR**: Heart rate, sampled at 1 Hz from HR.csv.
- **RMSSD**: Root mean square of successive differences of inter-beat intervals, used as the heart-rate variability measure, computed from IBI.csv.
- **IBI**: Inter-beat interval data from IBI.csv.
- **Skin_Temperature**: Wrist skin temperature, sampled at 4 Hz from TEMP.csv.
- **Accelerometer_Magnitude**: The combined-axis acceleration magnitude derived from ACC.csv, sampled at 32 Hz in units of 1/64 g.
- **Combined_Score**: A single deviation value per window, computed either as a directional sum of signed z-scores or as a Mahalanobis distance with Ledoit-Wolf shrinkage covariance.
- **Persistence_Gate**: The rule that a flag fires only when the Combined_Score exceeds threshold tau1 in at least 2 of the last 3 windows.
- **Severity_Tier**: One of four levels assigned to a flag: 0 None, 1 Watch, 2 Notable, 3 Strong.
- **tau1, tau2, tau3**: The three ordered Combined_Score thresholds (tau1 < tau2 < tau3) that separate severity tiers.
- **Hysteresis**: The rule that a tier steps down only after 2 consecutive windows below its threshold.
- **Movement_Check**: The rule that reduces a flag by one tier and labels it "explained by movement" when the accelerometer shows sustained or rhythmic motion.
- **Cohort_V1**: Participants whose identifiers begin with "S", who ran protocol version 1.
- **Cohort_V2**: Participants whose identifiers begin with "f", who ran protocol version 2.
- **WHOOP_Rule**: The single-signal out-of-range comparison rule modeled on WHOOP Health Monitor behavior.
- **LOSO**: Leave-one-subject-out cross-validation.

## Requirements

### Requirement 1: Load PhysioNet Empatica E4 signal files

**User Story:** As a data engineer, I want the Pipeline to load raw Empatica E4 CSV files, so that downstream components receive structured signal data with correct timing.

#### Acceptance Criteria

1. WHEN a signal CSV file for HR, Skin_Temperature, or Accelerometer_Magnitude is provided, THE Data_Loader SHALL read the first row as the session start time in UTC and the second row as the sampling rate in Hz.
2. WHEN an IBI file is provided, THE Data_Loader SHALL read the first row as the session start time in UTC and parse each subsequent row as an inter-beat interval measurement.
3. WHEN a tags file is provided, THE Data_Loader SHALL read each entry as a button-press timestamp.
4. WHEN a participant identifier begins with "S", THE Data_Loader SHALL label the participant as Cohort_V1.
5. WHEN a participant identifier begins with "f", THE Data_Loader SHALL label the participant as Cohort_V2.
6. WHEN the file `data_constraints.txt` or `Data_Dictionary.csv` is read, THE Data_Loader SHALL decode the file using the Windows-1252 (cp1252) character encoding.
7. IF a required signal file is missing for a participant and session, THEN THE Data_Loader SHALL record the absent signal and continue loading the remaining signals.

### Requirement 2: Apply documented data-cleaning rules

**User Story:** As a data engineer, I want the Pipeline to apply the known data issues documented in the dataset, so that corrupted or invalid segments do not distort results.

#### Acceptance Criteria

1. WHEN participant S02 stress-session signals are processed, THE Cleaner SHALL truncate each signal at the duplicate start row specified in `data_constraints.txt`.
2. WHEN participant f07 stress-session signals are processed, THE Cleaner SHALL exclude the participant from HR, RMSSD, and Skin_Temperature analysis.
3. WHEN participant f14 stress-session signals are processed, THE Cleaner SHALL join the baseline file segment and the rest file segment into one continuous recording.
4. WHEN participant S11 or S16 aerobic or anaerobic signals are processed, THE Cleaner SHALL join the split recording segments into one continuous recording.
5. WHEN participants S03, S07, or S06 aerobic or anaerobic signals are processed, THE Cleaner SHALL retain only the phases the participant completed.
6. WHEN participant S01 anaerobic signals are processed, THE Cleaner SHALL produce HR output and SHALL mark RMSSD as unavailable.
7. WHEN participants f14 through f18 are processed, THE Cleaner SHALL restrict analysis to the stress session.
8. WHERE a participant lacks demographic information, THE Cleaner SHALL retain the participant for signal analysis and SHALL mark demographics as unavailable.

### Requirement 3: Assign protocol phases from tags

**User Story:** As a data engineer, I want the Pipeline to map button-press tags to named protocol phases, so that each window can be labeled with its experimental phase.

#### Acceptance Criteria

1. WHEN tag timestamps and a session protocol version are provided, THE Windower SHALL map each inter-tag interval to a named phase according to the protocol timing table for that version.
2. WHERE a Cohort_V1 participant has an exercise session without a quiet baseline phase, THE Windower SHALL use the baseline segment from that participant's stress session as the baseline for the exercise session.
3. WHEN a window falls within a labeled phase, THE Windower SHALL record the phase name in the Per_Window_Table row for that window.

### Requirement 4: Window signals into the per-window table

**User Story:** As a data engineer, I want the Pipeline to segment signals into fixed rolling windows, so that scoring operates on consistent time units.

#### Acceptance Criteria

1. THE Windower SHALL advance the analysis window in steps of 10 seconds.
2. WHEN computing HR, Skin_Temperature, and Accelerometer_Magnitude for a window, THE Windower SHALL aggregate the samples within the trailing 30-second interval ending at the step time.
3. WHEN computing RMSSD for a window, THE Windower SHALL aggregate the inter-beat intervals within the trailing 120-second interval ending at the step time.
4. THE Windower SHALL produce a Per_Window_Table with one row per person per window containing person_id, cohort, protocol, phase, window time, HR, RMSSD, Skin_Temperature, and Accelerometer_Magnitude.
5. IF a signal has insufficient samples within its window interval to compute a value, THEN THE Windower SHALL mark that signal's value as unavailable for that window.

### Requirement 5: Compute per-person rest baselines

**User Story:** As a data scientist, I want the Pipeline to learn each person's resting normal per signal, so that deviation is measured in that person's own terms.

#### Acceptance Criteria

1. WHEN computing a participant's baseline, THE Baseline_Estimator SHALL use the participant's opening rest segment excluding the first 60 seconds of that segment.
2. THE Baseline_Estimator SHALL summarize each signal's baseline using the median and the median absolute deviation.
3. THE Baseline_Estimator SHALL shrink each participant's per-signal baseline spread toward the pooled cohort estimate using an empirical-Bayes estimate.
4. WHERE a Cohort_V1 participant lacks a rest baseline in an exercise session, THE Baseline_Estimator SHALL derive the baseline from the participant's stress-session rest segment.

### Requirement 6: Compute signed z-scores and combined scores

**User Story:** As a data scientist, I want the Pipeline to compute signed deviation scores and combine them, so that several small shifts together register as one larger signal.

#### Acceptance Criteria

1. WHEN computing the HR z-score for a window, THE Scorer SHALL assign a positive sign to an increase above baseline.
2. WHEN computing the RMSSD z-score for a window, THE Scorer SHALL assign a positive sign to a decrease below baseline.
3. WHEN computing the Skin_Temperature z-score for a window, THE Scorer SHALL assign a sign reflecting the direction of deviation from baseline.
4. THE Scorer SHALL compute each signed z-score as the baseline-centered signal value divided by the scaled median absolute deviation of that signal.
5. THE Scorer SHALL compute a directional-sum Combined_Score as the sum of the signed z-scores across HR, RMSSD, and Skin_Temperature.
6. THE Scorer SHALL compute a Mahalanobis Combined_Score using a Ledoit-Wolf shrinkage covariance estimate.
7. WHERE a signal value is unavailable for a window, THE Scorer SHALL compute the Combined_Score from the available signals for that window.

### Requirement 7: Apply persistence gate and severity tiers

**User Story:** As a data scientist, I want the Pipeline to fire a flag only when a shift persists and to rate it by size, so that momentary spikes are screened out and severity is meaningful.

#### Acceptance Criteria

1. THE Flagger SHALL fire a flag for a window only when the Combined_Score exceeds tau1 in at least 2 of the last 3 windows.
2. WHEN the Persistence_Gate is passed, THE Flagger SHALL assign the Severity_Tier from the median Combined_Score over the last 3 windows.
3. IF the Persistence_Gate is not passed or the median Combined_Score over the last 3 windows is below tau1, THEN THE Flagger SHALL assign Severity_Tier 0 None.
4. WHILE the median Combined_Score over the last 3 windows is at least tau1 and below tau2, THE Flagger SHALL assign Severity_Tier 1 Watch.
5. WHILE the median Combined_Score over the last 3 windows is at least tau2 and below tau3, THE Flagger SHALL assign Severity_Tier 2 Notable.
6. WHILE the median Combined_Score over the last 3 windows is at least tau3, THE Flagger SHALL assign Severity_Tier 3 Strong.
7. WHEN only one signal exceeds its deviation contribution, THE Flagger SHALL cap the assigned Severity_Tier at 1 Watch.
8. THE Flagger SHALL reduce an assigned tier by one level only after the Combined_Score is below that tier's threshold for 2 consecutive windows.

### Requirement 8: Apply movement check

**User Story:** As a data scientist, I want the Pipeline to check movement as a benign cause, so that exercise-driven shifts are not treated as the same signal as resting shifts.

#### Acceptance Criteria

1. WHEN the Accelerometer_Magnitude shows sustained movement across a flagged window, THE Flagger SHALL reduce the flag by one Severity_Tier and SHALL label the flag "explained by movement."
2. WHEN the Accelerometer_Magnitude shows rhythmic motion at a steady cadence across a flagged window, THE Flagger SHALL reduce the flag by one Severity_Tier and SHALL label the flag "explained by movement."
3. WHILE no sustained or rhythmic movement is present across a flagged window, THE Flagger SHALL leave the flag's Severity_Tier unchanged.

### Requirement 9: Explain each flag

**User Story:** As a user, I want each flag to name its top contributing signals in real units, so that I can understand what shifted.

#### Acceptance Criteria

1. WHEN a flag is fired, THE Explainer SHALL produce a text explanation that names the top contributing signals ranked by their z-score contribution.
2. WHEN a flag is fired, THE Explainer SHALL express each named signal's shift in that signal's real unit relative to the participant's rest baseline.
3. WHEN a flag is fired, THE Explainer SHALL include the Severity_Tier wording corresponding to the assigned tier.
4. WHEN a flag is fired, THE Explainer SHALL include the flag start time, the seconds the flag has persisted, the Combined_Score, and whether movement explains the flag.
5. THE Explainer SHALL produce explanation text that excludes diagnostic claims and color-coded severity labels.

### Requirement 10: Evaluate detection performance

**User Story:** As a data scientist, I want the Pipeline to measure detection quality against known events, so that the method's accuracy is reported rather than assumed.

#### Acceptance Criteria

1. THE Evaluator SHALL compute sensitivity as the share of stress-task phases that receive a flag at any tier, reported separately for long tasks and 30-second tasks.
2. THE Evaluator SHALL compute false alarms per hour of rest from flags occurring during baseline and rest segments, excluding the first 120 seconds after each task phase.
3. THE Evaluator SHALL compute detection delay as the number of seconds from a phase start to the first flag in that phase.
4. THE Evaluator SHALL compute tier agreement as the rank correlation between assigned Severity_Tier and self-reported stress, evaluated per phase.
5. THE Evaluator SHALL compute movement attribution as the share of exercise-session flags labeled "explained by movement" and the share of stress-session flags labeled "explained by movement."
6. THE Evaluator SHALL report each metric separately for Cohort_V1 and Cohort_V2.

### Requirement 11: Compare against a WHOOP-style baseline rule

**User Story:** As a data scientist, I want the Pipeline to compare combined flagging against a single-signal rule, so that the benefit of combining metrics is measured.

#### Acceptance Criteria

1. THE Evaluator SHALL compute every detection metric for the WHOOP_Rule single-signal out-of-range rule.
2. THE Evaluator SHALL compute every detection metric for the directional-sum Combined_Score.
3. THE Evaluator SHALL compute every detection metric for the Mahalanobis Combined_Score.
4. THE Evaluator SHALL report the detection metrics for the WHOOP_Rule, the directional-sum score, and the Mahalanobis score side by side.

### Requirement 12: Calibrate thresholds with leave-one-subject-out cross-validation

**User Story:** As a data scientist, I want the Pipeline to tune thresholds by cross-validation, so that thresholds are not hand-picked and generalize across participants.

#### Acceptance Criteria

1. THE Calibrator SHALL tune tau1, tau2, and tau3 using LOSO cross-validation.
2. THE Calibrator SHALL select tau1 so that false alarms during rest segments stay within the configured false-alarm budget.
3. THE Calibrator SHALL select tau2 and tau3 so that assigned tiers track self-reported stress and the aerobic test's increasing intensity stages.
4. THE Calibrator SHALL produce thresholds ordered such that tau1 is less than tau2 and tau2 is less than tau3.

### Requirement 13: Render the demo timeline

**User Story:** As a user, I want a per-person timeline view, so that I can see signals, score, and tiers together over time.

#### Acceptance Criteria

1. WHEN a participant is selected, THE Demo_App SHALL render the participant's signals, Combined_Score, Severity_Tier bands, and phase shading on a shared timeline.
2. WHEN a flag is present at a window, THE Demo_App SHALL display the flag's explanation for that window.

### Requirement 14: Export a flag in standard formats

**User Story:** As a developer, I want to export a high-severity flag in standard healthcare formats, so that a flag can be shared with downstream systems.

#### Acceptance Criteria

1. WHERE export is enabled and a flag is Severity_Tier 2 Notable or Severity_Tier 3 Strong, THE Exporter SHALL emit a FHIR Bundle containing one Observation per metric and one summary Observation for the flag.
2. WHERE export is enabled and a flag is Severity_Tier 2 Notable or Severity_Tier 3 Strong, THE Exporter SHALL emit a CMS care-update document populated with baseline and most-recent clinical measures and the flag explanation.
3. WHERE export is enabled, THE Exporter SHALL leave the care-update Assessment and Plan sections unpopulated.

### Requirement 15: Constrain claims and scope

**User Story:** As a product owner, I want the system to avoid medical claims, so that the system stays within its intended non-diagnostic scope.

#### Acceptance Criteria

1. THE Pipeline SHALL describe every flag as a sustained change rather than a diagnosis.
2. THE Pipeline SHALL present severity using neutral tier labels rather than color-coded ranges.
