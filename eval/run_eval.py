"""run_eval — end-to-end LOSO evaluation harness and headline figures.

This module is the *integration seam* of the Wearable Flag Pipeline: it wires
every component together and drives them against the PhysioNet protocol as the
validation oracle, then writes the project's two headline figures to ``output/``
(design "run_eval — eval/run_eval.py" and "Ground-truth-driven validation (LOSO
evaluation harness)"). Nothing new is computed here — the harness only *orches-
trates* the existing modules:

    Data_Loader  (data/physionet_load.py: load_participant, cohort_of)
      -> Cleaner (data/clean.py: load_constraints, clean)
      -> Windower (data/windows.py: protocol_key, assign_phases, window)
      -> Baseline_Estimator (flag/baseline.py: person_baseline, shrink_to_cohort)
      -> Scorer (flag/score.py: combined_score, window_signed_z, fit_ledoit_wolf)
      -> Flagger (flag/tiers.py: is_single_signal, via eval.metrics.tiers_from_scores)
      -> Movement_Check (flag/context.py: movement_label, apply_movement)
      -> Explainer (flag/explain.py: explain)
      -> Evaluator (eval/metrics.py: evaluate, compare_methods, tiers_from_scores,
                    whoop_window_scores, METHOD_WHOOP)
      -> Calibrator (eval/calibrate.py: calibrate_loso_folds, loso_folds, subjects_of)

Leave-one-subject-out (LOSO)
----------------------------
For each method (``WHOOP_Rule``, directional-sum, Mahalanobis) the harness runs
LOSO: every subject is held out exactly once; thresholds ``tau1 < tau2 < tau3``
are calibrated on the *other* subjects only (``calibrate_loso_folds``), and the
held-out subject is then flagged with those thresholds (score ->
``tiers_from_scores`` -> movement adjust). Per-window tiers for every held-out
subject are accumulated into one combined per-window tier vector per method, so
no subject is ever scored with thresholds tuned on itself (design: "LOSO
guarantees thresholds are never tuned on the subject they are evaluated on").
The accumulated tiers are then handed to :func:`eval.metrics.compare_methods`,
which runs :func:`eval.metrics.evaluate` per method *per cohort* (AC 10.6) and
returns the three methods side by side (AC 11.4).

Headline figures (written to ``output/``)
------------------------------------------
1. **False alarms per hour of rest vs detection delay** — one point per method
   per cohort, annotated with the method label. This is the project's primary
   trade-off figure (``output/false_alarms_vs_delay.png``).
2. **Tier vs self-reported stress** — per-phase assigned tier against the phase's
   self-report rating, per method (``output/tier_vs_self_report.png``).

The per-cohort metric tables are also written as CSV
(``output/metrics_by_method_cohort.csv``) so the numbers behind the figures are
inspectable. matplotlib runs on the non-interactive ``Agg`` backend so figures
are written headlessly.

Real data vs synthetic verification
-----------------------------------
The real Empatica E4 session CSVs live under ``data/sessions/...`` and the
reference files (``data_constraints.txt``, ``Stress_Level_v*.csv``) under
``data/external``. That tree is gitignored, so the harness is written to run
whether or not the recordings are present: :func:`main` loads whatever
participants it finds and logs/skips gracefully when a session folder is empty or
absent. To make the whole wiring verifiable *without* the real data, the harness
also accepts a pre-built list of :class:`~data.physionet_load.ParticipantSignals`
(via :func:`run_from_participants`) and ships :func:`simulate_participant`, a
small deterministic synthetic-signal generator used by the smoke test. Real data
simply drops into ``data/sessions`` and ``main`` picks it up.

A known real-data caveat (not in scope here, Task 1 territory): the shipped
``data/physionet_load.load_signal`` / ``load_tags`` parse the E4 header's first
row as an epoch-second float, but the on-disk files store a human-readable
datetime string. The synthetic path builds :class:`ParticipantSignals` with
epoch-second timing directly, so the end-to-end wiring is exercised regardless;
running on the real CSVs additionally depends on the loader handling that
datetime header, which this task does not modify.

Requirements covered:
- 10.6 Per-cohort aggregation of every detection metric (via
       :func:`eval.metrics.evaluate` / :func:`eval.metrics.compare_methods`).
- 11.4 WHOOP_Rule, directional-sum and Mahalanobis reported side by side
       (:func:`run_loso`, :func:`compare_methods`).
- 12.1 LOSO calibration — each subject held out once, thresholds tuned on the
       rest (:func:`run_loso` via :func:`eval.calibrate.calibrate_loso_folds`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# Headless, non-interactive backend: figures are written to disk, never shown.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

from data.clean import Constraints, clean, load_constraints
from data.physionet_load import (
    ParticipantSignals,
    SignalSeries,
    IBISeries,
    cohort_of,
    load_participant,
)
from data.schema import empty_per_window_table
from data.windows import (
    SESSION_AEROBIC,
    SESSION_ANAEROBIC,
    SESSION_STRESS,
    STEP_S,
    assign_phases,
    needs_external_baseline,
    protocol_key,
    window,
)
from flag.baseline import SCORED_SIGNALS, person_baseline, shrink_to_cohort
from flag.context import apply_movement, movement_label
from flag.score import (
    METHOD_DIRECTIONAL_SUM,
    METHOD_MAHALANOBIS,
    combined_score,
    fit_ledoit_wolf,
    window_signed_z,
)
from flag.tiers import is_single_signal
from eval.labels import load_stress_levels, phase_labels
from eval.metrics import (
    METHOD_WHOOP,
    ComparisonTable,
    compare_methods,
    tiers_from_scores,
    whoop_window_scores,
)
from eval.calibrate import calibrate_loso_folds, subjects_of

__all__ = [
    "DEFAULT_SESSIONS",
    "DEFAULT_BUDGET",
    "ACC_SAMPLE_RATE_HZ",
    "FIG_FALSE_ALARMS_VS_DELAY",
    "FIG_TIER_VS_SELF_REPORT",
    "CSV_METRICS",
    "BuiltTable",
    "LOSOResult",
    "EvalResult",
    "build_combined_table",
    "method_scores",
    "movement_flags",
    "run_loso",
    "run_eval",
    "run_from_participants",
    "simulate_participant",
    "write_headline_figures",
    "main",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sessions the harness builds for each participant (the Windower has a protocol
# timing table for each). f14..f18 are stress-only; the Cleaner drops their
# non-stress sessions (AC 2.7), so requesting all three here is safe.
DEFAULT_SESSIONS: tuple[str, ...] = (SESSION_STRESS, SESSION_AEROBIC, SESSION_ANAEROBIC)

# Default rest false-alarm budget (flags per hour of rest) used to calibrate
# tau1 (AC 12.2; README "<= 1 flag per hour of rest").
DEFAULT_BUDGET: float = 1.0

# E4 accelerometer sample rate (Hz), passed to the Movement_Check's cadence
# detection when raw ACC samples are available (data README).
ACC_SAMPLE_RATE_HZ: float = 32.0

# Figure / CSV filenames written under output/.
FIG_FALSE_ALARMS_VS_DELAY: str = "false_alarms_vs_delay.png"
FIG_TIER_VS_SELF_REPORT: str = "tier_vs_self_report.png"
CSV_METRICS: str = "metrics_by_method_cohort.csv"


# ---------------------------------------------------------------------------
# Combined table + baselines (step 1 of the design's LOSO oracle)
# ---------------------------------------------------------------------------


@dataclass
class BuiltTable:
    """The combined Per_Window_Table and the per-person baselines it was scored with.

    ``table`` is one concatenated :class:`~pandas.DataFrame` across all clean
    participants/sessions (one row per person per 10-second window, in a stable
    per-(person, session) block order). ``baselines`` maps ``person_id -> {signal
    -> SignalBaseline}`` (the cohort-shrunk resting baselines), and ``cohort``
    maps ``person_id -> "V1"/"V2"``. ``acc_windows`` optionally carries, per table
    row, the raw per-window accelerometer magnitude samples the Movement_Check
    needs for cadence detection; it is ``None`` for rows with no raw ACC.
    """

    table: pd.DataFrame
    baselines: dict[str, dict[str, object]]
    cohort: dict[str, str]
    acc_windows: list[np.ndarray | None] = field(default_factory=list)


def _session_table(signals: ParticipantSignals) -> pd.DataFrame:
    """Window one cleaned participant/session into its Per_Window_Table block.

    Assigns protocol phases from the participant's tags
    (:func:`data.windows.assign_phases`) and windows the signals
    (:func:`data.windows.window`). A participant with no usable tags/timeline
    yields an empty table block (the Windower handles the empty case).
    """
    tags = list(signals.tags or [])
    key = protocol_key(signals.person_id, signals.session)
    try:
        phases = assign_phases(tags, key)
    except KeyError:
        # Unknown (session, version) — no protocol timing table; skip phasing.
        phases = []
    return window(signals, phases)


def _acc_window_samples(
    signals: ParticipantSignals, t_wins: Sequence[float]
) -> list[np.ndarray | None]:
    """Per-window raw ACC magnitude samples for the Movement_Check (trailing 30 s).

    For each window end time ``t`` in ``t_wins``, slice the participant's raw ACC
    samples to the trailing ``(t-30, t]`` interval and reduce to per-sample vector
    magnitude ``sqrt(x^2+y^2+z^2)`` — exactly the samples
    :func:`flag.context.movement_label` expects for cadence detection. Returns
    ``None`` for a window when ACC is absent (the ``Per_Window_Table`` ``acc_mag``
    aggregate is a single number per window, so the Movement_Check genuinely needs
    these underlying samples; see ``flag/context.py``).
    """
    acc = signals.acc
    if not isinstance(acc, SignalSeries) or acc.values.size == 0:
        return [None] * len(t_wins)
    values = np.asarray(acc.values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        return [None] * len(t_wins)
    rate = float(acc.sample_rate_hz)
    if not np.isfinite(rate) or rate <= 0:
        return [None] * len(t_wins)
    mags = np.sqrt(np.sum(values * values, axis=1))
    times = acc.start_utc + np.arange(values.shape[0], dtype=float) / rate
    span = 30.0  # SHORT_WINDOW_S; HR/TEMP/ACC trailing interval
    out: list[np.ndarray | None] = []
    for t in t_wins:
        mask = (times > (t - span)) & (times <= t)
        out.append(mags[mask] if mask.any() else None)
    return out


def build_combined_table(
    participants: Mapping[str, Mapping[str, ParticipantSignals]],
) -> BuiltTable:
    """Build one combined Per_Window_Table across all clean participants (design step 1).

    ``participants`` maps ``person_id -> {session -> cleaned ParticipantSignals}``
    (already run through the Cleaner). For each participant/session the harness
    windows the signals into a Per_Window_Table block and concatenates the blocks,
    in a stable ``(person_id, session)`` order, into one table.

    Per-person baselines are then estimated (:func:`flag.baseline.person_baseline`)
    and cohort-shrunk (:func:`flag.baseline.shrink_to_cohort`). The V1 exercise
    external-baseline case (AC 3.2 / 5.4) is resolved here: when a V1 exercise
    session has no quiet baseline of its own
    (:func:`data.windows.needs_external_baseline`), the participant's
    **stress-session** table is passed as the external baseline source so the
    exercise baseline is read from the stress rest segment.

    Returns a :class:`BuiltTable` with the table, the shrunk baselines keyed per
    person (keyed by ``person_id`` — one baseline set per participant, taken from
    their baseline-bearing session), the cohort map, and the per-row raw ACC
    windows for the Movement_Check.
    """
    # Stable order: sort persons, and sessions in the canonical pipeline order.
    session_order = {s: i for i, s in enumerate(DEFAULT_SESSIONS)}

    blocks: list[pd.DataFrame] = []
    acc_windows: list[np.ndarray | None] = []
    # Per person: the stress table (external-baseline source) and per-session tables.
    stress_table_by_pid: dict[str, pd.DataFrame] = {}
    per_person_session_tables: dict[str, dict[str, pd.DataFrame]] = {}

    ordered_pids = sorted(participants)
    # First pass: window every session so stress tables are available for the
    # V1-exercise external baseline before we compute exercise baselines.
    for pid in ordered_pids:
        sessions = participants[pid]
        per_person_session_tables[pid] = {}
        for session in sorted(sessions, key=lambda s: session_order.get(s.upper(), 99)):
            sig = sessions[session]
            block = _session_table(sig)
            per_person_session_tables[pid][sig.session.upper()] = block
            if sig.session.upper() == SESSION_STRESS:
                stress_table_by_pid[pid] = block

    # Second pass: concatenate in stable order and compute raw ACC windows.
    for pid in ordered_pids:
        sessions = participants[pid]
        for session in sorted(sessions, key=lambda s: session_order.get(s.upper(), 99)):
            sig = sessions[session]
            block = per_person_session_tables[pid][sig.session.upper()]
            if len(block):
                blocks.append(block)
                acc_windows.extend(
                    _acc_window_samples(sig, block["t_win"].to_numpy(dtype=float))
                )

    if blocks:
        table = pd.concat(blocks, ignore_index=True)
    else:
        table = empty_per_window_table()

    # Per-person cohort-shrunk baselines.
    raw_baselines: dict[str, dict[str, object]] = {}
    counts: dict[str, dict[str, float]] = {}
    cohort: dict[str, str] = {}

    for pid in ordered_pids:
        try:
            cohort[pid] = cohort_of(pid)
        except ValueError:
            continue
        session_tables = per_person_session_tables[pid]
        # Prefer a baseline-bearing session: stress always has a quiet baseline.
        pid_baselines: dict[str, object] = {}
        for session, block in session_tables.items():
            if not len(block):
                continue
            needs_external = needs_external_baseline(protocol_key(pid, session))
            if needs_external:
                ext = stress_table_by_pid.get(pid)
                if ext is None or not len(ext):
                    continue
                try:
                    b = person_baseline(
                        block,
                        pid,
                        session=session,
                        external_table=ext,
                        external_pid=pid,
                    )
                except KeyError:
                    continue
            else:
                b = person_baseline(block, pid, session=session)
            # Merge: keep the first (stress) baseline per signal; it is the
            # canonical resting normal reused across that person's sessions.
            for signal, baseline in b.items():
                pid_baselines.setdefault(signal, baseline)
        if pid_baselines:
            raw_baselines[pid] = pid_baselines
            counts[pid] = {
                signal: float(_baseline_count(session_tables, signal))
                for signal in pid_baselines
            }

    shrunk = shrink_to_cohort(raw_baselines, cohort, counts=counts)

    return BuiltTable(
        table=table,
        baselines={pid: dict(sig) for pid, sig in shrunk.items()},
        cohort=cohort,
        acc_windows=acc_windows,
    )


def _baseline_count(session_tables: Mapping[str, pd.DataFrame], signal: str) -> int:
    """Rough per-signal baseline sample count across a person's sessions.

    Used only to weight the empirical-Bayes shrinkage (``shrink_to_cohort``):
    counts the finite values of ``signal`` in the baseline-phase rows available
    for the person. A coarse but monotone proxy — more rest windows => keep more
    of the person's own spread.
    """
    total = 0
    for block in session_tables.values():
        if signal in block.columns and "phase" in block.columns:
            col = block.loc[block["phase"] == "Baseline", signal]
            total += int(np.isfinite(col.to_numpy(dtype=float)).sum())
    return total


# ---------------------------------------------------------------------------
# Per-method per-window scores and movement flags
# ---------------------------------------------------------------------------


def method_scores(
    table: pd.DataFrame,
    baselines: Mapping[str, Mapping[str, object]],
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window scores and single-signal-cap flags for one method.

    Returns ``(scores, single_signal)`` aligned by table row index:

    - ``WHOOP_Rule`` (:data:`~eval.metrics.METHOD_WHOOP`): the single-signal
      out-of-range magnitude per window (:func:`eval.metrics.whoop_window_scores`);
      ``single_signal`` is all-``True`` (one contributing signal by construction,
      capped at Watch, AC 11.1 / 7.7).
    - ``directional_sum`` / ``mahalanobis``: the combined
      :func:`flag.score.combined_score` per window with the matching method, and
      the per-window :func:`flag.tiers.is_single_signal` cap flag from
      :func:`flag.score.window_signed_z` (AC 11.2, 11.3, 7.7). For Mahalanobis a
      Ledoit-Wolf covariance is fit once over the full available-signal z-matrix
      (:func:`flag.score.fit_ledoit_wolf`) and reused for every window.
    """
    n = len(table)
    if n == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=bool)

    if method == METHOD_WHOOP:
        scores = whoop_window_scores(table, baselines)
        return scores, np.ones(n, dtype=bool)

    if method not in (METHOD_DIRECTIONAL_SUM, METHOD_MAHALANOBIS):
        raise ValueError(f"unknown method {method!r}")

    # Per-window signed-z maps (used for both the single-signal cap and the
    # Mahalanobis covariance fit).
    z_maps: list[dict[str, float]] = []
    for pos in range(n):
        row = table.iloc[pos]
        pid = row["person_id"]
        z_maps.append(window_signed_z(row, baselines.get(pid, {})))

    single_signal = np.array([is_single_signal(z) for z in z_maps], dtype=bool)

    cov = None
    if method == METHOD_MAHALANOBIS:
        cov = _fit_full_cov(z_maps)

    scores = np.zeros(n, dtype=float)
    for pos in range(n):
        row = table.iloc[pos]
        pid = row["person_id"]
        scores[pos] = combined_score(
            row, baselines.get(pid, {}), method=method, cov=cov
        )
    return scores, single_signal


def _fit_full_cov(z_maps: Sequence[Mapping[str, float]]):
    """Fit one Ledoit-Wolf covariance over the full-signal z-vectors (AC 6.6).

    Builds the ``(n_windows, len(SCORED_SIGNALS))`` z-matrix from the windows that
    have *all* scored signals available (so the covariance is over a consistent
    dimensionality), and fits :func:`flag.score.fit_ledoit_wolf`. Returns ``None``
    when too few complete rows exist to estimate a covariance — the scorer then
    falls back to a Euclidean norm in z-space, still a valid non-negative
    distance.
    """
    rows = []
    for z in z_maps:
        if all(s in z and np.isfinite(z[s]) for s in SCORED_SIGNALS):
            rows.append([float(z[s]) for s in SCORED_SIGNALS])
    if len(rows) < 2:
        return None
    return fit_ledoit_wolf(np.asarray(rows, dtype=float))


def movement_flags(
    table: pd.DataFrame,
    tiers: np.ndarray,
    acc_windows: Sequence[np.ndarray | None] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the Movement_Check to per-window tiers (AC 8.1-8.3).

    For each window, label its raw ACC samples
    (:func:`flag.context.movement_label`) and adjust the pre-movement ``tier``
    with :func:`flag.context.apply_movement` — a sustained/rhythmic window drops
    one tier (floored at None) and is marked "explained by movement". Returns
    ``(adjusted_tiers, movement_explained)`` aligned by row index.

    When no raw ACC samples are available for a row (``acc_windows`` is ``None`` or
    the entry is ``None`` — e.g. only the ``Per_Window_Table`` ``acc_mag``
    aggregate exists), the Movement_Check cannot run for that window, so the tier
    is left unchanged and ``movement_explained`` is ``False`` for it. This is the
    documented limitation of running without raw per-window ACC: cadence detection
    needs the underlying samples, not the single trailing-30 s aggregate.
    """
    n = len(table)
    adjusted = np.asarray(tiers, dtype=int).copy()
    explained = np.zeros(n, dtype=bool)
    if acc_windows is None:
        return adjusted, explained

    for pos in range(n):
        samples = acc_windows[pos] if pos < len(acc_windows) else None
        if samples is None:
            continue
        mv = movement_label(samples, sample_rate_hz=ACC_SAMPLE_RATE_HZ)
        new_tier, moved = apply_movement(int(adjusted[pos]), mv)
        adjusted[pos] = new_tier
        explained[pos] = moved
    return adjusted, explained


# ---------------------------------------------------------------------------
# LOSO per method (design step 2)
# ---------------------------------------------------------------------------


@dataclass
class LOSOResult:
    """Accumulated per-window flag result for one method after LOSO.

    ``tiers`` and ``movement`` are aligned by combined-table row index: each
    held-out subject's rows carry the tier assigned by thresholds calibrated on
    the *other* subjects. Rows for subjects that were never a valid held-out fold
    (e.g. a single-subject dataset) stay at Tier 0. ``fold_thresholds`` records
    the ``held_out -> Thresholds`` chosen per fold for inspection.
    """

    method: str
    tiers: np.ndarray
    movement: np.ndarray
    fold_thresholds: dict[str, object] = field(default_factory=dict)


def run_loso(
    built: BuiltTable,
    labels,
    self_report: Mapping[tuple[str, str, int], float] | None,
    method: str,
    *,
    budget: float = DEFAULT_BUDGET,
) -> LOSOResult:
    """Run leave-one-subject-out for one method (AC 12.1, design step 2).

    Computes the method's per-window scores (:func:`method_scores`), then for each
    LOSO fold calibrates thresholds on the fold's training subjects
    (:func:`eval.calibrate.calibrate_loso_folds`) and tiers the **held-out**
    subject's windows with those thresholds (:func:`eval.metrics.tiers_from_scores`,
    the exact production Flagger). The held-out tiers are movement-adjusted
    (:func:`movement_flags`) and written into the combined per-window tier vector
    at the held-out subject's row positions, so the accumulated result holds every
    subject scored by thresholds never tuned on itself.

    Returns a :class:`LOSOResult` with the accumulated per-window tiers and
    movement flags for the method.
    """
    table = built.table
    n = len(table)
    scores, single_signal = method_scores(table, built.baselines, method)

    tiers = np.zeros(n, dtype=int)
    movement = np.zeros(n, dtype=bool)
    fold_thresholds: dict[str, object] = {}

    if n == 0 or not subjects_of(labels):
        return LOSOResult(method=method, tiers=tiers, movement=movement)

    # Per-window stream keys so the Flagger resets its rolling history at each
    # person/session boundary (never bleeds across participants).
    stream_ids = [(w.person_id, w.session) for w in labels.windows]
    person_ids = np.array([w.person_id for w in labels.windows], dtype=object)

    folds = calibrate_loso_folds(
        scores,
        labels,
        self_report,
        budget,
        single_signal=single_signal,
    )

    for fold, taus in folds:
        fold_thresholds[fold.held_out] = taus
        held_mask = person_ids == fold.held_out
        if not held_mask.any():
            continue
        # Tier the whole table with the fold thresholds, then keep only the
        # held-out subject's rows (the Flagger resets per stream, so a row's tier
        # depends only on its own stream's scores — slicing after is correct).
        all_tiers = tiers_from_scores(
            scores,
            taus,
            stream_ids=stream_ids,
            single_signal=single_signal,
        )
        held_idx = np.nonzero(held_mask)[0]
        adj_tiers, adj_moved = movement_flags(
            table.iloc[held_idx],
            all_tiers[held_idx],
            [built.acc_windows[i] if i < len(built.acc_windows) else None for i in held_idx],
        )
        tiers[held_idx] = adj_tiers
        movement[held_idx] = adj_moved

    return LOSOResult(
        method=method,
        tiers=tiers,
        movement=movement,
        fold_thresholds=fold_thresholds,
    )


# ---------------------------------------------------------------------------
# Full evaluation (design steps 2-4)
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """The complete LOSO evaluation outcome for the three methods.

    ``comparison`` is the :class:`~eval.metrics.ComparisonTable` holding each
    method's per-cohort :class:`~eval.metrics.MetricTable` (AC 10.6, 11.4).
    ``loso`` maps each method label to its accumulated :class:`LOSOResult`.
    ``built`` is the combined table / baselines the evaluation ran on, and
    ``labels`` the phase labels, both kept so the figure writer can reach the
    per-phase self-report and segment timing.
    """

    comparison: ComparisonTable
    loso: dict[str, LOSOResult]
    built: BuiltTable
    labels: object
    self_report: Mapping[tuple[str, str, int], float]


def run_eval(
    built: BuiltTable,
    self_report: Mapping[tuple[str, str, int], float] | None = None,
    *,
    budget: float = DEFAULT_BUDGET,
    methods: Sequence[str] = (METHOD_WHOOP, METHOD_DIRECTIONAL_SUM, METHOD_MAHALANOBIS),
) -> EvalResult:
    """Run LOSO for every method and compare them per cohort (AC 10.6, 11.4, 12.1).

    Derives phase labels for the combined table (:func:`eval.labels.phase_labels`),
    runs :func:`run_loso` per method to accumulate held-out per-window tiers, then
    hands the three tier/movement vectors to :func:`eval.metrics.compare_methods`,
    which evaluates every detection metric per method per cohort. Returns an
    :class:`EvalResult` carrying the comparison, the per-method LOSO results, and
    the built table / labels for figure writing.
    """
    sr: Mapping[tuple[str, str, int], float] = self_report or {}
    labels = phase_labels(built.table)

    loso: dict[str, LOSOResult] = {}
    tiers_by_method: dict[str, object] = {}
    for method in methods:
        result = run_loso(built, labels, sr, method, budget=budget)
        loso[method] = result
        tiers_by_method[method] = {
            "tier": result.tiers,
            "movement_explained": result.movement,
        }

    comparison = compare_methods(tiers_by_method, labels, sr)

    return EvalResult(
        comparison=comparison,
        loso=loso,
        built=built,
        labels=labels,
        self_report=sr,
    )


# ---------------------------------------------------------------------------
# Headline figures (design step 4)
# ---------------------------------------------------------------------------


def _metrics_dataframe(comparison: ComparisonTable) -> pd.DataFrame:
    """Flatten a ComparisonTable into a tidy per-method/per-cohort metric frame."""
    rows: list[dict[str, object]] = []
    for method in comparison.methods():
        table = comparison.by_method[method]
        for cohort in table.cohorts():
            cm = table.by_cohort[cohort]
            rows.append(
                {
                    "method": method,
                    "cohort": cohort,
                    "sensitivity_long": cm.sensitivity.long_sensitivity,
                    "sensitivity_short": cm.sensitivity.short_sensitivity,
                    "sensitivity_overall": cm.sensitivity.overall_sensitivity,
                    "false_alarms_per_rest_hour": cm.false_alarms.per_hour,
                    "false_alarms": cm.false_alarms.false_alarms,
                    "rest_hours": cm.false_alarms.rest_hours,
                    "detection_delay_s": cm.detection_delay.overall_mean,
                    "tier_agreement": cm.tier_agreement,
                    "movement_share_exercise": cm.movement_attribution.exercise_share,
                    "movement_share_stress": cm.movement_attribution.stress_share,
                    "n_windows": cm.n_windows,
                    "n_segments": cm.n_segments,
                }
            )
    return pd.DataFrame(rows)


def _plot_false_alarms_vs_delay(df: pd.DataFrame, path: str) -> None:
    """Headline figure 1: false alarms per rest hour vs detection delay (AC 11.4)."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    markers = {"V1": "o", "V2": "s"}
    colors = {
        METHOD_WHOOP: "#888888",
        METHOD_DIRECTIONAL_SUM: "#1f77b4",
        METHOD_MAHALANOBIS: "#d62728",
    }
    for _, r in df.iterrows():
        x = r["false_alarms_per_rest_hour"]
        y = r["detection_delay_s"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ax.scatter(
            x,
            y,
            marker=markers.get(str(r["cohort"]), "x"),
            color=colors.get(str(r["method"]), "black"),
            s=70,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            f"{r['method']}/{r['cohort']}",
            (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("False alarms per hour of rest")
    ax.set_ylabel("Detection delay (s)")
    ax.set_title("False alarms vs detection delay by method and cohort")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_tier_vs_self_report(result: EvalResult, path: str) -> None:
    """Headline figure 2: per-phase assigned tier vs self-reported stress (AC 10.4)."""
    from eval.labels import attach_self_report
    from eval.metrics import segment_max_tier

    labels = result.labels
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    colors = {
        METHOD_WHOOP: "#888888",
        METHOD_DIRECTIONAL_SUM: "#1f77b4",
        METHOD_MAHALANOBIS: "#d62728",
    }
    jitter = {METHOD_WHOOP: -0.12, METHOD_DIRECTIONAL_SUM: 0.0, METHOD_MAHALANOBIS: 0.12}

    plotted_any = False
    for method, loso in result.loso.items():
        joined = attach_self_report(labels.segments, result.self_report)
        xs: list[float] = []
        ys: list[float] = []
        for seg, rating in joined:
            if rating is None or (isinstance(rating, float) and np.isnan(rating)):
                continue
            xs.append(float(rating) + jitter.get(method, 0.0))
            ys.append(float(segment_max_tier(seg, loso.tiers)))
        if xs:
            plotted_any = True
            ax.scatter(
                xs,
                ys,
                color=colors.get(method, "black"),
                label=method,
                alpha=0.7,
                s=40,
                edgecolor="none",
            )
    ax.set_xlabel("Self-reported stress (0-10)")
    ax.set_ylabel("Assigned tier (max over phase)")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_title("Tier vs self-reported stress, per phase")
    if plotted_any:
        ax.legend(title="method", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_headline_figures(result: EvalResult, output_dir: str) -> list[str]:
    """Write the two headline figures and the metric CSV to ``output_dir`` (step 4).

    Produces, under ``output_dir`` (created if needed, used as a directory — the
    ``output/`` package dir is reused, never overwritten as a file):

    - :data:`FIG_FALSE_ALARMS_VS_DELAY` — false alarms per rest hour vs detection
      delay, one point per method per cohort (AC 11.4).
    - :data:`FIG_TIER_VS_SELF_REPORT` — per-phase tier vs self-reported stress,
      per method (AC 10.4).
    - :data:`CSV_METRICS` — the tidy per-method/per-cohort metric table (AC 10.6).

    Returns the list of written file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = _metrics_dataframe(result.comparison)

    csv_path = os.path.join(output_dir, CSV_METRICS)
    df.to_csv(csv_path, index=False)

    fa_path = os.path.join(output_dir, FIG_FALSE_ALARMS_VS_DELAY)
    _plot_false_alarms_vs_delay(df, fa_path)

    tier_path = os.path.join(output_dir, FIG_TIER_VS_SELF_REPORT)
    _plot_tier_vs_self_report(result, tier_path)

    return [csv_path, fa_path, tier_path]


# ---------------------------------------------------------------------------
# Loading + cleaning the real data (robust to a missing data/sessions tree)
# ---------------------------------------------------------------------------


def _sessions_root(data_root: str) -> str:
    """Locate the Wearable_Dataset session root under ``data_root`` if present."""
    candidate = os.path.join(
        data_root,
        "sessions",
        "Wearable_Dataset_sessions",
        "Wearable_Dataset",
    )
    return candidate


def _discover_participants(
    data_root: str,
    sessions: Sequence[str],
    constraints: Constraints,
) -> dict[str, dict[str, ParticipantSignals]]:
    """Load and clean every participant/session found under ``data_root``.

    Walks each requested session folder under the sessions root, loads each
    participant with :func:`data.physionet_load.load_participant`, and applies the
    Cleaner (:func:`data.clean.clean`). Split ``_a``/``_b`` folders (f14, S11,
    S16) are joined by passing the ``_b`` segment to the Cleaner. A session that
    the Cleaner drops (``None`` — e.g. f14..f18 non-stress, AC 2.7) is skipped.
    Returns ``{person_id -> {session -> cleaned ParticipantSignals}}``; an empty
    dict when no recordings are present (the harness then logs and exits cleanly).
    """
    root = _sessions_root(data_root)
    out: dict[str, dict[str, ParticipantSignals]] = {}
    if not os.path.isdir(root):
        return out

    for session in sessions:
        session_dir = os.path.join(root, session)
        if not os.path.isdir(session_dir):
            continue
        entries = sorted(os.listdir(session_dir))
        # Group split segments: "S11_a" / "S11_b" -> base pid "S11".
        base_pids: dict[str, dict[str, str]] = {}
        for name in entries:
            full = os.path.join(session_dir, name)
            if not os.path.isdir(full):
                continue
            if name.endswith("_a") or name.endswith("_b"):
                base, suffix = name[:-2], name[-1]
                base_pids.setdefault(base, {})[suffix] = name
            else:
                base_pids.setdefault(name, {})["whole"] = name

        for pid, parts in base_pids.items():
            try:
                cohort_of(pid)
            except ValueError:
                continue
            try:
                if "whole" in parts:
                    loaded = load_participant(session_dir, parts["whole"])
                    segment_b = None
                else:
                    a_name = parts.get("a")
                    b_name = parts.get("b")
                    if a_name is None:
                        continue
                    loaded = load_participant(session_dir, a_name)
                    loaded.person_id = pid  # normalize off the "_a" suffix
                    segment_b = (
                        load_participant(session_dir, b_name) if b_name else None
                    )
            except (ValueError, OSError) as exc:  # malformed/absent files
                print(f"  skip {pid}/{session}: load failed ({exc})")
                continue

            try:
                cleaned = clean(loaded, constraints, segment_b=segment_b)
            except Exception as exc:  # defensive: a bad constraint row
                print(f"  skip {pid}/{session}: clean failed ({exc})")
                continue
            if cleaned is None:
                continue  # dropped by stress-only restriction (AC 2.7)
            out.setdefault(pid, {})[cleaned.session.upper()] = cleaned

    return out


def run_from_participants(
    participants: Mapping[str, Mapping[str, ParticipantSignals]],
    self_report: Mapping[tuple[str, str, int], float] | None = None,
    output_dir: str | None = None,
    *,
    budget: float = DEFAULT_BUDGET,
) -> EvalResult:
    """Build the combined table from ``participants`` and run the full evaluation.

    The in-memory entry point: given already-cleaned
    ``{person_id -> {session -> ParticipantSignals}}`` (real or synthetic), build
    the combined Per_Window_Table (:func:`build_combined_table`), run LOSO per
    method (:func:`run_eval`), and — when ``output_dir`` is given — write the
    headline figures (:func:`write_headline_figures`). Returns the
    :class:`EvalResult`.
    """
    built = build_combined_table(participants)
    result = run_eval(built, self_report, budget=budget)
    if output_dir is not None:
        written = write_headline_figures(result, output_dir)
        for p in written:
            print(f"  wrote {p}")
    return result


# ---------------------------------------------------------------------------
# Synthetic signal generator (for verification without the real data)
# ---------------------------------------------------------------------------


def simulate_participant(
    pid: str,
    session: str,
    *,
    seed: int = 0,
    start_utc: float = 1_000_000.0,
    stressed: bool = True,
) -> ParticipantSignals:
    """Generate a small deterministic synthetic :class:`ParticipantSignals`.

    Builds a short, protocol-shaped recording with epoch-second timing so the
    whole harness (windowing, baselines, scoring, flagging, movement, evaluation,
    calibration) can be exercised end to end without the real CSVs. It is *not* a
    physiological model — just enough structure for the pipeline to produce a
    non-trivial table: a quiet baseline segment followed by task/exercise segments
    where HR rises and (for stress) RMSSD drops, with accelerometer motion during
    exercise sessions so the Movement_Check has something to attribute.

    The returned record carries HR (1 Hz), TEMP (4 Hz), ACC (32 Hz, ``(N, 3)``),
    IBI, and tags (epoch-second phase boundaries matching the session's protocol
    timing table), ready to pass straight to the Cleaner / Windower. Deterministic
    for a given ``(pid, session, seed)``.
    """
    rng = np.random.default_rng(abs(hash((pid, session, seed))) % (2**32))
    cohort = cohort_of(pid)
    session = session.upper()

    # Phase layout: a Baseline then a few task/exercise phases. Keep it short but
    # longer than the 60 s baseline warm-up exclusion + a few windows.
    if session == SESSION_STRESS:
        # Baseline, task, rest, task (durations in seconds).
        phases = [("Baseline", 200.0), ("task", 120.0), ("Rest", 150.0), ("task2", 90.0)]
    else:
        phases = [("Baseline", 200.0), ("stage", 180.0), ("stage2", 180.0), ("Rest", 120.0)]

    boundaries = [start_utc]
    t = start_utc
    for _, dur in phases:
        t += dur
        boundaries.append(t)
    end_utc = boundaries[-1]
    duration = end_utc - start_utc

    def _phase_elevation(ti: float) -> float:
        """Return a 0..1 'stress elevation' for absolute time ``ti``."""
        for (name, _), b0, b1 in zip(phases, boundaries, boundaries[1:]):
            if b0 <= ti < b1:
                if name == "Baseline":
                    return 0.0
                if name in ("Rest",):
                    return 0.1
                return 1.0 if stressed else 0.6
        return 0.0

    # HR at 1 Hz.
    hr_rate = 1.0
    n_hr = int(duration * hr_rate)
    hr_t = start_utc + np.arange(n_hr) / hr_rate
    hr_base = 65.0
    hr_vals = np.array(
        [hr_base + 20.0 * _phase_elevation(ti) + rng.normal(0, 1.2) for ti in hr_t],
        dtype=float,
    )
    hr = SignalSeries(name="HR", start_utc=start_utc, sample_rate_hz=hr_rate, values=hr_vals)

    # TEMP at 4 Hz.
    temp_rate = 4.0
    n_temp = int(duration * temp_rate)
    temp_t = start_utc + np.arange(n_temp) / temp_rate
    temp_vals = np.array(
        [33.0 - 0.6 * _phase_elevation(ti) + rng.normal(0, 0.05) for ti in temp_t],
        dtype=float,
    )
    temp = SignalSeries(
        name="TEMP", start_utc=start_utc, sample_rate_hz=temp_rate, values=temp_vals
    )

    # ACC at 32 Hz: quiet at rest; for exercise sessions a rhythmic cadence during
    # the "stage" phases so the Movement_Check can attribute motion.
    acc_rate = 32.0
    n_acc = int(duration * acc_rate)
    acc_t = start_utc + np.arange(n_acc) / acc_rate
    is_exercise = session in (SESSION_AEROBIC, SESSION_ANAEROBIC)
    acc_rows = np.zeros((n_acc, 3), dtype=float)
    for i, ti in enumerate(acc_t):
        elev = _phase_elevation(ti)
        if is_exercise and elev >= 1.0 - 1e-9 and stressed is not None:
            # Rhythmic ~2 Hz cadence superimposed on gravity (counts in 1/64 g).
            beat = 25.0 * np.sin(2.0 * np.pi * 2.0 * (ti - start_utc))
            acc_rows[i] = [64.0 + beat + rng.normal(0, 2.0), rng.normal(0, 2.0), rng.normal(0, 2.0)]
        else:
            acc_rows[i] = [64.0 + rng.normal(0, 1.0), rng.normal(0, 1.0), rng.normal(0, 1.0)]
    acc = SignalSeries(
        name="ACC", start_utc=start_utc, sample_rate_hz=acc_rate, values=acc_rows
    )

    # IBI: inter-beat intervals consistent with HR (shorter when HR higher), with
    # reduced variability (lower RMSSD) under stress.
    ibi_offsets: list[float] = []
    beat_t = start_utc
    while beat_t < end_utc:
        elev = _phase_elevation(beat_t)
        mean_ibi = 60.0 / (hr_base + 20.0 * elev)
        hrv = 0.04 * (1.0 - 0.6 * elev)  # RMSSD shrinks under stress
        interval = max(0.3, mean_ibi + rng.normal(0, hrv))
        ibi_offsets.append(interval)
        beat_t += interval
    ibi = IBISeries(start_utc=start_utc, offsets_s=np.asarray(ibi_offsets, dtype=float))

    return ParticipantSignals(
        person_id=pid,
        cohort=cohort,
        session=session,
        hr=hr,
        temp=temp,
        acc=acc,
        ibi=ibi,
        tags=list(boundaries),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    data_root: str | None = None,
    output_dir: str | None = None,
    *,
    sessions: Sequence[str] = DEFAULT_SESSIONS,
    budget: float = DEFAULT_BUDGET,
) -> EvalResult | None:
    """Load the real data, run the full LOSO evaluation, write the figures.

    Resolves ``data_root`` (default: the repo's ``data/`` directory) and
    ``output_dir`` (default: the repo's ``output/`` directory), loads the
    constraints and self-report tables from ``<data_root>/external``, discovers
    and cleans every participant/session under ``<data_root>/sessions`` (robust to
    a missing/empty tree), builds the combined table, runs LOSO per method, and
    writes the headline figures.

    Returns the :class:`EvalResult`, or ``None`` when no participants could be
    loaded (logged and skipped gracefully — the real recordings are gitignored, so
    a fresh checkout has none; drop them into ``data/sessions`` to run for real).
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_root is None:
        data_root = os.path.join(repo_root, "data")
    if output_dir is None:
        output_dir = os.path.join(repo_root, "output")

    external = os.path.join(data_root, "external")
    constraints_path = os.path.join(external, "data_constraints.txt")
    constraints = (
        load_constraints(constraints_path)
        if os.path.isfile(constraints_path)
        else Constraints()
    )

    self_report: Mapping[tuple[str, str, int], float] = {}
    v1_path = os.path.join(external, "Stress_Level_v1.csv")
    v2_path = os.path.join(external, "Stress_Level_v2.csv")
    if os.path.isfile(v1_path) and os.path.isfile(v2_path):
        try:
            self_report = load_stress_levels(v1_path, v2_path)
        except Exception as exc:  # defensive: malformed reference tables
            print(f"warning: could not load self-report tables ({exc})")

    print(f"Discovering participants under {_sessions_root(data_root)} ...")
    participants = _discover_participants(data_root, sessions, constraints)
    n_pairs = sum(len(s) for s in participants.values())
    print(f"Loaded {len(participants)} participant(s), {n_pairs} participant/session block(s).")

    if not participants:
        print(
            "No session recordings found. The real PhysioNet CSVs are gitignored; "
            "drop them into data/sessions/Wearable_Dataset_sessions/Wearable_Dataset "
            "to run the harness on real data. Nothing to evaluate."
        )
        return None

    result = run_from_participants(
        participants, self_report, output_dir, budget=budget
    )
    print("Done. Metric summary:")
    print(_metrics_dataframe(result.comparison).to_string(index=False))
    return result


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
