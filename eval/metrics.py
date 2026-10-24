"""Evaluator — detection metrics per cohort (Requirement 10).

This module turns a set of per-window flag results into the detection-quality
metrics the project reports: sensitivity on stress tasks, false alarms per hour
of rest, detection delay, tier-vs-self-report agreement, and movement
attribution — every one of them reported separately for Cohort_V1 and
Cohort_V2 (design "Evaluator — eval/metrics.py", AC 10.6).

It consumes, never recomputes, the upstream pieces:

- the per-window *labels* and per-phase *segments* from :mod:`eval.labels`
  (Task 12.1) — rest vs task, long vs 30-second, segment timing, and the
  self-report join (:func:`eval.labels.attach_self_report`);
- a per-window *flag result* produced by the Flagger + Movement_Check
  (``flag/tiers.py`` + ``flag/context.py``): for each window, the assigned
  Severity_Tier (``0..3``) and whether the flag was "explained by movement".

The WHOOP_Rule vs directional-sum vs Mahalanobis comparison (Requirement 11) is
*not* here — that is Task 13's ``compare_methods``. :func:`evaluate` is written
so Task 13 can call it once per method: it takes a ``method`` label (carried
onto the result for side-by-side reporting) and a single set of per-window
tiers/movement labels, and returns a :class:`MetricTable` holding the five
metrics for each cohort.

Flag-result input shape
-----------------------
The caller provides the per-window flag outcome **aligned by table row index**
to the ``Per_Window_Table`` that produced ``labels``. Any of these forms is
accepted by :func:`as_flag_frame` and the metric functions:

- a :class:`pandas.DataFrame` with columns ``tier`` (int ``0..3``) and
  ``movement_explained`` (bool), one row per window in table order. An optional
  ``session`` column is used by movement attribution; if absent, the session is
  taken from the labels;
- a mapping ``{"tier": array-like, "movement_explained": array-like}``;
- a pair/sequence of two array-likes ``(tier, movement_explained)``.

Row *i* of the flag input describes the window at ``labels.windows[i]`` /
positional row *i* of the table. A window is **flagged** when its ``tier >= 1``
(Tier 1 Watch or above); Tier 0 None is not a flag. This "any tier" rule is what
sensitivity (AC 10.1) and detection delay (AC 10.3) count against.

Per-phase tier (tier agreement, detection)
------------------------------------------
Several metrics need one representative tier per phase segment. We use the
**maximum** tier over the segment's windows. Max — not mean/median — because the
question each metric asks is "did this phase trigger, and how strongly at its
peak?": a phase that reaches Tier 3 for even a few windows is a strong detection
regardless of how many quiet windows surround it, and self-reported stress is a
single peak rating for the whole phase (:func:`eval.labels.attach_self_report`).
Detection (flagged-or-not) is the same rule at the Watch threshold: a phase is
flagged iff its max tier ``>= 1``.

Dependencies are numpy / pandas / scipy (``scipy.stats.spearmanr`` for the rank
correlation) / stdlib plus the sibling :mod:`eval.labels` and ``data`` records.
Every function is pure over its inputs; nothing reads the filesystem.

Requirements covered:
- 10.1 :func:`sensitivity` — share of stress-task phases flagged at any tier,
       split into long tasks and 30-second tasks.
- 10.2 :func:`false_alarms_per_rest_hour` — flags during baseline/rest segments
       per rest hour, excluding the first 120 s after each task phase (the
       excluded recovery-tail flags are reported separately).
- 10.3 :func:`detection_delay` — seconds from a stress-task phase start to the
       first flag in that phase (non-negative, <= phase duration; Property 24).
- 10.4 :func:`tier_agreement` — Spearman rank correlation between the per-phase
       tier and the per-phase self-reported stress.
- 10.5 :func:`movement_attribution` — share of exercise-session flags and share
       of stress-session flags labeled "explained by movement".
- 10.6 :func:`evaluate` — all of the above reported separately for Cohort_V1 and
       Cohort_V2, carrying the ``method`` label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data.schema import TIER_NONE, TIER_WATCH, Thresholds
from data.windows import STEP_S
from flag.baseline import SignalBaseline
from flag.score import window_signed_z
from flag.tiers import Flagger
from eval.labels import (
    PHASE_CLASS_EXERCISE,
    PHASE_CLASS_LONG_TASK,
    PHASE_CLASS_SHORT_TASK,
    PhaseLabels,
    PhaseSegment,
    attach_self_report,
    phase_labels,
    phase_segments,
)

__all__ = [
    "SESSION_STRESS",
    "EXERCISE_SESSIONS",
    "RECOVERY_TAIL_S",
    "FlagResult",
    "as_flag_frame",
    "segment_max_tier",
    "segment_is_flagged",
    "SensitivityResult",
    "sensitivity",
    "FalseAlarmResult",
    "false_alarms_per_rest_hour",
    "DelayResult",
    "detection_delay",
    "tier_agreement",
    "MovementAttributionResult",
    "movement_attribution",
    "CohortMetrics",
    "MetricTable",
    "evaluate",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Session names (match data/windows.py and the loader). Movement attribution
# (AC 10.5) contrasts exercise sessions against the stress session.
SESSION_STRESS = "STRESS"
SESSION_AEROBIC = "AEROBIC"
SESSION_ANAEROBIC = "ANAEROBIC"
EXERCISE_SESSIONS: frozenset[str] = frozenset({SESSION_AEROBIC, SESSION_ANAEROBIC})

# Recovery tail excluded from the rest false-alarm denominator (AC 10.2): the
# first 120 s after each task phase, where residual stress may linger, is not
# counted as clean rest.
RECOVERY_TAIL_S: float = 120.0


# ---------------------------------------------------------------------------
# Per-window flag-result normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlagResult:
    """Per-window flag outcome aligned by table row index.

    ``tier`` is the assigned Severity_Tier per window (int ``0..3``), ``movement``
    the "explained by movement" boolean per window. Both arrays are parallel to
    the ``Per_Window_Table`` rows (row *i* is window *i*). ``session`` is the
    per-window session label; when the caller does not supply one it is filled
    from the labels so movement attribution can split exercise vs stress.
    """

    tier: np.ndarray
    movement: np.ndarray
    session: np.ndarray

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self.tier.shape[0])


def as_flag_frame(
    flags: object, labels: PhaseLabels | None = None
) -> FlagResult:
    """Normalize any accepted flag-result input into a :class:`FlagResult`.

    Accepts (all aligned by table row index — see the module docstring):

    - a :class:`pandas.DataFrame` with ``tier`` / ``movement_explained`` columns
      (and optionally ``session``);
    - a mapping with keys ``tier`` / ``movement_explained`` (and optionally
      ``session``); the keys ``movement`` and ``movement_explained`` are both
      accepted;
    - a 2-sequence ``(tier, movement_explained)`` of array-likes.

    ``tier`` is coerced to int, ``movement_explained`` to bool. When no per-window
    ``session`` is given, it is taken from ``labels`` (one entry per window in the
    same order); if ``labels`` is also absent the session column is filled with
    the empty string (only movement attribution reads it).

    Raises ``ValueError`` on a shape mismatch (tier vs movement lengths, or a
    length that disagrees with ``labels``).
    """
    tier_arr: np.ndarray
    mv_arr: np.ndarray
    sess_arr: np.ndarray | None = None

    if isinstance(flags, FlagResult):
        return flags

    if isinstance(flags, pd.DataFrame):
        if "tier" not in flags.columns:
            raise ValueError("flag DataFrame must have a 'tier' column")
        mv_col = (
            "movement_explained"
            if "movement_explained" in flags.columns
            else ("movement" if "movement" in flags.columns else None)
        )
        if mv_col is None:
            raise ValueError(
                "flag DataFrame must have a 'movement_explained' (or 'movement') column"
            )
        tier_arr = flags["tier"].to_numpy()
        mv_arr = flags[mv_col].to_numpy()
        if "session" in flags.columns:
            sess_arr = flags["session"].astype(object).to_numpy()
    elif isinstance(flags, Mapping):
        if "tier" not in flags:
            raise ValueError("flag mapping must have a 'tier' entry")
        if "movement_explained" in flags:
            mv_src = flags["movement_explained"]
        elif "movement" in flags:
            mv_src = flags["movement"]
        else:
            raise ValueError(
                "flag mapping must have a 'movement_explained' (or 'movement') entry"
            )
        tier_arr = np.asarray(flags["tier"])
        mv_arr = np.asarray(mv_src)
        if "session" in flags:
            sess_arr = np.asarray(flags["session"], dtype=object)
    elif isinstance(flags, Sequence) and not isinstance(flags, (str, bytes)):
        if len(flags) != 2:
            raise ValueError(
                "a sequence flag input must be exactly (tier, movement_explained)"
            )
        tier_arr = np.asarray(flags[0])
        mv_arr = np.asarray(flags[1])
    else:
        raise TypeError(
            "flags must be a DataFrame, a mapping, a (tier, movement) sequence, "
            f"or a FlagResult; got {type(flags)!r}"
        )

    tier_arr = np.asarray(tier_arr, dtype=int).reshape(-1)
    mv_arr = np.asarray(mv_arr).astype(bool).reshape(-1)

    if tier_arr.shape[0] != mv_arr.shape[0]:
        raise ValueError(
            f"tier and movement lengths differ: {tier_arr.shape[0]} vs "
            f"{mv_arr.shape[0]}"
        )

    n = tier_arr.shape[0]

    if sess_arr is None:
        if labels is not None:
            sess_arr = np.array(
                [w.session for w in labels.windows], dtype=object
            )
        else:
            sess_arr = np.array([""] * n, dtype=object)
    else:
        sess_arr = np.asarray(sess_arr, dtype=object).reshape(-1)

    if labels is not None and len(labels.windows) != n:
        raise ValueError(
            f"flag input has {n} rows but labels describes {len(labels.windows)} "
            "windows; they must be aligned by table row index"
        )
    if sess_arr.shape[0] != n:
        raise ValueError(
            f"session length {sess_arr.shape[0]} does not match flag length {n}"
        )

    return FlagResult(tier=tier_arr, movement=mv_arr, session=sess_arr)


# ---------------------------------------------------------------------------
# Per-segment helpers
# ---------------------------------------------------------------------------


def segment_max_tier(segment: PhaseSegment, tier: np.ndarray) -> int:
    """Return the maximum per-window tier over a segment's windows.

    The representative per-phase tier (see the module docstring): the strongest
    tier reached anywhere in the phase. An empty segment (no row indices) yields
    :data:`~data.schema.TIER_NONE` (0).
    """
    if not segment.row_indices:
        return TIER_NONE
    idx = np.asarray(segment.row_indices, dtype=int)
    return int(np.max(tier[idx]))


def segment_is_flagged(segment: PhaseSegment, tier: np.ndarray) -> bool:
    """True if any window in the segment is flagged (tier >= Watch).

    A phase "receives a flag at any tier" (AC 10.1) exactly when its max tier is
    at least :data:`~data.schema.TIER_WATCH` (1).
    """
    return segment_max_tier(segment, tier) >= TIER_WATCH


def _first_flag_time(segment: PhaseSegment, tier: np.ndarray, t_win: np.ndarray) -> float | None:
    """Earliest window-end time in the segment whose tier >= Watch, else ``None``.

    Walks the segment's rows in time order (row indices are emitted in increasing
    ``t_win`` by the Windower) and returns the first flagged window's ``t_win``.
    """
    best: float | None = None
    for ri in segment.row_indices:
        if int(tier[ri]) >= TIER_WATCH:
            t = float(t_win[ri])
            if best is None or t < best:
                best = t
    return best


# ---------------------------------------------------------------------------
# Sensitivity (Requirement 10.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityResult:
    """Sensitivity split into long tasks and 30-second tasks (AC 10.1).

    For each task class: ``*_detected`` phases flagged at any tier, ``*_total``
    task phases present, and ``*_sensitivity`` the ratio (``NaN`` when there are
    no task phases of that class to score).
    """

    long_detected: int
    long_total: int
    long_sensitivity: float
    short_detected: int
    short_total: int
    short_sensitivity: float

    @property
    def overall_sensitivity(self) -> float:
        total = self.long_total + self.short_total
        if total == 0:
            return float("nan")
        return (self.long_detected + self.short_detected) / total


def _ratio(num: int, den: int) -> float:
    return float(num) / den if den else float("nan")


def sensitivity(flags: object, labels: PhaseLabels) -> SensitivityResult:
    """Share of stress-task phases flagged at any tier, long vs 30-second (AC 10.1).

    A stress-task phase segment is *detected* when any of its windows is flagged
    (tier >= Watch; :func:`segment_is_flagged`). The result is reported separately
    for long tasks (Stroop, TMCT) and 30-second tasks (RealOpinion,
    OppositeOpinion, Subtract), using the ``phase_class`` already assigned by
    :mod:`eval.labels`.
    """
    fr = as_flag_frame(flags, labels)
    long_total = long_det = 0
    short_total = short_det = 0

    for seg in labels.segments:
        if seg.phase_class == PHASE_CLASS_LONG_TASK:
            long_total += 1
            if segment_is_flagged(seg, fr.tier):
                long_det += 1
        elif seg.phase_class == PHASE_CLASS_SHORT_TASK:
            short_total += 1
            if segment_is_flagged(seg, fr.tier):
                short_det += 1

    return SensitivityResult(
        long_detected=long_det,
        long_total=long_total,
        long_sensitivity=_ratio(long_det, long_total),
        short_detected=short_det,
        short_total=short_total,
        short_sensitivity=_ratio(short_det, short_total),
    )


# ---------------------------------------------------------------------------
# False alarms per rest hour (Requirement 10.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FalseAlarmResult:
    """False alarms per hour of rest, with the recovery tail excluded (AC 10.2).

    ``false_alarms`` is the count of flagged windows in clean rest (baseline/rest
    segments, excluding the first :data:`RECOVERY_TAIL_S` seconds after each task
    phase). ``rest_hours`` is the clean-rest duration in hours (counted rest
    windows * :data:`~data.windows.STEP_S` / 3600). ``per_hour`` is the ratio
    (``NaN`` when there is no clean rest). ``excluded_tail_flags`` is the number
    of flagged rest windows that fell inside an excluded recovery tail — reported
    separately per AC 10.2's "excluding the first 120 seconds" note.
    """

    false_alarms: int
    rest_hours: float
    per_hour: float
    excluded_tail_flags: int
    excluded_tail_windows: int


def false_alarms_per_rest_hour(flags: object, labels: PhaseLabels) -> FalseAlarmResult:
    """False alarms per rest hour, skipping the 120 s post-task tail (AC 10.2).

    Counts flagged windows (tier >= Watch) that fall in rest/baseline segments,
    but a rest window is excluded from both the numerator and the denominator
    when it lies within :data:`RECOVERY_TAIL_S` seconds after the end of a task
    phase in the same ``(person_id, session)`` (the recovery tail). The clean
    rest hours are the retained rest-window count times the 10-second
    :data:`~data.windows.STEP_S`, in hours. The flags that fell inside an excluded
    tail are returned separately in ``excluded_tail_flags``.

    Task-phase end times are taken from the task segments' ``end_t``; a rest
    window at time ``t`` is in a recovery tail iff some task phase for the same
    person/session ended within the preceding ``RECOVERY_TAIL_S`` seconds
    (``task_end < t <= task_end + RECOVERY_TAIL_S``).
    """
    fr = as_flag_frame(flags, labels)

    # Task-phase end times keyed by (person, session), for the recovery tail.
    task_ends: dict[tuple[str, str], list[float]] = {}
    for seg in labels.segments:
        if seg.is_task:
            task_ends.setdefault((seg.person_id, seg.session), []).append(seg.end_t)

    def _in_recovery_tail(person: str, session: str, t: float) -> bool:
        for end_t in task_ends.get((person, session), ()):
            if end_t < t <= end_t + RECOVERY_TAIL_S:
                return True
        return False

    clean_rest_windows = 0
    false_alarms = 0
    excluded_windows = 0
    excluded_flags = 0

    for seg in labels.segments:
        if not seg.is_rest:
            continue
        for ri in seg.row_indices:
            w = labels.windows[ri]
            flagged = int(fr.tier[ri]) >= TIER_WATCH
            if _in_recovery_tail(w.person_id, w.session, w.t_win):
                excluded_windows += 1
                if flagged:
                    excluded_flags += 1
                continue
            clean_rest_windows += 1
            if flagged:
                false_alarms += 1

    rest_hours = clean_rest_windows * STEP_S / 3600.0
    per_hour = false_alarms / rest_hours if rest_hours > 0 else float("nan")

    return FalseAlarmResult(
        false_alarms=false_alarms,
        rest_hours=rest_hours,
        per_hour=per_hour,
        excluded_tail_flags=excluded_flags,
        excluded_tail_windows=excluded_windows,
    )


# ---------------------------------------------------------------------------
# Detection delay (Requirement 10.3, Property 24)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DelayResult:
    """Detection delay aggregated per task class (AC 10.3, Property 24).

    ``delays`` holds one ``(segment, delay_s)`` per *flagged* stress-task phase;
    unflagged phases carry no delay and are excluded from the aggregates.
    ``long_mean`` / ``short_mean`` / ``overall_mean`` are the mean delays in
    seconds over the flagged phases of each class (``NaN`` when none were
    flagged). Each delay is non-negative and no greater than its phase duration
    (Property 24).
    """

    delays: list[tuple[PhaseSegment, float]]
    long_mean: float
    short_mean: float
    overall_mean: float
    long_flagged: int
    short_flagged: int


def detection_delay(flags: object, labels: PhaseLabels) -> DelayResult:
    """Seconds from each stress-task phase start to its first flag (AC 10.3).

    For every long or 30-second task segment that is flagged (tier >= Watch at
    some window), the delay is ``first_flag_t_win - segment.start_t``. The delay
    is clamped into ``[0, phase_duration]`` so it satisfies Property 24: it is
    never negative and never exceeds the phase duration (``end_t - start_t``).
    Phases with no flag contribute no delay (they are excluded, not counted as a
    maximal delay). Delays are aggregated as the mean per task class.
    """
    fr = as_flag_frame(flags, labels)
    t_win = np.array([w.t_win for w in labels.windows], dtype=float)

    delays: list[tuple[PhaseSegment, float]] = []
    long_vals: list[float] = []
    short_vals: list[float] = []

    for seg in labels.segments:
        if seg.phase_class not in (PHASE_CLASS_LONG_TASK, PHASE_CLASS_SHORT_TASK):
            continue
        first_t = _first_flag_time(seg, fr.tier, t_win)
        if first_t is None:
            continue  # unflagged phase: no detection delay
        duration = max(0.0, seg.end_t - seg.start_t)
        delay = first_t - seg.start_t
        # Property 24: non-negative and no greater than the phase duration.
        delay = min(max(delay, 0.0), duration)
        delays.append((seg, delay))
        if seg.phase_class == PHASE_CLASS_LONG_TASK:
            long_vals.append(delay)
        else:
            short_vals.append(delay)

    def _mean(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else float("nan")

    return DelayResult(
        delays=delays,
        long_mean=_mean(long_vals),
        short_mean=_mean(short_vals),
        overall_mean=_mean(long_vals + short_vals),
        long_flagged=len(long_vals),
        short_flagged=len(short_vals),
    )


# ---------------------------------------------------------------------------
# Tier agreement (Requirement 10.4)
# ---------------------------------------------------------------------------


def tier_agreement(
    flags: object,
    self_report: Mapping[tuple[str, str, int], float],
    labels: PhaseLabels,
) -> float:
    """Spearman rank correlation of per-phase tier vs self-reported stress (AC 10.4).

    Agreement is judged *per phase* because the self-report is one rating per
    phase visit (:func:`eval.labels.attach_self_report`). For each phase segment
    that has a self-report rating, the representative tier is the **max** tier
    over the segment's windows (:func:`segment_max_tier` — the phase's strongest
    response, see the module docstring). The two per-phase vectors (assigned tier,
    self-report rating) are correlated with :func:`scipy.stats.spearmanr`.

    ``self_report`` is the ``(person_id, phase, occurrence) -> rating`` map from
    :func:`eval.labels.load_stress_levels`; segments with no entry, a ``None``, or
    a ``NaN`` rating are dropped. Returns the Spearman coefficient, or ``NaN``
    when fewer than two usable pairs exist or either vector has no variance (an
    undefined correlation).
    """
    fr = as_flag_frame(flags, labels)

    joined = attach_self_report(labels.segments, self_report)
    tiers: list[float] = []
    ratings: list[float] = []
    for seg, rating in joined:
        if rating is None:
            continue
        if isinstance(rating, float) and np.isnan(rating):
            continue
        tiers.append(float(segment_max_tier(seg, fr.tier)))
        ratings.append(float(rating))

    if len(tiers) < 2:
        return float("nan")
    # Spearman is undefined when either input is constant.
    if len(set(tiers)) < 2 or len(set(ratings)) < 2:
        return float("nan")

    rho, _ = spearmanr(tiers, ratings)
    return float(rho)


# ---------------------------------------------------------------------------
# Movement attribution (Requirement 10.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MovementAttributionResult:
    """Share of flags labeled "explained by movement" by session kind (AC 10.5).

    ``exercise_share`` is the fraction of *flagged* exercise-session windows
    (aerobic/anaerobic) whose flag was explained by movement; ``stress_share`` is
    the same for stress-session windows. Each is ``NaN`` when that session kind
    has no flagged windows. The counts behind the shares are carried alongside.
    """

    exercise_flagged: int
    exercise_movement: int
    exercise_share: float
    stress_flagged: int
    stress_movement: int
    stress_share: float


def movement_attribution(flags: object, labels: PhaseLabels) -> MovementAttributionResult:
    """Movement-explained share among exercise vs stress flags (AC 10.5).

    Over every flagged window (tier >= Watch), split by the window's session into
    exercise sessions (AEROBIC/ANAEROBIC) and the stress session, and report the
    share whose ``movement_explained`` is true for each. A window's session comes
    from the flag input when present, otherwise from the labels. Windows that are
    not flagged are ignored (movement attribution describes flags).
    """
    fr = as_flag_frame(flags, labels)
    sess = fr.session

    ex_flag = ex_mv = 0
    st_flag = st_mv = 0

    for i in range(len(fr.tier)):
        if int(fr.tier[i]) < TIER_WATCH:
            continue
        s = str(sess[i]).upper()
        moved = bool(fr.movement[i])
        if s in EXERCISE_SESSIONS:
            ex_flag += 1
            if moved:
                ex_mv += 1
        elif s == SESSION_STRESS:
            st_flag += 1
            if moved:
                st_mv += 1

    return MovementAttributionResult(
        exercise_flagged=ex_flag,
        exercise_movement=ex_mv,
        exercise_share=_ratio(ex_mv, ex_flag),
        stress_flagged=st_flag,
        stress_movement=st_mv,
        stress_share=_ratio(st_mv, st_flag),
    )


# ---------------------------------------------------------------------------
# Per-cohort evaluation (Requirement 10.6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortMetrics:
    """All five detection metrics for a single cohort.

    One of these per cohort inside a :class:`MetricTable`. Each field is the
    corresponding metric's result over the windows/segments of that cohort only.
    """

    cohort: str
    sensitivity: SensitivityResult
    false_alarms: FalseAlarmResult
    detection_delay: DelayResult
    tier_agreement: float
    movement_attribution: MovementAttributionResult
    n_windows: int
    n_segments: int


@dataclass(frozen=True)
class MetricTable:
    """Detection metrics for one method, reported per cohort (AC 10.6).

    ``method`` is the method label (e.g. the combined-score method name) carried
    through so Task 13's ``compare_methods`` can report methods side by side.
    ``by_cohort`` maps each cohort label (``V1`` / ``V2``) to its
    :class:`CohortMetrics`.
    """

    method: str
    by_cohort: dict[str, CohortMetrics] = field(default_factory=dict)

    def cohorts(self) -> list[str]:
        return sorted(self.by_cohort)


def _subset_labels(labels: PhaseLabels, keep: np.ndarray) -> tuple[PhaseLabels, np.ndarray]:
    """Restrict ``labels`` to the rows selected by boolean mask ``keep``.

    Returns a new :class:`PhaseLabels` whose windows and segments are reindexed to
    the compacted positions, plus the integer array of kept original row indices
    (so the caller can slice the parallel flag arrays the same way). Segments
    that lose all their rows are dropped; surviving segments keep their relative
    row order and have ``row_indices`` remapped to the new positions.
    """
    kept_idx = np.nonzero(keep)[0]
    remap = {old: new for new, old in enumerate(kept_idx)}

    new_windows = []
    for new_pos, old in enumerate(kept_idx):
        w = labels.windows[old]
        # WindowLabel is frozen; rebuild with the compacted row_index.
        new_windows.append(
            type(w)(
                row_index=new_pos,
                person_id=w.person_id,
                cohort=w.cohort,
                session=w.session,
                phase=w.phase,
                t_win=w.t_win,
                phase_class=w.phase_class,
            )
        )

    new_segments = []
    for seg in labels.segments:
        remapped = tuple(remap[r] for r in seg.row_indices if r in remap)
        if not remapped:
            continue
        new_segments.append(
            type(seg)(
                person_id=seg.person_id,
                cohort=seg.cohort,
                session=seg.session,
                phase=seg.phase,
                phase_class=seg.phase_class,
                occurrence=seg.occurrence,
                start_t=seg.start_t,
                end_t=seg.end_t,
                row_indices=remapped,
            )
        )

    return PhaseLabels(windows=new_windows, segments=new_segments), kept_idx


def evaluate(
    method: str,
    flags: object,
    labels: PhaseLabels,
    self_report: Mapping[tuple[str, str, int], float] | None = None,
) -> MetricTable:
    """Compute every detection metric for ``method``, split by cohort (AC 10.6).

    ``flags`` is the per-window flag result aligned by table row index (any form
    :func:`as_flag_frame` accepts); ``labels`` is the :class:`PhaseLabels` for the
    same table; ``self_report`` is the ``(person_id, phase, occurrence) -> rating``
    map from :func:`eval.labels.load_stress_levels` (tier agreement is ``NaN``
    when it is omitted).

    The windows are partitioned by their ``cohort`` label (``V1`` / ``V2``), and
    each of :func:`sensitivity`, :func:`false_alarms_per_rest_hour`,
    :func:`detection_delay`, :func:`tier_agreement`, and
    :func:`movement_attribution` is computed on that cohort's windows/segments
    alone. ``method`` is carried onto the returned :class:`MetricTable` so Task
    13 can report multiple methods side by side; the metric values depend only on
    the provided per-window tiers, not on ``method``.
    """
    fr = as_flag_frame(flags, labels)
    sr: Mapping[tuple[str, str, int], float] = self_report or {}

    cohorts = sorted({w.cohort for w in labels.windows})
    by_cohort: dict[str, CohortMetrics] = {}

    for cohort in cohorts:
        keep = np.array(
            [w.cohort == cohort for w in labels.windows], dtype=bool
        )
        sub_labels, kept_idx = _subset_labels(labels, keep)
        sub_fr = FlagResult(
            tier=fr.tier[kept_idx],
            movement=fr.movement[kept_idx],
            session=fr.session[kept_idx],
        )

        by_cohort[cohort] = CohortMetrics(
            cohort=cohort,
            sensitivity=sensitivity(sub_fr, sub_labels),
            false_alarms=false_alarms_per_rest_hour(sub_fr, sub_labels),
            detection_delay=detection_delay(sub_fr, sub_labels),
            tier_agreement=tier_agreement(sub_fr, sr, sub_labels),
            movement_attribution=movement_attribution(sub_fr, sub_labels),
            n_windows=len(sub_labels.windows),
            n_segments=len(sub_labels.segments),
        )

    return MetricTable(method=method, by_cohort=by_cohort)


# ---------------------------------------------------------------------------
# WHOOP_Rule single-signal baseline + method comparison (Requirement 11)
# ---------------------------------------------------------------------------
#
# Requirement 11 measures *the benefit of combining metrics* by putting the
# combined scores next to a single-signal rule. The combined methods
# (directional-sum, Mahalanobis) ask "do several metrics shift together?"; the
# ``WHOOP_Rule`` is the deliberate opposite — it fires on **any one** metric
# drifting out of its per-person range, modeled on WHOOP Health Monitor's
# per-metric out-of-range alerting. It is an *evaluation comparison baseline
# only* (design Introduction: "The single-signal `WHOOP_Rule` is implemented
# only as an evaluation comparison baseline"); it is never part of the live
# flagging path.
#
# Making the WHOOP_Rule comparable to the combined methods in :func:`evaluate`
# --------------------------------------------------------------------------
# ``evaluate`` scores a method from one set of per-window Severity_Tiers. To put
# the WHOOP_Rule on the same footing, we turn its single-signal out-of-range
# test into a per-window *score* and then run the **same Flagger logic** the
# combined methods use (persistence gate + median-of-3 tiering,
# ``flag/tiers.py``):
#
#   WHOOP per-window score := max over the available scored signals of the
#   signal's out-of-range magnitude |signed z| (:func:`whoop_window_score`).
#
# This is explicitly *not a combination*: the score is the single most
# out-of-range signal for that window (a ``max``, not a sum / Mahalanobis
# distance), so only one metric ever drives it — the single-signal nature the
# baseline is meant to represent. Because the magnitude is in the same signed-z
# units as a one-signal directional sum, the *same* ``Thresholds`` (tau1/tau2/
# tau3) and the *same* gate/median tiering apply, so the three methods'
# per-window tiers are produced identically from identically-calibrated
# thresholds and are directly comparable in ``evaluate`` (AC 11.1-11.4).
#
# One consequence falls straight out of the single-signal construction and the
# Flagger's single-signal cap (AC 7.7): a WHOOP window has exactly one
# contributing signal, so ``single_signal`` is always true and the WHOOP_Rule
# tier is capped at Watch (1). That is the intended, honest contrast — a rule
# that never sees metrics move together cannot earn the higher multi-metric
# tiers the combined score can.

__all__ += [
    "METHOD_WHOOP",
    "whoop_window_score",
    "whoop_window_scores",
    "tiers_from_scores",
    "ComparisonTable",
    "compare_methods",
]


# Method label for the WHOOP_Rule, carried onto its :class:`MetricTable` so the
# side-by-side report (AC 11.4) names it alongside the combined methods. The
# combined-method labels are reused from :mod:`flag.score`
# (``METHOD_DIRECTIONAL_SUM`` / ``METHOD_MAHALANOBIS``).
METHOD_WHOOP: str = "whoop_rule"


def whoop_window_score(
    window_row,
    baselines: Mapping[str, SignalBaseline],
) -> float:
    """Single-signal out-of-range magnitude for one window (WHOOP_Rule, AC 11.1).

    Returns the **largest** per-signal out-of-range magnitude over the window's
    available scored signals — ``max_s |signed z_s|`` — i.e. how far the most
    out-of-range single metric has drifted from that person's baseline. This is
    the WHOOP_Rule's "any one signal out of range" test expressed as a score:
    deliberately a ``max`` over signals, never a combination, so a single metric
    always drives it (the opposite of the combined Combined_Score).

    Uses :func:`flag.score.window_signed_z`, which already drops signals that are
    unavailable (``NaN``) or lack a baseline for this window (AC 6.7), and takes
    the magnitude of each signed z (an out-of-range drift counts regardless of
    direction). A window with no available signal scores ``0.0`` (nothing is out
    of range), matching the combined methods' empty-window behavior.
    """
    z = window_signed_z(window_row, baselines)
    if not z:
        return 0.0
    return max(abs(float(v)) for v in z.values())


def whoop_window_scores(
    table: pd.DataFrame,
    baselines_by_person: Mapping[str, Mapping[str, SignalBaseline]],
    *,
    person_col: str = "person_id",
) -> np.ndarray:
    """WHOOP_Rule per-window scores for a whole ``Per_Window_Table`` (AC 11.1).

    Applies :func:`whoop_window_score` to every row, picking each row's
    per-person ``{signal -> SignalBaseline}`` from ``baselines_by_person`` by the
    row's ``person_col`` value. A row whose person has no baseline entry scores
    ``0.0``. The returned float array is aligned by table row index, ready to feed
    :func:`tiers_from_scores` (and then :func:`evaluate`).
    """
    scores = np.zeros(len(table), dtype=float)
    for pos in range(len(table)):
        row = table.iloc[pos]
        pid = row[person_col]
        baselines = baselines_by_person.get(pid, {})
        scores[pos] = whoop_window_score(row, baselines)
    return scores


def tiers_from_scores(
    scores,
    taus: Thresholds,
    *,
    stream_ids=None,
    single_signal=None,
) -> np.ndarray:
    """Run the Flagger over per-window scores to produce per-window tiers.

    This is the shared bridge from a method's per-window Combined_Score (or the
    WHOOP_Rule's out-of-range score) to the per-window Severity_Tiers that
    :func:`evaluate` consumes. It applies exactly the production Flagger logic
    from :mod:`flag.tiers` — the 2-of-3 persistence gate, median-of-3 tiering at
    ``tau1 < tau2 < tau3``, the single-signal cap, and down-step hysteresis — so
    every method is tiered the same way and the resulting metrics are comparable
    (AC 11.1-11.4).

    Parameters
    ----------
    scores:
        Per-window scores aligned by table row index (one per window).
    taus:
        The ordered :class:`~data.schema.Thresholds` to tier against — the *same*
        thresholds for every compared method.
    stream_ids:
        Optional per-window stream key (e.g. ``(person_id, session)``) so the
        rolling last-3 history and hysteresis reset at each person/session
        boundary rather than bleeding across participants. When omitted, all rows
        are treated as one continuous stream. Must be aligned by row index.
    single_signal:
        Per-window single-signal-cap flags (AC 7.7), aligned by row index, or a
        single ``bool`` broadcast to every window. For the WHOOP_Rule pass
        ``True`` (one contributing signal by construction, so every window is
        capped at Watch); for a combined method pass the per-window
        ``is_single_signal(window_signed_z(...))`` array. When omitted, defaults
        to ``False`` for every window (no cap).

    Returns
    -------
    numpy.ndarray
        The per-window tiers (int ``0..3``), aligned by table row index.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n = scores.shape[0]

    if single_signal is None:
        ss = np.zeros(n, dtype=bool)
    elif isinstance(single_signal, (bool, np.bool_)):
        # A single bool broadcasts to every window (e.g. the WHOOP_Rule, which is
        # single-signal for every window by construction).
        ss = np.full(n, bool(single_signal), dtype=bool)
    else:
        ss = np.asarray(single_signal).astype(bool).reshape(-1)
        if ss.shape[0] != n:
            raise ValueError(
                f"single_signal length {ss.shape[0]} does not match scores "
                f"length {n}"
            )

    if stream_ids is None:
        keys = [None] * n
    else:
        keys = list(stream_ids)
        if len(keys) != n:
            raise ValueError(
                f"stream_ids length {len(keys)} does not match scores length {n}"
            )

    tiers = np.zeros(n, dtype=int)
    flaggers: dict[object, Flagger] = {}
    for i in range(n):
        key = keys[i]
        flagger = flaggers.get(key)
        if flagger is None:
            flagger = Flagger(taus)
            flaggers[key] = flagger
        tiers[i] = flagger.step(float(scores[i]), single_signal=bool(ss[i]))
    return tiers


@dataclass(frozen=True)
class ComparisonTable:
    """Side-by-side detection metrics across methods (AC 11.4).

    ``by_method`` maps each method label — :data:`METHOD_WHOOP`,
    :data:`~flag.score.METHOD_DIRECTIONAL_SUM`,
    :data:`~flag.score.METHOD_MAHALANOBIS` — to that method's
    :class:`MetricTable` (itself per-cohort, AC 10.6). Holding all three tables in
    one container is what lets a report place the WHOOP_Rule, directional-sum and
    Mahalanobis metrics next to each other for the same cohorts (AC 11.1-11.4).
    """

    by_method: dict[str, MetricTable] = field(default_factory=dict)

    def methods(self) -> list[str]:
        """The method labels present, WHOOP_Rule first then the combined methods."""
        order = {METHOD_WHOOP: 0, "directional_sum": 1, "mahalanobis": 2}
        return sorted(self.by_method, key=lambda m: (order.get(m, 99), m))

    def cohorts(self) -> list[str]:
        """Union of cohort labels across the method tables (sorted)."""
        cohorts: set[str] = set()
        for table in self.by_method.values():
            cohorts.update(table.by_cohort)
        return sorted(cohorts)


def compare_methods(
    tiers_by_method: Mapping[str, object],
    labels: PhaseLabels,
    self_report: Mapping[tuple[str, str, int], float] | None = None,
) -> ComparisonTable:
    """Evaluate every method's per-window tiers side by side (AC 11.1-11.4).

    This is the comparison entry point for Requirement 11. It takes **precomputed
    per-window tiers per method** and runs the existing :func:`evaluate` once per
    method, collecting the resulting :class:`MetricTable`s into one
    :class:`ComparisonTable` so they can be reported side by side (AC 11.4).

    Taking precomputed tiers (rather than scores) keeps this function a thin,
    composable aggregator and leaves the score -> tier step explicit at the call
    site, where each method's scores and the shared calibrated ``Thresholds`` live
    (the Calibrator, Task 14). The typical call site builds the three tier arrays
    from one ``Per_Window_Table`` + per-person baselines:

    - ``WHOOP_Rule``: ``tiers_from_scores(whoop_window_scores(table, baselines),
      taus, stream_ids=..., single_signal=True)`` — single-signal out-of-range
      (AC 11.1);
    - ``directional_sum`` / ``mahalanobis``: ``tiers_from_scores`` over the
      per-window :func:`flag.score.combined_score` for that method, with the
      per-window :func:`flag.tiers.is_single_signal` cap flag (AC 11.2, 11.3).

    Each value in ``tiers_by_method`` is any per-window flag input
    :func:`as_flag_frame` accepts — a ``(tier, movement_explained)`` pair, a
    mapping with ``tier`` / ``movement_explained`` keys, a DataFrame, or a
    :class:`FlagResult` — aligned by the same table row index as ``labels``. The
    movement-explained flags drive movement attribution (AC 10.5); the comparison
    baselines (WHOOP_Rule and the combined scores evaluated here) do not run the
    live Movement_Check, so their movement array is simply all-``False``.
    :func:`evaluate` computes the full metric set — every
    detection metric of Requirement 10 — for each method, split per cohort
    (AC 10.6), so each listed method gets *every* metric (AC 11.1-11.3).

    Parameters
    ----------
    tiers_by_method:
        ``{method_label -> per-window flag input}``. Typically the three keys
        :data:`METHOD_WHOOP`, ``"directional_sum"`` and ``"mahalanobis"``, but any
        set of methods is accepted.
    labels:
        The :class:`~eval.labels.PhaseLabels` for the shared table; passed through
        to each :func:`evaluate` call.
    self_report:
        The ``(person_id, phase, occurrence) -> rating`` map for tier agreement
        (AC 10.4); tier agreement is ``NaN`` for every method when omitted.

    Returns
    -------
    ComparisonTable
        One :class:`MetricTable` per method, keyed by method label.
    """
    by_method: dict[str, MetricTable] = {}
    for method, tiers in tiers_by_method.items():
        by_method[method] = evaluate(method, tiers, labels, self_report)
    return ComparisonTable(by_method=by_method)
