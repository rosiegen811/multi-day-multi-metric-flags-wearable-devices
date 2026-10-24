"""Calibrator — leave-one-subject-out threshold calibration (Requirement 12).

This module tunes the three ordered Combined_Score thresholds ``tau1 < tau2 <
tau3`` (``data/schema.Thresholds``) that the Flagger (``flag/tiers.py``) steps a
Severity_Tier at. Thresholds are *learned by cross-validation rather than
hand-picked* (design goal "Thresholds tuned by cross-validation, not
hand-picked"; design "Calibrator — eval/calibrate.py"): the Calibrator sweeps
candidate thresholds against the evaluation oracle — the rest false-alarm budget
and the per-phase self-reported stress / aerobic-intensity ordering — using
leave-one-subject-out (LOSO) folds so the chosen thresholds generalize across
participants.

Scope (Task 14.1)
-----------------
This module provides:

- :func:`loso_folds` — the LOSO fold generator: each subject is left out exactly
  once and each fold's training set is exactly the other subjects (Property 25,
  AC 12.1). The harness (Task 15, ``eval/run_eval.py``) uses a fold's training
  subjects to calibrate and its held-out subject to measure; the Calibrator here
  only *produces* the folds and the thresholds.
- :func:`calibrate_on_subset` — calibrate ``tau1, tau2, tau3`` from the windows
  of a given subject set (the per-fold training step):
    * ``tau1`` is the smallest swept candidate whose resulting rest false-alarm
      rate is within the configured budget (AC 12.2);
    * ``tau2``/``tau3`` are chosen so the assigned tiers track self-reported
      stress and the aerobic test's increasing-intensity stages (AC 12.3);
    * the returned triple is always strictly ordered ``tau1 < tau2 < tau3``
      (AC 12.4, Property 26), enforced by construction and by building a
      :class:`~data.schema.Thresholds` (which raises on a tie/misorder).
- :func:`calibrate_loso` — the top-level entry the design documents
  (``calibrate_loso(table, labels, budget) -> Thresholds``). It runs
  :func:`calibrate_on_subset` on **every** LOSO training set and returns the
  **median** of each threshold across folds (an aggregate that is robust to any
  single subject and that the harness can use as the deployed threshold set).
  The per-fold thresholds are available via :func:`calibrate_loso_folds` for the
  harness, which calibrates on each training set and evaluates on the matching
  held-out subject.

Inputs (documented contract)
----------------------------
Calibration needs, aligned by ``Per_Window_Table`` row index:

- ``scores`` — the per-window **Combined_Score** for the method being calibrated
  (directional-sum or Mahalanobis), as produced by ``flag/score.combined_score``
  over the table rows. The Calibrator never recomputes scores; it tiers them with
  ``eval.metrics.tiers_from_scores`` for candidate thresholds.
- ``labels`` — the :class:`~eval.labels.PhaseLabels` for the same table (rest vs
  task segments, per-window session/phase; from ``eval.labels.phase_labels``).
- ``self_report`` — the ``(person_id, phase, occurrence) -> rating`` map from
  ``eval.labels.load_stress_levels`` (AC 12.3 tier-vs-self-report tracking).
- optional ``single_signal`` — per-window single-signal-cap flags (AC 7.7),
  passed straight through to ``tiers_from_scores`` so candidate tiers match what
  the Flagger would assign.

Subjects/folds are derived from ``labels`` (the distinct ``person_id`` over the
windows). A LOSO fold is ``(train_subjects, held_out_subject)``.

Dependencies are numpy / stdlib plus the sibling ``eval`` and ``data`` modules;
every function is pure over its inputs and reads no files.

Requirements covered:
- 12.1 :func:`loso_folds` / :func:`calibrate_loso` — LOSO cross-validation,
       leaving each subject out once (Property 25).
- 12.2 ``tau1`` from the rest false-alarm budget (:func:`calibrate_on_subset`).
- 12.3 ``tau2``/``tau3`` so tiers track self-reported stress and aerobic
       intensity stages (:func:`calibrate_on_subset`).
- 12.4 output strictly ordered ``tau1 < tau2 < tau3`` (Property 26;
       :func:`_order_thresholds`, :class:`~data.schema.Thresholds`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from data.schema import Thresholds
from data.windows import STEP_S
from eval.labels import (
    PHASE_CLASS_EXERCISE,
    PHASE_CLASS_LONG_TASK,
    PHASE_CLASS_SHORT_TASK,
    PhaseLabels,
    attach_self_report,
)
from eval.metrics import (
    RECOVERY_TAIL_S,
    segment_max_tier,
    tiers_from_scores,
)

__all__ = [
    "DEFAULT_CANDIDATE_COUNT",
    "MIN_THRESHOLD_GAP",
    "LOSOFold",
    "subjects_of",
    "loso_folds",
    "aerobic_stage_rank",
    "calibrate_on_subset",
    "calibrate_loso_folds",
    "calibrate_loso",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many candidate thresholds to sweep across the observed score range when a
# caller does not supply an explicit grid. A quantile grid of this many points
# over the training scores is dense enough to resolve the budget crossing and the
# tier bands without being expensive.
DEFAULT_CANDIDATE_COUNT: int = 64

# The minimum gap enforced between consecutive thresholds when clamping a
# degenerate (tied or mis-ordered) triple back into strict order (AC 12.4). Small
# but strictly positive so ``Thresholds`` (which requires tau1 < tau2 < tau3)
# accepts the result even when the data would otherwise collapse the bands.
MIN_THRESHOLD_GAP: float = 1e-6


# ---------------------------------------------------------------------------
# LOSO folds (Requirement 12.1, Property 25)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LOSOFold:
    """One leave-one-subject-out fold.

    ``held_out`` is the single subject left out this fold; ``train`` is the tuple
    of all other subjects, in the stable subject order. Together they partition
    the subject set: ``train`` excludes exactly ``held_out`` (Property 25).
    """

    held_out: str
    train: tuple[str, ...]


def subjects_of(labels: PhaseLabels) -> list[str]:
    """Return the distinct subject ids (``person_id``) in a stable order.

    Subjects are the LOSO unit (AC 12.1). The order is first-appearance order over
    the table windows, deduplicated — deterministic so folds are reproducible.
    """
    seen: dict[str, None] = {}
    for w in labels.windows:
        if w.person_id not in seen:
            seen[w.person_id] = None
    return list(seen)


def loso_folds(subjects: Sequence[str]) -> Iterator[LOSOFold]:
    """Yield one :class:`LOSOFold` per subject, leaving each out exactly once.

    For a subject list ``[a, b, c]`` this yields folds with ``held_out`` ``a``,
    ``b``, ``c`` in turn, each carrying ``train`` = the other two in order. Every
    subject is held out exactly once and each training set is exactly the
    complement of its held-out subject (Property 25, AC 12.1).

    Duplicate ids in ``subjects`` are collapsed to their first occurrence so the
    partition is clean (a subject cannot be both held out and in training).
    ``subjects`` with fewer than one entry yields nothing; a single subject yields
    one fold with an empty training set (the caller decides how to handle a fold
    that cannot be trained).
    """
    # Preserve order, drop duplicates.
    unique: list[str] = []
    for s in subjects:
        if s not in unique:
            unique.append(s)
    for i, held in enumerate(unique):
        train = tuple(unique[:i] + unique[i + 1 :])
        yield LOSOFold(held_out=held, train=train)


# ---------------------------------------------------------------------------
# Aerobic-intensity ordering (Requirement 12.3)
# ---------------------------------------------------------------------------


def aerobic_stage_rank(phase: str | None) -> int | None:
    """Return an increasing intensity rank for an aerobic stage phase, else ``None``.

    The aerobic protocol steps through ``Stage<rpm>rpm`` phases of increasing
    pedaling cadence (``data/windows.py``: 60,65,…,110 rpm for V1; 70…95 for V2),
    so a higher rpm is a higher exercise intensity. tau2/tau3 should be set so
    that these increasing-intensity stages map to non-decreasing tiers (AC 12.3);
    this helper exposes the ordering as the integer rpm value, which is a monotone
    proxy for intensity.

    ``WarmUp`` / ``CoolDown`` / ``Sprint*`` and any non-stage phase return
    ``None`` — they are exercise windows but not part of the monotone
    increasing-cadence ramp, so they are not used to order the intensity
    objective. Returns ``None`` for ``None`` or a non-``Stage`` phase.
    """
    if not phase or not isinstance(phase, str):
        return None
    if not phase.startswith("Stage") or not phase.endswith("rpm"):
        return None
    mid = phase[len("Stage") : -len("rpm")]
    try:
        return int(mid)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Threshold ordering guard (Requirement 12.4, Property 26)
# ---------------------------------------------------------------------------


def _order_thresholds(tau1: float, tau2: float, tau3: float) -> Thresholds:
    """Clamp a candidate triple to strict order and return a :class:`Thresholds`.

    The sweep that picks tau2/tau3 is constrained to stay above tau1, but a sparse
    training sample can still produce a tie (e.g. tau2 == tau3 when the task-score
    distribution is degenerate). This guard sorts the triple and nudges any
    non-increasing step up by :data:`MIN_THRESHOLD_GAP` so the result is strictly
    ordered ``tau1 < tau2 < tau3`` (AC 12.4). Constructing the
    :class:`~data.schema.Thresholds` is the final assertion of the invariant — it
    raises if, against expectation, the triple is still not strictly ordered
    (Property 26). The ordering is never silently reordered away from the data: we
    only ever push a tied/inverted upper threshold *up*, preserving tau1.
    """
    a = float(tau1)
    b = float(tau2)
    c = float(tau3)
    # Ensure strictly increasing, preserving tau1 as the anchor.
    if not b > a:
        b = a + MIN_THRESHOLD_GAP
    if not c > b:
        c = b + MIN_THRESHOLD_GAP
    # Thresholds.__post_init__ enforces tau1 < tau2 < tau3 (fatal on violation).
    return Thresholds(tau1=a, tau2=b, tau3=c)


# ---------------------------------------------------------------------------
# Internal helpers: per-subject row masks and candidate grids
# ---------------------------------------------------------------------------


def _subject_mask(labels: PhaseLabels, subjects: Iterable[str]) -> np.ndarray:
    """Boolean per-window mask selecting rows whose subject is in ``subjects``."""
    keep = set(subjects)
    return np.array([w.person_id in keep for w in labels.windows], dtype=bool)


def _stream_ids(labels: PhaseLabels) -> list[tuple[str, str]]:
    """Per-window ``(person_id, session)`` stream keys for the Flagger reset.

    Passed to :func:`eval.metrics.tiers_from_scores` so the rolling last-3 history
    and hysteresis reset at each person/session boundary rather than bleeding
    across participants.
    """
    return [(w.person_id, w.session) for w in labels.windows]


def _candidate_grid(
    values: np.ndarray, candidates: Sequence[float] | int | None
) -> np.ndarray:
    """Build an ascending candidate-threshold grid from observed scores.

    ``candidates`` may be an explicit sequence of thresholds (used as-is, sorted
    ascending and de-duplicated), an int count (that many quantile points over the
    finite ``values``), or ``None`` (``DEFAULT_CANDIDATE_COUNT`` quantile points).
    A quantile grid concentrates candidates where the scores actually lie, so the
    budget crossing (tau1) and the tier bands (tau2/tau3) are resolved finely.
    """
    if isinstance(candidates, (Sequence, np.ndarray)) and not isinstance(
        candidates, (str, bytes)
    ):
        grid = np.asarray(list(candidates), dtype=float)
        grid = grid[np.isfinite(grid)]
        return np.unique(grid)

    count = DEFAULT_CANDIDATE_COUNT if candidates is None else int(candidates)
    count = max(count, 2)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([0.0])
    qs = np.linspace(0.0, 1.0, count)
    grid = np.quantile(finite, qs)
    return np.unique(grid)


# ---------------------------------------------------------------------------
# tau1 from the rest false-alarm budget (Requirement 12.2)
# ---------------------------------------------------------------------------


def _tau1_grid(
    rest_scores: np.ndarray, candidates: Sequence[float] | int | None
) -> np.ndarray:
    """Candidate tau1 grid over the rest-score floor, reaching past its maximum.

    Starts from the rest-score quantile grid (:func:`_candidate_grid`) so most
    candidates sit where the rest noise actually lies, then appends one candidate
    just **above** the maximum finite rest score. The persistence gate fires only
    on scores strictly above tau1, so a tau1 above every rest score flags no rest
    window — guaranteeing the budget is *reachable* even when the rest noise floor
    alone never drops the false-alarm rate to the budget (the lower quantiles can
    leave a residual rate above budget). Without this the smallest in-budget
    candidate could be missing and tau1 would be left at the rest maximum, still
    over budget. An explicit candidate sequence is honored as given (no extension).
    """
    if isinstance(candidates, (Sequence, np.ndarray)) and not isinstance(
        candidates, (str, bytes)
    ):
        return _candidate_grid(rest_scores, candidates)

    grid = _candidate_grid(rest_scores, candidates)
    finite = rest_scores[np.isfinite(rest_scores)]
    if finite.size:
        top = float(np.max(finite))
        # A point above every rest score (relative margin, robust to scale).
        margin = max(abs(top), 1.0) * 1e-3 + MIN_THRESHOLD_GAP
        grid = np.unique(np.concatenate([grid, [top + margin]]))
    return grid


def _rest_false_alarm_rate(
    labels: PhaseLabels,
    tiers: np.ndarray,
    mask: np.ndarray,
) -> float:
    """False alarms per hour of clean rest for the subset ``mask`` (AC 12.2 / 10.2).

    Counts flagged rest windows (tier >= 1 Watch) in Baseline/Rest segments whose
    windows belong to ``mask`` subjects, excluding the first
    :data:`~eval.metrics.RECOVERY_TAIL_S` (120 s) after each task phase — the same
    clean-rest definition the Evaluator uses (``eval.metrics`` AC 10.2). Returns
    flags / clean-rest-hours, or ``+inf`` when there is no clean rest to measure
    (so an empty-rest subject never looks "within budget").
    """
    # Task-phase end times per (person, session) for the recovery-tail exclusion.
    task_ends: dict[tuple[str, str], list[float]] = {}
    for seg in labels.segments:
        if seg.is_task:
            task_ends.setdefault((seg.person_id, seg.session), []).append(seg.end_t)

    def _in_tail(person: str, session: str, t: float) -> bool:
        for end_t in task_ends.get((person, session), ()):
            if end_t < t <= end_t + RECOVERY_TAIL_S:
                return True
        return False

    clean_windows = 0
    false_alarms = 0
    for seg in labels.segments:
        if not seg.is_rest:
            continue
        for ri in seg.row_indices:
            if not mask[ri]:
                continue
            w = labels.windows[ri]
            if _in_tail(w.person_id, w.session, w.t_win):
                continue
            clean_windows += 1
            if int(tiers[ri]) >= 1:
                false_alarms += 1

    rest_hours = clean_windows * STEP_S / 3600.0
    if rest_hours <= 0:
        return float("inf")
    return false_alarms / rest_hours


def _select_tau1(
    scores: np.ndarray,
    labels: PhaseLabels,
    mask: np.ndarray,
    stream_ids: list[tuple[str, str]],
    single_signal,
    budget: float,
    grid: np.ndarray,
) -> float:
    """Smallest candidate whose rest false-alarm rate is within ``budget`` (AC 12.2).

    Sweeps the ascending ``grid`` of candidate tau1 values. For each candidate the
    windows are tiered with ``tiers_from_scores`` using a provisional ordered
    threshold triple (tau1 = candidate, tau2/tau3 placed just above so tiering is
    well-defined), and the resulting rest false-alarm rate over the ``mask``
    subset is measured. tau1 is the **smallest** candidate whose rate is ``<=
    budget`` — the lowest threshold (most sensitive) that still respects the
    false-alarm budget. If every candidate exceeds the budget, the largest
    candidate (the strictest tau1 available) is returned as the closest feasible
    choice.

    A higher tau1 can only reduce or hold the number of flagged rest windows (a
    monotone screen), so the smallest in-budget candidate is the natural pick.
    """
    best: float | None = None
    for cand in grid:
        taus = _order_thresholds(cand, cand, cand)  # bands irrelevant for the Watch count
        tiers = tiers_from_scores(
            scores, taus, stream_ids=stream_ids, single_signal=single_signal
        )
        rate = _rest_false_alarm_rate(labels, tiers, mask)
        if rate <= budget:
            best = float(cand)
            break
    if best is None:
        best = float(grid[-1])
    return best


# ---------------------------------------------------------------------------
# tau2 / tau3 tracking self-report and aerobic intensity (Requirement 12.3)
# ---------------------------------------------------------------------------


def _tracking_objective(
    labels: PhaseLabels,
    tiers: np.ndarray,
    mask: np.ndarray,
    self_report: Mapping[tuple[str, str, int], float],
) -> float:
    """Score how well per-phase tiers track self-report and aerobic intensity (AC 12.3).

    Two concrete, documented components, each a correlation in ``[-1, 1]``, summed:

    1. **Self-reported stress tracking.** For every phase segment (of a ``mask``
       subject) with a self-report rating, pair the segment's representative tier
       (``segment_max_tier`` — the phase's peak response, matching the
       Evaluator's per-phase tier) with its rating and take the Pearson
       correlation across phases. Higher = tiers rise with reported stress.
    2. **Aerobic-intensity tracking.** For every ``Stage<rpm>rpm`` segment, pair
       its representative tier with the stage's rpm rank (:func:`aerobic_stage_rank`)
       and take the Pearson correlation across stages. Higher = higher-intensity
       stages earn higher tiers.

    Returns the sum of the two correlations (a component with fewer than two
    usable, non-constant pairs contributes ``0``). The tau2/tau3 sweep maximizes
    this objective, so the chosen bands are the ones under which assigned tiers
    best track *both* self-reported stress and the increasing exercise ramp.
    """
    # --- self-report component ---
    sr_tiers: list[float] = []
    sr_ratings: list[float] = []
    for seg in labels.segments:
        # Segment belongs to the subset iff its rows are in the mask subset
        # (segments never straddle subjects, so testing the first row suffices).
        if seg.row_indices and not mask[seg.row_indices[0]]:
            continue
        key = (seg.person_id, seg.phase, seg.occurrence)
        rating = self_report.get(key)
        if rating is None:
            continue
        if isinstance(rating, float) and np.isnan(rating):
            continue
        sr_tiers.append(float(segment_max_tier(seg, tiers)))
        sr_ratings.append(float(rating))

    # --- aerobic-intensity component ---
    ex_tiers: list[float] = []
    ex_ranks: list[float] = []
    for seg in labels.segments:
        if seg.row_indices and not mask[seg.row_indices[0]]:
            continue
        rank = aerobic_stage_rank(seg.phase)
        if rank is None:
            continue
        ex_tiers.append(float(segment_max_tier(seg, tiers)))
        ex_ranks.append(float(rank))

    return _pearson(sr_tiers, sr_ratings) + _pearson(ex_tiers, ex_ranks)


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation of two equal-length lists, or ``0.0`` if undefined.

    Undefined (returns ``0.0``, a neutral contribution to the objective) when
    there are fewer than two pairs or either side is constant (zero variance).
    """
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return 0.0 if not np.isfinite(r) else float(r)


def _select_tau2_tau3(
    scores: np.ndarray,
    labels: PhaseLabels,
    mask: np.ndarray,
    stream_ids: list[tuple[str, str]],
    single_signal,
    tau1: float,
    self_report: Mapping[tuple[str, str, int], float],
    grid: np.ndarray,
) -> tuple[float, float]:
    """Pick (tau2, tau3) above ``tau1`` that best track self-report + intensity (AC 12.3).

    Sweeps all ordered candidate pairs ``(c2, c3)`` from ``grid`` with ``tau1 < c2
    < c3`` and tiers the windows under ``Thresholds(tau1, c2, c3)``. The pair with
    the largest :func:`_tracking_objective` over the ``mask`` subset is chosen —
    i.e. the bands under which assigned tiers best track *both* self-reported
    stress and the aerobic increasing-intensity stages. Ties break toward the
    smaller (c2, c3) so the tiers are no more conservative than needed to achieve
    the best tracking.

    When the grid offers no candidate strictly above ``tau1`` (a degenerate
    training sample), falls back to ``(tau1 + gap, tau1 + 2*gap)``; the final
    :class:`Thresholds` construction still guarantees strict ordering (AC 12.4).
    """
    above = np.unique(grid[grid > tau1])
    best_obj = -np.inf
    best_pair: tuple[float, float] | None = None

    for i in range(len(above)):
        c2 = float(above[i])
        for j in range(i + 1, len(above)):
            c3 = float(above[j])
            taus = _order_thresholds(tau1, c2, c3)
            tiers = tiers_from_scores(
                scores, taus, stream_ids=stream_ids, single_signal=single_signal
            )
            obj = _tracking_objective(labels, tiers, mask, self_report)
            if obj > best_obj + 1e-12:
                best_obj = obj
                best_pair = (c2, c3)

    if best_pair is None:
        gap = MIN_THRESHOLD_GAP
        # Try a single band split if only one candidate sits above tau1.
        if above.size == 1:
            c2 = float(above[0])
            return c2, c2 + gap
        return tau1 + gap, tau1 + 2 * gap
    return best_pair


# ---------------------------------------------------------------------------
# Per-subset calibration (training step of one LOSO fold)
# ---------------------------------------------------------------------------


def calibrate_on_subset(
    scores,
    labels: PhaseLabels,
    self_report: Mapping[tuple[str, str, int], float] | None,
    budget: float,
    subjects: Iterable[str],
    *,
    single_signal=None,
    candidates: Sequence[float] | int | None = None,
) -> Thresholds:
    """Calibrate ``tau1, tau2, tau3`` from the windows of ``subjects`` (AC 12.2-12.4).

    This is one LOSO fold's **training** step: given the per-window ``scores`` and
    ``labels`` for the full table, restrict attention to the rows of ``subjects``
    (the training set) and choose:

    - ``tau1`` as the smallest swept candidate whose rest false-alarm rate over
      the training subjects is within ``budget`` flags per rest hour (AC 12.2,
      :func:`_select_tau1`);
    - ``tau2``/``tau3`` as the candidate pair above ``tau1`` that maximizes the
      tracking objective — tiers rising with self-reported stress and with the
      aerobic increasing-intensity stages (AC 12.3, :func:`_select_tau2_tau3`).

    The returned :class:`~data.schema.Thresholds` is strictly ordered ``tau1 <
    tau2 < tau3`` (AC 12.4, Property 26): the sweep constrains ``tau1 < tau2 <
    tau3`` and :func:`_order_thresholds` + the ``Thresholds`` constructor are the
    final guard.

    ``candidates`` controls the swept grid (explicit thresholds, an int count, or
    ``None`` for a default quantile grid over the training scores). ``single_signal``
    is passed through to :func:`eval.metrics.tiers_from_scores` so candidate tiers
    match the Flagger's single-signal cap (AC 7.7).
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.shape[0] != len(labels.windows):
        raise ValueError(
            f"scores has {scores.shape[0]} rows but labels describes "
            f"{len(labels.windows)} windows; they must align by row index"
        )
    sr: Mapping[tuple[str, str, int], float] = self_report or {}

    mask = _subject_mask(labels, subjects)
    stream_ids = _stream_ids(labels)
    train_scores = scores[mask]

    # Build candidate grids tailored to what each threshold separates.
    #
    # tau1 is the rest false-alarm screen (AC 12.2): its candidates are drawn from
    # the REST-window score distribution of the training subjects, so the grid
    # resolves finely across the rest noise floor where false alarms actually
    # cross the budget — a grid over all training scores would be dominated by the
    # much larger task/exercise scores and would place almost no candidates near
    # the rest floor. Falls back to the full training scores if there is no rest.
    rest_mask = _rest_mask(labels) & mask
    rest_scores = scores[rest_mask] if rest_mask.any() else train_scores
    tau1_grid = _tau1_grid(rest_scores, candidates)

    # tau2/tau3 separate the Notable/Strong bands, which fire on the stress tasks
    # and higher exercise stages: draw their candidates from the task/stage score
    # distribution of the training subjects.
    task_mask = _task_or_stage_mask(labels) & mask
    band_scores = scores[task_mask] if task_mask.any() else train_scores
    band_grid = _candidate_grid(band_scores, candidates)

    tau1 = _select_tau1(
        scores, labels, mask, stream_ids, single_signal, budget, tau1_grid
    )
    tau2, tau3 = _select_tau2_tau3(
        scores, labels, mask, stream_ids, single_signal, tau1, sr, band_grid
    )
    return _order_thresholds(tau1, tau2, tau3)


def _rest_mask(labels: PhaseLabels) -> np.ndarray:
    """Per-window mask of rest windows (Baseline/Rest), where false alarms occur.

    tau1 is chosen against the rest false-alarm budget, so its candidate grid is
    drawn from exactly these windows' scores (:func:`calibrate_on_subset`).
    """
    out = np.zeros(len(labels.windows), dtype=bool)
    for i, w in enumerate(labels.windows):
        if w.is_rest:
            out[i] = True
    return out


def _task_or_stage_mask(labels: PhaseLabels) -> np.ndarray:
    """Per-window mask of task and aerobic-stage windows (where upper tiers live).

    tau2/tau3 separate the Notable/Strong bands, which should fire on the stress
    tasks and the higher exercise stages — so the candidate grid for those bands
    is drawn from the score distribution on exactly those windows (long/short task
    phases and ``Stage<rpm>rpm`` exercise stages).
    """
    out = np.zeros(len(labels.windows), dtype=bool)
    for i, w in enumerate(labels.windows):
        if w.phase_class in (PHASE_CLASS_LONG_TASK, PHASE_CLASS_SHORT_TASK):
            out[i] = True
        elif w.phase_class == PHASE_CLASS_EXERCISE and aerobic_stage_rank(w.phase) is not None:
            out[i] = True
    return out


# ---------------------------------------------------------------------------
# LOSO orchestration (Requirement 12.1)
# ---------------------------------------------------------------------------


def calibrate_loso_folds(
    scores,
    labels: PhaseLabels,
    self_report: Mapping[tuple[str, str, int], float] | None,
    budget: float,
    *,
    single_signal=None,
    candidates: Sequence[float] | int | None = None,
) -> list[tuple[LOSOFold, Thresholds]]:
    """Calibrate one threshold set per LOSO fold (AC 12.1).

    Derives the subjects from ``labels``, generates the LOSO folds
    (:func:`loso_folds`), and runs :func:`calibrate_on_subset` on each fold's
    **training** subjects. Returns ``[(fold, thresholds), ...]`` — the thresholds
    the harness (Task 15) uses to flag that fold's held-out subject. Folds whose
    training set is empty (a single-subject dataset) are skipped, since there is
    nothing to calibrate on.

    Each returned :class:`~data.schema.Thresholds` is strictly ordered (AC 12.4).
    """
    subjects = subjects_of(labels)
    out: list[tuple[LOSOFold, Thresholds]] = []
    for fold in loso_folds(subjects):
        if not fold.train:
            continue
        taus = calibrate_on_subset(
            scores,
            labels,
            self_report,
            budget,
            fold.train,
            single_signal=single_signal,
            candidates=candidates,
        )
        out.append((fold, taus))
    return out


def calibrate_loso(
    scores,
    labels: PhaseLabels,
    self_report: Mapping[tuple[str, str, int], float] | None = None,
    budget: float = 1.0,
    *,
    single_signal=None,
    candidates: Sequence[float] | int | None = None,
) -> Thresholds:
    """Tune ``tau1, tau2, tau3`` by LOSO cross-validation (AC 12.1-12.4).

    The top-level Calibrator entry (design signature ``calibrate_loso(table,
    labels, budget) -> Thresholds``; here ``scores`` stands in for the per-window
    Combined_Score derived from the table, per the module docstring's input
    contract). It calibrates thresholds on **every** LOSO training set
    (:func:`calibrate_loso_folds`) and returns the **median** of each threshold
    across folds.

    *Aggregation choice (documented).* LOSO produces one calibrated triple per
    held-out subject; the deployed threshold set is the per-fold **median** of
    ``tau1``, ``tau2`` and ``tau3`` independently. The median is robust to any one
    subject's idiosyncratic scores (a single outlier fold cannot drag the deployed
    threshold), which is the point of cross-validating rather than hand-picking.
    Per-fold evaluation on the held-out subject (train on the rest, measure on the
    held-out subject) is the harness's job (Task 15) and uses
    :func:`calibrate_loso_folds` directly.

    With a single subject (no non-empty training fold) there is nothing to
    cross-validate on, so the thresholds are calibrated on that lone subject's own
    windows as a graceful fallback.

    The returned :class:`~data.schema.Thresholds` is strictly ordered ``tau1 <
    tau2 < tau3`` (AC 12.4, Property 26): each per-fold triple is ordered, and the
    independent medians are passed through :func:`_order_thresholds` so the
    aggregate is ordered too (component-wise medians could, in principle, tie).
    """
    folds = calibrate_loso_folds(
        scores,
        labels,
        self_report,
        budget,
        single_signal=single_signal,
        candidates=candidates,
    )

    if not folds:
        # Degenerate: 0 or 1 subject. Fall back to calibrating on all windows.
        subjects = subjects_of(labels)
        return calibrate_on_subset(
            scores,
            labels,
            self_report,
            budget,
            subjects,
            single_signal=single_signal,
            candidates=candidates,
        )

    tau1s = np.array([t.tau1 for _, t in folds], dtype=float)
    tau2s = np.array([t.tau2 for _, t in folds], dtype=float)
    tau3s = np.array([t.tau3 for _, t in folds], dtype=float)
    return _order_thresholds(
        float(np.median(tau1s)),
        float(np.median(tau2s)),
        float(np.median(tau3s)),
    )
