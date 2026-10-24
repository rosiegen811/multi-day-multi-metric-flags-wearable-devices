"""Evaluation labels and the self-reported-stress ground-truth join.

This module turns the shared ``Per_Window_Table`` (``data/schema.py``) plus the
dataset's per-phase self-reported stress tables into the labels the Evaluator
(``eval/metrics.py``, Task 12.2) needs: for each window and for each contiguous
phase segment, whether it is a REST phase or a stress TASK phase, and for tasks
whether it is a LONG task (Stroop, TMCT) or a 30-SECOND task (RealOpinion,
OppositeOpinion, Subtract), joined to the one-per-phase self-reported stress
rating used for tier agreement.

Scope of this module (Task 12.1): labels and the ground-truth join only. It does
**not** compute any metric — sensitivity, false-alarms-per-rest-hour, detection
delay, tier agreement, and movement attribution all live in ``eval/metrics.py``
(Task 12.2). What this module does provide is enough structure (per-phase class
and per-phase timing, plus per-window labels) for those metrics to be computed:

- Sensitivity (AC 10.1) reads each stress-task segment's ``phase_class`` to split
  long vs 30-second tasks.
- False-alarms-per-rest-hour (AC 10.2) reads the REST segments and their
  ``start_t`` / ``end_t`` so Task 12.2 can skip the first 120 s after each task
  phase; that 120 s exclusion is a *metric* concern and is intentionally not
  applied here.
- Detection delay (AC 10.3) reads each segment's ``start_t`` relative to the
  first flag in the segment.
- Tier agreement (AC 10.4) reads ``attach_self_report`` output, which carries the
  one-per-phase self-reported stress rating so agreement is judged per phase
  (the ratings are recorded once per phase, before/after — never per window; see
  the README "Evaluation" note).

"Phases from tags" (design "labels — eval/labels.py"): tags are mapped to named
phases *upstream* by the Windower (``data/windows.assign_phases`` /
``window``), which writes the phase name onto every ``Per_Window_Table`` row
(AC 3.3). This module therefore derives rest/task/long/short from that existing
``phase`` column rather than re-reading tags. :func:`phase_labels` accepts an
optional ``tags`` argument only to keep the design's documented signature; it is
not required and, when omitted, the table's ``phase`` column is the single source
of truth.

It is dependency-light: only ``pandas``, ``numpy``, the standard library, and the
sibling ``data`` modules (``data/schema.py`` for the column names,
``data/windows.py`` for the canonical phase names, ``data/physionet_load.py`` for
``read_cp1252`` / ``cohort_of``). Side effects are confined to
:func:`load_stress_levels`, which reads the two self-report CSVs.

Requirements covered:
- 10.1 Expose each stress-task segment's long-vs-30-second class so sensitivity
       can be reported separately for long and 30-second tasks.
- 10.2 Expose rest (baseline + rest) segments and their start/end times so the
       false-alarm-per-rest-hour metric can exclude the 120 s post-task tail.
- 10.3 Expose each segment's start time so detection delay (phase start to first
       flag) can be computed.
- 10.4 Load and join per-phase self-reported stress so tier agreement can be
       evaluated per phase.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from data.physionet_load import cohort_of, read_cp1252
from data.schema import COLUMNS
from data.windows import (
    LONG_TASK_PHASES,
    PHASE_BASELINE,
    PHASE_REST,
    SHORT_TASK_PHASES,
)

__all__ = [
    "PHASE_CLASS_REST",
    "PHASE_CLASS_LONG_TASK",
    "PHASE_CLASS_SHORT_TASK",
    "PHASE_CLASS_EXERCISE",
    "PHASE_CLASS_OTHER",
    "REST_PHASES",
    "STRESS_TASK_PHASES",
    "classify_phase",
    "is_rest",
    "is_task",
    "WindowLabel",
    "PhaseLabels",
    "phase_labels",
    "PhaseSegment",
    "phase_segments",
    "StressRating",
    "load_stress_levels",
    "attach_self_report",
]


# ---------------------------------------------------------------------------
# Phase-class vocabulary (Requirements 10.1, 10.2)
# ---------------------------------------------------------------------------
#
# The evaluation metrics care about four kinds of window, derived from the
# Windower's phase name:
#   REST         - a quiet segment: Baseline or Rest. Known-quiet time; the
#                  denominator for false alarms per rest hour (AC 10.2).
#   LONG_TASK    - a long cognitive stress task: Stroop or TMCT (AC 10.1).
#   SHORT_TASK   - a 30-second cognitive stress task: RealOpinion /
#                  OppositeOpinion / Subtract (AC 10.1).
#   EXERCISE     - an aerobic/anaerobic stage (WarmUp / Stage*rpm / Sprint* /
#                  CoolDown*). Not a stress task and not quiet rest, so it is
#                  neither a sensitivity target nor part of the rest false-alarm
#                  denominator; movement attribution (AC 10.5, Task 12.2) is the
#                  metric that reads exercise-session flags. Kept as its own
#                  class so Task 12.2 can treat it explicitly rather than having
#                  it masquerade as rest or task.
#   OTHER        - any window with no phase label (``phase is None``) or an
#                  unrecognized phase name. Excluded from stress-task and rest
#                  metrics alike.

PHASE_CLASS_REST = "rest"
PHASE_CLASS_LONG_TASK = "long_task"
PHASE_CLASS_SHORT_TASK = "short_task"
PHASE_CLASS_EXERCISE = "exercise"
PHASE_CLASS_OTHER = "other"

# The quiet phases (AC 10.2 false-alarm denominator) and the stress-task phases
# (AC 10.1 sensitivity numerator), by canonical Windower phase name.
REST_PHASES: frozenset[str] = frozenset({PHASE_BASELINE, PHASE_REST})
STRESS_TASK_PHASES: frozenset[str] = frozenset(LONG_TASK_PHASES) | frozenset(
    SHORT_TASK_PHASES
)

# Prefixes/names that identify an aerobic/anaerobic stage phase (see the
# Windower protocol timing tables: WarmUp, Stage<rpm>rpm, Sprint<n>, CoolDown<n>).
_EXERCISE_PREFIXES: tuple[str, ...] = ("WarmUp", "Stage", "Sprint", "CoolDown")


def classify_phase(phase: str | None) -> str:
    """Return the phase class for a Windower phase name.

    Maps the ``phase`` column value on a ``Per_Window_Table`` row to one of
    :data:`PHASE_CLASS_REST`, :data:`PHASE_CLASS_LONG_TASK`,
    :data:`PHASE_CLASS_SHORT_TASK`, :data:`PHASE_CLASS_EXERCISE`, or
    :data:`PHASE_CLASS_OTHER`. ``None`` (a window outside any labeled phase, per
    AC 3.3) and unrecognized names classify as ``OTHER``.
    """
    if phase is None or (isinstance(phase, float) and np.isnan(phase)):
        return PHASE_CLASS_OTHER
    if phase in REST_PHASES:
        return PHASE_CLASS_REST
    if phase in LONG_TASK_PHASES:
        return PHASE_CLASS_LONG_TASK
    if phase in SHORT_TASK_PHASES:
        return PHASE_CLASS_SHORT_TASK
    if phase.startswith(_EXERCISE_PREFIXES):
        return PHASE_CLASS_EXERCISE
    return PHASE_CLASS_OTHER


def is_rest(phase: str | None) -> bool:
    """True if ``phase`` is a quiet rest phase (Baseline or Rest) (AC 10.2)."""
    return classify_phase(phase) == PHASE_CLASS_REST


def is_task(phase: str | None) -> bool:
    """True if ``phase`` is a stress task (long or 30-second) (AC 10.1)."""
    return classify_phase(phase) in (PHASE_CLASS_LONG_TASK, PHASE_CLASS_SHORT_TASK)


# ---------------------------------------------------------------------------
# Per-window labels (Requirements 10.1, 10.2, 10.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowLabel:
    """Rest/task classification for a single ``Per_Window_Table`` row.

    ``row_index`` is the row's positional index in the table passed to
    :func:`phase_labels` (so Task 12.2 can line labels up with flags computed on
    the same table). ``phase`` is the raw Windower phase name (or ``None``);
    ``phase_class`` is its :func:`classify_phase` result. ``is_rest`` / ``is_task``
    are the boolean shortcuts the sensitivity and false-alarm metrics use.
    """

    row_index: int
    person_id: str
    cohort: str
    session: str
    phase: str | None
    t_win: float
    phase_class: str

    @property
    def is_rest(self) -> bool:
        return self.phase_class == PHASE_CLASS_REST

    @property
    def is_task(self) -> bool:
        return self.phase_class in (PHASE_CLASS_LONG_TASK, PHASE_CLASS_SHORT_TASK)


@dataclass(frozen=True)
class PhaseLabels:
    """Per-window rest/task labels plus the per-phase segments for a table.

    ``windows`` is one :class:`WindowLabel` per table row, in table order.
    ``segments`` is the list of contiguous per-phase segments from
    :func:`phase_segments`. Both views are derived from the same ``phase`` column
    so they always agree.
    """

    windows: list[WindowLabel]
    segments: list["PhaseSegment"]


def _normalize_phase(value: object) -> str | None:
    """Coerce a raw ``phase`` cell to ``str`` or ``None``.

    Treats pandas/NumPy missing values (``None``, ``NaN``, ``pd.NA``) as "no
    labeled phase" (``None``); everything else is stringified.
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def phase_labels(
    table: pd.DataFrame, tags: Sequence[float] | None = None
) -> PhaseLabels:
    """Derive rest-vs-task (and long-vs-30-second) labels for every window.

    For each row of ``table`` (a ``Per_Window_Table``), classify its ``phase``
    into a phase class (:func:`classify_phase`) and record a :class:`WindowLabel`.
    Also build the per-phase segments via :func:`phase_segments`. Returns both in
    a :class:`PhaseLabels`.

    The ``tags`` argument is accepted to match the design's documented signature
    (``phase_labels(table, tags)``) but is **not used**: phases are assigned from
    tags upstream by the Windower, which has already written the resulting phase
    name onto each row's ``phase`` column (AC 3.3). The table's ``phase`` column
    is therefore the single source of truth here. Passing ``tags`` is harmless.

    ``table`` is read only; it is not mutated. Rows are taken in table order and
    ``row_index`` is the 0-based positional index within ``table`` (independent
    of any DataFrame index labels), so Task 12.2 can align labels with flags
    computed over the same row order.
    """
    _require_columns(table)

    windows: list[WindowLabel] = []
    for pos in range(len(table)):
        row = table.iloc[pos]
        phase = _normalize_phase(row["phase"])
        windows.append(
            WindowLabel(
                row_index=pos,
                person_id=str(row["person_id"]),
                cohort=str(row["cohort"]),
                session=str(row["session"]),
                phase=phase,
                t_win=float(row["t_win"]),
                phase_class=classify_phase(phase),
            )
        )

    segments = phase_segments(table)
    return PhaseLabels(windows=windows, segments=segments)


# ---------------------------------------------------------------------------
# Per-phase segments (Requirements 10.1, 10.2, 10.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseSegment:
    """One contiguous run of same-phase windows for one person/session.

    A segment is a maximal run of consecutive table rows (in table order, within
    a single ``(person_id, session)``) that share the same ``phase`` name. The
    Windower emits windows in increasing ``t_win`` order per person/session, so a
    run of equal phase names is exactly one visit to that phase.

    - ``occurrence`` is the 1-based index of this segment among same-named
      segments for the same ``(person_id, session)``. It distinguishes repeated
      phases that share a Windower name — notably the two ``Rest`` visits, which
      the self-report tables record as "First Rest" and "Second Rest" (AC 10.4).
      The first ``Rest`` segment in protocol order is occurrence 1 (First Rest),
      the second is occurrence 2 (Second Rest).
    - ``start_t`` / ``end_t`` are the first and last window end times in the
      segment. They give Task 12.2 the timing it needs for detection delay (phase
      start to first flag, AC 10.3) and for excluding the first 120 s after a task
      phase from the rest false-alarm denominator (AC 10.2).
    - ``row_indices`` are the 0-based positional row indices (into the table
      passed to :func:`phase_segments`) that make up the segment, so flags aligned
      by row index can be bucketed per phase.
    """

    person_id: str
    cohort: str
    session: str
    phase: str
    phase_class: str
    occurrence: int
    start_t: float
    end_t: float
    row_indices: tuple[int, ...]

    @property
    def is_rest(self) -> bool:
        return self.phase_class == PHASE_CLASS_REST

    @property
    def is_task(self) -> bool:
        return self.phase_class in (PHASE_CLASS_LONG_TASK, PHASE_CLASS_SHORT_TASK)


def phase_segments(table: pd.DataFrame) -> list[PhaseSegment]:
    """Group a ``Per_Window_Table`` into contiguous per-phase segments.

    Walks the table in row order and starts a new :class:`PhaseSegment` whenever
    the ``(person_id, session)`` or the ``phase`` name changes. Windows with no
    labeled phase (``phase is None``) are skipped — they belong to no phase
    segment (``OTHER``) and are not part of any rest or task metric. The returned
    list is in table order.

    ``occurrence`` is assigned per ``(person_id, session, phase)`` in the order
    segments are encountered, so the two ``Rest`` visits become occurrence 1
    (First Rest) and 2 (Second Rest), matching the self-report column mapping in
    :func:`load_stress_levels` (AC 10.4).

    ``table`` is read only. Row indices in ``row_indices`` are 0-based positional
    indices into ``table``.
    """
    _require_columns(table)

    segments: list[PhaseSegment] = []
    occurrence_counts: dict[tuple[str, str, str], int] = {}

    # Accumulator for the segment currently being built.
    cur_key: tuple[str, str] | None = None
    cur_phase: str | None = None
    cur_rows: list[int] = []
    cur_person = cur_cohort = cur_session = ""
    cur_t: list[float] = []

    def _flush() -> None:
        nonlocal cur_rows, cur_t
        if cur_phase is None or not cur_rows:
            cur_rows = []
            cur_t = []
            return
        occ_key = (cur_person, cur_session, cur_phase)
        occurrence_counts[occ_key] = occurrence_counts.get(occ_key, 0) + 1
        segments.append(
            PhaseSegment(
                person_id=cur_person,
                cohort=cur_cohort,
                session=cur_session,
                phase=cur_phase,
                phase_class=classify_phase(cur_phase),
                occurrence=occurrence_counts[occ_key],
                start_t=min(cur_t),
                end_t=max(cur_t),
                row_indices=tuple(cur_rows),
            )
        )
        cur_rows = []
        cur_t = []

    for pos in range(len(table)):
        row = table.iloc[pos]
        person = str(row["person_id"])
        session = str(row["session"])
        phase = _normalize_phase(row["phase"])
        key = (person, session)

        # A phase break is a change of person/session or of phase name. An
        # unlabeled window (phase is None) always closes the current segment.
        if key != cur_key or phase != cur_phase or phase is None:
            _flush()
            cur_key = key
            cur_phase = phase
            cur_person = person
            cur_cohort = str(row["cohort"])
            cur_session = session

        if phase is not None:
            cur_rows.append(pos)
            cur_t.append(float(row["t_win"]))

    _flush()
    return segments


# ---------------------------------------------------------------------------
# Self-reported stress tables (Requirement 10.4)
# ---------------------------------------------------------------------------
#
# Stress_Level_v1.csv / Stress_Level_v2.csv hold one self-reported stress rating
# (0-10) per participant per protocol phase, recorded once per phase (before /
# after), never per window. Their human-readable column headers differ from the
# Windower phase names, so each header is mapped to a (phase_name, occurrence)
# pair. The two rest columns ("First Rest", "Second Rest") both map to the
# Windower "Rest" phase and are distinguished by occurrence (1 and 2), matching
# PhaseSegment.occurrence.

# V1 header -> (Windower phase name, occurrence). Protocol order (README):
# Baseline, Stroop, Rest(1), TMCT, Rest(2), RealOpinion, OppositeOpinion, Subtract.
_V1_COLUMN_MAP: Mapping[str, tuple[str, int]] = MappingProxyType(
    {
        "Baseline": (PHASE_BASELINE, 1),
        "Stroop": ("Stroop", 1),
        "First Rest": (PHASE_REST, 1),
        "TMCT": ("TMCT", 1),
        "Second Rest": (PHASE_REST, 2),
        "Real Opinion": ("RealOpinion", 1),
        "Opposite Opinion": ("OppositeOpinion", 1),
        "Subtract": ("Subtract", 1),
    }
)

# V2 header -> (Windower phase name, occurrence). Protocol order (README):
# Baseline, TMCT, Rest(1), RealOpinion, OppositeOpinion, Rest(2), Subtract.
_V2_COLUMN_MAP: Mapping[str, tuple[str, int]] = MappingProxyType(
    {
        "Baseline": (PHASE_BASELINE, 1),
        "TMCT": ("TMCT", 1),
        "First Rest": (PHASE_REST, 1),
        "Real Opinion": ("RealOpinion", 1),
        "Opposite Opinion": ("OppositeOpinion", 1),
        "Second Rest": (PHASE_REST, 2),
        "Subtract": ("Subtract", 1),
    }
)


@dataclass(frozen=True)
class StressRating:
    """One self-reported stress rating for a participant/phase visit.

    ``rating`` is the 0-10 self-report for the phase named ``phase`` on its
    ``occurrence``-th visit (1 = First Rest, 2 = Second Rest for the repeated
    ``Rest`` phase). ``rating`` is ``NaN`` when the source cell was blank.
    """

    person_id: str
    phase: str
    occurrence: int
    rating: float


# The join key used throughout: a (person_id, phase_name, occurrence) triple.
StressKey = tuple[str, str, int]


def _parse_stress_csv(text: str, column_map: Mapping[str, tuple[str, int]]) -> dict:
    """Parse one self-report CSV body into ``{(pid, phase, occ): rating}``.

    ``text`` is the already-decoded CSV text (first cell of the header row is
    empty; remaining header cells are the human-readable phase names; each data
    row is ``participant_id`` followed by one rating per phase column). Unknown
    header columns are ignored (defensive against extra columns). A blank cell
    becomes ``NaN``.
    """
    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return {}

    header = [cell.strip() for cell in rows[0]]
    # Column 0 is the (empty) participant-id corner; map the remaining columns.
    col_targets: list[tuple[int, tuple[str, int]]] = []
    for idx in range(1, len(header)):
        name = header[idx]
        if name in column_map:
            col_targets.append((idx, column_map[name]))

    out: dict[StressKey, float] = {}
    for data_row in rows[1:]:
        cells = [cell.strip() for cell in data_row]
        pid = cells[0]
        if not pid:
            continue
        for idx, (phase_name, occurrence) in col_targets:
            raw = cells[idx] if idx < len(cells) else ""
            rating = float(raw) if raw != "" else float("nan")
            out[(pid, phase_name, occurrence)] = rating
    return out


def load_stress_levels(path_v1: str, path_v2: str) -> dict:
    """Load both self-report tables into a ``(pid, phase, occurrence) -> rating`` map.

    Reads ``Stress_Level_v1.csv`` (Cohort_V1, ``S*`` participants) and
    ``Stress_Level_v2.csv`` (Cohort_V2, ``f*`` participants), mapping each CSV's
    human-readable phase column to the canonical Windower phase name and its
    occurrence (see ``_V1_COLUMN_MAP`` / ``_V2_COLUMN_MAP``). The two rest
    columns ("First Rest" / "Second Rest") both map to the ``Rest`` phase and are
    kept distinct by occurrence (1 and 2), matching
    :attr:`PhaseSegment.occurrence` so :func:`attach_self_report` can join them
    (AC 10.4).

    The files are decoded with :func:`~data.physionet_load.read_cp1252` for
    consistency with the dataset's other reference files; the stress CSVs are
    ASCII, so this is a safe superset. Returns a plain dict keyed on
    ``(person_id, phase_name, occurrence)`` with float ratings (``NaN`` for a
    blank cell). Keys from the two files never collide because V1 ids start with
    ``S`` and V2 ids with ``f``.
    """
    merged: dict[StressKey, float] = {}
    merged.update(_parse_stress_csv(read_cp1252(path_v1), _V1_COLUMN_MAP))
    merged.update(_parse_stress_csv(read_cp1252(path_v2), _V2_COLUMN_MAP))
    return merged


def attach_self_report(
    segments: Iterable[PhaseSegment],
    stress_levels: Mapping[StressKey, float],
) -> list[tuple[PhaseSegment, float | None]]:
    """Join per-phase self-reported stress onto phase segments (AC 10.4).

    For each :class:`PhaseSegment`, look up the self-report rating for
    ``(person_id, phase, occurrence)`` in ``stress_levels`` (as produced by
    :func:`load_stress_levels`). Returns a list of ``(segment, rating)`` pairs in
    the input order, where ``rating`` is the float 0-10 self-report, or ``None``
    when the dataset has no rating for that phase visit (e.g. exercise stages,
    the ``OTHER`` class, or an early-stopped participant who never reached the
    phase). A ``NaN`` rating recorded as a blank cell in the source table is
    passed through as ``NaN`` (distinct from ``None`` = no entry at all).

    Tier agreement (AC 10.4) is judged per phase precisely because the rating is
    one-per-phase; Task 12.2 pairs each segment's rating with the tier assigned
    over that segment and computes the rank correlation across segments.
    """
    joined: list[tuple[PhaseSegment, float | None]] = []
    for seg in segments:
        key: StressKey = (seg.person_id, seg.phase, seg.occurrence)
        rating = stress_levels.get(key, None)
        joined.append((seg, rating))
    return joined


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_columns(table: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``table`` lacks the Per_Window_Table columns used here.

    Only the columns this module reads are required (``person_id``, ``cohort``,
    ``session``, ``phase``, ``t_win``); the full-contract check lives in
    ``data/schema.validate_per_window_table``. Checking here gives a clear error
    instead of a ``KeyError`` deep in a loop.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(table)!r}")
    needed = ("person_id", "cohort", "session", "phase", "t_win")
    missing = [c for c in needed if c not in table.columns]
    if missing:
        raise ValueError(
            f"table is missing required Per_Window_Table columns {missing}; "
            f"expected at least {list(needed)} (full contract: {list(COLUMNS)})"
        )
