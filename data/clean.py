"""Cleaner for the Wearable Flag Pipeline.

Applies the documented known-data-issue cases from ``data_constraints.txt`` to a
loaded :class:`~data.physionet_load.ParticipantSignals` record so that corrupted
or invalid segments do not distort downstream results. The constraint text is the
authoritative list of issues (dataset README / ``data/external/data_constraints.txt``);
it is parsed once into a :class:`Constraints` structure keyed on
``(participant, session)`` and then each documented case is applied as an
*independently testable rule* via a dispatch table (design "Cleaner —
data/clean.py" and its case table).

Side effects (reading the cp1252-encoded constraint file) are isolated to
:func:`parse_constraints` / :func:`load_constraints`; the cleaning logic itself is
pure over in-memory ``ParticipantSignals`` and is testable with synthetic records
(the real ``data/sessions`` tree is empty / gitignored). This module depends only
on ``numpy``, the standard library, and the sibling ``physionet_load`` records.

The documented cases and the acceptance criterion each satisfies (Requirement 2):

================================  ===========================================  ====
Case                              Rule                                         AC
================================  ===========================================  ====
S02 stress                        Truncate each signal at its duplicate-start  2.1
                                  row from ``data_constraints.txt``.
f07 stress                        Exclude from HR/RMSSD/Skin_Temperature       2.2
                                  analysis (keep EDA/ACC): dock covered the
                                  PPG + temperature sensors.
f14 stress                        Join the baseline file (``_a``) and rest     2.3
                                  file (``_b``) segments into one recording.
S11, S16 aerobic/anaerobic        Join the split ``_a``/``_b`` segments.       2.4
S03, S07, S06 aerobic/anaerobic   Keep only the phases the participant         2.5
                                  completed (ended early).
S01 anaerobic                     Produce HR, mark RMSSD unavailable           2.6
                                  (the downloaded IBI file is empty).
f14 through f18                   Restrict analysis to the stress session.     2.7
missing demographics              Retain for signal analysis, mark             2.8
                                  demographics unavailable.
================================  ===========================================  ====

Sessions are the uppercase folder names used by the loader: ``STRESS``,
``AEROBIC``, ``ANAEROBIC``. Participant ids follow the dataset convention
(uppercase ``S..`` for Cohort_V1, lowercase ``f..`` for Cohort_V2).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping

import numpy as np

from .physionet_load import (
    IBISeries,
    ParticipantSignals,
    SignalSeries,
    read_cp1252,
)

__all__ = [
    "SESSION_STRESS",
    "SESSION_AEROBIC",
    "SESSION_ANAEROBIC",
    "SignalTruncation",
    "ParticipantConstraint",
    "Constraints",
    "parse_constraints",
    "load_constraints",
    "truncate_signals",
    "exclude_signals",
    "join_segments",
    "keep_completed_phases",
    "mark_rmssd_unavailable",
    "restrict_to_stress",
    "mark_demographics_unavailable",
    "clean",
]


# Session folder names, matching the loader / schema convention.
SESSION_STRESS = "STRESS"
SESSION_AEROBIC = "AEROBIC"
SESSION_ANAEROBIC = "ANAEROBIC"

# The signal attributes on ParticipantSignals that carry physiological channels.
# Order is the loader's expected-signal order.
_SIGNAL_ATTRS: tuple[str, ...] = ("hr", "temp", "acc", "ibi", "tags")

# The signal names (and their ParticipantSignals attribute) excluded for the
# dock-on f07 case: HR, RMSSD (derived from IBI), and Skin_Temperature. EDA and
# ACC remain valid (data_constraints.txt f07).
_F07_EXCLUDED: tuple[tuple[str, str], ...] = (
    ("HR", "hr"),
    ("RMSSD", "ibi"),
    ("Skin_Temperature", "temp"),
)

# Cohort_V2 participants f14..f18 are restricted to the stress session (AC 2.7).
_STRESS_ONLY_PIDS: frozenset[str] = frozenset(f"f{n:02d}" for n in range(14, 19))


# ---------------------------------------------------------------------------
# Constraint records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalTruncation:
    """A documented duplicate-start row for one signal file.

    ``signal`` is the ParticipantSignals attribute to truncate (e.g. ``"acc"``,
    ``"temp"``). ``start_row`` is the 1-based data row, as written in
    ``data_constraints.txt``, at which duplicated raw values begin; samples from
    that row onward are dropped. A signal whose duplicate-start row is ambiguous
    (HR/IBI, derived from BVP for S02) is intentionally absent from the map.
    """

    signal: str
    start_row: int


@dataclass
class ParticipantConstraint:
    """All documented issues for one ``(participant, session)`` pair.

    Flags map directly onto the dispatch rules; defaults mean "no issue". The
    structure is keyed by :class:`Constraints` so each rule can be looked up and
    applied independently.
    """

    participant: str
    session: str
    # AC 2.1: per-signal duplicate-start rows (signal attr -> truncation).
    truncations: dict[str, SignalTruncation] = field(default_factory=dict)
    # AC 2.2: dock-on; exclude HR/RMSSD/Skin_Temperature.
    exclude_dock_on: bool = False
    # AC 2.3, 2.4: recording was split across two files and must be joined.
    split_recording: bool = False
    # AC 2.5: participant ended the protocol early; keep completed phases only.
    ended_early: bool = False
    # AC 2.6: IBI file is empty; produce HR, mark RMSSD unavailable.
    empty_ibi: bool = False


class Constraints:
    """Documented data issues keyed on ``(participant, session)``.

    Built by :func:`parse_constraints` / :func:`load_constraints` from the
    cp1252 constraint text. Lookups are tolerant: a pair with no documented issue
    returns an empty :class:`ParticipantConstraint`, so callers need not special-
    case the common "nothing wrong" path.
    """

    def __init__(self, entries: Mapping[tuple[str, str], ParticipantConstraint] | None = None):
        self._entries: dict[tuple[str, str], ParticipantConstraint] = dict(entries or {})

    def get(self, participant: str, session: str) -> ParticipantConstraint:
        """Return the constraint for ``(participant, session)`` or an empty one."""
        key = (participant, session.upper())
        entry = self._entries.get(key)
        if entry is None:
            return ParticipantConstraint(participant=participant, session=session.upper())
        return entry

    def __contains__(self, key: tuple[str, str]) -> bool:
        participant, session = key
        return (participant, session.upper()) in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def items(self):
        return self._entries.items()


# ---------------------------------------------------------------------------
# Constraint parsing (cp1252 side effect isolated here)
# ---------------------------------------------------------------------------

# Session headers appear on their own line in data_constraints.txt.
_SESSION_HEADERS = {SESSION_STRESS, SESSION_AEROBIC, SESSION_ANAEROBIC}

# A participant entry opens with its id ("S02:", "f07:"); a leading bullet glyph
# (cp1252 0x95) and surrounding whitespace are stripped before matching.
_ENTRY_RE = re.compile(r"^(?P<pid>[A-Za-z]\d+)\s*:\s*(?P<body>.*)$")

# Duplicate-start rows in the S02 note, e.g. "ACC.csv: row 49,545" or
# "EDA.csv and TEMP.csv: row 6,195". File stem -> ParticipantSignals attribute.
_FILE_TO_ATTR = {"ACC": "acc", "TEMP": "temp", "EDA": "eda", "BVP": "bvp", "HR": "hr", "IBI": "ibi"}
# Only these signals are represented on ParticipantSignals and are truncatable
# with an unambiguous documented row for S02 (ACC, TEMP; EDA/BVP are not loaded
# as ParticipantSignals attributes, HR/IBI are ambiguous for S02).
_TRUNCATABLE_ATTRS = frozenset({"acc", "temp"})

_DUP_ROW_RE = re.compile(
    r"([A-Za-z0-9 ]+?\.csv(?:\s+and\s+[A-Za-z0-9 ]+?\.csv)*)\s*:?\s*row\s*([\d,]+)",
    re.IGNORECASE,
)
_CSV_NAME_RE = re.compile(r"([A-Za-z0-9]+)\.csv", re.IGNORECASE)


def _strip_bullet(line: str) -> str:
    """Remove a leading cp1252 bullet glyph and surrounding whitespace."""
    return line.lstrip("\u2022\u00b7\x95\t -").strip()


def _parse_truncations(body: str) -> dict[str, SignalTruncation]:
    """Extract per-signal duplicate-start rows from an S02-style note body."""
    truncations: dict[str, SignalTruncation] = {}
    for files_part, row_text in _DUP_ROW_RE.findall(body):
        start_row = int(row_text.replace(",", ""))
        for stem in _CSV_NAME_RE.findall(files_part):
            attr = _FILE_TO_ATTR.get(stem.upper())
            if attr in _TRUNCATABLE_ATTRS:
                truncations[attr] = SignalTruncation(signal=attr, start_row=start_row)
    return truncations


def _mentions_split_files(body: str) -> bool:
    """Return True if the note references companion ``_a``/``_b`` file segments.

    f14, S11, and S16 all phrase the split as ``<pid>_a`` / ``<pid>_b`` (e.g.
    "File S11_a ... and S11_b ..."), so the presence of both suffixes marks the
    recording as split.
    """
    low = body.lower()
    return ("_a" in low) and ("_b" in low)


def parse_constraints(text: str) -> Constraints:
    """Parse the ``data_constraints.txt`` contents into a :class:`Constraints`.

    ``text`` is the already-decoded (cp1252) constraint document. Section headers
    (``STRESS`` / ``AEROBIC`` / ``ANAEROBIC``) set the current session; each
    subsequent ``<pid>: <note>`` line is classified into the documented cases by
    keyword. Unrecognized note wording still yields a :class:`ParticipantConstraint`
    for the pair (so the participant/session is at least known), with no flags set.
    """
    entries: dict[tuple[str, str], ParticipantConstraint] = {}
    current_session: str | None = None

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped in _SESSION_HEADERS:
            current_session = stripped
            continue

        entry_line = _strip_bullet(raw_line)
        m = _ENTRY_RE.match(entry_line)
        if not m or current_session is None:
            continue

        pid = m.group("pid")
        body = m.group("body")
        low = body.lower()

        constraint = ParticipantConstraint(participant=pid, session=current_session)

        if "duplicated" in low or "duplicate" in low:
            constraint.truncations = _parse_truncations(body)
        if "dock" in low:
            constraint.exclude_dock_on = True
        # Split across companion _a/_b files (f14, S11, S16). The notes describe
        # the split either by naming the files (e.g. "File S11_a ... and S11_b")
        # or by the cause ("Bluetooth connection was lost ... File ..._a ...").
        if _mentions_split_files(body):
            constraint.split_recording = True
        if ("ended the protocol early" in low) or ("ended early" in low) or (
            "missing" in low and "sprint" in low
        ):
            constraint.ended_early = True
        if "ibi" in low and "empty" in low:
            constraint.empty_ibi = True

        entries[(pid, current_session)] = constraint

    return Constraints(entries)


def load_constraints(path: str) -> Constraints:
    """Read and parse ``data_constraints.txt`` from ``path`` (cp1252).

    Thin wrapper that decodes the file with :func:`~data.physionet_load.read_cp1252`
    (Requirement 1.6) and delegates to :func:`parse_constraints`.
    """
    return parse_constraints(read_cp1252(path))


# ---------------------------------------------------------------------------
# Independent cleaning rules (each an AC, each independently testable)
# ---------------------------------------------------------------------------


def _copy_signals(signals: ParticipantSignals) -> ParticipantSignals:
    """Return a deep-ish copy so rules do not mutate the caller's record.

    Numpy arrays are copied; the dataclass and its list fields are duplicated so
    truncation/exclusion on the result leaves the input untouched.
    """
    return copy.deepcopy(signals)


def _truncate_signal_series(series: SignalSeries, start_row: int) -> SignalSeries:
    """Return ``series`` with samples from 1-based ``start_row`` onward dropped.

    ``start_row`` counts data-sample rows (the samples in ``values``), matching the
    "row N" convention of the duplicate-start notes: rows ``1..start_row-1`` are
    kept, ``start_row..`` are discarded.
    """
    keep = max(start_row - 1, 0)
    return replace(series, values=series.values[:keep].copy())


def truncate_signals(
    signals: ParticipantSignals, constraint: ParticipantConstraint
) -> ParticipantSignals:
    """AC 2.1 - truncate each signal at its documented duplicate-start row (S02).

    For every signal with a documented truncation (e.g. ACC at row 49545, TEMP at
    row 6195 for S02), drop the duplicated tail. Signals without a documented,
    unambiguous row (HR/IBI for S02) are left unchanged.
    """
    result = _copy_signals(signals)
    for attr, trunc in constraint.truncations.items():
        series = getattr(result, attr, None)
        if isinstance(series, SignalSeries):
            setattr(result, attr, _truncate_signal_series(series, trunc.start_row))
    return result


def exclude_signals(signals: ParticipantSignals) -> ParticipantSignals:
    """AC 2.2 - exclude f07 stress from HR/RMSSD/Skin_Temperature analysis.

    The protection dock covered the PPG and temperature sensors, so only EDA and
    ACC are valid. HR, IBI (the source of RMSSD), and TEMP are cleared to ``None``
    and recorded in ``absent`` so downstream analysis skips them; ACC is retained.
    """
    result = _copy_signals(signals)
    for name, attr in _F07_EXCLUDED:
        setattr(result, attr, None)
        if name not in result.absent:
            result.absent.append(name)
    return result


def _concat_signal_series(a: SignalSeries, b: SignalSeries) -> SignalSeries:
    """Concatenate two signal segments in time order into one series.

    Timing and name are taken from the earlier-starting segment; the two value
    arrays are concatenated along the sample axis (works for both 1-D single-
    channel and 2-D ``(N, 3)`` ACC arrays).
    """
    first, second = (a, b) if a.start_utc <= b.start_utc else (b, a)
    values = np.concatenate([first.values, second.values], axis=0)
    return replace(first, values=values)


def _concat_ibi_series(a: IBISeries, b: IBISeries) -> IBISeries:
    """Concatenate two IBI segments in time order into one series."""
    first, second = (a, b) if a.start_utc <= b.start_utc else (b, a)
    offsets = np.concatenate([first.offsets_s, second.offsets_s], axis=0)
    return replace(first, offsets_s=offsets)


def join_segments(
    segment_a: ParticipantSignals, segment_b: ParticipantSignals
) -> ParticipantSignals:
    """AC 2.3, 2.4 - join two split recording segments into one recording.

    Models the ``_a``/``_b`` split documented for f14 (stress: baseline file +
    rest file), S11 and S16 (aerobic/anaerobic: Bluetooth drop split the block).
    The two ``ParticipantSignals`` are joined signal-by-signal in time order:
    ``SignalSeries`` and ``IBISeries`` are concatenated; tag timestamp lists are
    merged and sorted; a signal present in only one segment is carried through. A
    signal absent from both remains recorded in ``absent``.

    The design's ``load_participant`` loads a single folder, so the join is modeled
    here at the Cleaner level (see task note) and dispatched from :func:`clean`
    when a constraint marks a split.
    """
    result = _copy_signals(segment_a)

    for attr in ("hr", "temp", "acc"):
        sa = getattr(segment_a, attr)
        sb = getattr(segment_b, attr)
        if isinstance(sa, SignalSeries) and isinstance(sb, SignalSeries):
            setattr(result, attr, _concat_signal_series(sa, sb))
        elif isinstance(sb, SignalSeries):
            setattr(result, attr, copy.deepcopy(sb))
        elif isinstance(sa, SignalSeries):
            setattr(result, attr, copy.deepcopy(sa))

    ia, ib = segment_a.ibi, segment_b.ibi
    if isinstance(ia, IBISeries) and isinstance(ib, IBISeries):
        result.ibi = _concat_ibi_series(ia, ib)
    elif isinstance(ib, IBISeries):
        result.ibi = copy.deepcopy(ib)
    elif isinstance(ia, IBISeries):
        result.ibi = copy.deepcopy(ia)

    ta, tb = segment_a.tags, segment_b.tags
    if ta is not None or tb is not None:
        merged = list(ta or []) + list(tb or [])
        result.tags = sorted(merged)

    # A signal is absent only if absent from both segments.
    present_attrs = {
        "HR": result.hr,
        "TEMP": result.temp,
        "ACC": result.acc,
        "IBI": result.ibi,
        "tags": result.tags,
    }
    result.absent = [name for name, value in present_attrs.items() if value is None]
    return result


def keep_completed_phases(
    signals: ParticipantSignals, completed_tags: int | None = None
) -> ParticipantSignals:
    """AC 2.5 - keep only the phases the participant completed (early stop).

    For S03/S07 (aerobic, stopped at a lower rpm) and S06 (anaerobic, last sprint
    missing), the participant ended the protocol early, so only completed phases
    should be analyzed. Phase boundaries are defined by button-press tags; this
    rule trims the tag list to the ``completed_tags`` boundaries that were actually
    reached. ``None`` leaves the tags as loaded (the recording already stops where
    the participant stopped - no synthetic tail to remove), which is the common
    real-data case.

    Trimming the tags is sufficient because the Windower derives phases from tags;
    samples past the last completed phase simply fall outside any labeled phase.
    """
    result = _copy_signals(signals)
    if completed_tags is not None and result.tags is not None:
        result.tags = result.tags[: max(completed_tags, 0)]
    return result


def mark_rmssd_unavailable(signals: ParticipantSignals) -> ParticipantSignals:
    """AC 2.6 - produce HR, mark RMSSD unavailable for S01 anaerobic.

    The downloaded IBI file is empty, so RMSSD (derived from IBI) cannot be
    computed. HR is retained for analysis; the IBI slot is cleared to ``None`` and
    ``IBI`` is recorded in ``absent`` so RMSSD-dependent windows become unavailable
    (NaN) downstream, while HR windows are produced normally.
    """
    result = _copy_signals(signals)
    result.ibi = None
    if "IBI" not in result.absent:
        result.absent.append("IBI")
    return result


def restrict_to_stress(signals: ParticipantSignals) -> ParticipantSignals | None:
    """AC 2.7 - restrict f14..f18 analysis to the stress session.

    These Cohort_V2 participants have no (usable) exercise session, so only their
    stress-session recording is analyzed. A stress-session record passes through
    unchanged; any non-stress session for these participants is dropped by
    returning ``None`` so the caller excludes it from analysis.
    """
    if signals.session.upper() == SESSION_STRESS:
        return _copy_signals(signals)
    return None


def mark_demographics_unavailable(signals: ParticipantSignals) -> ParticipantSignals:
    """AC 2.8 - retain for signal analysis, mark demographics unavailable.

    A participant lacking demographic information is still analyzed on their
    signals; only the demographics are flagged as unavailable. ``ParticipantSignals``
    carries no demographic fields, so this rule records the gap in ``absent`` under
    the ``"demographics"`` sentinel and otherwise leaves the signals intact.
    """
    result = _copy_signals(signals)
    if "demographics" not in result.absent:
        result.absent.append("demographics")
    return result


# ---------------------------------------------------------------------------
# Dispatch table + driver
# ---------------------------------------------------------------------------

# Participants with no demographic information (retained, demographics marked
# unavailable per AC 2.8). f14..f18 lack a usable exercise session; the dataset
# notes them as the no-exercise cohort tail.
_MISSING_DEMOGRAPHICS: frozenset[str] = _STRESS_ONLY_PIDS


def clean(
    signals: ParticipantSignals,
    constraints: Constraints,
    segment_b: ParticipantSignals | None = None,
) -> ParticipantSignals | None:
    """Apply every documented cleaning rule for one participant/session.

    Dispatches by ``(participant, session)`` through the constraint entry and the
    small per-rule functions above, in a fixed, independently testable order:

    1. AC 2.7 - f14..f18 restricted to stress; a non-stress session returns ``None``.
    2. AC 2.3 / 2.4 - if the recording is split and ``segment_b`` is supplied, join
       the two segments first so later rules see the joined recording.
    3. AC 2.1 - truncate signals at documented duplicate-start rows (S02).
    4. AC 2.2 - exclude dock-on f07 from HR/RMSSD/Skin_Temperature.
    5. AC 2.6 - empty IBI (S01 anaerobic): keep HR, mark RMSSD unavailable.
    6. AC 2.5 - ended early (S03/S07/S06): keep completed phases.
    7. AC 2.8 - missing demographics: retain, mark demographics unavailable.

    ``segment_b`` is the optional companion recording for a split (``_a``/``_b``)
    case; it is required to actually join (the split flag without a companion leaves
    the single segment unchanged). Returns the cleaned record, or ``None`` when the
    session is dropped by the stress-only restriction (AC 2.7).

    The input ``signals`` is never mutated; a cleaned copy is returned.
    """
    pid = signals.person_id
    session = signals.session.upper()

    # AC 2.7: f14..f18 are stress-only; drop any non-stress session.
    if pid in _STRESS_ONLY_PIDS:
        restricted = restrict_to_stress(signals)
        if restricted is None:
            return None
        current = restricted
    else:
        current = _copy_signals(signals)

    constraint = constraints.get(pid, session)

    # AC 2.3 / 2.4: join split segments before signal-level rules.
    if constraint.split_recording and segment_b is not None:
        current = join_segments(current, segment_b)

    # AC 2.1: duplicate-start truncation (S02 stress).
    if constraint.truncations:
        current = truncate_signals(current, constraint)

    # AC 2.2: dock-on exclusion (f07 stress).
    if constraint.exclude_dock_on:
        current = exclude_signals(current)

    # AC 2.6: empty IBI -> HR kept, RMSSD unavailable (S01 anaerobic).
    if constraint.empty_ibi:
        current = mark_rmssd_unavailable(current)

    # AC 2.5: ended early -> keep completed phases (S03/S07/S06).
    if constraint.ended_early:
        current = keep_completed_phases(current)

    # AC 2.8: missing demographics -> retain, mark demographics unavailable.
    if pid in _MISSING_DEMOGRAPHICS:
        current = mark_demographics_unavailable(current)

    return current


# The dispatch table, exposed for callers/tests that want to invoke a single
# documented rule by name without re-deriving it from the constraint flags. Each
# entry is an independent, directly callable rule (design: "each rule
# independently testable").
_RULE_DISPATCH: dict[str, Callable[..., object]] = {
    "truncate": truncate_signals,
    "exclude_dock_on": exclude_signals,
    "join_segments": join_segments,
    "keep_completed_phases": keep_completed_phases,
    "empty_ibi": mark_rmssd_unavailable,
    "restrict_to_stress": restrict_to_stress,
    "missing_demographics": mark_demographics_unavailable,
}
