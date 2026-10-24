"""Windower for the Wearable Flag Pipeline — protocol phase assignment.

This module maps button-press tags to named protocol phases. It is the
phase-assignment half of the Windower component (design "Windower —
data/windows.py"); the trailing-window segmentation into the ``Per_Window_Table``
(``window(...)``) is added by Task 4.2 and is intentionally *not* implemented
here.

Phases come from the dataset's protocol timing table (project README, "Protocol
timing"), one ordered phase sequence per ``(session, protocol version)`` pair.
The button-press ``tags.csv`` timestamps delimit phase boundaries; each inter-tag
interval is one phase, named in protocol order. Downstream, Task 4.2 labels each
window with the phase whose interval contains the window end time (AC 3.3), and
the Baseline_Estimator (Task 6) reads the baseline phase for a session.

It is dependency-light: only ``numpy``, the standard library, and sibling
records. Side effects are none — phase assignment is pure over the tag list.

Requirements covered:
- 3.1  Map each inter-tag interval to a named phase using the protocol timing
       table for that session's protocol version.
- 3.2  A Cohort_V1 (``S*``) exercise session has NO quiet baseline phase; its
       baseline must be sourced from that participant's stress session. The
       phase-assignment side encodes this: V1 aerobic/anaerobic protocol tables
       contain no baseline phase, and :func:`baseline_phase` /
       :func:`needs_external_baseline` expose to the Baseline_Estimator (Task 6)
       where the baseline comes from (Open Decision 1: reuse the stress-session
       rest baseline).
- 3.3  The returned ordered :class:`Phase` list covers the tag-delimited
       intervals so the windower (Task 4.2) can record each window's phase name.

Protocol versions and sessions
-------------------------------
A participant id maps to a protocol version via the dataset convention (``S*`` ->
V1, ``f*`` -> V2; see :func:`~data.physionet_load.cohort_of`). A ``protocol``
identifier here is the ``(session, version)`` pair — the session folder name
(``STRESS`` / ``AEROBIC`` / ``ANAEROBIC``) plus the protocol version (``V1`` /
``V2``). Use :func:`protocol_key` to build one from a participant id and session,
or construct a :class:`ProtocolKey` directly.

Task vs 30-second task naming
-----------------------------
Evaluation (Task 12) reports long tasks and 30-second tasks separately, so the
long cognitive tasks (``Stroop``, ``TMCT``) and the three 30-second tasks
(``RealOpinion``, ``OppositeOpinion``, ``Subtract``) carry distinct phase names;
:data:`SHORT_TASK_PHASES` / :data:`LONG_TASK_PHASES` enumerate each group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .physionet_load import (
    COHORT_V1,
    COHORT_V2,
    IBISeries,
    ParticipantSignals,
    SignalSeries,
    cohort_of,
)
from .schema import make_per_window_table

__all__ = [
    "SESSION_STRESS",
    "SESSION_AEROBIC",
    "SESSION_ANAEROBIC",
    "VERSION_V1",
    "VERSION_V2",
    "PHASE_BASELINE",
    "PHASE_REST",
    "SHORT_TASK_PHASES",
    "LONG_TASK_PHASES",
    "Phase",
    "PhaseSpec",
    "ProtocolKey",
    "protocol_key",
    "version_of",
    "PROTOCOL_TIMING",
    "phase_specs",
    "has_quiet_baseline",
    "baseline_phase",
    "needs_external_baseline",
    "assign_phases",
    "STEP_S",
    "SHORT_WINDOW_S",
    "RMSSD_WINDOW_S",
    "MIN_SHORT_SAMPLES",
    "MIN_RMSSD_INTERVALS",
    "window",
]


# ---------------------------------------------------------------------------
# Session / version labels (match the loader, cleaner, and schema conventions)
# ---------------------------------------------------------------------------

SESSION_STRESS = "STRESS"
SESSION_AEROBIC = "AEROBIC"
SESSION_ANAEROBIC = "ANAEROBIC"

# Protocol version labels, aligned with the cohort labels from physionet_load
# (``S*`` -> Cohort_V1 ran protocol V1; ``f*`` -> Cohort_V2 ran protocol V2).
VERSION_V1 = COHORT_V1  # "V1"
VERSION_V2 = COHORT_V2  # "V2"

_EXERCISE_SESSIONS: frozenset[str] = frozenset({SESSION_AEROBIC, SESSION_ANAEROBIC})

# Canonical phase names reused across protocols.
PHASE_BASELINE = "Baseline"
PHASE_REST = "Rest"

# The three 30-second cognitive tasks and the two long cognitive tasks. Named
# distinctly so evaluation (Task 12) can report long vs 30-second tasks apart.
SHORT_TASK_PHASES: tuple[str, ...] = ("RealOpinion", "OppositeOpinion", "Subtract")
LONG_TASK_PHASES: tuple[str, ...] = ("Stroop", "TMCT")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseSpec:
    """One entry in a protocol timing table: an ordered, named phase.

    ``name`` is the phase name (e.g. ``"Baseline"``, ``"Stroop"``). ``duration_s``
    is the nominal phase length in seconds from the README Protocol timing table,
    or ``None`` when the README gives no explicit duration (e.g. the V2 stress
    baseline). Durations are documentary: the authoritative boundaries at runtime
    are the button-press tags, so :func:`assign_phases` uses tag times, not these
    nominal durations, to set each phase's start/end.
    """

    name: str
    duration_s: float | None = None


@dataclass(frozen=True)
class Phase:
    """A named phase spanning one tag-delimited interval.

    ``name`` is the protocol phase name; ``start_t`` and ``end_t`` are epoch
    seconds (UTC), taken from consecutive button-press tags. The interval is
    half-open ``[start_t, end_t)`` so that adjacent phases do not both claim a
    boundary instant; Task 4.2 labels a window by the phase whose interval
    contains the window end time (AC 3.3).
    """

    name: str
    start_t: float
    end_t: float


@dataclass(frozen=True)
class ProtocolKey:
    """Identifies a protocol timing table: a ``(session, version)`` pair.

    ``session`` is the session folder name (``STRESS`` / ``AEROBIC`` /
    ``ANAEROBIC``); ``version`` is the protocol version (``V1`` / ``V2``).
    """

    session: str
    version: str

    def normalized(self) -> "ProtocolKey":
        """Return a copy with session upper-cased and version upper-cased."""
        return ProtocolKey(session=self.session.upper(), version=self.version.upper())


# ---------------------------------------------------------------------------
# Protocol timing tables (README "Protocol timing")
# ---------------------------------------------------------------------------
#
# Each table is the ordered phase sequence for one (session, version). Durations
# are the README's nominal values in seconds (documentary only — tags set the
# real boundaries). Repeated exercise blocks are expanded into individually named
# phases (e.g. ``Sprint1`` / ``CoolDown1`` ...) so each tag-delimited interval
# maps to a unique phase name.

# --- STRESS V1 (S01–S18): has a 3-minute quiet baseline ---------------------
_STRESS_V1: tuple[PhaseSpec, ...] = (
    PhaseSpec(PHASE_BASELINE, 180.0),
    PhaseSpec("Stroop", 120.0),
    PhaseSpec(PHASE_REST, 300.0),
    PhaseSpec("TMCT", 180.0),
    PhaseSpec(PHASE_REST, 300.0),
    PhaseSpec("RealOpinion", 30.0),
    PhaseSpec("OppositeOpinion", 30.0),
    PhaseSpec("Subtract", 30.0),
)

# --- STRESS V2 (f01–f18): has a quiet baseline (README gives no duration) ---
_STRESS_V2: tuple[PhaseSpec, ...] = (
    PhaseSpec(PHASE_BASELINE, None),
    PhaseSpec("TMCT", 180.0),
    PhaseSpec(PHASE_REST, 600.0),
    PhaseSpec("RealOpinion", 30.0),
    PhaseSpec("OppositeOpinion", 30.0),
    PhaseSpec(PHASE_REST, 600.0),
    PhaseSpec("Subtract", 30.0),
)

# --- AEROBIC V1: NO quiet baseline (reuse stress-session baseline) ----------
# Warm-up 3:00 -> 60..110 rpm in 10 stages (2–3 min each) -> Cool-down 4:00 ->
# Rest 2:00. The 10 rpm stages step 60,65,...,110 (5 rpm increments).
_AEROBIC_V1: tuple[PhaseSpec, ...] = (
    PhaseSpec("WarmUp", 180.0),
    *tuple(
        PhaseSpec(f"Stage{rpm}rpm", None)
        for rpm in range(60, 111, 5)  # 60,65,...,110 -> 11 labels
    ),
    PhaseSpec("CoolDown", 240.0),
    PhaseSpec(PHASE_REST, 120.0),
)

# --- AEROBIC V2: has a 4.5-minute quiet baseline ----------------------------
_AEROBIC_V2: tuple[PhaseSpec, ...] = (
    PhaseSpec(PHASE_BASELINE, 270.0),
    PhaseSpec("WarmUp", 135.0),
    PhaseSpec("Stage70rpm", 90.0),
    PhaseSpec("Stage75rpm", 90.0),
    PhaseSpec("Stage80rpm", 90.0),
    PhaseSpec("Stage85rpm", 675.0),
    PhaseSpec("Stage90rpm", 270.0),
    PhaseSpec("Stage95rpm", 270.0),
    PhaseSpec("CoolDown", 180.0),
    PhaseSpec(PHASE_REST, 120.0),
)

# --- ANAEROBIC V1: NO quiet baseline (reuse stress-session baseline) --------
# Warm-up 3:00 -> 3 x (Sprint 0:30 + Cool-down 4:00) -> Rest 2:00.
_ANAEROBIC_V1: tuple[PhaseSpec, ...] = (
    PhaseSpec("WarmUp", 180.0),
    PhaseSpec("Sprint1", 30.0),
    PhaseSpec("CoolDown1", 240.0),
    PhaseSpec("Sprint2", 30.0),
    PhaseSpec("CoolDown2", 240.0),
    PhaseSpec("Sprint3", 30.0),
    PhaseSpec("CoolDown3", 240.0),
    PhaseSpec(PHASE_REST, 120.0),
)

# --- ANAEROBIC V2: has a 4.5-minute quiet baseline --------------------------
# Baseline 4:30 -> Warm-up 4:30 -> 4 x (Sprint 0:45 + Cool-down ~3:45) -> Rest 2:00.
_ANAEROBIC_V2: tuple[PhaseSpec, ...] = (
    PhaseSpec(PHASE_BASELINE, 270.0),
    PhaseSpec("WarmUp", 270.0),
    PhaseSpec("Sprint1", 45.0),
    PhaseSpec("CoolDown1", 225.0),
    PhaseSpec("Sprint2", 45.0),
    PhaseSpec("CoolDown2", 225.0),
    PhaseSpec("Sprint3", 45.0),
    PhaseSpec("CoolDown3", 225.0),
    PhaseSpec("Sprint4", 45.0),
    PhaseSpec("CoolDown4", 225.0),
    PhaseSpec(PHASE_REST, 120.0),
)


# The protocol timing registry keyed on a normalized ``(session, version)`` pair.
PROTOCOL_TIMING: Mapping[ProtocolKey, tuple[PhaseSpec, ...]] = {
    ProtocolKey(SESSION_STRESS, VERSION_V1): _STRESS_V1,
    ProtocolKey(SESSION_STRESS, VERSION_V2): _STRESS_V2,
    ProtocolKey(SESSION_AEROBIC, VERSION_V1): _AEROBIC_V1,
    ProtocolKey(SESSION_AEROBIC, VERSION_V2): _AEROBIC_V2,
    ProtocolKey(SESSION_ANAEROBIC, VERSION_V1): _ANAEROBIC_V1,
    ProtocolKey(SESSION_ANAEROBIC, VERSION_V2): _ANAEROBIC_V2,
}


# ---------------------------------------------------------------------------
# Protocol lookup helpers
# ---------------------------------------------------------------------------


def version_of(participant_id: str) -> str:
    """Return the protocol version (``V1`` / ``V2``) a participant ran.

    Thin alias over :func:`~data.physionet_load.cohort_of`: ``S*`` participants
    ran protocol V1 and ``f*`` participants ran protocol V2 (Requirements 1.4,
    1.5). Kept here so callers that think in terms of *protocol version* need not
    import the cohort mapping directly.
    """
    return cohort_of(participant_id)


def protocol_key(participant_id: str, session: str) -> ProtocolKey:
    """Build the :class:`ProtocolKey` for a participant's session.

    Combines the session folder name with the participant's protocol version
    (derived from the id prefix). ``session`` is case-insensitive and is
    normalized to the upper-case folder convention.
    """
    return ProtocolKey(session=session.upper(), version=version_of(participant_id))


def phase_specs(protocol: ProtocolKey) -> tuple[PhaseSpec, ...]:
    """Return the ordered protocol timing table for a ``(session, version)``.

    Raises ``KeyError`` for an unknown session/version combination (e.g. a
    V1 exercise session is valid, but a misspelled session name is not).
    """
    key = protocol.normalized()
    try:
        return PROTOCOL_TIMING[key]
    except KeyError:
        raise KeyError(
            f"no protocol timing table for session={key.session!r}, "
            f"version={key.version!r}; known keys: "
            f"{sorted((k.session, k.version) for k in PROTOCOL_TIMING)}"
        )


def has_quiet_baseline(protocol: ProtocolKey) -> bool:
    """Return True if this protocol has its own quiet baseline phase.

    V2 sessions and the V1 stress session begin with a ``Baseline`` phase; V1
    exercise sessions (aerobic, anaerobic) do not (README: "V1 exercise sessions
    have no quiet baseline"). This is the phase-assignment witness of AC 3.2.
    """
    specs = phase_specs(protocol)
    return any(spec.name == PHASE_BASELINE for spec in specs)


def needs_external_baseline(protocol: ProtocolKey) -> bool:
    """Return True if the baseline must be sourced from another session.

    True exactly for a Cohort_V1 exercise session, which has no quiet baseline of
    its own (AC 3.2). In that case the Baseline_Estimator (Task 6) must reuse the
    participant's stress-session rest baseline (Open Decision 1). For every other
    protocol the baseline comes from this session's own :func:`baseline_phase`.
    """
    key = protocol.normalized()
    return key.session in _EXERCISE_SESSIONS and not has_quiet_baseline(key)


def baseline_phase(protocol: ProtocolKey) -> str | None:
    """Return the phase name that serves as this protocol's resting baseline.

    - V1/V2 stress and V2 exercise sessions: the ``Baseline`` phase.
    - V1 exercise sessions (no quiet baseline): ``None`` — the caller must source
      the baseline externally (see :func:`needs_external_baseline`), reusing the
      participant's stress-session rest baseline per Open Decision 1 / AC 3.2.
    """
    if has_quiet_baseline(protocol):
        return PHASE_BASELINE
    return None


# ---------------------------------------------------------------------------
# Phase assignment (Requirement 3.1, 3.3)
# ---------------------------------------------------------------------------


def assign_phases(tags: list[float], protocol: ProtocolKey) -> list[Phase]:
    """Map tag-delimited intervals to named protocol phases (AC 3.1, 3.3).

    ``tags`` are the button-press timestamps (epoch seconds UTC) from
    ``tags.csv``, each marking a phase boundary. ``protocol`` identifies the
    ``(session, version)`` whose timing table names the phases in order.

    With *n* tags there are *n − 1* inter-tag intervals; interval *i* (between
    ``tags[i]`` and ``tags[i+1]``) is assigned the *i*-th phase name from the
    protocol timing table. The returned phases are ordered and each spans the
    half-open interval ``[start_t, end_t)`` so adjacent phases share no instant;
    Task 4.2 labels each window by the phase whose interval contains the window
    end time (AC 3.3).

    Behavior and edge cases:

    - Tags are sorted ascending before pairing (the loader returns them in file
      order; the cleaner may merge split-file tags). A zero-length or negative
      interval (duplicate / out-of-order tags that survive sorting as equal) is
      skipped rather than producing an empty or inverted phase.
    - Fewer intervals than phases (the participant stopped early — e.g. S03/S07/
      S06, AC 2.5): only the phases reached are returned, in order.
    - More intervals than phases (extra trailing button presses): the surplus
      trailing intervals are dropped, since the protocol defines no phase for
      them. Partial coverage is preferred over inventing unnamed phases.
    - Fewer than two tags: no interval can be formed, so an empty list is
      returned.

    This function does not source the V1 exercise baseline itself; that is the
    Baseline_Estimator's job (Task 6), which consults
    :func:`needs_external_baseline` / :func:`baseline_phase`. Here, a V1 exercise
    protocol table simply has no ``Baseline`` phase to assign (AC 3.2).
    """
    specs = phase_specs(protocol)

    if len(tags) < 2:
        return []

    ordered = sorted(float(t) for t in tags)

    phases: list[Phase] = []
    spec_idx = 0
    # Walk consecutive tag pairs, consuming one phase name per *non-degenerate*
    # interval. A degenerate (zero/negative) interval from duplicate or equal
    # tags is skipped without consuming a phase name, so the remaining phase
    # mapping stays aligned. Stop once either the intervals or the phase names
    # run out (early-stopped sessions yield fewer intervals; surplus trailing
    # presses yield more — both are handled by this bound).
    for start_t, end_t in zip(ordered, ordered[1:]):
        if spec_idx >= len(specs):
            break
        if end_t <= start_t:
            continue
        phases.append(Phase(name=specs[spec_idx].name, start_t=start_t, end_t=end_t))
        spec_idx += 1

    return phases


# ---------------------------------------------------------------------------
# Trailing-window segmentation into the Per_Window_Table (Requirement 4)
# ---------------------------------------------------------------------------
#
# The analysis window advances in fixed 10-second steps (AC 4.1). At each step
# time ``t`` (a window *end* time) the signals are aggregated over their trailing
# intervals: HR, Skin_Temperature and Accelerometer_Magnitude over the short
# 30-second interval ``(t-30, t]`` (AC 4.2), and RMSSD over the longer
# 120-second interval ``(t-120, t]`` (AC 4.3). Each resulting row is one window;
# the full set of rows is the Per_Window_Table built through
# :func:`~data.schema.make_per_window_table` so the column set, dtypes, and the
# one-row-per-person-per-window-step invariant are enforced (AC 4.4).

# Window geometry (seconds). Step is the spacing between consecutive window end
# times (AC 4.1); the two trailing-interval lengths are the aggregation spans.
STEP_S: float = 10.0
SHORT_WINDOW_S: float = 30.0   # HR / Skin_Temperature / Accelerometer_Magnitude
RMSSD_WINDOW_S: float = 120.0  # RMSSD

# Minimum samples required in a trailing interval to compute an aggregate;
# otherwise the cell is marked unavailable (NaN) for that window (AC 4.5).
#
# - HR / Skin_Temperature / Accelerometer_Magnitude take a mean, so a single
#   in-window sample already yields a defined value -> require at least 1.
# - RMSSD is the root-mean-square of *successive differences* of inter-beat
#   intervals, so it needs at least one difference, i.e. at least 2 intervals
#   in the trailing interval -> require at least 2.
MIN_SHORT_SAMPLES: int = 1
MIN_RMSSD_INTERVALS: int = 2


def _sample_times(series: SignalSeries) -> np.ndarray:
    """Absolute UTC epoch-second time of each sample in a :class:`SignalSeries`.

    Sample ``i`` is taken at ``start_utc + i / sample_rate_hz`` (AC 1.1 timing
    convention). Works for both single-channel (1-D ``values``) and 3-axis ACC
    (``(N, 3)`` ``values``) because only the sample count (``len``) matters here.
    """
    n = len(series.values)
    if n == 0 or not np.isfinite(series.sample_rate_hz) or series.sample_rate_hz <= 0:
        return np.empty(0, dtype=float)
    idx = np.arange(n, dtype=float)
    return series.start_utc + idx / float(series.sample_rate_hz)


def _trailing_mask(times: np.ndarray, t: float, span_s: float) -> np.ndarray:
    """Boolean mask of samples whose time falls in the half-open ``(t-span, t]``.

    The interval is half-open on the left and closed on the right so that a
    sample exactly at the window end ``t`` is included while one exactly at the
    far edge ``t-span`` is excluded, matching the design's ``(t-30, t]`` /
    ``(t-120, t]`` membership rule (AC 4.2, 4.3; Property 5).
    """
    if times.size == 0:
        return np.zeros(0, dtype=bool)
    return (times > (t - span_s)) & (times <= t)


def _short_mean(
    series: SignalSeries | None, t: float, values_1d: np.ndarray | None
) -> float:
    """Mean of a single-channel signal over the trailing ``(t-30, t]`` interval.

    Returns ``NaN`` when the signal is absent or the interval holds fewer than
    :data:`MIN_SHORT_SAMPLES` samples (AC 4.5). ``values_1d`` is the already
    reduced-to-1-D value array for ``series`` (identical to ``series.values`` for
    HR/TEMP; the per-sample vector magnitude for ACC).
    """
    if series is None or values_1d is None or values_1d.size == 0:
        return float("nan")
    times = _sample_times(series)
    mask = _trailing_mask(times, t, SHORT_WINDOW_S)
    if int(mask.sum()) < MIN_SHORT_SAMPLES:
        return float("nan")
    return float(np.mean(values_1d[mask]))


def _acc_magnitude(acc: SignalSeries | None) -> np.ndarray | None:
    """Per-sample accelerometer vector magnitude ``sqrt(x^2 + y^2 + z^2)``.

    Returns ``None`` when ACC is absent. The returned 1-D array lines up index
    for index with the ACC sample rows, so it can be masked by
    :func:`_trailing_mask` just like a single-channel signal.
    """
    if acc is None:
        return None
    values = np.asarray(acc.values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        return np.empty(0, dtype=float)
    return np.sqrt(np.sum(values * values, axis=1))


def _ibi_beat_times(ibi: IBISeries) -> np.ndarray:
    """Absolute UTC epoch-second time of each recorded heartbeat.

    ``IBISeries.offsets_s`` holds successive inter-beat interval measurements in
    seconds. The cumulative sum of those intervals, added to ``start_utc``, gives
    the time of each beat; beat ``k`` lands at
    ``start_utc + sum(offsets_s[:k+1])``.
    """
    offsets = np.asarray(ibi.offsets_s, dtype=float)
    if offsets.size == 0:
        return np.empty(0, dtype=float)
    return ibi.start_utc + np.cumsum(offsets)


def _rmssd(ibi: IBISeries | None, t: float) -> float:
    """RMSSD over inter-beat intervals in the trailing ``(t-120, t]`` interval.

    RMSSD is the root-mean-square of successive differences of the inter-beat
    intervals whose beats fall in the trailing 120-second interval (AC 4.3).
    Returns ``NaN`` when IBI is absent or fewer than :data:`MIN_RMSSD_INTERVALS`
    intervals fall in the interval (too few to form a single difference, AC 4.5).
    """
    if ibi is None:
        return float("nan")
    offsets = np.asarray(ibi.offsets_s, dtype=float)
    if offsets.size == 0:
        return float("nan")
    beat_times = _ibi_beat_times(ibi)
    mask = _trailing_mask(beat_times, t, RMSSD_WINDOW_S)
    intervals = offsets[mask]
    if intervals.size < MIN_RMSSD_INTERVALS:
        return float("nan")
    diffs = np.diff(intervals)
    return float(np.sqrt(np.mean(diffs * diffs)))


def _phase_at(t: float, phases: list[Phase]) -> str | None:
    """Return the name of the phase whose ``[start_t, end_t)`` interval holds ``t``.

    Phases are the tag-delimited, half-open intervals from :func:`assign_phases`
    (adjacent phases share no instant). A window end time ``t`` inside exactly one
    interval takes that interval's name (AC 3.3, Property 3); a ``t`` outside every
    labeled phase has no phase (``None``).
    """
    for phase in phases:
        if phase.start_t <= t < phase.end_t:
            return phase.name
    return None


def _recording_span(signals: ParticipantSignals) -> tuple[float, float] | None:
    """Earliest start and latest end time across all present signals.

    Returns ``(start_utc, end_utc)`` in epoch seconds, where ``start_utc`` is the
    earliest start over HR/TEMP/ACC/IBI and ``end_utc`` the latest last-sample
    time, so the window timeline covers the whole recording (Property 4). Returns
    ``None`` when no signal carries a usable timeline.
    """
    starts: list[float] = []
    ends: list[float] = []

    for series in (signals.hr, signals.temp, signals.acc):
        if isinstance(series, SignalSeries):
            times = _sample_times(series)
            if times.size:
                starts.append(float(times[0]))
                ends.append(float(times[-1]))

    if isinstance(signals.ibi, IBISeries):
        beat_times = _ibi_beat_times(signals.ibi)
        # Always anchor the start at the IBI start time even if no beats parsed.
        starts.append(float(signals.ibi.start_utc))
        if beat_times.size:
            ends.append(float(beat_times[-1]))

    if not starts or not ends:
        return None
    return min(starts), max(ends)


def window(signals: ParticipantSignals, phases: list[Phase]) -> pd.DataFrame:
    """Segment a cleaned participant's signals into the Per_Window_Table (Req 4).

    Advances a window end time ``t`` in fixed :data:`STEP_S` (10 s) steps across
    the recording span (AC 4.1). For each step it aggregates:

    - **HR**, **Skin_Temperature** (TEMP) and **Accelerometer_Magnitude** (ACC)
      as the mean over the trailing short interval ``(t-30, t]`` (AC 4.2); ACC is
      the mean of the per-sample vector magnitude ``sqrt(x^2+y^2+z^2)``;
    - **RMSSD** as the root-mean-square of successive differences of the inter-beat
      intervals whose beats fall in the trailing ``(t-120, t]`` interval (AC 4.3).

    A cell is marked unavailable (``NaN``) when its trailing interval holds too
    few samples — at least :data:`MIN_SHORT_SAMPLES` for the short signals and
    :data:`MIN_RMSSD_INTERVALS` inter-beat intervals for RMSSD (AC 4.5). An absent
    signal (``None`` on ``signals``) yields an all-``NaN`` column. Each window's
    ``phase`` is the labeled phase containing ``t`` (AC 3.3), or ``None`` outside
    every phase.

    Window timeline (documented choice): end times start at ``span_start + STEP_S``
    and advance by exactly ``STEP_S`` up to and including the recording's last
    sample time, so consecutive end times differ by exactly 10 s and cover the
    span (Property 4). The first window end is one step in — its trailing 30 s /
    120 s interval simply holds whatever early samples exist (fewer samples near
    the start naturally yield more ``NaN`` RMSSD cells, which is the intended
    AC 4.5 behavior rather than a special case).

    The rows are assembled through :func:`~data.schema.make_per_window_table`,
    which enforces the full column set, the dtypes, and one row per person per
    window step (AC 4.4). An empty/timeline-less input yields an empty table.

    ``signals`` is read only; no input is mutated.
    """
    span = _recording_span(signals)
    if span is None:
        return make_per_window_table([])

    span_start, span_end = span

    # Precompute the 1-D value arrays used by the short-window mean so they are
    # built once rather than per step.
    hr = signals.hr if isinstance(signals.hr, SignalSeries) else None
    temp = signals.temp if isinstance(signals.temp, SignalSeries) else None
    acc = signals.acc if isinstance(signals.acc, SignalSeries) else None
    ibi = signals.ibi if isinstance(signals.ibi, IBISeries) else None

    hr_vals = np.asarray(hr.values, dtype=float) if hr is not None else None
    temp_vals = np.asarray(temp.values, dtype=float) if temp is not None else None
    acc_mag_vals = _acc_magnitude(acc)

    rows: list[dict[str, object]] = []
    # Step end times: span_start + STEP_S, +2*STEP_S, ... up to span_end.
    # Use integer step counts to keep the spacing exact (Property 4).
    n_steps = int(np.floor((span_end - span_start) / STEP_S + 1e-9))
    for k in range(1, n_steps + 1):
        t = span_start + k * STEP_S
        rows.append(
            {
                "person_id": signals.person_id,
                "cohort": signals.cohort,
                "protocol": signals.cohort,  # protocol version label == cohort (V1/V2)
                "session": signals.session,
                "phase": _phase_at(t, phases),
                "t_win": t,
                "hr": _short_mean(hr, t, hr_vals),
                "rmssd": _rmssd(ibi, t),
                "skin_temp": _short_mean(temp, t, temp_vals),
                "acc_mag": _short_mean(acc, t, acc_mag_vals),
            }
        )

    return make_per_window_table(rows)
