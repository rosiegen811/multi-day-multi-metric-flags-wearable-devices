"""Baseline_Estimator for the Wearable Flag Pipeline — per-person rest baselines.

This module learns each participant's *resting normal* per scored signal from
the opening rest segment of a session, then shrinks the per-person spread toward
the cohort pooled estimate. Downstream the Scorer (Task 7) z-scores each window
against these baselines, so "deviation" is always measured in that person's own
terms (design "Baseline_Estimator — flag/baseline.py", Requirement 5).

It consumes the shared ``Per_Window_Table`` (``data/schema.py``) and produces
:class:`~data.schema.SignalBaseline` records (``median``, ``mad``, ``shrunk_mad``)
keyed by signal name. Only the three *scored* signals carry a baseline — HR
(``hr``), RMSSD (``rmssd``) and Skin_Temperature (``skin_temp``); the
accelerometer magnitude (``acc_mag``) drives the movement check, not the z-score,
so it has no resting baseline here.

Dependencies are numpy / pandas / stdlib plus sibling records. Every function is
pure over its inputs (no I/O, no mutation of the passed table).

Requirements covered
--------------------
- 5.1  Use the opening rest segment, *excluding its first 60 seconds* while HR
       settles. :func:`person_baseline` drops rows whose ``t_win`` is within 60 s
       of the segment's first window end time (Property 8).
- 5.2  Summarize each signal by the **median** and the **scaled MAD**
       (``1.4826 * MAD``), the robust center/spread the design calls for
       (Property 9). Scaling MAD by ``1.4826`` makes it a consistent estimator of
       the standard deviation for normally distributed data.
- 5.3  Shrink each participant's per-signal spread toward the **cohort pooled
       estimate** via an empirical-Bayes weight. The shrunk spread stays between
       the per-person spread and the pooled spread, finite and positive
       (:func:`shrink_to_cohort`, Property 10).
- 5.4  WHERE a Cohort_V1 participant lacks a rest baseline in an exercise session
       (``needs_external_baseline`` is True), derive the baseline from that
       participant's **stress-session** rest segment (Open Decision 1 — reuse the
       stress-session rest baseline). :func:`person_baseline` accepts an optional
       ``external_table``/``external_pid`` for the caller to supply that session.

Research note (design "Robust baselines")
-----------------------------------------
Short rest segments (a few minutes) give unstable per-person spread estimates, so
the per-person MAD is stabilized by shrinking it toward the cohort pooled MAD —
the same empirical-Bayes variance-stabilization idea used by genomics tooling
(DESeq2 / limma) for small-sample estimates. The shrinkage weight here grows with
the per-person sample count, so a participant with a long, informative rest
segment keeps most of their own spread while a participant with only a handful of
baseline windows is pulled strongly toward the cohort estimate. Content was
rephrased for compliance with licensing restrictions.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from data.schema import SignalBaseline
from data.windows import (
    PHASE_BASELINE,
    baseline_phase,
    needs_external_baseline,
    protocol_key,
)

__all__ = [
    "SCORED_SIGNALS",
    "WARMUP_EXCLUDE_S",
    "MAD_SCALE",
    "MAD_FLOOR",
    "scaled_mad",
    "person_baseline",
    "shrink_to_cohort",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The signals that get a resting baseline and are z-scored downstream. The
# accelerometer magnitude (``acc_mag``) is intentionally excluded: it feeds the
# movement check (Task 9), not the deviation score.
SCORED_SIGNALS: tuple[str, ...] = ("hr", "rmssd", "skin_temp")

# Warm-up exclusion window: skip the first 60 s of the opening rest segment while
# HR settles (AC 5.1, Property 8).
WARMUP_EXCLUDE_S: float = 60.0

# MAD -> standard-deviation consistency scale for normally distributed data
# (AC 5.2). ``scaled_mad = 1.4826 * median(|x - median(x)|)``.
MAD_SCALE: float = 1.4826

# Small positive floor for a degenerate (zero / non-finite) spread, so the
# divide-by-spread z-score downstream stays finite (design Error Handling:
# "zero-MAD baselines ... guarded with a small floor so z-scores stay finite").
MAD_FLOOR: float = 1e-9


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------


def scaled_mad(values: np.ndarray) -> float:
    """Return the scaled median absolute deviation ``1.4826 * MAD`` (AC 5.2).

    ``MAD`` is ``median(|x - median(x)|)`` over the finite entries of ``values``.
    Non-finite entries (``NaN`` / ``inf``) are dropped first so an unavailable
    window cell does not poison the estimate. The result is floored at
    :data:`MAD_FLOOR`: a genuinely zero spread (a flat baseline segment) would
    otherwise make the downstream z-score divide by zero, so it is replaced with
    a small positive value (design Error Handling, Property 9/Property 10).
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return MAD_FLOOR
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    scaled = MAD_SCALE * mad
    if not np.isfinite(scaled) or scaled <= 0.0:
        return MAD_FLOOR
    return scaled


def _robust_center(values: np.ndarray) -> float:
    """Median of the finite entries of ``values`` (``NaN`` when none are finite)."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


# ---------------------------------------------------------------------------
# Opening-rest-segment selection
# ---------------------------------------------------------------------------


def _opening_rest_rows(table: pd.DataFrame, pid: str) -> pd.DataFrame:
    """Return the warm-up-excluded opening rest (baseline-phase) rows for ``pid``.

    The opening rest segment is the set of ``Per_Window_Table`` rows for ``pid``
    whose ``phase`` is the baseline phase (:data:`~data.windows.PHASE_BASELINE`),
    ordered by window end time ``t_win``. The first
    :data:`WARMUP_EXCLUDE_S` seconds of that segment are then dropped: a row is
    kept only when ``t_win > first_t_win + WARMUP_EXCLUDE_S`` (strictly greater,
    so the boundary instant exactly 60 s in is itself excluded — AC 5.1,
    Property 8). An empty frame is returned when ``pid`` has no baseline-phase
    rows.
    """
    rows = table[table["person_id"] == pid]
    rows = rows[rows["phase"] == PHASE_BASELINE]
    if rows.empty:
        return rows.sort_values("t_win")
    rows = rows.sort_values("t_win")
    first_t = float(rows["t_win"].iloc[0])
    cutoff = first_t + WARMUP_EXCLUDE_S
    return rows[rows["t_win"] > cutoff]


def _baselines_from_rows(rows: pd.DataFrame) -> dict[str, SignalBaseline]:
    """Compute per-signal ``SignalBaseline`` records from rest-segment rows.

    For each scored signal present in ``rows`` with at least one finite value,
    the baseline is the median (center) and scaled MAD (spread) of that column.
    ``shrunk_mad`` is initialized equal to ``mad``; :func:`shrink_to_cohort`
    replaces it with the empirical-Bayes-shrunk spread once the cohort is known.
    A signal whose column is entirely unavailable (all ``NaN`` — e.g. S01
    anaerobic RMSSD, AC 2.6) is omitted rather than given a ``NaN`` baseline.
    """
    baselines: dict[str, SignalBaseline] = {}
    for signal in SCORED_SIGNALS:
        if signal not in rows.columns:
            continue
        values = rows[signal].to_numpy(dtype=float)
        center = _robust_center(values)
        if not np.isfinite(center):
            continue
        spread = scaled_mad(values)
        baselines[signal] = SignalBaseline(
            median=center, mad=spread, shrunk_mad=spread
        )
    return baselines


def person_baseline(
    table: pd.DataFrame,
    pid: str,
    *,
    session: str | None = None,
    external_table: pd.DataFrame | None = None,
    external_pid: str | None = None,
) -> dict[str, SignalBaseline]:
    """Compute ``pid``'s per-signal resting baseline (AC 5.1, 5.2, 5.4).

    Returns a mapping ``{signal_name: SignalBaseline}`` over the scored signals
    (:data:`SCORED_SIGNALS`) that have a usable baseline. Each baseline is the
    median and scaled MAD (``1.4826 * MAD``) of the opening rest segment —
    the baseline-phase windows — with the first :data:`WARMUP_EXCLUDE_S` seconds
    excluded (AC 5.1, 5.2). ``shrunk_mad`` is set equal to ``mad`` here; call
    :func:`shrink_to_cohort` afterward to fill the empirical-Bayes-shrunk spread.

    V1 exercise baseline sourcing (AC 5.4, Open Decision 1)
    -------------------------------------------------------
    A Cohort_V1 exercise session has no quiet baseline phase of its own. When the
    caller passes ``session`` and :func:`~data.windows.needs_external_baseline`
    reports that this ``(session, version)`` needs an external baseline, the
    baseline is instead computed from the participant's **stress-session** rest
    segment, which the caller supplies via ``external_table`` /
    ``external_pid`` (defaulting ``external_pid`` to ``pid``). This realizes the
    design's Open Decision 1: reuse the stress-session rest baseline for V1
    exercise sessions.

    Parameters
    ----------
    table:
        The ``Per_Window_Table`` holding ``pid``'s rows for the session of
        interest.
    pid:
        The participant id whose baseline to compute.
    session:
        Optional session label (``STRESS`` / ``AEROBIC`` / ``ANAEROBIC``). Only
        used to detect the V1-exercise external-baseline case; omit it (or pass a
        session that owns a baseline phase) to always read the baseline from
        ``table`` itself.
    external_table:
        The participant's stress-session ``Per_Window_Table``. Required to
        satisfy AC 5.4 when ``session`` is a V1 exercise session; a ``KeyError``
        is raised if it is needed but not supplied, so the missing baseline is
        surfaced loudly rather than silently producing an empty result.
    external_pid:
        The participant id within ``external_table`` (defaults to ``pid``).

    Notes
    -----
    The function reads ``table`` (and ``external_table``) only; neither is
    mutated. A participant with no baseline-phase rows in the resolved table
    yields an empty mapping.
    """
    use_external = False
    if session is not None:
        key = protocol_key(pid, session)
        if needs_external_baseline(key):
            use_external = True

    if use_external:
        if external_table is None:
            raise KeyError(
                f"participant {pid!r} session {session!r} has no quiet baseline "
                f"(Cohort_V1 exercise session); an external stress-session table "
                f"must be supplied to source the baseline (AC 5.4 / Open "
                f"Decision 1)"
            )
        source_table = external_table
        source_pid = external_pid if external_pid is not None else pid
    else:
        source_table = table
        source_pid = pid

    rows = _opening_rest_rows(source_table, source_pid)
    return _baselines_from_rows(rows)


# ---------------------------------------------------------------------------
# Empirical-Bayes shrinkage toward the cohort pooled spread (AC 5.3)
# ---------------------------------------------------------------------------


def _pooled_spread(spreads: list[float], counts: list[float]) -> float:
    """Cohort pooled spread: the sample-count-weighted mean of per-person spreads.

    ``spreads`` are the per-person scaled-MAD spreads for one signal across the
    cohort and ``counts`` the matching baseline sample counts (windows used).
    Weighting by sample count lets participants with longer, more informative
    rest segments count more toward the pooled estimate. The result is floored at
    :data:`MAD_FLOOR` so a cohort of entirely flat baselines still yields a
    positive pooled spread.
    """
    s = np.asarray(spreads, dtype=float)
    w = np.asarray(counts, dtype=float)
    mask = np.isfinite(s) & np.isfinite(w) & (w > 0)
    s = s[mask]
    w = w[mask]
    if s.size == 0 or float(w.sum()) <= 0.0:
        return MAD_FLOOR
    pooled = float(np.sum(s * w) / np.sum(w))
    if not np.isfinite(pooled) or pooled <= 0.0:
        return MAD_FLOOR
    return pooled


def _shrink_one(person_spread: float, pooled_spread: float, n: float) -> float:
    """Empirical-Bayes shrink one spread toward the pooled spread (Property 10).

    Returns a convex combination
    ``shrunk = w * person_spread + (1 - w) * pooled_spread`` with the weight
    ``w = n / (n + 1)`` growing from 0 toward 1 as the per-person baseline sample
    count ``n`` grows. A participant with many baseline windows keeps most of
    their own spread; one with few is pulled toward the cohort estimate — the
    small-sample variance-stabilization behavior the design borrows from DESeq2 /
    limma.

    Because the result is a convex combination of two positive values it always
    lies between ``person_spread`` and ``pooled_spread`` (inclusive) and stays
    finite and positive (Property 10). Non-finite / non-positive inputs fall back
    to the other operand or :data:`MAD_FLOOR`.
    """
    p = float(person_spread)
    c = float(pooled_spread)
    if not np.isfinite(p) or p <= 0.0:
        p = c if (np.isfinite(c) and c > 0.0) else MAD_FLOOR
    if not np.isfinite(c) or c <= 0.0:
        c = p
    n = float(n)
    if not np.isfinite(n) or n < 0.0:
        n = 0.0
    weight = n / (n + 1.0)
    shrunk = weight * p + (1.0 - weight) * c
    if not np.isfinite(shrunk) or shrunk <= 0.0:
        return MAD_FLOOR
    return shrunk


def shrink_to_cohort(
    per_person: Mapping[str, Mapping[str, SignalBaseline]],
    cohort: Mapping[str, str],
    *,
    counts: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, dict[str, SignalBaseline]]:
    """Fill each baseline's ``shrunk_mad`` by shrinking toward the cohort (AC 5.3).

    ``per_person`` maps ``participant_id -> {signal -> SignalBaseline}`` (the raw
    per-person baselines from :func:`person_baseline`). ``cohort`` maps
    ``participant_id -> cohort label`` (``"V1"`` / ``"V2"`` from
    :func:`~data.physionet_load.cohort_of`). For each cohort and each signal the
    pooled spread is computed from that cohort's per-person spreads, and every
    participant's ``shrunk_mad`` is set to its spread shrunk toward that pooled
    value via :func:`_shrink_one` (Property 10).

    ``counts`` optionally supplies the per-person, per-signal baseline sample
    count used both to weight the pooled estimate and to set the shrinkage
    weight; when omitted every participant is weighted equally (``n = 1``), which
    still yields a valid bounded shrinkage (halfway to the pooled spread). The
    returned mapping is a fresh, deep-ish copy — new ``SignalBaseline`` records
    with ``median``/``mad`` preserved and ``shrunk_mad`` filled — leaving the
    input untouched.

    Only the shrinkage of the spread changes; the median center and the raw
    ``mad`` are carried through unchanged (shrinkage stabilizes *spread*, AC 5.3).
    """
    # Group per-person spreads/counts by cohort and signal for pooling.
    pooled: dict[str, dict[str, float]] = {}
    grouped: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for pid, signals in per_person.items():
        cohort_label = cohort.get(pid)
        if cohort_label is None:
            continue
        for signal, baseline in signals.items():
            n = _count_for(counts, pid, signal)
            grouped.setdefault(cohort_label, {}).setdefault(signal, []).append(
                (float(baseline.mad), n)
            )

    for cohort_label, signals in grouped.items():
        pooled[cohort_label] = {}
        for signal, pairs in signals.items():
            spreads = [p for p, _ in pairs]
            cnts = [c for _, c in pairs]
            pooled[cohort_label][signal] = _pooled_spread(spreads, cnts)

    # Build the shrunk output.
    out: dict[str, dict[str, SignalBaseline]] = {}
    for pid, signals in per_person.items():
        cohort_label = cohort.get(pid)
        out[pid] = {}
        for signal, baseline in signals.items():
            if cohort_label is not None and signal in pooled.get(cohort_label, {}):
                pooled_spread = pooled[cohort_label][signal]
                n = _count_for(counts, pid, signal)
                shrunk = _shrink_one(baseline.mad, pooled_spread, n)
            else:
                # No cohort peers to pool with: nothing to shrink toward, so the
                # shrunk spread is the (floored) per-person spread.
                shrunk = baseline.mad if baseline.mad > 0.0 else MAD_FLOOR
            out[pid][signal] = SignalBaseline(
                median=baseline.median, mad=baseline.mad, shrunk_mad=shrunk
            )
    return out


def _count_for(
    counts: Mapping[str, Mapping[str, float]] | None, pid: str, signal: str
) -> float:
    """Per-person/per-signal baseline sample count for weighting (default 1.0)."""
    if counts is None:
        return 1.0
    per_signal = counts.get(pid)
    if per_signal is None:
        return 1.0
    n = per_signal.get(signal)
    if n is None or not np.isfinite(float(n)) or float(n) < 0.0:
        return 1.0
    return float(n)
