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

Why the shrinkage has real teeth (regularization note)
------------------------------------------------------
A few-minute rest segment yields on the order of a dozen-to-twenty baseline
windows, and some rest segments are *nearly flat* — e.g. a wrist skin-temperature
baseline whose scaled MAD is only a few thousandths of a degree. An unregularized
spread that tiny makes the downstream z-score (deviation / spread) explode: a
routine +0.4 C stress shift over a 0.004 C spread is a z-score near 90, which
alone swamps the directional-sum Combined_Score and makes that one signal the
permanent top contributor. That defeats the whole point of combining several
metrics.

The empirical-Bayes shrinkage is what is supposed to tame this, so the shrinkage
weight must *not* saturate at ~1 for the modest window counts a short baseline
yields. The weight is therefore ``w = n / (n + PRIOR_STRENGTH)`` with a prior
strength :data:`PRIOR_STRENGTH` set to a meaningful fraction of a typical
baseline window count (so a ~18-window baseline is pulled roughly halfway toward
the cohort spread) rather than the old ``w = n / (n + 1)`` which left
``shrunk_mad == mad`` in practice. On top of that, a *degenerate* per-person
spread — one that is an outlier-small fraction (< :data:`DEGENERATE_RATIO`) of the
cohort pooled spread for that signal — is treated as unreliable: its effective
prior is boosted (:data:`DEGENERATE_PRIOR_STRENGTH`) so the shrunk spread lands
close to the pooled spread. Because the result is always a convex combination of
the per-person and pooled spreads, it stays within ``[person, pooled]`` and
Property 10 continues to hold — the regularization is achieved by shrinking
*toward* pooled, never by a hard floor that could jump outside that interval.
:data:`MAD_FLOOR` remains only as a last-resort positive floor for the truly-zero
case so divisions stay finite; it is a safety net, not the regularizer.
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
    "PRIOR_STRENGTH",
    "DEGENERATE_RATIO",
    "DEGENERATE_PRIOR_STRENGTH",
    "scaled_mad",
    "person_baseline",
    "shrink_to_cohort",
    "pooled_spreads",
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

# Small positive floor for a *truly zero / non-finite* spread, so the
# divide-by-spread z-score downstream stays finite (design Error Handling:
# "zero-MAD baselines ... guarded with a small floor so z-scores stay finite").
# This is a last-resort safety net only — the real regularization of small
# spreads is the empirical-Bayes shrinkage below, not this floor.
MAD_FLOOR: float = 1e-9

# Empirical-Bayes prior strength for the shrinkage weight ``w = n / (n + n0)``
# (AC 5.3, Property 10). ``n`` is the per-person baseline sample count and ``n0``
# this prior strength, expressed in the same units (windows). A few-minute rest
# segment yields on the order of a dozen-to-twenty baseline windows, so a prior
# strength of 15 means a ~15-window baseline is pulled halfway toward the cohort
# pooled spread and an ~18-window baseline a little past halfway toward it — i.e.
# the shrinkage has real effect at realistic baseline lengths, unlike the old
# ``n0 = 1`` which left ``w ~= 1`` (``shrunk_mad == mad``) for any usable
# baseline. Chosen as a meaningful fraction of a typical baseline window count.
PRIOR_STRENGTH: float = 15.0

# Degeneracy guard. A per-person scaled MAD that is below this fraction of the
# cohort pooled spread for that signal is treated as an unreliable, near-constant
# rest segment (e.g. f13 skin-temp at ~0.004 C against a cohort pooled spread of
# a few tenths of a degree, a ratio well under 0.1). Such a spread would produce
# z-scores in the tens or hundreds, so it is strongly shrunk toward pooled.
DEGENERATE_RATIO: float = 0.1

# Effective prior strength applied to a degenerate spread (see DEGENERATE_RATIO).
# Much larger than PRIOR_STRENGTH so the shrinkage weight ``w = n / (n + n0)``
# stays small even for a long-but-flat baseline, landing the shrunk spread close
# to — but, being a convex combination, never above — the cohort pooled spread.
# Property 10 still holds: the result stays within ``[person, pooled]``.
DEGENERATE_PRIOR_STRENGTH: float = 200.0


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
    ``w = n / (n + n0)`` growing from 0 toward 1 as the per-person baseline sample
    count ``n`` grows, where ``n0`` is the prior strength. A participant with many
    baseline windows keeps most of their own spread; one with few is pulled toward
    the cohort estimate — the small-sample variance-stabilization behavior the
    design borrows from DESeq2 / limma.

    Two refinements give the shrinkage the teeth the real data needs (regularizer
    note in the module docstring):

    - The prior strength is :data:`PRIOR_STRENGTH` (``= 15`` windows), not ``1``.
      With ``n0 = 15`` a ~15-window rest segment is shrunk halfway toward the
      cohort pooled spread, so the shrinkage actually moves ``shrunk_mad`` instead
      of leaving it equal to ``mad`` (as the old ``n0 = 1`` did for any usable
      baseline, where ``w`` was ~1).
    - A **degenerate** per-person spread — one below :data:`DEGENERATE_RATIO`
      (``0.1``) times the pooled spread — is treated as an unreliable near-flat
      baseline and shrunk much harder: its effective prior strength jumps to
      :data:`DEGENERATE_PRIOR_STRENGTH` (``= 200``), so even a long flat baseline
      lands close to the pooled spread. This is what stops a near-constant rest
      segment (f13 skin-temp ~0.004 C) from producing z-scores in the tens or
      hundreds.

    Because the result is always a convex combination of two positive values
    (``0 <= w <= 1``), it lies between ``person_spread`` and ``pooled_spread``
    (inclusive) and stays finite and positive — Property 10 holds for both the
    ordinary and the degenerate-guard path, since the guard only *raises* the
    prior strength (lowers ``w``) and never floors the result outside the
    ``[person, pooled]`` interval. Non-finite / non-positive inputs fall back to
    the other operand or :data:`MAD_FLOOR`.
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

    # Degeneracy guard: a per-person spread that is an outlier-small fraction of
    # the pooled spread is unreliable, so lean much harder on the cohort estimate
    # by boosting the prior strength. Still a convex combination => Property 10.
    prior_strength = PRIOR_STRENGTH
    if p < DEGENERATE_RATIO * c:
        prior_strength = DEGENERATE_PRIOR_STRENGTH

    weight = n / (n + prior_strength)
    shrunk = weight * p + (1.0 - weight) * c
    if not np.isfinite(shrunk) or shrunk <= 0.0:
        return MAD_FLOOR
    return shrunk


def shrink_to_cohort(
    per_person: Mapping[str, Mapping[str, SignalBaseline]],
    cohort: Mapping[str, str],
    *,
    counts: Mapping[str, Mapping[str, float]] | None = None,
    pooled: Mapping[str, Mapping[str, float]] | None = None,
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
    count used both to weight the pooled estimate and to set the shrinkage weight
    ``w = n / (n + n0)`` (:func:`_shrink_one`); when omitted every participant is
    weighted equally (``n = 1``), which with the prior strength
    :data:`PRIOR_STRENGTH` pulls the spread almost all the way to the pooled value
    — still a valid bounded shrinkage (Property 10). The returned mapping is a
    fresh, deep-ish copy — new ``SignalBaseline`` records
    with ``median``/``mad`` preserved and ``shrunk_mad`` filled — leaving the
    input untouched.

    Robust pooled target (``pooled``)
    ---------------------------------
    The per-person ``mad`` carried in ``per_person`` is the AC-5.1
    warm-up-excluded *opening-rest-tail* spread, which on real data is only a
    handful of windows long and, for a near-constant signal like skin temperature,
    is degenerately tiny *for every participant* — so a pooled target computed
    from those tails is itself tiny and cannot regularize anything (the whole
    cohort is degenerate at once). ``pooled`` lets the caller supply a **more
    informative per-cohort, per-signal pooled spread** to shrink toward —
    typically the cohort-pooled scaled MAD over the *full* baseline-phase windows
    (warm-up included), which carries the real resting variation. See
    :func:`pooled_spreads`, the helper that computes exactly this from a
    Per_Window_Table. When ``pooled`` is given it overrides the tail-derived
    pooled target for every ``(cohort, signal)`` it covers; signals it omits fall
    back to the tail-pooled value. When ``pooled`` is ``None`` the original
    tail-pooled behavior is used unchanged (backward compatible).

    This keeps AC 5.3 — the shrinkage target is still the *cohort pooled estimate*
    of resting spread, just estimated robustly — and keeps Property 10: the
    per-person ``mad`` is unchanged and the shrunk value is a convex combination of
    that ``mad`` and the (now larger) pooled target, so it stays within
    ``[mad, pooled]``.

    Only the shrinkage of the spread changes; the median center and the raw
    ``mad`` are carried through unchanged (shrinkage stabilizes *spread*, AC 5.3).
    """
    # Group per-person spreads/counts by cohort and signal for the (fallback)
    # tail-derived pooled estimate.
    tail_pooled: dict[str, dict[str, float]] = {}
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
        tail_pooled[cohort_label] = {}
        for signal, pairs in signals.items():
            spreads = [p for p, _ in pairs]
            cnts = [c for _, c in pairs]
            tail_pooled[cohort_label][signal] = _pooled_spread(spreads, cnts)

    # Resolve the effective pooled target per (cohort, signal): prefer the
    # caller-supplied robust pooled, fall back to the tail-derived estimate.
    def _pooled_target(cohort_label: str, signal: str) -> float | None:
        if pooled is not None:
            per_cohort = pooled.get(cohort_label)
            if per_cohort is not None:
                v = per_cohort.get(signal)
                if v is not None and np.isfinite(float(v)) and float(v) > 0.0:
                    return float(v)
        per_cohort = tail_pooled.get(cohort_label)
        if per_cohort is None:
            return None
        return per_cohort.get(signal)

    # Build the shrunk output.
    out: dict[str, dict[str, SignalBaseline]] = {}
    for pid, signals in per_person.items():
        cohort_label = cohort.get(pid)
        out[pid] = {}
        for signal, baseline in signals.items():
            pooled_spread = (
                _pooled_target(cohort_label, signal)
                if cohort_label is not None
                else None
            )
            if pooled_spread is not None:
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


# ---------------------------------------------------------------------------
# Robust cohort pooled shrinkage target (AC 5.3 — the estimate shrunk toward)
# ---------------------------------------------------------------------------


def pooled_spreads(
    table: pd.DataFrame,
    cohort: Mapping[str, str],
) -> dict[str, dict[str, float]]:
    """Robust per-cohort, per-signal resting spread to shrink toward (AC 5.3).

    Returns ``{cohort_label: {signal: pooled_scaled_MAD}}`` for the scored signals
    (:data:`SCORED_SIGNALS`), suitable as the ``pooled`` argument of
    :func:`shrink_to_cohort`.

    Why not just pool the per-person tails
    --------------------------------------
    The per-person baseline ``mad`` is the AC-5.1 warm-up-excluded *opening-rest
    tail* — on the real data only a handful of windows, and for a near-constant
    signal like skin temperature it is degenerately tiny for *every* participant.
    Pooling those tails therefore yields a tiny, uninformative target that cannot
    regularize a degenerate per-person spread. This helper instead estimates the
    cohort pooled resting spread from the **full baseline-phase windows** (the
    whole opening rest, warm-up *included* — :data:`~data.windows.PHASE_BASELINE`
    rows), which carry the person's real resting variation. It remains a *cohort
    pooled estimate of resting spread* (AC 5.3); only its robustness changes.

    For each cohort and signal the pooled spread is the median-across-participants
    of each participant's full-baseline scaled MAD (:func:`scaled_mad`),
    restricted to participants with a positive, finite estimate. The median across
    participants (rather than a count-weighted mean) keeps one unusually flat or
    unusually noisy participant from dragging the cohort target, which matters
    precisely because some participants are degenerate. A cohort/signal with no
    usable participant is simply omitted, and :func:`shrink_to_cohort` then falls
    back to its tail-pooled estimate for that ``(cohort, signal)``.

    The ``table`` is read only (never mutated). It must carry ``person_id``,
    ``phase`` and the scored-signal columns; a missing column or an absent
    baseline phase for a participant just drops that contribution.
    """
    if "person_id" not in table.columns or "phase" not in table.columns:
        return {}

    base_rows = table[table["phase"] == PHASE_BASELINE]

    # Per participant, per signal: the full-baseline scaled MAD.
    per_person_full: dict[str, dict[str, float]] = {}
    for pid, rows in base_rows.groupby("person_id"):
        pid = str(pid)
        for signal in SCORED_SIGNALS:
            if signal not in rows.columns:
                continue
            values = rows[signal].to_numpy(dtype=float)
            if not np.any(np.isfinite(values)):
                continue
            spread = scaled_mad(values)
            if np.isfinite(spread) and spread > 0.0:
                per_person_full.setdefault(pid, {})[signal] = spread

    # Group by cohort and take the median across participants per signal.
    grouped: dict[str, dict[str, list[float]]] = {}
    for pid, signals in per_person_full.items():
        cohort_label = cohort.get(pid)
        if cohort_label is None:
            continue
        for signal, spread in signals.items():
            grouped.setdefault(cohort_label, {}).setdefault(signal, []).append(spread)

    out: dict[str, dict[str, float]] = {}
    for cohort_label, signals in grouped.items():
        out[cohort_label] = {}
        for signal, spreads in signals.items():
            arr = np.asarray(spreads, dtype=float)
            arr = arr[np.isfinite(arr) & (arr > 0.0)]
            if arr.size == 0:
                continue
            pooled = float(np.median(arr))
            if np.isfinite(pooled) and pooled > 0.0:
                out[cohort_label][signal] = pooled
    return out
