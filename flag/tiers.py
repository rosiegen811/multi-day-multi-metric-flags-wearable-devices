"""Flagger — persistence gate, severity-tier assignment, and tier hysteresis.

This module decides *whether* a window flags and *how severe* the flag is. It sits
between the Scorer (``flag/score.py``, which turns each window into one
``Combined_Score``) and the Movement_Check (``flag/context.py``, Task 9): the
Flagger screens out momentary spikes, rates the size of a sustained shift, and
holds a tier steady against flicker — then the Movement_Check adjusts the result
as a benign cause (design "Flagger — flag/tiers.py", Requirement 7).

The pipeline mirrors how consumer products (Oura Symptom Radar, WHOOP) require a
signal to persist before flagging: a **2-of-3-window persistence gate** plus
**tier hysteresis** prevents a single spike or a one-window dip from moving the
tier around (design "Key design decisions — Persistence over spikes").

Three primitives, one flow
---------------------------
- :func:`persistence_gate` — the fire/don't-fire screen: pass iff at least 2 of
  the last 3 Combined_Scores exceed ``tau1`` (AC 7.1). This is a **count** over
  the raw last-3 scores and is deliberately a *different* computation from the
  median used for tiering.
- :func:`assign_tier` — the raw candidate tier from the **median** Combined_Score
  over the last 3 windows (AC 7.2), stepped at the ordered thresholds ``tau1 <
  tau2 < tau3`` (AC 7.3–7.6), with the single-signal cap at Watch (AC 7.7).
- :func:`apply_hysteresis` — the anti-flicker rule: a tier steps **down** only
  after the Combined_Score has sat below that tier's threshold for 2 consecutive
  windows; **up**-steps apply immediately (AC 7.8).

:class:`TierHistory` carries the small state the hysteresis needs between windows,
and :class:`Flagger` / :func:`flag_window` tie the three primitives together into
one call: gate → median-of-3 tier → single-signal cap → hysteresis, returning the
final tier. The **movement check is applied after this** in ``flag/context.py``
(Task 9); :func:`flag_window` returns the pre-movement tier as the clear
integration point and does not touch the accelerometer here.

Why median for tiering but a count for the gate
------------------------------------------------
Tiering off the **median** of the last 3 windows means a single spike cannot set
the tier (AC 7.2 / Property 16). The gate, by contrast, counts how many of the
last 3 exceeded ``tau1`` (AC 7.1 / Property 15) — a window can pass the gate (2 of
3 above ``tau1``) yet have a median below ``tau1`` (e.g. scores ``[2, 0, 2]`` with
``tau1 = 1`` → median ``2`` actually above; but ``[2, 2, 0]`` sorts to median
``2`` too — the two computations simply answer different questions and are kept
distinct per the design).

Dependencies are numpy / stdlib plus the shared records in ``data/schema.py``
(``Thresholds`` with its ``tau1 < tau2 < tau3`` invariant, and the tier constants
``TIER_NONE`` … ``TIER_STRONG``). Every function is pure over its inputs except
:meth:`Flagger.step`, which advances the carried :class:`TierHistory`.

Requirements covered
--------------------
- 7.1  persistence gate = 2-of-3 last scores above ``tau1`` (:func:`persistence_gate`).
- 7.2  tier from the median of the last 3 window scores (:func:`assign_tier`).
- 7.3  gate failed / median below ``tau1`` → tier 0 None (:func:`assign_tier`).
- 7.4  median in ``[tau1, tau2)`` → tier 1 Watch (:func:`assign_tier`).
- 7.5  median in ``[tau2, tau3)`` → tier 2 Notable (:func:`assign_tier`).
- 7.6  median ``>= tau3`` → tier 3 Strong (:func:`assign_tier`).
- 7.7  only one contributing signal caps the tier at Watch (:func:`assign_tier`).
- 7.8  a tier steps down only after 2 consecutive sub-threshold windows
       (:func:`apply_hysteresis`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from data.schema import (
    TIER_NONE,
    TIER_NOTABLE,
    TIER_STRONG,
    TIER_WATCH,
    Thresholds,
)

__all__ = [
    "GATE_WINDOW",
    "GATE_MIN_ABOVE",
    "HYSTERESIS_WINDOWS",
    "persistence_gate",
    "assign_tier",
    "tier_threshold",
    "TierHistory",
    "apply_hysteresis",
    "is_single_signal",
    "flag_window",
    "Flagger",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The persistence gate looks at the last 3 windows and fires when at least 2 of
# them exceed tau1 (AC 7.1): a "2-of-3" rule.
GATE_WINDOW: int = 3
GATE_MIN_ABOVE: int = 2

# Hysteresis holds a tier until the score has been below that tier's threshold for
# this many consecutive windows (AC 7.8).
HYSTERESIS_WINDOWS: int = 2


# ---------------------------------------------------------------------------
# Persistence gate (AC 7.1)
# ---------------------------------------------------------------------------


def persistence_gate(scores_last3: Sequence[float], tau1: float) -> bool:
    """Return ``True`` when the 2-of-3 persistence gate fires (AC 7.1).

    The gate passes iff **at least 2** of the last 3 Combined_Scores **strictly
    exceed** ``tau1`` (``score > tau1``). This is a plain count over the raw
    last-3 scores — not the median used for tiering — so a window can pass the
    gate while its median tier is low (the two answer different questions; see the
    module docstring).

    Fewer than 3 scores (pipeline warm-up — the first 1–2 windows of a recording)
    are handled gracefully: the rule still reads as "2-of-the-available", so the
    gate can only fire once at least 2 scores exist and at least 2 of them exceed
    ``tau1``. With 0 or 1 scores the gate never fires. Only the **last** 3 scores
    are considered when more are supplied, matching "the last 3 windows".

    Parameters
    ----------
    scores_last3:
        The recent Combined_Scores, most recent last. Only the final up-to-3 are
        used. Non-finite entries (``NaN`` / ``inf``) never count as "above".
    tau1:
        The lowest threshold (``Thresholds.tau1``).

    Returns
    -------
    bool
        ``True`` iff at least 2 of the last up-to-3 finite scores exceed ``tau1``.
    """
    recent = list(scores_last3)[-GATE_WINDOW:]
    above = sum(1 for s in recent if np.isfinite(s) and float(s) > tau1)
    return above >= GATE_MIN_ABOVE


# ---------------------------------------------------------------------------
# Tier assignment from the median of the last 3 windows (AC 7.2–7.7)
# ---------------------------------------------------------------------------


def assign_tier(
    median3: float,
    taus: Thresholds,
    single_signal: bool,
) -> int:
    """Assign a Severity_Tier from the median Combined_Score (AC 7.2–7.7).

    ``median3`` is the **median of the Combined_Scores over the last 3 windows**
    (AC 7.2) — using the median, not the latest score, means one spike cannot set
    the tier (Property 16). It is stepped at the ordered thresholds ``taus``:

    - ``median3 < tau1`` → ``TIER_NONE`` (0) (AC 7.3).
    - ``tau1 <= median3 < tau2`` → ``TIER_WATCH`` (1) (AC 7.4).
    - ``tau2 <= median3 < tau3`` → ``TIER_NOTABLE`` (2) (AC 7.5).
    - ``median3 >= tau3`` → ``TIER_STRONG`` (3) (AC 7.6).

    The step is monotone in ``median3`` (Property 17). A non-finite ``median3``
    (an all-unavailable window) is treated as below ``tau1`` → ``TIER_NONE``.

    Single-signal cap (AC 7.7): when ``single_signal`` is ``True`` — only **one**
    signal contributes the deviation for this window — the result is capped at
    ``TIER_WATCH`` (1) regardless of how large ``median3`` is, because a shift in a
    single metric is weaker evidence than a coherent multi-metric shift. Derive
    ``single_signal`` from the count of contributing signals, e.g.
    ``is_single_signal(window_signed_z(row, baselines))`` (see
    :func:`is_single_signal`); the caller passes the resulting bool.

    This function assigns the *raw candidate* tier. The persistence gate (AC 7.1,
    :func:`persistence_gate`) and hysteresis (AC 7.8, :func:`apply_hysteresis`)
    are applied around it by :func:`flag_window` / :class:`Flagger`; a caller that
    already knows the gate failed should force ``TIER_NONE`` (per AC 7.3) rather
    than call this with a high median.
    """
    if not np.isfinite(median3) or median3 < taus.tau1:
        tier = TIER_NONE
    elif median3 < taus.tau2:
        tier = TIER_WATCH
    elif median3 < taus.tau3:
        tier = TIER_NOTABLE
    else:
        tier = TIER_STRONG

    if single_signal and tier > TIER_WATCH:
        tier = TIER_WATCH
    return tier


def tier_threshold(tier: int, taus: Thresholds) -> float:
    """Return the Combined_Score threshold that defines ``tier``'s lower edge.

    The threshold a tier must stay above to hold its level (used by the
    down-step hysteresis, AC 7.8):

    - ``TIER_WATCH`` (1) → ``tau1``
    - ``TIER_NOTABLE`` (2) → ``tau2``
    - ``TIER_STRONG`` (3) → ``tau3``

    ``TIER_NONE`` (0) has no lower threshold; it returns ``-inf`` so "below this
    tier's threshold" is never true for tier 0 (a tier-0 flag has nothing to step
    down from).
    """
    if tier == TIER_WATCH:
        return taus.tau1
    if tier == TIER_NOTABLE:
        return taus.tau2
    if tier == TIER_STRONG:
        return taus.tau3
    return float("-inf")


# ---------------------------------------------------------------------------
# Hysteresis (AC 7.8)
# ---------------------------------------------------------------------------


@dataclass
class TierHistory:
    """The small carried state the down-step hysteresis needs (AC 7.8).

    Hysteresis only constrains **down**-steps, and only needs to know two things
    between windows:

    - ``tier`` — the tier currently *held* (the output of the previous window
      after hysteresis). Starts at ``TIER_NONE``.
    - ``below_count`` — how many **consecutive** recent windows have had a
      Combined_Score **below the held tier's threshold**. Reset to ``0`` whenever
      the score is at or above that threshold (the tier is still "earned"), and
      reset on every up-step (a higher tier brings a fresh threshold to fall
      below). A down-step is permitted only once this reaches
      :data:`HYSTERESIS_WINDOWS` (2).

    This is a mutable record advanced by :func:`apply_hysteresis` /
    :meth:`Flagger.step`; create one per flag stream (per person/session).
    """

    tier: int = TIER_NONE
    below_count: int = 0


def apply_hysteresis(
    history: TierHistory,
    candidate: int,
    taus: Thresholds,
    *,
    score: float | None = None,
) -> int:
    """Apply down-step hysteresis and advance ``history`` in place (AC 7.8).

    Given the tier currently held (``history.tier``) and the fresh ``candidate``
    tier from :func:`assign_tier` for this window, decide the tier the flag
    actually shows:

    - **Up-step or hold** (``candidate >= history.tier``): adopt ``candidate``
      immediately — a rise in severity is never delayed — and reset the
      below-threshold streak to ``0``.
    - **Down-step** (``candidate < history.tier``): the held tier is kept *unless*
      the Combined_Score has been below the held tier's threshold for
      :data:`HYSTERESIS_WINDOWS` (2) **consecutive** windows. The streak is
      counted here: each call with the score below the held tier's threshold
      increments ``below_count``; a score at or above the threshold resets it to
      ``0`` (and the tier holds). Once the streak reaches 2, the tier steps down
      — to the ``candidate`` tier — and the streak resets.

    The threshold compared against is the **held** tier's lower edge
    (:func:`tier_threshold`), so "below this tier's threshold" means the held tier
    is no longer earned. ``score`` is the current window's Combined_Score; when it
    is ``None`` the ``candidate``'s relationship to the held tier is used as a
    proxy (``candidate < history.tier`` implies the score fell below the held
    tier's band), which is enough to drive the 2-consecutive-window count.

    Steps down by **one evaluation**, not one level at a time: a drop from Strong
    straight to None is allowed once the 2-window condition is met, because the
    candidate already reflects where the median now sits. The design constrains
    *when* a down-step may occur (after 2 sub-threshold windows), not its size.

    Returns the final tier for this window and mutates ``history`` to carry the
    new held tier and streak forward.
    """
    held = history.tier

    # Up-step or hold at the same level: adopt immediately, clear the streak.
    if candidate >= held:
        history.tier = candidate
        history.below_count = 0
        return history.tier

    # Candidate is lower than the held tier — a down-step is being requested.
    # Count only windows whose score is genuinely below the held tier's threshold.
    threshold = tier_threshold(held, taus)
    if score is None:
        below = True  # candidate < held implies the median fell below the band
    else:
        below = np.isfinite(score) and float(score) < threshold

    if below:
        history.below_count += 1
    else:
        # Score recovered to at/above the held tier's threshold: the tier is
        # still earned, so hold it and reset the streak.
        history.below_count = 0
        history.tier = held
        return history.tier

    if history.below_count >= HYSTERESIS_WINDOWS:
        # Sustained below-threshold for the required number of windows: step down.
        history.tier = candidate
        history.below_count = 0
        return history.tier

    # Not yet sustained: hold the current tier (no premature down-step).
    history.tier = held
    return history.tier


# ---------------------------------------------------------------------------
# Single-signal detection (AC 7.7 helper)
# ---------------------------------------------------------------------------


def is_single_signal(contributions: Mapping[str, float]) -> bool:
    """Return ``True`` when exactly one signal contributes the deviation (AC 7.7).

    ``contributions`` is the per-signal signed-z mapping for the window, typically
    :func:`~flag.score.window_signed_z` (which already drops unavailable signals).
    The single-signal cap fires when the count of contributing signals is exactly
    one — a lone-metric shift is weaker evidence than a coherent multi-metric one.
    An empty mapping (no available signals) is not "single" (there is no deviation
    to cap), so this returns ``False``.
    """
    return len(contributions) == 1


# ---------------------------------------------------------------------------
# Cohesive flow: gate -> median-of-3 tier -> single-signal cap -> hysteresis
# ---------------------------------------------------------------------------


def flag_window(
    scores_last3: Sequence[float],
    taus: Thresholds,
    *,
    single_signal: bool,
    history: TierHistory,
) -> int:
    """Final pre-movement tier for one window (AC 7.1–7.8).

    Combines the three primitives in order, advancing ``history`` in place:

    1. **Persistence gate** (AC 7.1) — if fewer than 2 of the last 3 scores exceed
       ``tau1`` the window does not flag; the candidate tier is forced to
       ``TIER_NONE`` (AC 7.3).
    2. **Median-of-3 tier** (AC 7.2–7.6) — otherwise the candidate tier is
       :func:`assign_tier` on the **median** of the last 3 scores.
    3. **Single-signal cap** (AC 7.7) — folded into :func:`assign_tier` via
       ``single_signal`` (derive it from the contributing-signal count, e.g.
       :func:`is_single_signal`).
    4. **Hysteresis** (AC 7.8) — :func:`apply_hysteresis` decides whether a
       down-step from the previously held tier is allowed yet.

    The returned tier is the **pre-movement** tier. The Movement_Check
    (``flag/context.py``, Task 9) is applied *after* this — ``apply_movement``
    may drop this tier by one and label the flag "explained by movement". This
    function deliberately does not read the accelerometer; that is the clean
    integration point for Task 9.

    Parameters
    ----------
    scores_last3:
        Recent Combined_Scores, most recent last (only the last up-to-3 are used).
    taus:
        The ordered :class:`~data.schema.Thresholds` (``tau1 < tau2 < tau3``).
    single_signal:
        Whether only one signal contributes the deviation for this window.
    history:
        The carried :class:`TierHistory` for this flag stream; mutated in place.

    Returns
    -------
    int
        The final tier for this window after gate, median tiering, single-signal
        cap and hysteresis — before the movement check.
    """
    recent = list(scores_last3)[-GATE_WINDOW:]
    current_score = recent[-1] if recent else float("nan")

    if persistence_gate(recent, taus.tau1):
        finite = [float(s) for s in recent if np.isfinite(s)]
        median3 = float(np.median(finite)) if finite else float("nan")
        candidate = assign_tier(median3, taus, single_signal)
    else:
        # Gate failed: no flag this window (AC 7.3).
        candidate = TIER_NONE

    return apply_hysteresis(history, candidate, taus, score=current_score)


@dataclass
class Flagger:
    """Stateful per-stream Flagger tying gate → tier → cap → hysteresis together.

    One :class:`Flagger` tracks a single flag stream (one person/session). It
    keeps the recent Combined_Scores (a rolling window of the last 3) and the
    :class:`TierHistory` so callers can feed one window at a time without managing
    that state themselves.

    Usage::

        flagger = Flagger(taus)
        for row in per_window_rows:               # Per_Window_Table rows in order
            score = combined_score(row, baselines, method)
            contributions = window_signed_z(row, baselines)
            tier = flagger.step(score, single_signal=is_single_signal(contributions))
            # ... Movement_Check (Task 9) adjusts `tier` afterwards ...

    :meth:`step` returns the **pre-movement** tier (AC 7.1–7.8); the movement
    check is applied by the caller via ``flag/context.py``.
    """

    taus: Thresholds
    history: TierHistory = field(default_factory=TierHistory)
    _recent: list[float] = field(default_factory=list)

    def step(self, score: float, *, single_signal: bool) -> int:
        """Advance by one window with Combined_Score ``score``; return the tier.

        Appends ``score`` to the rolling last-3 history, then runs
        :func:`flag_window` over the current last-3 scores. ``single_signal`` is
        the single-signal-cap flag for this window (AC 7.7); derive it from the
        contributing-signal count (:func:`is_single_signal`).
        """
        self._recent.append(float(score))
        if len(self._recent) > GATE_WINDOW:
            del self._recent[:-GATE_WINDOW]
        return flag_window(
            self._recent,
            self.taus,
            single_signal=single_signal,
            history=self.history,
        )

    @property
    def tier(self) -> int:
        """The tier currently held after the most recent :meth:`step`."""
        return self.history.tier
