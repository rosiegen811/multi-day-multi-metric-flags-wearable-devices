"""Explainer for the Wearable Flag Pipeline — plain-language flag explanations.

This module turns a fired :class:`~data.schema.Flag` into a human-readable
explanation: it names the signals that drove the flag, ranked by how much each
contributed, expresses each signal's shift in that signal's *real unit* relative
to the participant's rest baseline, and carries the flag's severity-tier wording
and context (start time, how long it persisted, the Combined_Score, and whether
movement explains it). The wording is deliberately neutral and non-diagnostic —
a *sustained change* from the person's resting normal, never a diagnosis, and
never a color-coded severity label (design "Explainer — flag/explain.py",
Requirement 9; scope constraints 15.1/15.2).

It consumes the :class:`~data.schema.SignalBaseline` records produced by the
Baseline_Estimator (``flag/baseline.py``), the per-signal *signed z* mapping
produced by the Scorer (:func:`flag.score.window_signed_z`) for the flagged
window, and the window's raw real-unit signal values. Every function is pure over
its inputs (no I/O, no mutation).

Dependencies are numpy / stdlib plus sibling records (``data/schema`` for
:class:`~data.schema.Flag` / :class:`~data.schema.SignalBaseline` and the tier
wording, ``flag/score`` for :data:`~flag.score.SCORED_SIGNALS`). No pandas is
required.

Ranking (AC 9.1)
----------------
Signals are named in **non-increasing order of their z-score contribution**,
where "contribution" is the *magnitude* of the signed z (``abs(z)``) — a large
negative z (e.g. a big HRV drop) contributes as much as an equally large positive
one. Ties (equal magnitude) are broken by the canonical scored-signal order
:data:`~flag.score.SCORED_SIGNALS` (``hr``, ``rmssd``, ``skin_temp``) so the
ordering is deterministic.

Real-unit shifts (AC 9.2)
-------------------------
Each named signal's shift is expressed in its own native unit, relative to the
participant's rest baseline: ``delta = window_value - baseline.median``. This is
computed directly from the window's raw real-unit values (passed in via
``values``) and the baseline center, rather than reconstructed from the z-score —
that keeps the real-unit arithmetic exact and preserves each signal's natural
direction (e.g. HRV genuinely *dropping* reads as a negative ms delta). The
``(signal, real_unit_delta)`` pairs match the shape of
:attr:`data.schema.Flag.top_signals`.

Display names and units
-----------------------
- ``hr`` — "heart rate", unit **bpm** (beats per minute).
- ``rmssd`` — "HRV", unit **ms** (milliseconds).
- ``skin_temp`` — "skin temperature", unit **C** (degrees Celsius).

Requirements covered
--------------------
- 9.1  Name the top contributing signals ranked by z-score contribution
       (non-increasing magnitude) (:func:`rank_contributions`, :func:`explain`).
- 9.2  Express each named signal's shift in that signal's real unit relative to
       the rest baseline (:func:`real_unit_delta`, :func:`explain`).
- 9.3  Include the Severity_Tier wording for the assigned tier
       (:func:`~data.schema.tier_wording`).
- 9.4  Include the flag start time, seconds persisted, Combined_Score, and the
       movement-explained boolean (:class:`Explanation`, :func:`explain`).
- 9.5  Exclude diagnostic claims and color-coded severity labels.
- 15.1 Describe the flag as a sustained change, never a diagnosis.
- 15.2 Use neutral tier wording rather than color-coded ranges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from data.schema import Flag, SignalBaseline, tier_wording
from flag.score import SCORED_SIGNALS

__all__ = [
    "SIGNAL_DISPLAY",
    "SIGNAL_UNIT",
    "SignalShift",
    "Explanation",
    "signal_display_name",
    "signal_unit",
    "real_unit_delta",
    "format_delta",
    "rank_contributions",
    "explain",
]


# ---------------------------------------------------------------------------
# Display names and units (documented above)
# ---------------------------------------------------------------------------

# Human-readable name shown for each scored signal.
SIGNAL_DISPLAY: Mapping[str, str] = {
    "hr": "heart rate",
    "rmssd": "HRV",
    "skin_temp": "skin temperature",
}

# Real-unit label for each scored signal: HR in bpm, RMSSD in ms, skin
# temperature in degrees Celsius.
SIGNAL_UNIT: Mapping[str, str] = {
    "hr": "bpm",
    "rmssd": "ms",
    "skin_temp": "C",
}


def signal_display_name(signal: str) -> str:
    """Return the human-readable display name for ``signal`` (fallback: the name)."""
    return SIGNAL_DISPLAY.get(signal, signal)


def signal_unit(signal: str) -> str:
    """Return the real-unit label for ``signal`` (fallback: empty string)."""
    return SIGNAL_UNIT.get(signal, "")


# ---------------------------------------------------------------------------
# Structured records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalShift:
    """One named contributing signal and its real-unit shift from rest (AC 9.1, 9.2).

    ``signal`` is the raw signal key (``hr`` / ``rmssd`` / ``skin_temp``);
    ``display`` is its human-readable name and ``unit`` its real-unit label.
    ``z`` is the signed z-score contribution for the flagged window (ranking uses
    its magnitude). ``delta`` is the real-unit shift ``window_value -
    baseline.median`` in ``unit`` (positive = above rest, negative = below rest).
    ``text`` is the rendered phrase, e.g. ``"heart rate +18 bpm vs your rest"``.
    """

    signal: str
    display: str
    unit: str
    z: float
    delta: float
    text: str


@dataclass(frozen=True)
class Explanation:
    """A plain-language, non-diagnostic explanation of a fired flag (Requirement 9).

    Structured fields carry everything AC 9.1–9.4 require; :attr:`text` is the
    rendered human-readable rendering of the same content. The wording describes a
    *sustained change* from the person's resting normal and uses only neutral tier
    words — no diagnosis, no color-coded severity labels (AC 9.5, 15.1, 15.2).

    Fields
    ------
    person_id:
        The participant the flag belongs to.
    start_t:
        Flag start time, in seconds (AC 9.4).
    persisted_s:
        How many seconds the flag has persisted (AC 9.4).
    combined_score:
        The flag's Combined_Score (AC 9.4).
    tier:
        The assigned Severity_Tier, ``0..3``.
    tier_wording:
        The neutral plain-language wording for ``tier`` (AC 9.3), from
        :data:`~data.schema.TIER_WORDING`.
    movement_explained:
        Whether movement explains the flag (AC 9.4).
    shifts:
        The named contributing signals, ranked by z-score contribution in
        non-increasing magnitude (AC 9.1), each with its real-unit shift (AC 9.2).
    text:
        The full rendered explanation string.
    """

    person_id: str
    start_t: float
    persisted_s: float
    combined_score: float
    tier: int
    tier_wording: str
    movement_explained: bool
    shifts: list[SignalShift] = field(default_factory=list)
    text: str = ""


# ---------------------------------------------------------------------------
# Real-unit shifts (AC 9.2)
# ---------------------------------------------------------------------------


def real_unit_delta(value: float, b: SignalBaseline) -> float:
    """Real-unit shift of ``value`` from the rest baseline: ``value - b.median`` (AC 9.2).

    The delta is expressed in the signal's native unit (no z-scaling): positive
    means above the participant's resting normal, negative means below. Returns
    ``nan`` when ``value`` or the baseline center is non-finite.
    """
    x = float(value)
    center = float(b.median)
    if not np.isfinite(x) or not np.isfinite(center):
        return float("nan")
    return x - center


def format_delta(signal: str, delta: float) -> str:
    """Render a real-unit delta with an explicit sign and unit, e.g. ``"+18 bpm"``.

    The magnitude is rounded to one decimal place and trailing ``.0`` is trimmed,
    so ``+18.0`` reads as ``+18`` while ``+0.4`` is kept. The sign is always shown
    (``+`` for zero or above, ``-`` below) so the direction of the shift is
    explicit. A non-finite delta renders as ``"n/a"``.
    """
    unit = signal_unit(signal)
    d = float(delta)
    if not np.isfinite(d):
        return "n/a"

    rounded = round(d, 1)
    # Normalize -0.0 to 0.0 so the sign reads "+".
    if rounded == 0.0:
        rounded = 0.0
    sign = "+" if rounded >= 0.0 else "-"
    magnitude = abs(rounded)
    # Trim a trailing ".0" (18.0 -> "18") but keep real decimals (0.4 -> "0.4").
    if magnitude == int(magnitude):
        mag_text = str(int(magnitude))
    else:
        mag_text = f"{magnitude:.1f}"
    unit_suffix = f" {unit}" if unit else ""
    return f"{sign}{mag_text}{unit_suffix}"


# ---------------------------------------------------------------------------
# Contributor ranking (AC 9.1)
# ---------------------------------------------------------------------------


def rank_contributions(contributions: Mapping[str, float]) -> list[str]:
    """Rank contributing signals by z-score contribution, non-increasing (AC 9.1).

    Returns the signal names ordered by ``abs(z)`` descending — the magnitude of
    the signed z-score, so a large drop (negative z) ranks alongside an equally
    large rise. Non-finite contributions are dropped. Ties on magnitude are broken
    by the canonical scored-signal order (:data:`~flag.score.SCORED_SIGNALS`) and
    then alphabetically, so the ranking is deterministic.
    """
    order = {name: i for i, name in enumerate(SCORED_SIGNALS)}

    finite = {
        signal: float(z)
        for signal, z in contributions.items()
        if np.isfinite(float(z))
    }

    def sort_key(signal: str):
        # Higher magnitude first; then canonical signal order; then name.
        return (-abs(finite[signal]), order.get(signal, len(order)), signal)

    return sorted(finite, key=sort_key)


# ---------------------------------------------------------------------------
# Explanation (Requirement 9)
# ---------------------------------------------------------------------------


def _render_text(
    shifts: Sequence[SignalShift],
    tier_words: str,
    start_t: float,
    persisted_s: float,
    combined_score: float,
    movement_explained: bool,
) -> str:
    """Render the full plain-language explanation string (AC 9.1–9.5, 15.1, 15.2).

    The wording frames the flag as a *sustained change* from the person's resting
    normal (never a diagnosis) and uses only the neutral tier wording — no
    color-coded severity terms.
    """
    lines: list[str] = []

    # Lead with the neutral tier wording when present (tier 0 wording is empty).
    if tier_words:
        lines.append(tier_words + ".")

    if shifts:
        named = ", ".join(s.text for s in shifts)
        lines.append(f"Top contributing signals: {named}.")
    else:
        lines.append("No individual signal shift is available to report.")

    movement_phrase = (
        "yes — this change lines up with movement"
        if movement_explained
        else "no"
    )
    lines.append(
        f"This is a sustained change from your resting normal "
        f"(not a diagnosis). It began at {start_t:g} s and has persisted for "
        f"{persisted_s:g} s. Combined score {combined_score:g}. "
        f"Explained by movement: {movement_phrase}."
    )

    return " ".join(lines)


def explain(
    flag: Flag,
    contributions: Mapping[str, float],
    baselines: Mapping[str, SignalBaseline],
    values: Mapping[str, float] | None = None,
) -> Explanation:
    """Build a plain-language :class:`Explanation` for a fired ``flag`` (Requirement 9).

    Parameters
    ----------
    flag:
        The fired :class:`~data.schema.Flag`. Supplies ``person_id``, ``start_t``,
        ``persisted_s``, ``combined_score``, ``tier``, ``movement_explained``
        (AC 9.3, 9.4) and, when ``values`` is not supplied, the real-unit deltas
        via its ``top_signals`` (``(signal, real_unit_delta)`` pairs).
    contributions:
        The per-signal *signed z* mapping for the flagged window (from
        :func:`flag.score.window_signed_z`). Drives the contributor ranking
        (AC 9.1). Signals absent here are not named.
    baselines:
        The participant's ``{signal -> SignalBaseline}``; the baseline center
        (``median``) is the rest reference for the real-unit shift (AC 9.2).
    values:
        Optional ``{signal -> raw real-unit value}`` for the flagged window (the
        window's ``hr`` / ``rmssd`` / ``skin_temp``). When given, each named
        signal's real-unit shift is computed as ``value - baseline.median``
        (AC 9.2). When omitted, the delta falls back to ``flag.top_signals`` (which
        already carries ``(signal, real_unit_delta)`` per the schema).

    Returns
    -------
    Explanation
        Structured fields plus a rendered ``text``. The signals in ``shifts`` are
        ranked by z-score contribution in non-increasing magnitude (AC 9.1), each
        carrying its real-unit shift (AC 9.2). The explanation includes the tier
        wording (AC 9.3) and the start time / persisted seconds / Combined_Score /
        movement-explained flag (AC 9.4), and contains no diagnostic claim or
        color-coded severity term (AC 9.5, 15.1, 15.2).
    """
    # Fallback real-unit deltas from the flag's top_signals, if provided there.
    flag_deltas: dict[str, float] = {}
    for item in flag.top_signals or []:
        try:
            name, delta = item
        except (TypeError, ValueError):
            continue
        flag_deltas[str(name)] = float(delta)

    ranked = rank_contributions(contributions)

    shifts: list[SignalShift] = []
    for signal in ranked:
        z = float(contributions[signal])

        # Real-unit delta: prefer the raw window value vs the baseline center;
        # otherwise fall back to the delta carried on the flag (AC 9.2).
        delta = float("nan")
        b = baselines.get(signal)
        if values is not None and signal in values and b is not None:
            delta = real_unit_delta(values[signal], b)
        if not np.isfinite(delta) and signal in flag_deltas:
            delta = flag_deltas[signal]

        display = signal_display_name(signal)
        unit = signal_unit(signal)
        delta_text = format_delta(signal, delta)
        text = f"{display} {delta_text} vs your rest"
        shifts.append(
            SignalShift(
                signal=signal,
                display=display,
                unit=unit,
                z=z,
                delta=delta,
                text=text,
            )
        )

    tier_words = tier_wording(flag.tier)

    text = _render_text(
        shifts,
        tier_words,
        float(flag.start_t),
        float(flag.persisted_s),
        float(flag.combined_score),
        bool(flag.movement_explained),
    )

    return Explanation(
        person_id=str(flag.person_id),
        start_t=float(flag.start_t),
        persisted_s=float(flag.persisted_s),
        combined_score=float(flag.combined_score),
        tier=int(flag.tier),
        tier_wording=tier_words,
        movement_explained=bool(flag.movement_explained),
        shifts=shifts,
        text=text,
    )
