"""Demo_App for the Wearable Flag Pipeline — per-person timeline rendering.

This module renders a *single selected participant's* story on one shared time
axis (design "Demo_App — app/demo.py", Requirement 13): the raw signals, the
Combined_Score, the Severity_Tier bands, and the protocol-phase shading, stacked
as subplots that all share the window end time ``t_win`` as their x-axis (AC
13.1). Where a flag is present at a window, the flag's plain-language explanation
is shown next to the timeline (AC 13.2).

It is a *view* only — it computes no new science. It consumes exactly what the
rest of the pipeline already produces:

- a ``Per_Window_Table`` (``data/schema.py``: ``person_id`` / ``session`` /
  ``phase`` / ``t_win`` / ``hr`` / ``rmssd`` / ``skin_temp`` / ``acc_mag``) for
  the selected person,
- a per-window ``Combined_Score`` vector and a per-window ``Severity_Tier``
  vector (``0..3``) aligned to that table's rows (as produced by, e.g.,
  ``eval.run_eval.method_scores`` + ``eval.metrics.tiers_from_scores``, or any
  equivalent source),
- optionally a per-window ``movement_explained`` boolean vector,
- the participant's ``{signal -> SignalBaseline}`` (``flag/baseline.py``) so the
  Explainer can express each flagged window's shift in real units (AC 9.2), and
- the phase segments (``eval.labels.phase_segments``) for the background shading.

Shared time axis and layout (AC 13.1)
-------------------------------------
Four vertically stacked panels share the x-axis (``t_win`` in seconds, re-based
to 0 at the participant's first window for readability):

1. **Signals** — ``hr`` (bpm), ``rmssd`` (ms), ``skin_temp`` (C) on twin y-axes,
   plus ``acc_mag`` when ``show_acc`` is set. Gaps (``NaN``) are left as gaps.
2. **Combined_Score** — the per-window score over time.
3. **Severity_Tier** — a step plot of the tier (``0..3``) over time, labelled
   with the neutral tier names (``None`` / ``Watch`` / ``Notable`` / ``Strong``,
   ``data/schema.TIER_LABELS``) — no color-coded ranges (AC 15.2).

Protocol-phase shading spans *all* panels via ``axvspan`` (one shaded band per
:class:`~eval.labels.PhaseSegment`), so the phase context lines up across the
signals, the score, and the tier (AC 13.1).

Flag explanations (AC 13.2)
---------------------------
A window is "flagged" when its tier is at least Watch (``tier >= 1``). For each
flagged window the Explainer (:func:`flag.explain.explain`) is run on that
window's signed-z contributions (:func:`flag.score.window_signed_z`), raw values,
and baselines to produce the neutral, non-diagnostic explanation text (AC 9.x,
15.x). To keep a long recording readable, by default **one** explanation is shown
per contiguous flagged run — the first flagged window of the run, which is the
onset the user cares about — and those onsets are both annotated on the tier
panel and listed in a side text panel. Set ``annotate_all=True`` to annotate
every flagged window instead. Either way, flagged-window explanations are shown
(AC 13.2); the per-run default is a documented clutter-reduction choice.

Headless by construction
------------------------
matplotlib runs on the non-interactive ``Agg`` backend (``matplotlib.use("Agg")``
before ``pyplot`` import), so a figure is built and saved to a PNG without a
display. :func:`render_timeline` returns the :class:`~matplotlib.figure.Figure`
and, when ``output_path`` is given, also writes it there.

Requirements covered
--------------------
- 13.1 Render the participant's signals, Combined_Score, Severity_Tier bands, and
       phase shading on a shared timeline (:func:`render_timeline`).
- 13.2 Display the flag's explanation at a flagged window
       (:func:`collect_flag_explanations`, :func:`render_timeline`).
- 9.x / 15.x wording is inherited verbatim from :func:`flag.explain.explain`
       (neutral, non-diagnostic — never a diagnosis, never a color-coded label).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# Headless, non-interactive backend: figures are written to disk, never shown.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

from data.schema import (
    Flag,
    SignalBaseline,
    TIER_LABELS,
    TIER_NONE,
    TIER_WATCH,
    tier_label,
)
from eval.labels import PhaseSegment, phase_segments
from flag.explain import Explanation, explain
from flag.score import SCORED_SIGNALS, window_signed_z

__all__ = [
    "SIGNAL_SERIES",
    "PHASE_SHADE_COLORS",
    "FlaggedWindow",
    "collect_flag_explanations",
    "render_timeline",
    "render_participant_timeline",
    "NIGHTLY_SIGNAL_PANELS",
    "TIER_STRIP_SHADES",
    "render_nightly_timeline",
]


# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------

# The raw signal series drawn on the signals panel: (column, display, unit,
# color). HR / RMSSD / skin temperature are the scored signals; acc_mag is the
# movement context, drawn only when ``show_acc`` is set.
SIGNAL_SERIES: tuple[tuple[str, str, str, str], ...] = (
    ("hr", "heart rate", "bpm", "#d62728"),
    ("rmssd", "HRV", "ms", "#1f77b4"),
    ("skin_temp", "skin temperature", "C", "#2ca02c"),
)

# Distinct, deliberately non-"severity" background shades cycled per phase
# segment. These color *protocol phases* (context), not severity — severity tiers
# are never color-coded (AC 15.2).
PHASE_SHADE_COLORS: tuple[str, ...] = (
    "#f0f0f0",
    "#e6f2ff",
    "#fff2e6",
    "#eaf7ea",
    "#f3e6ff",
    "#fdeef4",
)


# ---------------------------------------------------------------------------
# Flagged-window explanations (AC 13.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlaggedWindow:
    """One flagged window and its plain-language explanation (AC 13.2).

    ``row_index`` is the 0-based positional row into the per-person table passed
    to :func:`collect_flag_explanations`; ``t_win`` is that window's end time (in
    the table's native epoch seconds, before any re-basing done for display).
    ``tier`` is the window's Severity_Tier (``>= 1`` — only flagged windows are
    collected). ``explanation`` is the Explainer's output for the window and
    ``explanation.text`` the rendered neutral, non-diagnostic string.
    """

    row_index: int
    t_win: float
    tier: int
    explanation: Explanation


def _as_float_array(values: Sequence[float] | np.ndarray, n: int, name: str) -> np.ndarray:
    """Coerce ``values`` to a length-``n`` float array, raising on a length mismatch."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size != n:
        raise ValueError(
            f"{name} has length {arr.size} but the table has {n} row(s); "
            f"per-window vectors must align one-to-one with the table rows"
        )
    return arr


def _as_int_array(values: Sequence[int] | np.ndarray, n: int, name: str) -> np.ndarray:
    """Coerce ``values`` to a length-``n`` int array, raising on a length mismatch."""
    arr = np.asarray(values).ravel()
    if arr.size != n:
        raise ValueError(
            f"{name} has length {arr.size} but the table has {n} row(s); "
            f"per-window vectors must align one-to-one with the table rows"
        )
    return arr.astype(int)


def collect_flag_explanations(
    table: pd.DataFrame,
    scores: Sequence[float] | np.ndarray,
    tiers: Sequence[int] | np.ndarray,
    baselines: Mapping[str, SignalBaseline],
    *,
    movement: Sequence[bool] | np.ndarray | None = None,
    first_per_run: bool = True,
) -> list[FlaggedWindow]:
    """Build Explainer output for the table's flagged windows (AC 13.2).

    A window is *flagged* when its tier is at least Watch (``tier >= TIER_WATCH``).
    For each such window a :class:`~data.schema.Flag` is reconstructed from the
    per-window inputs and passed to :func:`flag.explain.explain` with that
    window's signed-z contributions (:func:`flag.score.window_signed_z`), raw
    real-unit values, and the participant's baselines, so the explanation names
    the top contributing signals and their real-unit shift from rest (AC 9.1,
    9.2) with neutral tier wording (AC 9.3, 15.x).

    ``persisted_s`` for a flagged window is the span from the start of its
    contiguous flagged run to the window's own ``t_win`` — a simple, honest "how
    long this flag has been up" read straight off the timeline (the production
    Flagger owns the authoritative persistence; here we only *display* it).

    Parameters
    ----------
    table:
        The selected participant's ``Per_Window_Table`` rows, in ``t_win`` order.
    scores, tiers:
        Per-window Combined_Score and Severity_Tier vectors, aligned to ``table``.
    baselines:
        The participant's ``{signal -> SignalBaseline}`` for real-unit shifts.
    movement:
        Optional per-window ``movement_explained`` flags; ``False`` when omitted.
    first_per_run:
        When ``True`` (default) only the first window of each contiguous flagged
        run is returned (the onset), to keep the timeline readable; when ``False``
        every flagged window is returned.

    Returns
    -------
    list[FlaggedWindow]
        One entry per reported flagged window, in table order.
    """
    n = len(table)
    if n == 0:
        return []

    score_arr = _as_float_array(scores, n, "scores")
    tier_arr = _as_int_array(tiers, n, "tiers")
    if movement is None:
        move_arr = np.zeros(n, dtype=bool)
    else:
        move_arr = _as_int_array(movement, n, "movement").astype(bool)

    t_wins = table["t_win"].to_numpy(dtype=float)
    person_ids = table["person_id"].astype(str).to_numpy()

    out: list[FlaggedWindow] = []
    run_start_t: float | None = None
    prev_flagged = False

    for pos in range(n):
        tier = int(tier_arr[pos])
        flagged = tier >= TIER_WATCH

        if flagged and not prev_flagged:
            # Onset of a new flagged run.
            run_start_t = float(t_wins[pos])
        prev_flagged = flagged

        if not flagged:
            continue
        if first_per_run and run_start_t is not None and float(t_wins[pos]) != run_start_t:
            # Not the onset of this run; skip when collapsing to one-per-run.
            continue

        row = table.iloc[pos]
        start_t = run_start_t if run_start_t is not None else float(t_wins[pos])
        persisted_s = float(t_wins[pos]) - float(start_t)

        contributions = window_signed_z(row, baselines)

        # Real-unit shift per available signal: value - baseline.median. The
        # Explainer recomputes this from ``values`` + ``baselines`` too; carrying
        # it on the Flag keeps a sensible fallback.
        top_signals: list[tuple[str, float]] = []
        values: dict[str, float] = {}
        for signal in SCORED_SIGNALS:
            try:
                raw = float(row[signal])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(raw):
                continue
            values[signal] = raw
            b = baselines.get(signal)
            if b is not None and np.isfinite(float(b.median)):
                top_signals.append((signal, raw - float(b.median)))

        flag = Flag(
            person_id=str(person_ids[pos]),
            start_t=float(start_t),
            persisted_s=float(persisted_s),
            combined_score=float(score_arr[pos]),
            tier=tier,
            top_signals=top_signals,
            movement_explained=bool(move_arr[pos]),
        )
        explanation = explain(flag, contributions, baselines, values=values)
        out.append(
            FlaggedWindow(
                row_index=pos,
                t_win=float(t_wins[pos]),
                tier=tier,
                explanation=explanation,
            )
        )

    return out


# ---------------------------------------------------------------------------
# Timeline figure (AC 13.1, 13.2)
# ---------------------------------------------------------------------------


def _shade_phases(
    axes: Sequence[plt.Axes],
    segments: Sequence[PhaseSegment],
    t0: float,
) -> list[tuple[str, float, float, str]]:
    """Shade each protocol-phase segment across ``axes`` via ``axvspan`` (AC 13.1).

    Returns the legend entries ``(phase, start_rel, end_rel, color)`` (times
    re-based so ``t0`` is 0) so the caller can build a phase legend. The caller
    passes the *signal* mini-panels here (not the score panel): colored phase
    shading is the protocol *context*, kept visually distinct from the greyscale
    *severity-tier* shading the score panel reserves for itself.
    """
    legend: list[tuple[str, float, float, str]] = []
    for i, seg in enumerate(segments):
        color = PHASE_SHADE_COLORS[i % len(PHASE_SHADE_COLORS)]
        start_rel = float(seg.start_t) - t0
        end_rel = float(seg.end_t) - t0
        # Guard a zero-width segment (single window) so the band is visible.
        if end_rel <= start_rel:
            end_rel = start_rel + 1e-6
        for ax in axes:
            ax.axvspan(start_rel, end_rel, color=color, alpha=0.5, zorder=0, linewidth=0)
        legend.append((seg.phase, start_rel, end_rel, color))
    return legend


def _draw_signal_panel(
    ax: plt.Axes,
    t_rel: np.ndarray,
    table: pd.DataFrame,
    col: str,
    display: str,
    unit: str,
    color: str,
    baselines: Mapping[str, SignalBaseline],
) -> None:
    """Draw one signal as a small-multiple mini-panel on its own y-scale.

    Mirrors the readable WHOOP nightly look (:func:`render_nightly_timeline`):
    each signal gets its own panel in its own real unit rather than being crammed
    onto one shared twin axis, with a faint dashed baseline-median reference line
    from ``baselines`` when available (the "your resting normal" anchor; AC 9.2
    context). Missing data (``NaN``) is left as a gap.
    """
    if col in table.columns:
        y = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
        ax.plot(t_rel, y, color=color, linewidth=1.2)
        b = baselines.get(col)
        if b is not None and np.isfinite(float(b.median)):
            ax.axhline(
                float(b.median), color=color, linewidth=0.8, alpha=0.35, linestyle="--"
            )
    ax.set_ylabel(f"{display}\n({unit})", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.tick_params(labelsize=8)


def render_timeline(
    table: pd.DataFrame,
    person_id: str,
    scores: Sequence[float] | np.ndarray,
    tiers: Sequence[int] | np.ndarray,
    baselines: Mapping[str, SignalBaseline],
    *,
    movement: Sequence[bool] | np.ndarray | None = None,
    segments: Sequence[PhaseSegment] | None = None,
    show_acc: bool = False,
    annotate_all: bool = False,
    taus=None,
    output_path: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Render one participant's timeline and (optionally) save it as a PNG (Req. 13).

    Restyled to match the readable WHOOP nightly look
    (:func:`render_nightly_timeline`), adapted to PhysioNet's per-*window* data:

    - **Signal small-multiples.** Each scored signal — ``hr`` (bpm), ``rmssd``
      (ms), ``skin_temp`` (C) — gets its own mini-panel on its own real-unit
      y-scale with a faint dashed baseline-median reference line (from
      ``baselines`` when present), instead of being crammed onto one shared twin
      axis. ``acc_mag`` gets a 4th mini-panel only when ``show_acc`` is set.
    - **Merged score + tier panel.** The Combined_Score line and the
      Severity_Tier are drawn on one panel: the tier is painted as muted-grey
      background shading (:data:`TIER_STRIP_SHADES`, tier 0 left unshaded) behind
      the score line, and when ``taus`` is supplied its ``tau1/tau2/tau3`` are
      drawn as labelled threshold lines so a viewer can see *why* a window
      crossed into a tier. Flagged windows (tier ``>= Watch``) are marked and a
      neutral tier legend is shown. There is no separate tier strip panel.
    - **Phase context stays colored, severity stays greyscale.** Because the
      score panel now uses greyscale for the *severity tier*, the colored
      protocol-**phase** shading (PhysioNet-specific) is drawn on the *signal*
      mini-panels via ``axvspan`` with the existing phase legend. Keeping phase
      (colored context) and tier (greyscale severity) on different panels and
      palettes keeps the two meanings visually distinct (AC 13.1, 15.2).
    - **Flag explanations** for flagged windows remain visible in a text panel
      (AC 13.2).

    The x-axis is PhysioNet elapsed **seconds** ("time since first window (s)"),
    re-based to 0 at the first window — *not* calendar dates — with a
    ``MaxNLocator`` so a multi-thousand-second session is not cramped.

    Parameters
    ----------
    table:
        A ``Per_Window_Table``. Only the rows for ``person_id`` are plotted; the
        table may hold other people (they are filtered out here). Rows are sorted
        by ``t_win`` for a monotone axis.
    person_id:
        The participant to render. ``ValueError`` if absent from ``table``.
    scores, tiers:
        Per-window Combined_Score and Severity_Tier (``0..3``) vectors aligned to
        ``table`` *as passed in* (before the per-person filter). They are filtered
        alongside the rows so they stay aligned.
    baselines:
        The participant's ``{signal -> SignalBaseline}`` for the real-unit shifts
        in the explanations (AC 9.2) and the per-signal reference lines.
    movement:
        Optional per-window ``movement_explained`` flags, aligned to ``table``.
    segments:
        Optional precomputed phase segments for ``person_id``. When omitted they
        are derived from the filtered per-person table via
        :func:`eval.labels.phase_segments`.
    show_acc:
        Also draw ``acc_mag`` as a 4th signal mini-panel.
    annotate_all:
        Annotate every flagged window rather than one per contiguous flagged run
        (the default collapses to the run onset to reduce clutter; AC 13.2).
    taus:
        Optional calibrated :class:`~data.schema.Thresholds`. When provided the
        ``tau1/tau2/tau3`` are drawn as labelled lines on the score panel; when
        ``None`` only the score line and tier shading are drawn (no fabricated
        thresholds).
    output_path:
        When given, the figure is saved here as a PNG (parent dirs created).
    title:
        Optional figure suptitle; a sensible default is used otherwise.

    Returns
    -------
    matplotlib.figure.Figure
        The built figure (also written to ``output_path`` when provided).
    """
    if "person_id" not in table.columns:
        raise ValueError("table is missing the 'person_id' column")

    n_all = len(table)
    score_all = _as_float_array(scores, n_all, "scores")
    tier_all = _as_int_array(tiers, n_all, "tiers")
    move_all = (
        np.zeros(n_all, dtype=bool)
        if movement is None
        else _as_int_array(movement, n_all, "movement").astype(bool)
    )

    # Filter to the selected person, keeping the per-window vectors aligned, then
    # sort by t_win for a monotone shared axis.
    mask = table["person_id"].astype(str).to_numpy() == str(person_id)
    if not mask.any():
        raise ValueError(f"person_id {person_id!r} not found in table")

    sub = table.loc[mask].copy()
    sub_scores = score_all[mask]
    sub_tiers = tier_all[mask]
    sub_move = move_all[mask]

    order = np.argsort(sub["t_win"].to_numpy(dtype=float), kind="stable")
    sub = sub.iloc[order].reset_index(drop=True)
    sub_scores = sub_scores[order]
    sub_tiers = sub_tiers[order]
    sub_move = sub_move[order]

    if segments is None:
        segments = phase_segments(sub)

    t_win = sub["t_win"].to_numpy(dtype=float)
    t0 = float(t_win[0]) if t_win.size else 0.0
    t_rel = t_win - t0

    # Flagged-window explanations (AC 13.2).
    flagged = collect_flag_explanations(
        sub,
        sub_scores,
        sub_tiers,
        baselines,
        movement=sub_move,
        first_per_run=not annotate_all,
    )

    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    # Which signals get a mini-panel: the three scored signals, plus acc_mag only
    # when show_acc is set. Each on its own real-unit y-scale (small multiples).
    panel_specs = [spec for spec in NIGHTLY_SIGNAL_PANELS if spec[0] in sub.columns]
    if show_acc and "acc_mag" in sub.columns:
        panel_specs = panel_specs + [("acc_mag", "movement", "acc", "#7f7f7f")]
    n_sig = len(panel_specs)

    # Layout: n_sig signal mini-panels, one merged score+tier panel, one flag-text
    # panel. The time panels (signals + score) share the x-axis; the text panel is
    # kept independent so its layout is not stretched by the shared axis.
    height_ratios = [1.2] * n_sig + [3.0, 2.2]
    fig, axes = plt.subplots(
        n_sig + 2,
        1,
        figsize=(11.0, 2.0 + 1.25 * n_sig + 5.0),
        sharex=False,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.35},
    )
    sig_axes = list(axes[:n_sig])
    ax_score = axes[n_sig]
    ax_text = axes[n_sig + 1]

    for ax in sig_axes:
        ax.sharex(ax_score)

    # Colored protocol-PHASE shading on the SIGNAL mini-panels (AC 13.1). The
    # score panel reserves greyscale for the SEVERITY TIER (below), so phase
    # context (color) and severity (greyscale) never share a palette or a panel.
    phase_legend = _shade_phases(sig_axes, segments, t0) if sig_axes else []

    # Signal mini-panels (each its own y-scale + faint baseline reference line).
    for ax, (col, display, unit, color) in zip(sig_axes, panel_specs):
        _draw_signal_panel(ax, t_rel, sub, col, display, unit, color, baselines)
    if sig_axes:
        sig_axes[0].set_title("Signals (each on its own scale)", fontsize=10, loc="left")

    # --- Merged Combined_Score + Severity_Tier panel. ---
    # Tier painted as muted-grey background bands (one axvspan per contiguous run
    # of same-tier windows; tier 0 unshaded). Neutral greyscale, never a
    # red/green health verdict (AC 15.2).
    if len(t_rel) >= 1:
        if len(t_rel) >= 2:
            half = float(np.median(np.diff(t_rel))) / 2.0
        else:
            half = 0.5
        start = 0
        for pos in range(1, len(sub_tiers) + 1):
            if pos == len(sub_tiers) or int(sub_tiers[pos]) != int(sub_tiers[start]):
                tier = min(max(int(sub_tiers[start]), 0), 3)
                if tier > 0:  # leave None unshaded for a cleaner look
                    left = t_rel[start] - half
                    right = t_rel[pos - 1] + half
                    ax_score.axvspan(
                        left, right,
                        facecolor=TIER_STRIP_SHADES[tier], alpha=0.45,
                        edgecolor="none", zorder=0,
                    )
                start = pos

    ax_score.plot(t_rel, sub_scores, color="#111111", linewidth=1.3, zorder=4)

    # Tuned threshold lines, only when supplied (never fabricated).
    if taus is not None:
        threshold_lines = (
            (float(taus.tau1), "Watch (tau1)", "#9467bd"),
            (float(taus.tau2), "Notable (tau2)", "#ff7f0e"),
            (float(taus.tau3), "Strong (tau3)", "#8c564b"),
        )
        for level, label, color in threshold_lines:
            ax_score.axhline(
                level, color=color, linewidth=1.0, linestyle="--", alpha=0.9, zorder=3
            )
            ax_score.text(
                1.005, level, label,
                transform=ax_score.get_yaxis_transform(),
                va="center", ha="left", fontsize=7, color=color,
            )

    # Mark flagged windows on the score line (AC 13.2).
    flagged_mask = sub_tiers >= TIER_WATCH
    if flagged_mask.any():
        ax_score.scatter(
            t_rel[flagged_mask], sub_scores[flagged_mask],
            s=34, facecolors="none", edgecolors="#d62728", linewidths=1.2, zorder=5,
        )
    # Annotate each reported flagged window (onset per run by default) so the
    # text-panel numbering lines up with the marks on the score panel. ``fw.row_index``
    # is the positional row into ``sub`` (collect_flag_explanations builds it from
    # the same per-person table), so x/y come straight off that window.
    for i, fw in enumerate(flagged):
        ridx = int(fw.row_index)
        if 0 <= ridx < len(t_rel):
            ax_score.annotate(
                f"#{i + 1}",
                (float(t_rel[ridx]), float(sub_scores[ridx])),
                textcoords="offset points",
                xytext=(2, 7),
                fontsize=7,
                color="#d62728",
                zorder=6,
            )

    ax_score.set_title(
        "Combined score with severity-tier shading" + (" and tuned thresholds" if taus is not None else ""),
        fontsize=10, loc="left",
    )
    ax_score.set_ylabel("combined\nscore", fontsize=8)
    ax_score.set_xlabel("time since first window (s)")
    ax_score.grid(True, linestyle=":", alpha=0.4)
    ax_score.tick_params(labelsize=8)
    ax_score.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=False))

    # Neutral severity-tier legend (muted greyscale — NOT a health verdict).
    tier_handles = [
        Patch(facecolor=TIER_STRIP_SHADES[t], edgecolor="#bbbbbb", label=tier_label(t))
        for t in sorted(TIER_LABELS)
    ]
    ax_score.legend(
        handles=tier_handles, loc="upper right", ncol=4, fontsize=7, framealpha=0.9,
        title="severity tier shading (neutral, not a health verdict)",
        title_fontsize=7,
    )

    # Flag-explanation text panel (AC 13.2).
    ax_text.axis("off")
    ax_text.set_title("Flag explanations", loc="left", fontsize=10)
    if flagged:
        lines: list[str] = []
        shown = flagged[:6]
        for i, fw in enumerate(shown):
            rel = float(fw.t_win) - t0
            lines.append(
                f"#{i + 1}  t+{rel:g}s  [{tier_label(fw.tier)}]  {fw.explanation.text}"
            )
        if len(flagged) > len(shown):
            lines.append(f"... and {len(flagged) - len(shown)} more flagged window(s).")
        ax_text.text(
            0.0,
            1.0,
            "\n\n".join(_wrap(line, 120) for line in lines),
            transform=ax_text.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            family="DejaVu Sans",
        )
    else:
        ax_text.text(
            0.0,
            1.0,
            "No flags fired for this participant on this timeline.",
            transform=ax_text.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )

    # Colored protocol-phase legend on the top signal mini-panel (context, not
    # severity). Kept distinct from the greyscale tier legend on the score panel.
    if phase_legend and sig_axes:
        seen: dict[str, str] = {}
        for phase, _s, _e, color in phase_legend:
            seen.setdefault(phase, color)
        handles = [Patch(facecolor=c, edgecolor="none", label=p) for p, c in seen.items()]
        sig_axes[0].legend(
            handles=handles,
            loc="upper right",
            fontsize=6.5,
            framealpha=0.8,
            ncol=3,
            title="protocol phase (context)",
            title_fontsize=6.5,
        )

    session = str(sub["session"].iloc[0]) if "session" in sub.columns and len(sub) else ""
    default_title = f"Timeline — {person_id}" + (f" ({session})" if session else "")
    fig.suptitle(title or default_title, fontsize=13)
    fig.subplots_adjust(top=0.94, right=0.88)

    if output_path is not None:
        parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")

    return fig


def _wrap(text: str, width: int) -> str:
    """Soft-wrap ``text`` to ``width`` columns (keeps the text panel inside the axes)."""
    import textwrap

    return "\n    ".join(textwrap.wrap(text, width=width)) or text


# ---------------------------------------------------------------------------
# Convenience: render straight from a BuiltTable + per-window score/tier vectors
# ---------------------------------------------------------------------------


def render_participant_timeline(
    built,
    person_id: str,
    scores: Sequence[float] | np.ndarray,
    tiers: Sequence[int] | np.ndarray,
    *,
    movement: Sequence[bool] | np.ndarray | None = None,
    show_acc: bool = False,
    annotate_all: bool = False,
    taus=None,
    output_path: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Render a timeline from an ``eval.run_eval.BuiltTable`` for one person (Req. 13).

    Convenience wrapper around :func:`render_timeline` that takes the combined
    :class:`~eval.run_eval.BuiltTable` (its ``table`` + per-person ``baselines``)
    and the full-table per-window ``scores`` / ``tiers`` vectors — exactly what
    :func:`eval.run_eval.method_scores` and :func:`eval.metrics.tiers_from_scores`
    produce for a method. The combined-table rows and the vectors are filtered to
    ``person_id`` by :func:`render_timeline`, so pass the vectors aligned to the
    *full* ``built.table`` (not pre-sliced).

    ``built.baselines`` is ``{person_id -> {signal -> SignalBaseline}}``; the
    selected person's baseline set is handed to the Explainer for real-unit shifts
    (AC 9.2). A person with no baseline set renders with empty baselines (the
    explanations then fall back to the flag-carried deltas).

    ``taus`` is an optional calibrated :class:`~data.schema.Thresholds`, threaded
    through to :func:`render_timeline` so the score panel can draw labelled
    ``tau1/tau2/tau3`` threshold lines; pass ``None`` to omit them.
    """
    table = built.table
    baselines = dict(built.baselines.get(str(person_id), {}))
    return render_timeline(
        table,
        person_id,
        scores,
        tiers,
        baselines,
        movement=movement,
        show_acc=show_acc,
        annotate_all=annotate_all,
        taus=taus,
        output_path=output_path,
        title=title,
    )


# ---------------------------------------------------------------------------
# Nightly timeline (WHOOP real-run view) — a *readable* calendar-dated variant
# ---------------------------------------------------------------------------
#
# ``render_timeline`` above is the PhysioNet per-window view: a shared x-axis in
# elapsed SECONDS, the signals as small-multiple mini-panels, a merged score+tier
# panel, and colored protocol-phase shading on the signal panels. A WHOOP export,
# by contrast, is hundreds of nights spaced ~1 day apart with no protocol phases,
# so ``render_nightly_timeline`` below is a *separate* function that plots the
# same merged-panel idea on a human DATE axis instead of seconds. It does not
# touch ``render_timeline``. It computes no new science — it draws exactly the
# per-night scores, tiers, tuned thresholds and baselines handed to it by
# ``scripts/whoop_run.py``.

# One mini-panel per scored signal so each has its own y-scale in its own real
# unit (resting HR bpm, HRV ms, skin temp C) — no cramming incomparable units
# onto one axis: (column, display, unit, color).
NIGHTLY_SIGNAL_PANELS: tuple[tuple[str, str, str, str], ...] = (
    ("hr", "resting HR", "bpm", "#d62728"),
    ("rmssd", "HRV", "ms", "#1f77b4"),
    ("skin_temp", "skin temp", "C", "#2ca02c"),
)

# Muted, sequential grey shades for the severity-tier background shading on the
# nightly score panel — deliberately NOT a red/amber/green health verdict (the
# project avoids alarming medical color coding; the shading carries a neutral
# "severity tier" legend). Index by tier 0..3: darker = higher tier.
TIER_STRIP_SHADES: tuple[str, ...] = (
    "#f5f5f5",  # 0 None
    "#c7c7c7",  # 1 Watch
    "#8a8a8a",  # 2 Notable
    "#3f3f3f",  # 3 Strong
)


def _epoch_to_datetimes(t_win: np.ndarray) -> np.ndarray:
    """Convert epoch-second window times to an array of ``datetime`` objects (UTC).

    The nightly x-axis is human dates, not seconds; matplotlib plots
    ``datetime`` objects directly and formats month/day ticks itself.
    """
    from datetime import datetime, timezone

    out: list = []
    for t in t_win:
        try:
            out.append(datetime.fromtimestamp(float(t), tz=timezone.utc))
        except (ValueError, OverflowError, OSError):
            out.append(None)
    return np.array(out, dtype=object)


def render_nightly_timeline(
    table: pd.DataFrame,
    person_id: str,
    scores: Sequence[float] | np.ndarray,
    tiers: Sequence[int] | np.ndarray,
    baselines: Mapping[str, SignalBaseline],
    taus,
    *,
    movement: Sequence[bool] | np.ndarray | None = None,
    last_n: int | None = 90,
    max_annotations: int = 4,
    output_path: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Render a *readable* nightly WHOOP timeline on a human DATE axis.

    A separate, calendar-dated companion to :func:`render_timeline` for the
    hundreds-of-nights WHOOP real run. Two groups of panels share a date x-axis:

    1. Three signal mini-panels — resting HR (bpm), HRV (ms), skin temp (C) — each
       on its own y-scale in its own unit (small multiples, never one shared axis
       of incomparable units).
    2. The combined score, with the tuned ``taus.tau1 / tau2 / tau3`` drawn as
       labelled horizontal lines (Watch / Notable / Strong) so a viewer can *see*
       why a night crossed into a tier. The severity tier is overlaid onto this
       same panel as subtle background shading — one muted-grey band per contiguous
       run of same-tier nights (darker = higher tier) — so "score rose above tau2
       and the background darkens to Notable" reads in a single panel. The bands
       use a neutral greyscale legend, not a red/green health verdict (AC 15.2).

    Flagged nights (tier >= Watch) are marked on the score panel; the few
    highest-tier nights are annotated with their date and top contributing signal
    in real units.

    Parameters
    ----------
    table, scores, tiers, baselines, movement:
        Same per-night inputs as the WHOOP run produces (aligned to the full
        ``table``; filtered to ``person_id`` and sorted by ``t_win`` here).
    taus:
        The tuned :class:`~data.schema.Thresholds` for the threshold lines.
    last_n:
        Show only the most recent ``last_n`` nights (default 90) for legibility;
        ``None`` shows the full history.
    max_annotations:
        Annotate at most this many of the highest-tier flagged nights.
    output_path:
        When given, the figure is saved here as a PNG (parent dirs created).
    title:
        Optional figure suptitle.
    """
    if "person_id" not in table.columns:
        raise ValueError("table is missing the 'person_id' column")

    n_all = len(table)
    score_all = _as_float_array(scores, n_all, "scores")
    tier_all = _as_int_array(tiers, n_all, "tiers")
    move_all = (
        np.zeros(n_all, dtype=bool)
        if movement is None
        else _as_int_array(movement, n_all, "movement").astype(bool)
    )

    mask = table["person_id"].astype(str).to_numpy() == str(person_id)
    if not mask.any():
        raise ValueError(f"person_id {person_id!r} not found in table")

    sub = table.loc[mask].copy()
    sub_scores = score_all[mask]
    sub_tiers = tier_all[mask]
    sub_move = move_all[mask]

    order = np.argsort(sub["t_win"].to_numpy(dtype=float), kind="stable")
    sub = sub.iloc[order].reset_index(drop=True)
    sub_scores = sub_scores[order]
    sub_tiers = sub_tiers[order]
    sub_move = sub_move[order]

    # Trim to the most recent window so the detail is legible.
    if last_n is not None and len(sub) > last_n:
        sub = sub.iloc[-last_n:].reset_index(drop=True)
        sub_scores = sub_scores[-last_n:]
        sub_tiers = sub_tiers[-last_n:]
        sub_move = sub_move[-last_n:]

    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    dates = _epoch_to_datetimes(sub["t_win"].to_numpy(dtype=float))

    n_sig = len(NIGHTLY_SIGNAL_PANELS)
    fig, axes = plt.subplots(
        n_sig + 1,
        1,
        figsize=(12.0, 9.0),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.3] * n_sig + [2.8],
            "hspace": 0.28,
        },
    )
    sig_axes = list(axes[:n_sig])
    ax_score = axes[n_sig]

    # --- Signal mini-panels: each signal on its own real-unit y-scale. ---
    for ax, (col, display, unit, color) in zip(sig_axes, NIGHTLY_SIGNAL_PANELS):
        if col in sub.columns:
            y = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            ax.plot(dates, y, color=color, linewidth=1.3, marker=".", markersize=3)
            # Baseline center as a faint reference line (your recent-nights rest).
            b = baselines.get(col)
            if b is not None and np.isfinite(float(b.median)):
                ax.axhline(
                    float(b.median), color=color, linewidth=0.8, alpha=0.35,
                    linestyle="--",
                )
        ax.set_ylabel(f"{display}\n({unit})", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.tick_params(labelsize=8)

    # --- Combined score + severity-tier shading + tuned threshold lines. ---
    # Overlay the severity tier as subtle background bands behind the score line:
    # one muted-grey span per contiguous run of same-tier nights (darker = higher
    # tier). This is deliberately a neutral greyscale, NOT a red/green health
    # verdict (AC 15.2); tier 0 (None) is left unshaded (white) to keep the score
    # and threshold lines clearly readable on top.
    xnum = mdates.date2num([d for d in dates])
    if len(xnum) >= 1:
        # Half the median night spacing, used to extend band edges to each
        # night's boundary so adjacent runs meet without gaps.
        if len(xnum) >= 2:
            half = float(np.median(np.diff(xnum))) / 2.0
        else:
            half = 0.5
        start = 0
        for pos in range(1, len(sub_tiers) + 1):
            if pos == len(sub_tiers) or int(sub_tiers[pos]) != int(sub_tiers[start]):
                tier = min(max(int(sub_tiers[start]), 0), 3)
                if tier > 0:  # leave None unshaded for a cleaner look
                    left = xnum[start] - half
                    right = xnum[pos - 1] + half
                    ax_score.axvspan(
                        left, right,
                        facecolor=TIER_STRIP_SHADES[tier], alpha=0.4,
                        edgecolor="none", zorder=0,
                    )
                start = pos

    ax_score.plot(
        dates, sub_scores, color="#111111", linewidth=1.3,
        marker=".", markersize=3, zorder=4,
    )
    threshold_lines = (
        (float(taus.tau1), "Watch (tau1)", "#9467bd"),
        (float(taus.tau2), "Notable (tau2)", "#ff7f0e"),
        (float(taus.tau3), "Strong (tau3)", "#8c564b"),
    )
    for level, label, color in threshold_lines:
        ax_score.axhline(level, color=color, linewidth=1.0, linestyle="--", alpha=0.9, zorder=3)
        ax_score.text(
            1.005, level, label,
            transform=ax_score.get_yaxis_transform(),
            va="center", ha="left", fontsize=7, color=color,
        )
    # Mark flagged nights.
    flagged_mask = sub_tiers >= TIER_WATCH
    if flagged_mask.any():
        ax_score.scatter(
            dates[flagged_mask], sub_scores[flagged_mask],
            s=36, facecolors="none", edgecolors="#d62728", linewidths=1.2, zorder=5,
        )
    ax_score.set_ylabel("combined\nscore", fontsize=8)
    ax_score.grid(True, linestyle=":", alpha=0.4)
    ax_score.tick_params(labelsize=8)

    # Annotate the highest-tier flagged nights with date + top signal (real unit).
    flagged_idx = [i for i in range(len(sub)) if int(sub_tiers[i]) >= TIER_WATCH]
    flagged_idx.sort(key=lambda i: (int(sub_tiers[i]), float(sub_scores[i])), reverse=True)
    # Stagger annotation offsets so clustered flagged nights don't overprint.
    _offsets = [(0, 26), (0, 46), (0, 14), (0, 36)]
    for ann_i, i in enumerate(flagged_idx[:max_annotations]):
        row = sub.iloc[i]
        contributions = window_signed_z(row, baselines)
        ranked = sorted(
            contributions.items(), key=lambda kv: abs(float(kv[1])), reverse=True
        )
        note = tier_label(int(sub_tiers[i]))
        if ranked:
            top_signal = ranked[0][0]
            disp = {"hr": "HR", "rmssd": "HRV", "skin_temp": "temp"}.get(top_signal, top_signal)
            unit = {"hr": "bpm", "rmssd": "ms", "skin_temp": "C"}.get(top_signal, "")
            b = baselines.get(top_signal)
            try:
                raw = float(row[top_signal])
            except (KeyError, TypeError, ValueError):
                raw = float("nan")
            if b is not None and np.isfinite(raw) and np.isfinite(float(b.median)):
                delta = raw - float(b.median)
                note = f"{note}: {disp} {delta:+.1f} {unit}"
        d = dates[i]
        label = d.strftime("%b %d") if d is not None else ""
        xytext = _offsets[ann_i % len(_offsets)]
        ax_score.annotate(
            f"{label} · {note}",
            (dates[i], float(sub_scores[i])),
            textcoords="offset points", xytext=xytext,
            fontsize=6.5, ha="center", color="#333333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.85),
            arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.6),
        )

    # --- Severity-tier legend (muted greyscale, neutral — not a health verdict).
    # The tier itself is now drawn as background shading on this same score panel
    # (above); only the explanatory legend lives down here, relocated from the
    # retired strip panel. One swatch per tier, labelled with the neutral names.
    strip_legend = [
        Patch(facecolor=TIER_STRIP_SHADES[t], edgecolor="#bbbbbb", label=tier_label(t))
        for t in sorted(TIER_LABELS)
    ]
    ax_score.legend(
        handles=strip_legend, loc="upper center", bbox_to_anchor=(0.5, -0.22),
        ncol=4, fontsize=7, framealpha=0.9,
        title="severity tier shading (neutral, not a health verdict)",
        title_fontsize=7,
    )

    # Date formatting on the shared x-axis, now anchored on the bottom-most
    # (score) panel: one tick per month, no duplicates.
    ax_score.xaxis.set_major_locator(mdates.MonthLocator())
    ax_score.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_score.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    for lab in ax_score.get_xticklabels():
        lab.set_rotation(0)
    ax_score.set_xlabel("night (date)", fontsize=9, labelpad=2)

    sig_axes[0].set_title("Nightly signals (each on its own scale)", fontsize=10, loc="left")
    ax_score.set_title(
        "Combined deviation score with severity-tier shading and tuned thresholds",
        fontsize=10, loc="left",
    )

    default_title = f"WHOOP nightly timeline — {person_id}"
    if last_n is not None and len(sub):
        default_title += f" (last {len(sub)} nights)"
    fig.suptitle(title or default_title, fontsize=13)
    fig.subplots_adjust(top=0.93, right=0.88, bottom=0.16)

    if output_path is not None:
        parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")

    return fig
