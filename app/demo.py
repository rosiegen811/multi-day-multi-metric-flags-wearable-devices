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
    """Shade each phase segment across all ``axes`` via ``axvspan`` (AC 13.1).

    Returns the legend entries ``(phase, start_rel, end_rel, color)`` (times
    re-based so ``t0`` is 0) so the caller can build a phase legend. The shading
    spans every panel, so a phase lines up vertically across the signals, the
    score, and the tier.
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
            ax.axvspan(start_rel, end_rel, color=color, alpha=0.6, zorder=0, linewidth=0)
        legend.append((seg.phase, start_rel, end_rel, color))
    return legend


def _draw_signals(ax: plt.Axes, t_rel: np.ndarray, table: pd.DataFrame, show_acc: bool) -> None:
    """Draw the raw signals on ``ax`` (HR / HRV / skin temp on twin axes; AC 13.1)."""
    host = ax
    # HR on the host axis.
    axes_for_signals = [host]
    host.set_ylabel("bpm / ms / C")
    handles: list = []

    for col, display, unit, color in SIGNAL_SERIES:
        if col not in table.columns:
            continue
        y = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
        (line,) = host.plot(
            t_rel, y, color=color, linewidth=1.2, label=f"{display} ({unit})"
        )
        handles.append(line)

    if show_acc and "acc_mag" in table.columns:
        acc_ax = host.twinx()
        axes_for_signals.append(acc_ax)
        y = pd.to_numeric(table["acc_mag"], errors="coerce").to_numpy(dtype=float)
        (line,) = acc_ax.plot(
            t_rel, y, color="#7f7f7f", linewidth=0.8, alpha=0.7, label="acc_mag"
        )
        acc_ax.set_ylabel("acc_mag", color="#7f7f7f")
        acc_ax.tick_params(axis="y", labelcolor="#7f7f7f")
        handles.append(line)

    host.set_title("Signals")
    host.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.8)
    host.grid(True, linestyle=":", alpha=0.4)


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
    output_path: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Render one participant's timeline and (optionally) save it as a PNG (Req. 13).

    Builds a headless matplotlib figure of four vertically stacked, x-shared
    panels — signals, Combined_Score, Severity_Tier bands, and a flag-explanation
    text panel — with protocol-phase shading spanning the first three panels (AC
    13.1). Flagged windows (tier ``>= Watch``) carry the Explainer's neutral,
    non-diagnostic explanation, annotated on the tier panel and listed in the text
    panel (AC 13.2).

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
        in the explanations (AC 9.2).
    movement:
        Optional per-window ``movement_explained`` flags, aligned to ``table``.
    segments:
        Optional precomputed phase segments for ``person_id``. When omitted they
        are derived from the filtered per-person table via
        :func:`eval.labels.phase_segments`.
    show_acc:
        Also plot ``acc_mag`` on a twin y-axis of the signals panel.
    annotate_all:
        Annotate every flagged window rather than one per contiguous flagged run
        (the default collapses to the run onset to reduce clutter; AC 13.2).
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

    fig, (ax_sig, ax_score, ax_tier, ax_text) = plt.subplots(
        4,
        1,
        figsize=(11.0, 10.0),
        sharex=False,
        gridspec_kw={"height_ratios": [3.0, 2.0, 1.4, 2.4], "hspace": 0.35},
    )

    # The first three panels share the time axis explicitly (sharex on the text
    # panel would stretch it; we keep the text panel independent).
    ax_score.sharex(ax_sig)
    ax_tier.sharex(ax_sig)

    # Phase shading spans the three time panels (AC 13.1).
    phase_legend = _shade_phases((ax_sig, ax_score, ax_tier), segments, t0)

    # Panel 1: signals.
    _draw_signals(ax_sig, t_rel, sub, show_acc)

    # Panel 2: Combined_Score.
    ax_score.plot(t_rel, sub_scores, color="#111111", linewidth=1.3)
    ax_score.set_title("Combined score")
    ax_score.set_ylabel("score")
    ax_score.grid(True, linestyle=":", alpha=0.4)

    # Panel 3: Severity_Tier bands (neutral labels, never color-coded; AC 15.2).
    ax_tier.step(t_rel, sub_tiers, where="post", color="#111111", linewidth=1.3)
    ax_tier.fill_between(t_rel, 0, sub_tiers, step="post", color="#111111", alpha=0.12)
    ax_tier.set_ylim(-0.3, 3.3)
    ax_tier.set_yticks(sorted(TIER_LABELS))
    ax_tier.set_yticklabels([tier_label(t) for t in sorted(TIER_LABELS)])
    ax_tier.set_title("Severity tier")
    ax_tier.set_ylabel("tier")
    ax_tier.set_xlabel("time since first window (s)")
    ax_tier.grid(True, linestyle=":", alpha=0.4)

    # Annotate flagged windows on the tier panel (AC 13.2).
    for i, fw in enumerate(flagged):
        x = float(fw.t_win) - t0
        ax_tier.axvline(x, color="#d62728", linewidth=0.9, alpha=0.7, zorder=4)
        ax_tier.annotate(
            f"#{i + 1}",
            (x, float(fw.tier)),
            textcoords="offset points",
            xytext=(2, 4),
            fontsize=7,
            color="#d62728",
        )

    # Panel 4: flag explanations (AC 13.2).
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

    # Phase legend across the top of the signals panel.
    if phase_legend:
        from matplotlib.patches import Patch

        seen: dict[str, str] = {}
        for phase, _s, _e, color in phase_legend:
            seen.setdefault(phase, color)
        handles = [Patch(facecolor=c, edgecolor="none", label=p) for p, c in seen.items()]
        ax_sig.legend(
            handles=ax_sig.get_legend().legend_handles + handles
            if ax_sig.get_legend()
            else handles,
            loc="upper right",
            fontsize=7,
            framealpha=0.8,
            ncol=2,
        )

    session = str(sub["session"].iloc[0]) if "session" in sub.columns and len(sub) else ""
    default_title = f"Timeline — {person_id}" + (f" ({session})" if session else "")
    fig.suptitle(title or default_title, fontsize=13)
    # Reserve headroom for the suptitle without ``tight_layout`` (which warns on
    # the signals panel's twin axis when ``show_acc`` is set).
    fig.subplots_adjust(top=0.94)

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
        output_path=output_path,
        title=title,
    )
