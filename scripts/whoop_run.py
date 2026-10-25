#!/usr/bin/env python3
"""Real-run entry point for the wearable-flag-pipeline on your own WHOOP data.

PhysioNet is the built-in *demo* (``scripts/demo.py``); this script is the
*real-run* path: it scores a single person's own WHOOP export, one row per night,
through the **same downstream pipeline** — "the same pipeline, nights in place of
minutes" from the project pitch. Nothing in ``flag/`` is re-implemented here; the
only new pieces are the WHOOP loader (``data/whoop_load.py``) and this
orchestration.

Flow
----
1. **Load** ``data/whoop/physiological_cycles.csv`` into a nightly
   ``Per_Window_Table`` (:func:`data.whoop_load.load_whoop_cycles` +
   :func:`data.whoop_load.to_per_window_table`). If the CSVs are absent, print
   how to export WHOOP data and where to drop the files, then exit cleanly (0) —
   mirroring the ``scripts/demo.py --synthetic`` guidance.
2. **Rolling baseline.** Unlike PhysioNet (which has a protocol rest phase), each
   night is scored against the person's *trailing* ``--baseline-window`` nights
   (default 14) of recent normal: for night ``i`` the per-signal median and
   scaled MAD come from nights ``[i-window, i-1]`` (the night being scored is
   excluded). Spread uses :func:`flag.baseline.scaled_mad`; the signed-z and
   combined score reuse :func:`flag.score.signed_z` /
   :func:`flag.score.combined_score` with the identical sign conventions (HR up
   positive, HRV down positive, skin temp signed).
3. **Tiers.** The per-night combined scores are fed to the production
   :class:`flag.tiers.Flagger`, so the 2-of-3 persistence gate, median-of-3
   tiering, single-signal cap and hysteresis all apply unchanged — only the unit
   of a "window" changes from 10 s to one night. ``--persist 2of3`` is therefore
   "2 of the last 3 nights". The reused production Flagger fixes its gate at
   2-of-3, so a different ``--persist`` (e.g. ``3of4``) is recorded and reported
   but does not change the reused gate — honest reuse over re-implementation.
4. **Benign cause.** The accelerometer movement check does not apply to nightly
   summaries, so instead a lightweight **high-exertion** check uses Day Strain:
   a night that follows a high-strain day has its flag dropped one tier and is
   labeled "explained by high exertion" (the nightly analogue of the movement
   check).
5. **Render + summarize.** A readable nightly timeline is rendered with
   :func:`app.demo.render_nightly_timeline` (x-axis = human dates, signals on
   small-multiple panels, the tuned thresholds drawn as labelled lines, and the
   muted severity tier overlaid as background shading on the score panel) to
   ``output/whoop_timeline.png`` — defaulting to
   the most recent ~90 nights for legibility (``--last N`` / ``--all``). A
   flagged-nights table is also written to ``output/whoop_flagged_nights.csv``,
   and a plain-text summary prints the nights, the date range, the calibrated
   thresholds, the flag rate (flags/month), the tier counts, and the
   explanations for the most recent flagged nights.

Thresholds are **calibrated to a target flag rate on the person's own nightly
score distribution** (``--target-flags-per-month``, default ~2), since a single
person's export has no cohort for leave-one-subject-out; ``--tau1/--tau2/--tau3``
override that for power users.

Determinism: the pipeline is deterministic given the input CSV. matplotlib's
headless ``Agg`` backend is selected by the imported modules, so no display is
needed. Privacy: ``data/whoop`` is gitignored — your export stays local and is
never read into committed files.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# Make the package root importable whether run as "python scripts/whoop_run.py"
# from the repo root or from anywhere else.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.schema import (  # noqa: E402
    Flag,
    SignalBaseline,
    TIER_LABELS,
    TIER_WATCH,
    Thresholds,
    tier_label,
)
from data.whoop_load import (  # noqa: E402
    WHOOP_PERSON_ID,
    WhoopNight,
    load_whoop_cycles,
    load_whoop_sleeps,
    to_per_window_table,
)
from flag.baseline import MAD_FLOOR, SCORED_SIGNALS, scaled_mad  # noqa: E402
from flag.explain import explain  # noqa: E402
from flag.score import (  # noqa: E402
    METHOD_DIRECTIONAL_SUM,
    METHOD_MAHALANOBIS,
    combined_score,
    fit_ledoit_wolf,
    window_signed_z,
)
from flag.tiers import Flagger, is_single_signal  # noqa: E402
from app.demo import render_nightly_timeline  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

CYCLES_FILENAME = "physiological_cycles.csv"
SLEEPS_FILENAME = "sleeps.csv"
TIMELINE_FILENAME = "whoop_timeline.png"
FLAGGED_CSV_FILENAME = "whoop_flagged_nights.csv"

DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, "data", "whoop")
DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "output")
DEFAULT_METHOD = METHOD_DIRECTIONAL_SUM
DEFAULT_BASELINE_WINDOW = 14
DEFAULT_PERSIST = "2of3"

_METHOD_CHOICES = (METHOD_DIRECTIONAL_SUM, METHOD_MAHALANOBIS)

# Average nights per month, used to turn a "flags per month" target into a flag
# *rate* (fraction of nights) over however many nights the export holds.
NIGHTS_PER_MONTH = 30.4375

# Default target flag rate, expressed as flagged nights per month. ~2 flags/month
# is roughly 1 flagged night in ~15, i.e. a few percent of nights — a realistic
# "occasionally something is genuinely off" rate, not the ~23% the old
# hand-picked fixed thresholds produced.
DEFAULT_TARGET_FLAGS_PER_MONTH = 2.0

# Where tau2 / tau3 sit, as upper quantiles of the person's own nightly score
# distribution, once tau1 is placed from the monthly target. Notable and Strong
# are progressively rarer than the tau1 flag rate, so Strong genuinely means a
# large, uncommon multi-signal shift rather than a routine wobble.
TAU2_QUANTILE = 0.95
TAU3_QUANTILE = 0.98

# Day Strain is WHOOP's 0-21 logarithmic exertion score. A night following a day
# at/above this strain is treated as "high exertion" for the benign-cause drop.
HIGH_STRAIN_THRESHOLD = 14.0


# ---------------------------------------------------------------------------
# Threshold calibration against the person's OWN nightly score distribution
# ---------------------------------------------------------------------------
#
# PhysioNet tunes tau1<tau2<tau3 with leave-one-subject-out over a cohort of
# participants and a false-alarm budget on known rest segments. A single person's
# WHOOP export has NO cohort of peers and NO labelled rest/stress ground truth —
# so LOSO is not available. Instead we calibrate against the ONE distribution we
# do have: the person's own per-night combined scores. We pick tau1 as the score
# quantile that leaves only the target number of nights above it (so the expected
# flag rate matches ~target flags/month given the export length), then place tau2
# and tau3 at progressively higher quantiles. This is an explicit,
# per-person-history calibration, documented as such — not a hidden rescale.


def calibrate_thresholds(
    scores: np.ndarray,
    *,
    target_flags_per_month: float = DEFAULT_TARGET_FLAGS_PER_MONTH,
    tau2_quantile: float = TAU2_QUANTILE,
    tau3_quantile: float = TAU3_QUANTILE,
) -> tuple[Thresholds, dict[str, float]]:
    """Calibrate tau1<tau2<tau3 to a target flag rate on the person's own scores.

    ``target_flags_per_month`` is converted to a target *fraction* of flagged
    nights using :data:`NIGHTS_PER_MONTH`; ``tau1`` is set to the ``(1 - frac)``
    quantile of the finite nightly ``scores`` so that, before the persistence
    gate, roughly ``frac`` of nights sit above ``tau1`` (the 2-of-3 gate then
    trims this further, so the realised flag rate is a touch lower — a
    conservative, honest outcome). ``tau2`` / ``tau3`` are the higher
    ``tau2_quantile`` / ``tau3_quantile`` of the same distribution so Notable and
    Strong are progressively rarer.

    The three are nudged apart if a flat score distribution collapses them, to
    honour the strict ``tau1 < tau2 < tau3`` invariant
    (:class:`data.schema.Thresholds`). Returns ``(Thresholds, info)`` where
    ``info`` records the quantiles actually used and the target fraction, for
    reporting.
    """
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        # Degenerate: no scores. Fall back to a tiny ordered triple.
        return Thresholds(tau1=1.0, tau2=2.0, tau3=3.0), {
            "target_frac": 0.0,
            "tau1_quantile": float("nan"),
            "tau2_quantile": tau2_quantile,
            "tau3_quantile": tau3_quantile,
        }

    frac = (float(target_flags_per_month) / NIGHTS_PER_MONTH)
    frac = float(min(max(frac, 0.0), 0.5))  # clamp to a sane [0, 50%] range
    tau1_quantile = 1.0 - frac
    # Keep tau1's quantile below tau2's so the ordering has room.
    tau1_quantile = min(tau1_quantile, tau2_quantile - 1e-6)

    tau1 = float(np.quantile(finite, tau1_quantile))
    tau2 = float(np.quantile(finite, tau2_quantile))
    tau3 = float(np.quantile(finite, tau3_quantile))

    tau1, tau2, tau3 = _enforce_order(tau1, tau2, tau3, finite)

    info = {
        "target_frac": frac,
        "tau1_quantile": tau1_quantile,
        "tau2_quantile": tau2_quantile,
        "tau3_quantile": tau3_quantile,
    }
    return Thresholds(tau1=tau1, tau2=tau2, tau3=tau3), info


def _enforce_order(
    tau1: float, tau2: float, tau3: float, scores: np.ndarray
) -> tuple[float, float, float]:
    """Nudge thresholds apart so tau1<tau2<tau3 holds even on a flat distribution.

    Quantiles of a nearly-constant score vector can coincide; the ``Thresholds``
    invariant forbids ties. We separate them by a small epsilon scaled to the
    score spread so the thresholds stay on the score's own scale.
    """
    spread = float(np.std(scores)) if scores.size else 1.0
    eps = max(spread * 1e-3, 1e-6)
    if not tau2 > tau1:
        tau2 = tau1 + eps
    if not tau3 > tau2:
        tau3 = tau2 + eps
    return tau1, tau2, tau3


# ---------------------------------------------------------------------------
# Persistence spec ("2of3" / "3of4")
# ---------------------------------------------------------------------------


def parse_persist(spec: str) -> tuple[int, int]:
    """Parse a ``"<k>of<n>"`` persistence spec into ``(k, n)``.

    ``"2of3"`` -> ``(2, 3)`` ("2 of the last 3 nights"). Raises ``ValueError`` on
    a malformed spec or an out-of-range ``k``/``n`` (need ``1 <= k <= n``).
    """
    text = spec.strip().lower().replace(" ", "")
    if "of" not in text:
        raise ValueError(f"persist spec {spec!r} must look like '2of3'")
    k_text, _, n_text = text.partition("of")
    try:
        k, n = int(k_text), int(n_text)
    except ValueError:
        raise ValueError(f"persist spec {spec!r} must look like '2of3'") from None
    if not (1 <= k <= n):
        raise ValueError(f"persist spec {spec!r} must satisfy 1 <= k <= n")
    return k, n


# ---------------------------------------------------------------------------
# Rolling per-person baseline (the nightly analogue of the rest baseline)
# ---------------------------------------------------------------------------


def rolling_baseline(
    table: pd.DataFrame,
    index: int,
    window: int,
) -> dict[str, SignalBaseline]:
    """Per-signal baseline for night ``index`` from its trailing ``window`` nights.

    Uses nights ``[index-window, index-1]`` (the scored night itself is excluded)
    to estimate, per scored signal, a robust center (median) and spread (scaled
    MAD, :func:`flag.baseline.scaled_mad`). A signal with no finite value in the
    trailing window is omitted (the Scorer then simply drops it for that night).
    ``shrunk_mad`` is set equal to ``mad`` — a single person's own history has no
    cohort to shrink toward, so no empirical-Bayes shrinkage applies; the spread
    is floored at :data:`flag.baseline.MAD_FLOOR` to keep z-scores finite.

    Returns ``{}`` for the earliest nights that have no trailing history yet (the
    pipeline then scores them at ~0 and they cannot flag — the intended warm-up
    behavior, analogous to the first windows of a PhysioNet recording).
    """
    lo = max(0, index - window)
    if lo >= index:
        return {}
    trailing = table.iloc[lo:index]

    baselines: dict[str, SignalBaseline] = {}
    for signal in SCORED_SIGNALS:
        if signal not in trailing.columns:
            continue
        values = trailing[signal].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        center = float(np.median(finite))
        spread = scaled_mad(finite)
        if not np.isfinite(spread) or spread <= 0.0:
            spread = MAD_FLOOR
        baselines[signal] = SignalBaseline(median=center, mad=spread, shrunk_mad=spread)
    return baselines


def nightly_scores(
    table: pd.DataFrame,
    window: int,
    method: str,
) -> tuple[np.ndarray, list[dict[str, SignalBaseline]], list[dict[str, float]]]:
    """Combined deviation score per night against its rolling baseline.

    For each night (row) in ``table`` (assumed t_win-sorted), build the trailing
    rolling baseline (:func:`rolling_baseline`), compute the per-signal signed z
    (:func:`flag.score.window_signed_z`) and the combined score
    (:func:`flag.score.combined_score`) with ``method``. For the Mahalanobis
    method a single Ledoit-Wolf covariance is fit over the nights that have all
    three signals available and reused for every night (a subset-signal night is
    marginalized by the scorer).

    Returns ``(scores, baselines_per_night, signed_z_per_night)``, each aligned to
    the table rows, so the caller can reuse the baselines/contributions for the
    tier assignment and the explanations.
    """
    n = len(table)
    scores = np.zeros(n, dtype=float)
    baselines_per_night: list[dict[str, SignalBaseline]] = []
    z_per_night: list[dict[str, float]] = []

    for i in range(n):
        baselines = rolling_baseline(table, i, window)
        baselines_per_night.append(baselines)
        z_per_night.append(window_signed_z(table.iloc[i], baselines))

    cov = None
    if method == METHOD_MAHALANOBIS:
        rows = [
            [float(z[s]) for s in SCORED_SIGNALS]
            for z in z_per_night
            if all(s in z and np.isfinite(z[s]) for s in SCORED_SIGNALS)
        ]
        if len(rows) >= 2:
            cov = fit_ledoit_wolf(np.asarray(rows, dtype=float))

    for i in range(n):
        scores[i] = combined_score(
            table.iloc[i], baselines_per_night[i], method=method, cov=cov
        )
    return scores, baselines_per_night, z_per_night


# ---------------------------------------------------------------------------
# Tiers via the production Flagger (gate is unit-agnostic: nights, not windows)
# ---------------------------------------------------------------------------


def nightly_tiers(
    scores: np.ndarray,
    z_per_night: Sequence[Mapping[str, float]],
    taus: Thresholds,
) -> np.ndarray:
    """Assign a severity tier per night with the production :class:`flag.tiers.Flagger`.

    The Flagger's 2-of-3 persistence gate, median-of-3 tiering, single-signal cap
    and hysteresis are all unit-agnostic — feeding one night at a time makes
    "the last 3 windows" mean "the last 3 nights". ``z_per_night`` supplies the
    per-night single-signal-cap flag (:func:`flag.tiers.is_single_signal`).
    """
    flagger = Flagger(taus)
    tiers = np.zeros(len(scores), dtype=int)
    for i, score in enumerate(scores):
        single = is_single_signal(z_per_night[i])
        tiers[i] = flagger.step(float(score), single_signal=single)
    return tiers


# ---------------------------------------------------------------------------
# Benign-cause: high-exertion (Day Strain) analogue of the movement check
# ---------------------------------------------------------------------------


def strain_benign_cause(
    nights: Sequence[WhoopNight],
    tiers: np.ndarray,
    *,
    threshold: float = HIGH_STRAIN_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop one tier on a flagged night that follows a high-strain day.

    The nightly analogue of the accelerometer movement check (``flag/context.py``):
    where ACC is unavailable in nightly summaries, Day Strain stands in as the
    exertion proxy. A night whose own Day Strain is at/above ``threshold`` is
    treated as "high exertion"; if that night is flagged (tier >= Watch) the tier
    is dropped by one (floored at 0) and the night is marked
    ``exertion_explained``. ``nights`` must be t_win-sorted to align with ``tiers``.

    Returns ``(adjusted_tiers, exertion_explained)`` aligned to ``tiers``.
    """
    timed = sorted((n for n in nights if np.isfinite(n.t_win)), key=lambda n: n.t_win)
    adjusted = np.asarray(tiers, dtype=int).copy()
    explained = np.zeros(len(adjusted), dtype=bool)

    for i in range(min(len(timed), len(adjusted))):
        strain = timed[i].day_strain
        high = np.isfinite(strain) and float(strain) >= threshold
        if high and adjusted[i] >= TIER_WATCH:
            adjusted[i] = max(0, int(adjusted[i]) - 1)
            explained[i] = True
    return adjusted, explained


# ---------------------------------------------------------------------------
# Absent-data guidance
# ---------------------------------------------------------------------------


def _print_absent_guidance(data_dir: str) -> None:
    """Explain how to export WHOOP data and where to drop the files."""
    print(
        "No WHOOP export found.\n\n"
        f"Expected these files in: {data_dir}\n"
        f"  - {CYCLES_FILENAME}\n"
        f"  - {SLEEPS_FILENAME}  (optional; naps are filtered out)\n\n"
        "To get them, export your data from WHOOP:\n"
        "  WHOOP app  ->  Account  ->  App Settings  ->  Data Export\n"
        "  (you receive a ZIP containing physiological_cycles.csv and sleeps.csv)\n\n"
        f"Unzip and drop those CSVs into {data_dir} (this folder is gitignored, so\n"
        "your data stays local and is never committed), then re-run:\n"
        "  python scripts/whoop_run.py\n\n"
        "To try the built-in PhysioNet demo instead (no personal data needed):\n"
        "  python scripts/demo.py --synthetic"
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Run the WHOOP nightly flow; return a process exit code."""
    data_dir = args.data_dir
    cycles_path = os.path.join(data_dir, CYCLES_FILENAME)
    sleeps_path = os.path.join(data_dir, SLEEPS_FILENAME)

    if not os.path.isfile(cycles_path):
        _print_absent_guidance(data_dir)
        return 0

    k, n_persist = parse_persist(args.persist)

    print("=" * 72)
    print("WEARABLE FLAG PIPELINE — WHOOP NIGHTLY RUN")
    print("=" * 72)
    print(f"Reading cycles from: {cycles_path}")

    nights = load_whoop_cycles(cycles_path)
    if os.path.isfile(sleeps_path):
        # Read + nap-filter sleeps to honor the contract; the nightly score uses
        # the cycles file. We only report that naps were excluded.
        sleeps = load_whoop_sleeps(sleeps_path)
        print(f"Read {len(sleeps)} non-nap sleep row(s) from {SLEEPS_FILENAME} (naps ignored).")

    table = to_per_window_table(nights, person_id=args.person_id)
    n_nights = len(table)
    if n_nights == 0:
        print("No usable nights found in the export (all rows blank or untimed).")
        return 0

    date_lo = _date_label(float(table["t_win"].min()))
    date_hi = _date_label(float(table["t_win"].max()))
    print(f"Nights loaded: {n_nights}   date range: {date_lo} -> {date_hi}")
    print(
        f"Rolling baseline: trailing {args.baseline_window} nights; method: {args.method}"
    )
    from flag.tiers import GATE_MIN_ABOVE, GATE_WINDOW

    if (k, n_persist) != (GATE_MIN_ABOVE, GATE_WINDOW):
        print(
            f"note: the production Flagger's persistence gate is fixed at "
            f"{GATE_MIN_ABOVE} of the last {GATE_WINDOW} nights; the requested "
            f"'{args.persist}' is recorded but the reused gate stays "
            f"{GATE_MIN_ABOVE}of{GATE_WINDOW}."
        )
    else:
        print(f"Persistence gate: {k} of the last {n_persist} nights.")

    # Rolling-baseline scores (needed before tiering so we can calibrate the
    # thresholds against this person's OWN nightly score distribution).
    scores, baselines_per_night, z_per_night = nightly_scores(
        table, args.baseline_window, args.method
    )

    # --- Calibrate (or override) tau1<tau2<tau3. ---
    taus, calib = _resolve_thresholds(args, scores)
    if calib is None:
        print(
            f"Thresholds (manual override): tau1={taus.tau1:.3f}  "
            f"tau2={taus.tau2:.3f}  tau3={taus.tau3:.3f}"
        )
    else:
        print(
            f"Thresholds calibrated to ~{args.target_flags_per_month:g} flags/month "
            f"on your own {n_nights}-night score distribution "
            f"(no cohort -> per-person-history calibration):"
        )
        print(
            f"  tau1={taus.tau1:.3f} (q={calib['tau1_quantile']:.3f})  "
            f"tau2={taus.tau2:.3f} (q={calib['tau2_quantile']:.3f})  "
            f"tau3={taus.tau3:.3f} (q={calib['tau3_quantile']:.3f})"
        )

    # Tiers -> strain benign-cause.
    tiers = nightly_tiers(scores, z_per_night, taus)
    tiers, exertion = strain_benign_cause(nights, tiers)

    # --- Flag-rate / tier summary. ---
    n_flagged = int(np.sum(tiers >= TIER_WATCH))
    peak = int(np.max(tiers, initial=0))
    flags_per_month = (n_flagged / n_nights) * NIGHTS_PER_MONTH if n_nights else 0.0
    counts = {t: int(np.sum(tiers == t)) for t in sorted(TIER_LABELS)}
    print(
        f"Flagged nights (tier>=Watch): {n_flagged}/{n_nights} "
        f"({100.0 * n_flagged / n_nights:.1f}%, ~{flags_per_month:.1f} flags/month)   "
        f"peak tier: {tier_label(peak)}"
    )
    print(
        "Tier counts: "
        + "  ".join(f"{tier_label(t)}={counts[t]}" for t in sorted(TIER_LABELS))
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Flagged-nights CSV (read results as a table, not just a figure). ---
    csv_path = os.path.join(args.output_dir, FLAGGED_CSV_FILENAME)
    n_csv = _write_flagged_csv(
        csv_path, table, scores, tiers, exertion, baselines_per_night, z_per_night
    )
    print(f"Wrote flagged-nights CSV: {csv_path} ({n_csv} flagged night(s))")

    # --- Readable nightly timeline (human date axis, recent window by default). ---
    timeline_path = os.path.join(args.output_dir, TIMELINE_FILENAME)
    last_n = None if args.all else args.last
    # Baselines passed to the renderer are the MOST RECENT night's rolling
    # baseline — the best "your recent nights" reference for the shift wording.
    recent_baselines = _last_nonempty(baselines_per_night)
    try:
        render_nightly_timeline(
            table,
            args.person_id,
            scores,
            tiers,
            recent_baselines,
            taus,
            movement=exertion,
            last_n=last_n,
            output_path=timeline_path,
            title=f"WHOOP nightly timeline — {args.person_id} ({args.method})",
        )
        span = "all nights" if last_n is None else f"most recent {min(last_n, n_nights)} nights"
        print(f"Wrote timeline: {timeline_path} ({span})")
    except Exception as exc:  # pragma: no cover - rendering best effort
        print(f"note: could not render timeline ({exc})")

    _print_flagged_explanations(
        table, scores, tiers, exertion, baselines_per_night, z_per_night
    )
    return 0


def _resolve_thresholds(
    args: argparse.Namespace, scores: np.ndarray
) -> tuple[Thresholds, dict[str, float] | None]:
    """Return ``(taus, calib_info)``; ``calib_info`` is ``None`` for a manual override.

    A manual override requires all three of ``--tau1/--tau2/--tau3``; otherwise
    the thresholds are calibrated to ``--target-flags-per-month`` against the
    person's own nightly ``scores`` (:func:`calibrate_thresholds`).
    """
    manual = (args.tau1, args.tau2, args.tau3)
    if all(v is not None for v in manual):
        return Thresholds(tau1=float(args.tau1), tau2=float(args.tau2), tau3=float(args.tau3)), None
    if any(v is not None for v in manual):
        # Partial override is ambiguous; calibrate but warn.
        print(
            "note: --tau1/--tau2/--tau3 must be given together to override; "
            "falling back to calibration since only some were provided."
        )
    return calibrate_thresholds(
        scores, target_flags_per_month=args.target_flags_per_month
    )


def _write_flagged_csv(
    path: str,
    table: pd.DataFrame,
    scores: np.ndarray,
    tiers: np.ndarray,
    exertion: np.ndarray,
    baselines_per_night: Sequence[Mapping[str, SignalBaseline]],
    z_per_night: Sequence[Mapping[str, float]],
) -> int:
    """Write one row per flagged night to ``path`` and return the row count.

    Columns: ``date, tier, combined_score, top_signals, explained_by_exertion``.
    ``top_signals`` is a human string of each available signal's real-unit shift
    from the night's rolling baseline (e.g. ``"HR +6.0 bpm; HRV -14.0 ms"``),
    ranked by contribution. Private by construction: ``output/whoop_*.csv`` is
    gitignored, so the user's own values never leave ``output/``.
    """
    import csv as _csv

    disp = {"hr": "HR", "rmssd": "HRV", "skin_temp": "temp"}
    unit = {"hr": "bpm", "rmssd": "ms", "skin_temp": "C"}

    rows: list[dict[str, object]] = []
    for i in range(len(table)):
        if int(tiers[i]) < TIER_WATCH:
            continue
        row = table.iloc[i]
        baselines = baselines_per_night[i]
        contributions = z_per_night[i]
        ranked = sorted(
            contributions.items(), key=lambda kv: abs(float(kv[1])), reverse=True
        )
        parts: list[str] = []
        for signal, _z in ranked:
            b = baselines.get(signal)
            try:
                raw = float(row[signal])
            except (KeyError, TypeError, ValueError):
                continue
            if b is None or not np.isfinite(raw) or not np.isfinite(float(b.median)):
                continue
            delta = raw - float(b.median)
            parts.append(f"{disp.get(signal, signal)} {delta:+.1f} {unit.get(signal, '')}".strip())
        rows.append(
            {
                "date": _date_label(float(row["t_win"])),
                "tier": tier_label(int(tiers[i])),
                "combined_score": f"{float(scores[i]):.3f}",
                "top_signals": "; ".join(parts) if parts else "n/a",
                "explained_by_exertion": "yes" if bool(exertion[i]) else "no",
            }
        )

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(
            fh,
            fieldnames=[
                "date",
                "tier",
                "combined_score",
                "top_signals",
                "explained_by_exertion",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _last_nonempty(
    baselines_per_night: Sequence[Mapping[str, SignalBaseline]],
) -> dict[str, SignalBaseline]:
    """Return the most recent non-empty rolling baseline (else ``{}``)."""
    for baselines in reversed(baselines_per_night):
        if baselines:
            return dict(baselines)
    return {}


def _print_flagged_explanations(
    table: pd.DataFrame,
    scores: np.ndarray,
    tiers: np.ndarray,
    exertion: np.ndarray,
    baselines_per_night: Sequence[Mapping[str, SignalBaseline]],
    z_per_night: Sequence[Mapping[str, float]],
    *,
    max_show: int = 5,
) -> None:
    """Print neutral, real-unit explanations for the most recent flagged nights.

    Each flagged night (tier >= Watch) is explained with the Explainer
    (:func:`flag.explain.explain`) using that night's signed-z contributions, its
    rolling baseline, and its raw values, so each line reads in real units
    relative to the person's recent nights, e.g. "heart rate +6 bpm vs your rest",
    "HRV -14 ms vs your rest". Shown most-recent-first.
    """
    flagged_idx = [i for i in range(len(table)) if int(tiers[i]) >= TIER_WATCH]
    if not flagged_idx:
        print("\nNo flagged nights to explain.")
        return

    print(f"\nMost recent flagged nights (showing up to {max_show}):")
    for i in reversed(flagged_idx[-max_show:]):
        row = table.iloc[i]
        baselines = baselines_per_night[i]
        contributions = z_per_night[i]
        values: dict[str, float] = {}
        top_signals: list[tuple[str, float]] = []
        for signal in SCORED_SIGNALS:
            raw = float(row[signal]) if np.isfinite(float(row[signal])) else float("nan")
            if not np.isfinite(raw):
                continue
            values[signal] = raw
            b = baselines.get(signal)
            if b is not None and np.isfinite(float(b.median)):
                top_signals.append((signal, raw - float(b.median)))

        flag = Flag(
            person_id=str(row["person_id"]),
            start_t=float(row["t_win"]),
            persisted_s=0.0,
            combined_score=float(scores[i]),
            tier=int(tiers[i]),
            top_signals=top_signals,
            movement_explained=bool(exertion[i]),
        )
        ex = explain(flag, contributions, baselines, values=values)
        # Date label for readability (t_win is epoch seconds).
        label = _date_label(float(row["t_win"]))
        tier_txt = tier_label(int(tiers[i]))
        exert_txt = " [explained by high exertion]" if bool(exertion[i]) else ""
        named = ", ".join(s.text for s in ex.shifts) or "no individual signal shift"
        print(f"  {label}  [{tier_txt}]{exert_txt}  {named}")


def _date_label(t_win: float) -> str:
    """Render an epoch-second ``t_win`` as a ``YYYY-MM-DD`` UTC date for display."""
    from datetime import datetime, timezone

    try:
        return datetime.fromtimestamp(float(t_win), tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OverflowError, OSError):
        return f"t={t_win:g}s"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the WHOOP run CLI."""
    parser = argparse.ArgumentParser(
        prog="scripts/whoop_run.py",
        description=(
            "Score your own WHOOP nightly export through the wearable-flag "
            "pipeline (the same combined flag as the PhysioNet demo, nights in "
            "place of minutes) and write output/whoop_timeline.png."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory holding physiological_cycles.csv and sleeps.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write whoop_timeline.png into.",
    )
    parser.add_argument(
        "--person-id",
        default=WHOOP_PERSON_ID,
        help="Grouping id for the single person in the export.",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=_METHOD_CHOICES,
        help="Combined-score method.",
    )
    parser.add_argument(
        "--baseline-window",
        type=int,
        default=DEFAULT_BASELINE_WINDOW,
        help="Trailing nights used for each night's rolling baseline.",
    )
    parser.add_argument(
        "--persist",
        default=DEFAULT_PERSIST,
        help="Persistence gate as '<k>of<n>' nights (e.g. 2of3, 3of4).",
    )
    parser.add_argument(
        "--target-flags-per-month",
        type=float,
        default=DEFAULT_TARGET_FLAGS_PER_MONTH,
        help=(
            "Target flagged nights per month used to calibrate tau1 against your "
            "own nightly score distribution (tau2/tau3 sit at higher quantiles). "
            "Lower = rarer flags."
        ),
    )
    parser.add_argument(
        "--tau1",
        type=float,
        default=None,
        help="Manual Watch threshold (override; requires --tau2 and --tau3 too).",
    )
    parser.add_argument(
        "--tau2",
        type=float,
        default=None,
        help="Manual Notable threshold (override; requires --tau1 and --tau3 too).",
    )
    parser.add_argument(
        "--tau3",
        type=float,
        default=None,
        help="Manual Strong threshold (override; requires --tau1 and --tau2 too).",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=90,
        help="Show the most recent N nights in the timeline for legibility.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render the full night history instead of the most recent --last nights.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        parse_persist(args.persist)
    except ValueError as exc:
        parser.error(str(exc))
    if args.baseline_window < 1:
        parser.error("--baseline-window must be >= 1")
    if args.target_flags_per_month <= 0:
        parser.error("--target-flags-per-month must be > 0")
    if args.last < 1:
        parser.error("--last must be >= 1")
    manual = (args.tau1, args.tau2, args.tau3)
    if all(v is not None for v in manual):
        try:
            Thresholds(tau1=float(args.tau1), tau2=float(args.tau2), tau3=float(args.tau3))
        except (ValueError, TypeError) as exc:
            parser.error(str(exc))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
