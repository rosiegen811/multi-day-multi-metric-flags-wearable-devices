#!/usr/bin/env python3
"""One-command DEMO entry point for the wearable-flag-pipeline.

This is a *presentation* deliverable that extends the Demo_App (``app/demo.py``,
task 17): it runs the **full, real pipeline** end to end on the PhysioNet
Empatica E4 recordings and writes a compelling, reproducible set of artifacts to
``output/`` — the two headline figures, the per-method/per-cohort metric CSV, and
two per-participant timeline PNGs (a strong stress responder "hero" and a calm
contrast). It also prints a concise plain-text summary with example flag
explanations.

It is glue only. Every scientific step is delegated to the existing modules —
nothing here re-implements loading, cleaning, windowing, baselining, scoring,
flagging, movement attribution, explanation, evaluation, or calibration:

    data.physionet_load / data.clean      load + clean the real E4 CSVs
    eval.run_eval                         build the combined Per_Window_Table,
                                          run LOSO evaluation, write figures,
                                          per-method per-window scores/tiers,
                                          the synthetic cohort generator
    eval.metrics                          tiers_from_scores + method constants
    eval.labels                           phase labels + self-report join
    eval.calibrate                        LOSO / all-subject threshold calibration
    app.demo                              per-participant timeline rendering,
                                          flag-explanation collection

Run it::

    python scripts/demo.py                 # real data (default participant set)
    python scripts/demo.py --synthetic     # no real data: synthetic cohort
    python scripts/demo.py --hero S05 --calm S04 --method mahalanobis

Artifacts written to ``--output-dir`` (default ``output/``):

    false_alarms_vs_delay.png       headline figure 1 (false alarms vs delay)
    tier_vs_self_report.png         headline figure 2 (tier vs self-report)
    metrics_by_method_cohort.csv    the numbers behind the figures
    timeline_<hero>.png             hero participant timeline (flags fire)
    timeline_<calm>.png             calm-contrast participant timeline

Determinism: the pipeline is deterministic given the inputs, and the synthetic
generator is seeded, so repeated runs reproduce the same artifacts. matplotlib's
headless ``Agg`` backend is selected by the imported modules, so no display is
needed.

A demo-only calibration note (step 3)
-------------------------------------
Thresholds are normally tuned by leave-one-subject-out (LOSO): a subject is
flagged only with thresholds calibrated on the *other* subjects
(``eval.run_eval.run_loso``). On a small hand-picked demo cohort LOSO can be too
strict for a single held-out subject — the folds see few peers, so the chosen
``tau1`` sits above even the hero's stress-task scores and no tier>=1 window
fires. When that happens this demo *falls back* to calibrating thresholds on
**all** subjects (``eval.calibrate.calibrate_on_subset``) purely so the hero's
timeline shows a non-trivial flag for the presentation. This fallback is a
demo-only visualization choice and is clearly labelled in the printed summary;
the headline evaluation figures always use the honest LOSO numbers.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Mapping, Sequence

import numpy as np

# The pipeline package root is the parent of this scripts/ directory. Make it
# importable whether the demo is run as "python scripts/demo.py" from the repo
# root or from anywhere else.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.clean import (  # noqa: E402
    SESSION_AEROBIC,
    SESSION_ANAEROBIC,
    SESSION_STRESS,
    Constraints,
    clean,
    load_constraints,
)
from data.physionet_load import (  # noqa: E402
    ParticipantSignals,
    cohort_of,
    load_participant,
)
from eval.calibrate import calibrate_on_subset  # noqa: E402
from eval.labels import load_stress_levels, phase_labels  # noqa: E402
from eval.metrics import METHOD_WHOOP, tiers_from_scores  # noqa: E402
from flag.score import METHOD_DIRECTIONAL_SUM, METHOD_MAHALANOBIS  # noqa: E402
from eval.run_eval import (  # noqa: E402
    DEFAULT_BUDGET,
    _metrics_dataframe,
    build_combined_table,
    method_scores,
    movement_flags,
    run_eval,
    run_loso,
    simulate_participant,
    write_headline_figures,
)
from app.demo import (  # noqa: E402
    collect_flag_explanations,
    render_participant_timeline,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Default demo cohort. f13 and S05 are strong stress responders whose flags fire
# under honest LOSO (no calibration fallback needed); S11/f11 add V1/V2 coverage;
# S04 is a calm contrast whose timeline stays near Tier 0.
#
# S07 is deliberately NOT in the default set even though the pitch lists it as a
# strong responder: its STRESS recording carries a corrupt skin-temperature spike
# (to ~221 C) that both yields a non-credible explanation ("+177 C vs your rest")
# and inflates the shared threshold calibration for every other participant. It
# can still be requested explicitly via --participants / --hero for inspection.
DEFAULT_PARTICIPANTS: tuple[str, ...] = ("f13", "S05", "f11", "S11", "S04")
DEFAULT_HERO = "f13"
DEFAULT_CALM = "S04"
DEFAULT_METHOD = METHOD_DIRECTIONAL_SUM

# The sessions loaded per participant: STRESS is the timeline focus; AEROBIC /
# ANAEROBIC are loaded too when present (they feed the movement-attribution and
# aerobic-intensity parts of the evaluation).
DEMO_SESSIONS: tuple[str, ...] = (SESSION_STRESS, SESSION_AEROBIC, SESSION_ANAEROBIC)

_METHOD_CHOICES = (METHOD_WHOOP, METHOD_DIRECTIONAL_SUM, METHOD_MAHALANOBIS)


def _sessions_root(data_root: str) -> str:
    """Return the Wearable_Dataset session root under ``data_root``."""
    return os.path.join(
        data_root, "sessions", "Wearable_Dataset_sessions", "Wearable_Dataset"
    )


# ---------------------------------------------------------------------------
# Real-data loading (robust, per-participant try/except)
# ---------------------------------------------------------------------------


def _load_session(
    session_dir: str, pid: str, constraints: Constraints
) -> ParticipantSignals | None:
    """Load + clean one participant/session folder, joining any ``_a``/``_b`` split.

    Returns the cleaned :class:`ParticipantSignals`, or ``None`` when the folder
    is absent, the load fails, or the Cleaner drops the session (e.g. f14..f18
    non-stress). Never raises — a bad folder is skipped, not fatal.
    """
    whole = os.path.join(session_dir, pid)
    a_dir = os.path.join(session_dir, f"{pid}_a")
    b_dir = os.path.join(session_dir, f"{pid}_b")

    try:
        if os.path.isdir(whole):
            loaded = load_participant(session_dir, pid)
            segment_b = None
        elif os.path.isdir(a_dir):
            loaded = load_participant(session_dir, f"{pid}_a")
            loaded.person_id = pid  # normalize off the "_a" suffix
            segment_b = (
                load_participant(session_dir, f"{pid}_b")
                if os.path.isdir(b_dir)
                else None
            )
        else:
            return None
    except (ValueError, OSError) as exc:
        print(f"  skip {pid}/{os.path.basename(session_dir)}: load failed ({exc})")
        return None

    try:
        cleaned = clean(loaded, constraints, segment_b=segment_b)
    except Exception as exc:  # defensive: a bad constraint row must not crash
        print(f"  skip {pid}/{os.path.basename(session_dir)}: clean failed ({exc})")
        return None
    return cleaned


def load_real_participants(
    data_root: str,
    participant_ids: Sequence[str],
    constraints: Constraints,
) -> dict[str, dict[str, ParticipantSignals]]:
    """Load + clean the requested participants from the real session tree.

    Walks only the requested ``participant_ids`` (not the whole tree) across the
    demo sessions, skipping any missing participant/session folder. Returns
    ``{person_id -> {session -> cleaned ParticipantSignals}}``. A participant with
    no usable session at all is reported and omitted.
    """
    root = _sessions_root(data_root)
    out: dict[str, dict[str, ParticipantSignals]] = {}

    for pid in participant_ids:
        try:
            cohort_of(pid)
        except ValueError:
            print(f"  skip {pid}: unrecognized participant id (want S.. or f..)")
            continue

        sessions: dict[str, ParticipantSignals] = {}
        for session in DEMO_SESSIONS:
            session_dir = os.path.join(root, session)
            if not os.path.isdir(session_dir):
                continue
            cleaned = _load_session(session_dir, pid, constraints)
            if cleaned is not None:
                sessions[cleaned.session.upper()] = cleaned

        if sessions:
            out[pid] = sessions
            print(f"  loaded {pid}: {', '.join(sorted(sessions))}")
        else:
            print(f"  skip {pid}: no usable session folders found")

    return out


def build_synthetic_participants(
    participant_ids: Sequence[str],
) -> dict[str, dict[str, ParticipantSignals]]:
    """Build a deterministic synthetic cohort when real data is absent.

    Uses :func:`eval.run_eval.simulate_participant` to make a protocol-shaped
    recording per participant/session. The calm contrast (``DEFAULT_CALM``) is
    generated unstressed so it stays near Tier 0; everyone else is a stressed
    responder so flags fire. Deterministic for a given participant set.
    """
    out: dict[str, dict[str, ParticipantSignals]] = {}
    for pid in participant_ids:
        try:
            cohort_of(pid)
        except ValueError:
            print(f"  skip {pid}: unrecognized participant id (want S.. or f..)")
            continue
        stressed = pid != DEFAULT_CALM
        sessions: dict[str, ParticipantSignals] = {}
        for session in DEMO_SESSIONS:
            sig = simulate_participant(pid, session, seed=7, stressed=stressed)
            sessions[session] = sig
        out[pid] = sessions
        print(f"  synthesized {pid}: {', '.join(sorted(sessions))} (stressed={stressed})")
    return out


# ---------------------------------------------------------------------------
# Hero tiers: LOSO with a documented all-subject calibration fallback
# ---------------------------------------------------------------------------


def hero_tiers(
    built,
    labels,
    self_report: Mapping[tuple[str, str, int], float],
    method: str,
    hero: str,
    *,
    budget: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Per-window scores + tiers + movement for the hero, with a demo-only fallback.

    Returns ``(scores, tiers, movement, used_fallback)`` where every array is
    aligned to ``built.table`` rows:

    1. Compute the method's per-window ``scores`` (and single-signal cap flags).
    2. Run LOSO (``run_loso``) and keep the held-out tiers for the hero's rows.
    3. If the hero has **no** tier>=1 window under LOSO (LOSO was too strict on
       this small demo cohort), fall back to calibrating thresholds on **all**
       subjects (``calibrate_on_subset``) and re-tier so the hero shows a flag.
       This fallback is a demo-only visualization choice (see the module
       docstring) — ``used_fallback`` records that it happened so the summary can
       say so.
    """
    table = built.table
    scores, single_signal = method_scores(table, built.baselines, method)

    loso = run_loso(built, labels, self_report, method, budget=budget)
    tiers = loso.tiers.copy()
    movement = loso.movement.copy()

    person_ids = table["person_id"].astype(str).to_numpy()
    hero_mask = person_ids == str(hero)

    used_fallback = False
    if hero_mask.any() and int(np.max(tiers[hero_mask], initial=0)) < 1:
        # LOSO produced no flag for the hero on this small cohort. Calibrate on
        # all subjects so the hero gets a non-trivial tier for the timeline.
        used_fallback = True
        try:
            taus = calibrate_on_subset(
                scores,
                labels,
                self_report,
                budget,
                subjects=sorted(set(person_ids)),
                single_signal=single_signal,
            )
            stream_ids = [(w.person_id, w.session) for w in labels.windows]
            all_tiers = tiers_from_scores(
                scores, taus, stream_ids=stream_ids, single_signal=single_signal
            )
            idx = np.nonzero(hero_mask)[0]
            adj_tiers, adj_moved = movement_flags(
                table.iloc[idx],
                all_tiers[idx],
                [
                    built.acc_windows[i] if i < len(built.acc_windows) else None
                    for i in idx
                ],
            )
            tiers[idx] = adj_tiers
            movement[idx] = adj_moved
        except Exception as exc:  # pragma: no cover - fallback best effort
            print(f"  note: all-subject calibration fallback failed ({exc})")

    return scores, tiers, movement, used_fallback


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_summary(
    participants: Mapping[str, Mapping[str, ParticipantSignals]],
    result,
    hero: str,
    method: str,
    hero_scores: np.ndarray,
    hero_tier_vec: np.ndarray,
    hero_movement: np.ndarray,
    used_fallback: bool,
) -> None:
    """Print the concise plain-text demo summary (participants, metrics, examples)."""
    built = result.built
    table = built.table

    _print_header("WEARABLE FLAG PIPELINE — DEMO SUMMARY")

    pid_list = ", ".join(sorted(participants))
    print(f"Participants ({len(participants)}): {pid_list}")
    print(f"Combined Per_Window_Table: {len(table)} windows across all sessions")
    by_session = table["session"].value_counts().to_dict() if len(table) else {}
    if by_session:
        parts = ", ".join(f"{s}={n}" for s, n in sorted(by_session.items()))
        print(f"  windows by session: {parts}")

    _print_header("HEADLINE METRICS (LOSO, per method x cohort)")
    df = _metrics_dataframe(result.comparison)
    if len(df):
        cols = [
            "method",
            "cohort",
            "sensitivity_overall",
            "false_alarms_per_rest_hour",
            "detection_delay_s",
            "tier_agreement",
            "n_windows",
        ]
        cols = [c for c in cols if c in df.columns]
        with_pd = df[cols].copy()
        for c in with_pd.columns:
            if with_pd[c].dtype.kind == "f":
                with_pd[c] = with_pd[c].round(3)
        print(with_pd.to_string(index=False))
    else:
        print("(no metric rows — no labeled phases in this cohort)")

    _print_header(f"HERO TIMELINE — {hero} (method: {method})")
    person_ids = table["person_id"].astype(str).to_numpy()
    hero_mask = person_ids == str(hero)
    n_flagged = int(np.sum(hero_tier_vec[hero_mask] >= 1)) if hero_mask.any() else 0
    max_tier = int(np.max(hero_tier_vec[hero_mask], initial=0)) if hero_mask.any() else 0
    print(f"Hero windows: {int(hero_mask.sum())}")
    print(f"Flagged windows (tier>=1): {n_flagged}   peak tier: {max_tier}")
    if used_fallback:
        print(
            "NOTE: LOSO was too strict on this small demo cohort, so the hero's "
            "tiers use an all-subject calibration fallback (demo-only "
            "visualization choice; headline metrics above remain honest LOSO)."
        )

    # Example flag explanations for the hero (neutral, real-unit lines).
    if hero_mask.any():
        hero_table = table.loc[hero_mask].reset_index(drop=True)
        hero_baselines = dict(built.baselines.get(str(hero), {}))
        flagged = collect_flag_explanations(
            hero_table,
            hero_scores[hero_mask],
            hero_tier_vec[hero_mask],
            hero_baselines,
            movement=hero_movement[hero_mask],
            first_per_run=True,
        )
        if flagged:
            print()
            print("Example flag explanations (one per flagged run):")
            for i, fw in enumerate(flagged[:4]):
                print(f"  [{i + 1}] {fw.explanation.text}")
        else:
            print("No flagged runs to explain for the hero.")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_demo(args: argparse.Namespace) -> int:
    """Run the full demo flow; return a process exit code."""
    participant_ids = [p.strip() for p in args.participants.split(",") if p.strip()]
    output_dir = args.output_dir
    data_root = args.data_root

    _print_header("LOADING PARTICIPANTS")
    external = os.path.join(data_root, "external")
    constraints_path = os.path.join(external, "data_constraints.txt")

    sessions_root = _sessions_root(data_root)
    have_real = os.path.isdir(sessions_root)

    if args.synthetic or not have_real:
        if not args.synthetic and not have_real:
            print(
                "No real session tree found at:\n"
                f"  {sessions_root}\n"
                "The real PhysioNet CSVs are gitignored. Either drop them into "
                "that path, or re-run with --synthetic to use a generated cohort:\n"
                "  python scripts/demo.py --synthetic"
            )
            return 2
        print("Building a deterministic SYNTHETIC cohort (no real data used).")
        participants = build_synthetic_participants(participant_ids)
        self_report: Mapping[tuple[str, str, int], float] = {}
        constraints = Constraints()
    else:
        constraints = (
            load_constraints(constraints_path)
            if os.path.isfile(constraints_path)
            else Constraints()
        )
        participants = load_real_participants(data_root, participant_ids, constraints)
        if not participants:
            print(
                "None of the requested participants could be loaded from the real "
                "tree. Re-run with --synthetic to use a generated cohort."
            )
            return 2

        self_report = {}
        v1 = os.path.join(external, "Stress_Level_v1.csv")
        v2 = os.path.join(external, "Stress_Level_v2.csv")
        if os.path.isfile(v1) and os.path.isfile(v2):
            try:
                self_report = load_stress_levels(v1, v2)
            except Exception as exc:  # defensive: malformed reference tables
                print(f"  warning: could not load self-report tables ({exc})")

    # --- Build the combined table + run the full LOSO evaluation ---------
    _print_header("BUILDING COMBINED TABLE + RUNNING EVALUATION")
    built = build_combined_table(participants)
    if not len(built.table):
        print("The combined table is empty (no usable windows). Nothing to do.")
        return 2
    print(f"Combined table: {len(built.table)} windows.")

    result = run_eval(built, self_report, budget=args.budget)

    written: list[str] = []
    written.extend(write_headline_figures(result, output_dir))

    # --- Hero + calm timelines ------------------------------------------
    _print_header("RENDERING TIMELINES")
    labels = result.labels
    method = args.method

    hero = args.hero
    calm = args.calm
    person_ids = set(built.table["person_id"].astype(str))

    hero_scores = hero_tier_vec = hero_movement = None
    used_fallback = False

    if hero in person_ids:
        hero_scores, hero_tier_vec, hero_movement, used_fallback = hero_tiers(
            built, labels, self_report, method, hero, budget=args.budget
        )
        hero_path = os.path.join(output_dir, f"timeline_{hero}.png")
        try:
            render_participant_timeline(
                built,
                hero,
                hero_scores,
                hero_tier_vec,
                movement=hero_movement,
                output_path=hero_path,
                title=f"Hero timeline — {hero} ({method})",
            )
            written.append(hero_path)
            print(f"  wrote {hero_path}")
        except Exception as exc:
            print(f"  note: could not render hero timeline ({exc})")
    else:
        print(f"  hero {hero!r} not among loaded participants; skipping hero timeline")

    if calm in person_ids:
        # The calm contrast uses the honest LOSO tiers (no fallback): the point
        # is that it stays quiet.
        calm_scores, _ = method_scores(built.table, built.baselines, method)
        calm_loso = run_loso(built, labels, self_report, method, budget=args.budget)
        calm_path = os.path.join(output_dir, f"timeline_{calm}.png")
        try:
            render_participant_timeline(
                built,
                calm,
                calm_scores,
                calm_loso.tiers,
                movement=calm_loso.movement,
                output_path=calm_path,
                title=f"Calm contrast — {calm} ({method})",
            )
            written.append(calm_path)
            print(f"  wrote {calm_path}")
        except Exception as exc:
            print(f"  note: could not render calm timeline ({exc})")
    else:
        print(f"  calm {calm!r} not among loaded participants; skipping calm timeline")

    # --- Summary ---------------------------------------------------------
    if hero_tier_vec is not None:
        print_summary(
            participants,
            result,
            hero,
            method,
            hero_scores,
            hero_tier_vec,
            hero_movement,
            used_fallback,
        )

    # --- Artifact manifest ----------------------------------------------
    _print_header("ARTIFACTS WRITTEN")
    for p in written:
        print(f"  {p}")

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the demo's argparse CLI."""
    default_data_root = os.path.join(_REPO_ROOT, "data")
    parser = argparse.ArgumentParser(
        prog="scripts/demo.py",
        description=(
            "Run the full wearable-flag pipeline on real PhysioNet data (or a "
            "synthetic cohort) and write presentation artifacts to output/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        default=default_data_root,
        help="Repo data/ directory holding sessions/ and external/.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_REPO_ROOT, "output"),
        help="Directory to write PNGs and the metric CSV into.",
    )
    parser.add_argument(
        "--participants",
        default=",".join(DEFAULT_PARTICIPANTS),
        help="Comma-separated participant ids to load.",
    )
    parser.add_argument("--hero", default=DEFAULT_HERO, help="Hero participant id.")
    parser.add_argument("--calm", default=DEFAULT_CALM, help="Calm-contrast participant id.")
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=_METHOD_CHOICES,
        help="Scoring method for the per-participant timelines.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_BUDGET,
        help="Rest false-alarm budget (flags per hour of rest) for calibration.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a deterministic synthetic cohort instead of the real data.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
