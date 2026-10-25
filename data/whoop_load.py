"""WHOOP nightly Data_Loader for the Wearable Flag Pipeline.

This module is the *second* data front-end of the pipeline, alongside the
PhysioNet Empatica E4 loader (``data/physionet_load.py``). PhysioNet remains the
built-in demo (short lab sessions windowed every 10 s); this loader instead reads
a single person's own **WHOOP export** — one row per cycle (≈ one per night) —
and maps each night to one row of the shared ``Per_Window_Table``
(``data/schema.py``). That mapping is the whole point: once a night is a
"window", every downstream module (Baseline_Estimator, Scorer, Flagger,
Explainer, Demo_App) runs completely unchanged — "the same pipeline, nights in
place of minutes" from the project pitch.

It reads the two CSVs a WHOOP data export produces:

- ``physiological_cycles.csv`` — one row per physiological cycle, carrying the
  nightly summary metrics this pipeline scores (resting heart rate, heart-rate
  variability, skin temperature) plus day-level context (Day Strain, energy
  burned) used only as a benign-cause exertion proxy downstream.
- ``sleeps.csv`` — the per-sleep rows, with a trailing ``Nap`` boolean. Naps are
  intraday and are **not** a nightly cycle, so they are ignored for the nightly
  baseline; the cycles file is the nightly source. ``sleeps.csv`` is read only to
  honor that contract (naps filtered out) and is otherwise optional.

Signal mapping (one cycle row -> one Per_Window_Table row)
----------------------------------------------------------
- ``person_id``  = a single id, default ``"WHOOP"`` (a single-person run).
- ``cohort``     = :data:`WHOOP_COHORT` (``"WHOOP"``) — a grouping label only.
- ``protocol``   = :data:`WHOOP_PROTOCOL` (``"NIGHT"``) — a grouping label only.
- ``session``    = :data:`WHOOP_SESSION` (``"NIGHTLY"``).
- ``phase``      = ``None`` (no protocol phases exist for nightly data).
- ``t_win``      = the cycle start time parsed to epoch seconds, via the shared
  :func:`data.physionet_load._parse_epoch_seconds` convention (reused, not
  re-implemented, so WHOOP and PhysioNet timing stay consistent). Nights are
  spaced ~1 day apart rather than 10 s; downstream treats ``t_win`` as opaque
  time, so the wide spacing is fine.
- ``hr``         = Resting heart rate (bpm).
- ``rmssd``      = Heart rate variability (ms). WHOOP reports HRV directly, so no
  IBI/RMSSD derivation is needed (unlike the PhysioNet path).
- ``skin_temp``  = Skin temp (celsius).
- ``acc_mag``    = ``NaN`` — nightly summaries carry no accelerometer, so the
  movement check simply never fires from ACC. Day-level exertion is carried
  separately (:attr:`WhoopNight.day_strain` / :attr:`WhoopNight.energy_burned`)
  for the strain-based benign-cause step in the run script, not in ``acc_mag``.

Any blank metric cell becomes ``NaN`` (the Scorer / Baseline_Estimator already
drop ``NaN`` per-signal), and a fully blank / placeholder row (no usable start
time and no finite metric) is dropped entirely.

Cohort labels downstream: the metrics / calibration code groups by whatever
label is present and does not hard-require ``{"V1","V2"}`` — so ``"WHOOP"`` is a
valid single-person cohort. (The one place that maps an id prefix to a cohort,
:func:`data.physionet_load.cohort_of`, is only used on the PhysioNet path and is
not touched here.)

Dependencies are stdlib + ``numpy`` + ``pandas`` only, matching the house style
of ``data/physionet_load.py``; side effects (file reads) are confined to
:func:`load_whoop_cycles` / :func:`load_whoop_sleeps`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.physionet_load import _parse_epoch_seconds
from data.schema import make_per_window_table

__all__ = [
    "WHOOP_PERSON_ID",
    "WHOOP_COHORT",
    "WHOOP_PROTOCOL",
    "WHOOP_SESSION",
    "CYCLE_START_COLUMN",
    "COLUMN_MAP",
    "WhoopNight",
    "load_whoop_cycles",
    "load_whoop_sleeps",
    "to_per_window_table",
]


# ---------------------------------------------------------------------------
# Constant grouping labels for a single-person nightly run
# ---------------------------------------------------------------------------

# Default participant id. A WHOOP export is one person's own data, so a single id
# is the natural grouping key; callers may override it (``--person-id``).
WHOOP_PERSON_ID: str = "WHOOP"

# Cohort / protocol / session labels. These are grouping labels only downstream
# (the metrics / calibration code groups by whatever is present); a single-person
# nightly run has exactly one of each.
WHOOP_COHORT: str = "WHOOP"
WHOOP_PROTOCOL: str = "NIGHT"
WHOOP_SESSION: str = "NIGHTLY"


# ---------------------------------------------------------------------------
# WHOOP CSV header contract
# ---------------------------------------------------------------------------

# The cycle-start column is both the row's timeline anchor (``t_win``) and the
# field that decides whether a row is a real cycle (a parseable start time) or a
# blank placeholder to drop.
CYCLE_START_COLUMN: str = "Cycle start time"

# WHOOP physiological_cycles.csv header -> the scored Per_Window_Table signal it
# supplies. Only these three are z-scored downstream; everything else in the CSV
# is either context (Day Strain, energy burned — carried separately for the
# benign-cause step) or unused here.
COLUMN_MAP: dict[str, str] = {
    "Resting heart rate (bpm)": "hr",
    "Heart rate variability (ms)": "rmssd",
    "Skin temp (celsius)": "skin_temp",
}

# Day-level exertion context columns (not Per_Window_Table signals). Carried on
# :class:`WhoopNight` for the strain-based benign-cause check in the run script.
_DAY_STRAIN_COLUMN: str = "Day Strain"
_ENERGY_BURNED_COLUMN: str = "Energy burned (cal)"

# The trailing boolean on sleeps.csv marking a nap (intraday) rather than the
# main nightly sleep. Naps are ignored for the nightly baseline.
_NAP_COLUMN: str = "Nap"


# ---------------------------------------------------------------------------
# Parsed nightly record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WhoopNight:
    """One parsed WHOOP cycle (≈ one night) in pipeline terms.

    ``t_win`` is the cycle start time in epoch seconds (via the shared
    ``_parse_epoch_seconds`` convention). ``hr`` / ``rmssd`` / ``skin_temp`` are
    the three scored signals (``NaN`` when the source cell was blank). ``day_strain``
    and ``energy_burned`` are day-level exertion context used only by the
    benign-cause step downstream (``NaN`` when blank); they are deliberately *not*
    mapped into ``acc_mag`` so the accelerometer movement check never fires from
    nightly data.
    """

    t_win: float
    hr: float
    rmssd: float
    skin_temp: float
    day_strain: float
    energy_burned: float


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _to_float(cell: str | None) -> float:
    """Parse a metric cell to ``float``; a blank / unparseable cell becomes ``NaN``.

    WHOOP leaves metric fields blank on missing-data nights; those become ``NaN``
    so the Scorer / Baseline_Estimator drop that signal for the night rather than
    poisoning the estimate.
    """
    if cell is None:
        return float("nan")
    text = str(cell).strip()
    if text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _parse_t_win(cell: str | None) -> float:
    """Parse a cycle start-time cell to epoch seconds, or ``NaN`` when unusable.

    Reuses the shared :func:`data.physionet_load._parse_epoch_seconds` so WHOOP's
    naive ``"YYYY-MM-DD HH:MM:SS"`` datetime strings land on the same epoch
    convention as the PhysioNet headers (only relative timing is used downstream,
    so the fixed-UTC interpretation is consistent across both paths). A blank or
    unparseable start time returns ``NaN`` so the row is treated as a placeholder
    and dropped.
    """
    if cell is None:
        return float("nan")
    text = str(cell).strip()
    if text == "":
        return float("nan")
    try:
        return float(_parse_epoch_seconds(text))
    except ValueError:
        return float("nan")


def _read_dicts(path: str) -> list[dict[str, str]]:
    """Read a WHOOP CSV into a list of header-keyed row dicts (UTF-8).

    A thin wrapper over :class:`csv.DictReader`. WHOOP exports are UTF-8 (unlike
    the cp1252 PhysioNet reference text), so no special decoding is needed.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_whoop_cycles(path: str) -> list[WhoopNight]:
    """Load ``physiological_cycles.csv`` into parsed nightly records.

    Reads each cycle row and maps it to a :class:`WhoopNight`: the cycle start
    time to ``t_win`` (epoch seconds), and the resting-HR / HRV / skin-temp cells
    to the three scored signals (blank -> ``NaN``). Day Strain and energy burned
    are carried for the benign-cause step. Rows with no parseable start time
    *and* no finite scored signal are treated as blank placeholders and dropped;
    a row with a valid start time is kept even if some metrics are blank (that is
    a legitimate missing-data night). The returned list is in file order (WHOOP
    exports newest-first; ordering by time is left to the caller / table build).
    """
    nights: list[WhoopNight] = []
    for row in _read_dicts(path):
        t_win = _parse_t_win(row.get(CYCLE_START_COLUMN))
        hr = _to_float(row.get("Resting heart rate (bpm)"))
        rmssd = _to_float(row.get("Heart rate variability (ms)"))
        skin_temp = _to_float(row.get("Skin temp (celsius)"))
        day_strain = _to_float(row.get(_DAY_STRAIN_COLUMN))
        energy_burned = _to_float(row.get(_ENERGY_BURNED_COLUMN))

        # Drop a fully blank / placeholder row: no timeline anchor and no signal.
        any_signal = any(np.isfinite(v) for v in (hr, rmssd, skin_temp))
        if not np.isfinite(t_win) and not any_signal:
            continue

        nights.append(
            WhoopNight(
                t_win=t_win,
                hr=hr,
                rmssd=rmssd,
                skin_temp=skin_temp,
                day_strain=day_strain,
                energy_burned=energy_burned,
            )
        )
    return nights


def load_whoop_sleeps(path: str) -> list[dict[str, str]]:
    """Load ``sleeps.csv``, keeping only the main nightly sleep (``Nap=false``).

    Naps are intraday and are not a nightly cycle, so they must be excluded from
    the nightly baseline. This returns the non-nap sleep rows as raw header-keyed
    dicts; the nightly pipeline scores off ``physiological_cycles.csv`` (one row
    per cycle), so this loader exists mainly to honor the nap-filtering contract
    and is optional. A missing ``Nap`` column is treated conservatively as "not a
    nap" (the row is kept).
    """
    kept: list[dict[str, str]] = []
    for row in _read_dicts(path):
        raw = row.get(_NAP_COLUMN)
        is_nap = str(raw).strip().lower() in {"true", "1", "yes"} if raw is not None else False
        if is_nap:
            continue
        kept.append(row)
    return kept


# ---------------------------------------------------------------------------
# Per_Window_Table build
# ---------------------------------------------------------------------------


def to_per_window_table(
    rows: list[WhoopNight],
    person_id: str = WHOOP_PERSON_ID,
) -> pd.DataFrame:
    """Build a validated nightly ``Per_Window_Table`` from parsed WHOOP nights.

    Maps each :class:`WhoopNight` to one table row (see the module docstring for
    the full field mapping): constant ``person_id`` / :data:`WHOOP_COHORT` /
    :data:`WHOOP_PROTOCOL` / :data:`WHOOP_SESSION`, ``phase = None``, the night's
    ``t_win`` and the three scored signals, and ``acc_mag = NaN``. Rows with a
    non-finite ``t_win`` are skipped (they have no place on the timeline), and the
    kept rows are sorted ascending by ``t_win`` so the nightly timeline is
    monotone (WHOOP exports are newest-first). The result is run through
    :func:`data.schema.make_per_window_table`, so the full column set, dtypes, and
    the one-row-per-person-per-window-step invariant are enforced (an all-``NaN``
    ``t_win`` or an empty input yields an empty, still-valid table).
    """
    timed = [n for n in rows if np.isfinite(n.t_win)]
    timed.sort(key=lambda n: n.t_win)

    table_rows: list[dict[str, object]] = []
    for night in timed:
        table_rows.append(
            {
                "person_id": person_id,
                "cohort": WHOOP_COHORT,
                "protocol": WHOOP_PROTOCOL,
                "session": WHOOP_SESSION,
                "phase": None,
                "t_win": float(night.t_win),
                "hr": float(night.hr),
                "rmssd": float(night.rmssd),
                "skin_temp": float(night.skin_temp),
                "acc_mag": float("nan"),
            }
        )

    return make_per_window_table(table_rows)
