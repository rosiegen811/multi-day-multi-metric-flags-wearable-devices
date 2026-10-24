"""Shared schema and records for the Wearable Flag Pipeline.

This module is the single source of truth for the ``Per_Window_Table`` contract
and the typed records shared across all workstreams (``data/windows.py``,
``flag/*``, ``eval/*``). It is intentionally dependency-light: it depends only on
``numpy``, ``pandas``, ``dataclasses``, and the standard library.

The ``Per_Window_Table`` holds one row per person per 10-second window. See the
design document's "Data Models" section for the authoritative column list and
the neutral, non-diagnostic tier wording.

Requirements covered:
- 4.4  full column set, one row per person per window step
- 9.3  severity-tier wording map
- 12.4 ``Thresholds`` ordering invariant ``tau1 < tau2 < tau3``
- 15.1 flags described as sustained change, never a diagnosis
- 15.2 neutral tier labels rather than color-coded ranges
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

__all__ = [
    "COLUMNS",
    "COLUMN_DTYPES",
    "FLOAT_COLUMNS",
    "KEY_COLUMNS",
    "TIER_NONE",
    "TIER_WATCH",
    "TIER_NOTABLE",
    "TIER_STRONG",
    "TIER_LABELS",
    "TIER_WORDING",
    "tier_label",
    "tier_wording",
    "Thresholds",
    "Flag",
    "MovementState",
    "SignalBaseline",
    "empty_per_window_table",
    "make_per_window_table",
    "validate_per_window_table",
    "SchemaError",
]


# ---------------------------------------------------------------------------
# Per_Window_Table column contract
# ---------------------------------------------------------------------------

# Column order is part of the contract (design "Data Models" table).
COLUMNS: tuple[str, ...] = (
    "person_id",
    "cohort",
    "protocol",
    "session",
    "phase",
    "t_win",
    "hr",
    "rmssd",
    "skin_temp",
    "acc_mag",
)

# NaN-capable float measurement columns. ``NaN`` marks an unavailable signal for
# a window (Requirement 4.5).
FLOAT_COLUMNS: tuple[str, ...] = ("hr", "rmssd", "skin_temp", "acc_mag")

# Columns that together identify a window uniquely. One row per person per
# window step (Requirement 4.4): a person is identified by ``person_id`` and a
# window within that person's recording by ``session`` + ``t_win`` (a person may
# appear in more than one session, each with its own window timeline).
KEY_COLUMNS: tuple[str, ...] = ("person_id", "session", "t_win")

# pandas dtypes for each column. Object-typed string columns keep ``None``
# available for ``phase`` (a window outside any labeled phase, per AC 3.3).
COLUMN_DTYPES: Mapping[str, str] = MappingProxyType(
    {
        "person_id": "object",
        "cohort": "object",
        "protocol": "object",
        "session": "object",
        "phase": "object",
        "t_win": "float64",
        "hr": "float64",
        "rmssd": "float64",
        "skin_temp": "float64",
        "acc_mag": "float64",
    }
)


# ---------------------------------------------------------------------------
# Severity tiers and neutral, non-diagnostic wording (Requirements 9.3, 15.1, 15.2)
# ---------------------------------------------------------------------------

TIER_NONE: int = 0
TIER_WATCH: int = 1
TIER_NOTABLE: int = 2
TIER_STRONG: int = 3

# Short neutral labels (no color-coded terms), per AC 15.2.
TIER_LABELS: Mapping[int, str] = MappingProxyType(
    {
        TIER_NONE: "None",
        TIER_WATCH: "Watch",
        TIER_NOTABLE: "Notable",
        TIER_STRONG: "Strong",
    }
)

# Plain-language wording shown with a flag. Wording describes a sustained change,
# never a diagnosis (AC 15.1), and uses no color-coded severity terms (AC 15.2).
# Tier 0 shows nothing.
TIER_WORDING: Mapping[int, str] = MappingProxyType(
    {
        TIER_NONE: "",
        TIER_WATCH: "Some of your signals have shifted from your resting normal",
        TIER_NOTABLE: "Several signals are clearly off your resting normal",
        TIER_STRONG: "A large shift across several signals",
    }
)


def tier_label(tier: int) -> str:
    """Return the neutral short label for a severity tier (0..3)."""
    try:
        return TIER_LABELS[tier]
    except KeyError:
        raise ValueError(f"tier must be one of {sorted(TIER_LABELS)}, got {tier!r}")


def tier_wording(tier: int) -> str:
    """Return the plain-language, non-diagnostic wording for a severity tier."""
    try:
        return TIER_WORDING[tier]
    except KeyError:
        raise ValueError(f"tier must be one of {sorted(TIER_WORDING)}, got {tier!r}")


# ---------------------------------------------------------------------------
# Shared derived records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    """The three ordered Combined_Score thresholds.

    Invariant (Requirement 12.4): ``tau1 < tau2 < tau3``. The ordering is
    enforced in ``__post_init__`` and never silently reordered; a violation is a
    programming/calibration error and raises ``ValueError``.
    """

    tau1: float
    tau2: float
    tau3: float

    def __post_init__(self) -> None:
        taus = (self.tau1, self.tau2, self.tau3)
        for name, value in zip(("tau1", "tau2", "tau3"), taus):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a real number, got {value!r}")
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if not (self.tau1 < self.tau2 < self.tau3):
            raise ValueError(
                "Thresholds must satisfy tau1 < tau2 < tau3, got "
                f"tau1={self.tau1!r}, tau2={self.tau2!r}, tau3={self.tau3!r}"
            )


@dataclass(frozen=True)
class SignalBaseline:
    """A per-person, per-signal resting baseline.

    ``median`` is the robust center; ``mad`` is the scaled median absolute
    deviation (``1.4826 * MAD``) used as the spread; ``shrunk_mad`` is the
    empirical-Bayes spread shrunk toward the cohort pooled estimate and is the
    spread used downstream for z-scoring.
    """

    median: float
    mad: float
    shrunk_mad: float


@dataclass(frozen=True)
class MovementState:
    """Accelerometer-derived movement context for a window.

    ``sustained`` marks continuous motion; ``rhythmic`` marks steady-cadence
    motion; ``cadence_hz`` is the detected cadence when rhythmic, else ``None``.
    """

    sustained: bool
    rhythmic: bool
    cadence_hz: float | None = None


@dataclass(frozen=True)
class Flag:
    """A fired flag describing a sustained, multi-metric change (never a diagnosis).

    ``tier`` is the assigned Severity_Tier in ``0..3``. ``top_signals`` lists the
    top contributing signals as ``(signal_name, real_unit_delta)`` ordered by
    contribution. ``movement_explained`` is set when the movement check
    attributes the flag to motion.
    """

    person_id: str
    start_t: float
    persisted_s: float
    combined_score: float
    tier: int
    top_signals: list[tuple[str, float]] = field(default_factory=list)
    movement_explained: bool = False

    def __post_init__(self) -> None:
        if self.tier not in TIER_LABELS:
            raise ValueError(
                f"tier must be one of {sorted(TIER_LABELS)}, got {self.tier!r}"
            )


# ---------------------------------------------------------------------------
# Per_Window_Table construction and validation
# ---------------------------------------------------------------------------


class SchemaError(ValueError):
    """Raised when a table does not satisfy the Per_Window_Table contract."""


def empty_per_window_table() -> pd.DataFrame:
    """Return an empty Per_Window_Table with the full column set and dtypes."""
    return pd.DataFrame(
        {name: pd.Series(dtype=COLUMN_DTYPES[name]) for name in COLUMNS}
    )


def make_per_window_table(rows) -> pd.DataFrame:
    """Build a validated Per_Window_Table from an iterable of row mappings.

    Each row must supply exactly the columns in ``COLUMNS``. The result is
    column-ordered, typed per ``COLUMN_DTYPES``, and validated to hold one row
    per person per window step (Requirement 4.4). Float measurement columns may
    hold ``NaN`` to mark an unavailable signal (Requirement 4.5).

    Raises ``SchemaError`` if any row is missing or has unexpected columns, or if
    the resulting table violates the one-row-per-person-per-window invariant.
    """
    rows = list(rows)
    expected = set(COLUMNS)
    for i, row in enumerate(rows):
        keys = set(row.keys())
        missing = expected - keys
        extra = keys - expected
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing {sorted(missing)}")
            if extra:
                parts.append(f"unexpected {sorted(extra)}")
            raise SchemaError(f"row {i} has {' and '.join(parts)}")

    if rows:
        table = pd.DataFrame(rows, columns=list(COLUMNS))
    else:
        table = empty_per_window_table()

    table = _coerce_dtypes(table)
    validate_per_window_table(table)
    return table


def validate_per_window_table(table: pd.DataFrame) -> None:
    """Validate that ``table`` satisfies the Per_Window_Table contract.

    Enforces (Requirement 4.4):
    - the full column set, in the contract order, with no extra columns;
    - exactly one row per person per window step (unique over ``KEY_COLUMNS``).

    Raises ``SchemaError`` on any violation. Returns ``None`` when valid.
    """
    if not isinstance(table, pd.DataFrame):
        raise SchemaError(f"expected a pandas DataFrame, got {type(table)!r}")

    actual = tuple(table.columns)
    if actual != COLUMNS:
        missing = [c for c in COLUMNS if c not in table.columns]
        extra = [c for c in table.columns if c not in COLUMNS]
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        if not details:
            details.append(f"wrong order: expected {list(COLUMNS)}, got {list(actual)}")
        raise SchemaError("Per_Window_Table column set invalid: " + "; ".join(details))

    # One row per person per window step.
    if len(table) and table.duplicated(subset=list(KEY_COLUMNS)).any():
        dupes = table[table.duplicated(subset=list(KEY_COLUMNS), keep=False)]
        offending = (
            dupes[list(KEY_COLUMNS)].drop_duplicates().to_dict(orient="records")
        )
        raise SchemaError(
            "Per_Window_Table must hold one row per person per window step; "
            f"duplicate keys found for {offending}"
        )


def _coerce_dtypes(table: pd.DataFrame) -> pd.DataFrame:
    """Return ``table`` with each column cast to its contract dtype."""
    out = table.copy()
    for name in COLUMNS:
        if name in FLOAT_COLUMNS:
            out[name] = pd.to_numeric(out[name], errors="coerce").astype("float64")
        elif COLUMN_DTYPES[name] == "float64":
            out[name] = pd.to_numeric(out[name], errors="coerce").astype("float64")
        else:
            out[name] = out[name].astype(COLUMN_DTYPES[name])
    return out[list(COLUMNS)]
