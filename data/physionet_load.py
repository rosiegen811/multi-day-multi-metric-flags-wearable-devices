"""Data_Loader for the Wearable Flag Pipeline.

Reads raw Empatica E4 CSV files, button-press tags, and the dataset's
cp1252-encoded reference text into in-memory structures for downstream
components. Side effects (file reads) are isolated to this module's edges; the
parsing logic itself is pure and testable.

This module implements the signal/IBI/tags/cohort/cp1252 loaders of the
Data_Loader component (design "Data_Loader — data/physionet_load.py"). It is
intentionally dependency-light: it depends only on ``numpy``, ``pandas``, and
the standard library.

Requirements covered:
- 1.1  HR/TEMP/ACC: row 1 is the session start time (UTC epoch seconds),
       row 2 is the sampling rate (Hz), remaining rows are samples.
- 1.2  IBI: row 1 is the session start time; each subsequent row is an
       inter-beat interval measurement.
- 1.3  tags: each entry is a button-press timestamp (epoch seconds).
- 1.4  participant id beginning with "S" -> Cohort_V1.
- 1.5  participant id beginning with "f" -> Cohort_V2.
- 1.6  ``data_constraints.txt`` / ``Data_Dictionary.csv`` decode as cp1252.
- 1.7  a required signal file missing for a participant/session is recorded as
       absent and loading of the remaining signals continues (no exception).

The Empatica E4 CSV header convention (see the dataset README and design
"Header contract"):

- HR.csv / TEMP.csv / EDA.csv (single channel): row 1 holds the start
  timestamp, row 2 holds the sample rate, subsequent rows are samples.
- ACC.csv (three axes): row 1 holds the start timestamp repeated once per axis
  (3 columns), row 2 holds the per-axis sample rate, subsequent rows are the
  ``x, y, z`` triples. ``values`` is returned with shape ``(N, 3)``.
- IBI.csv: row 1 is ``<start_timestamp>, IBI``; each subsequent row is
  ``<seconds_since_start>, <ibi_interval_seconds>``. The interval measurements
  (the second column) are stored in ``offsets_s``.
- tags.csv: one Unix timestamp per line, with no header rows.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SignalSeries",
    "IBISeries",
    "ParticipantSignals",
    "load_signal",
    "load_ibi",
    "load_tags",
    "cohort_of",
    "read_cp1252",
    "load_participant",
    "COHORT_V1",
    "COHORT_V2",
]


# Cohort labels (Requirements 1.4, 1.5). Kept as named constants so downstream
# code and tests can reference them without hard-coding the strings.
COHORT_V1 = "V1"
COHORT_V2 = "V2"


# ---------------------------------------------------------------------------
# Shared loader records
# ---------------------------------------------------------------------------


@dataclass
class SignalSeries:
    """A loaded single- or multi-channel E4 signal.

    ``name`` is one of ``"HR"``, ``"TEMP"``, ``"ACC"`` (and by extension any
    single-channel signal such as ``"EDA"``/``"BVP"``). ``start_utc`` is the
    session start time in epoch seconds UTC (row 1 of the CSV).
    ``sample_rate_hz`` is the sampling rate in Hz (row 2). ``values`` holds the
    remaining rows: a 1-D array for single-channel signals and a 2-D ``(N, 3)``
    array for ACC.
    """

    name: str
    start_utc: float
    sample_rate_hz: float
    values: np.ndarray


@dataclass
class IBISeries:
    """A loaded inter-beat-interval series from ``IBI.csv``.

    ``start_utc`` is the session start time in epoch seconds UTC (row 1).
    ``offsets_s`` holds the successive-row inter-beat interval measurements in
    seconds (the second column of each subsequent row).
    """

    start_utc: float
    offsets_s: np.ndarray


@dataclass
class ParticipantSignals:
    """All loaded signals for one participant in one session.

    Assembled by :func:`load_participant` from the per-signal loaders. Each
    signal slot holds a loaded record (``SignalSeries`` for HR/TEMP/ACC, an
    ``IBISeries`` for IBI, a list of timestamps for tags) or ``None`` when the
    corresponding file was absent. The names of any absent expected signals are
    collected in ``absent`` so downstream components can continue without the
    loader raising (Requirement 1.7).

    ``person_id`` is the participant identifier (e.g. ``"S01"``), ``cohort`` is
    its cohort label from :func:`cohort_of` (``"V1"`` or ``"V2"``), and
    ``session`` is the session folder name (e.g. ``"STRESS"``, ``"AEROBIC"``,
    ``"ANAEROBIC"``).

    The per-signal ``start_utc`` / ``sample_rate_hz`` of HR, TEMP, and ACC are
    available directly on their ``SignalSeries`` records; ``start_utc`` for the
    IBI series is on ``ibi``. They are not duplicated here so there is a single
    source of truth per signal.
    """

    person_id: str
    cohort: str
    session: str
    hr: SignalSeries | None = None
    temp: SignalSeries | None = None
    acc: SignalSeries | None = None
    ibi: IBISeries | None = None
    tags: list[float] | None = None
    absent: list[str] = field(default_factory=list)


# The expected signal files for a participant/session folder, mapped to the
# ``ParticipantSignals`` attribute each populates. Order fixes the order in
# which signals are attempted and, consequently, the order of ``absent``.
_EXPECTED_SIGNALS: tuple[tuple[str, str], ...] = (
    ("HR", "hr"),
    ("TEMP", "temp"),
    ("ACC", "acc"),
    ("IBI", "ibi"),
    ("tags", "tags"),
)


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------


def _read_rows(path: str) -> list[list[str]]:
    """Read ``path`` as CSV, returning a list of non-empty rows of cells.

    Empatica E4 files are plain comma-separated. Trailing blank lines (common
    when a file ends with a newline) are dropped so they are not mistaken for a
    data row. Individual cells are stripped of surrounding whitespace.
    """
    rows: list[list[str]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.reader(fh):
            cells = [cell.strip() for cell in raw]
            if not cells or all(cell == "" for cell in cells):
                continue
            rows.append(cells)
    return rows


def load_signal(path: str, name: str) -> SignalSeries:
    """Load an E4 signal CSV (HR/TEMP/ACC and other single-channel signals).

    Reads row 1 as ``start_utc`` (epoch seconds UTC) and row 2 as
    ``sample_rate_hz`` (Hz), then the remaining rows as samples (Requirement
    1.1). For ACC the header rows carry one value per axis (3 columns) and the
    sample rows are ``x, y, z`` triples, so ``values`` has shape ``(N, 3)``;
    for single-channel signals ``values`` is a 1-D array.

    ``name`` is stored on the returned record and also decides channel shape:
    ``"ACC"`` (case-insensitive) is treated as 3-channel, everything else as
    single-channel.
    """
    rows = _read_rows(path)
    if len(rows) < 2:
        raise ValueError(
            f"signal file {path!r} must have at least a start-time row and a "
            f"sample-rate row, found {len(rows)} row(s)"
        )

    is_acc = name.strip().upper() == "ACC"

    start_row = [float(c) for c in rows[0]]
    rate_row = [float(c) for c in rows[1]]

    # The E4 header repeats the start time / rate once per channel. Take the
    # first column as the scalar start time and sample rate.
    start_utc = start_row[0]
    sample_rate_hz = rate_row[0]

    sample_rows = rows[2:]
    if is_acc:
        if sample_rows:
            values = np.array(
                [[float(c) for c in row] for row in sample_rows], dtype=float
            )
            if values.ndim != 2 or values.shape[1] != 3:
                raise ValueError(
                    f"ACC file {path!r} must have 3 columns per sample row, "
                    f"got shape {values.shape}"
                )
        else:
            values = np.empty((0, 3), dtype=float)
    else:
        # Single-channel: take the first column of each sample row.
        values = np.array([float(row[0]) for row in sample_rows], dtype=float)

    return SignalSeries(
        name=name,
        start_utc=start_utc,
        sample_rate_hz=sample_rate_hz,
        values=values,
    )


def load_ibi(path: str) -> IBISeries:
    """Load an ``IBI.csv`` inter-beat-interval file.

    Reads row 1 as ``start_utc`` (epoch seconds UTC) and parses each subsequent
    row as an inter-beat interval measurement (Requirement 1.2). The real
    Empatica ``IBI.csv`` has two columns per sample row: the time since start
    and the IBI value; the IBI value (second column) is stored in ``offsets_s``.
    A single-column variant (interval only) is also accepted.
    """
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"IBI file {path!r} is empty")

    # Row 1: "<start_timestamp>" possibly followed by a header label ("IBI").
    start_utc = float(rows[0][0])

    intervals: list[float] = []
    for row in rows[1:]:
        # Two-column Empatica format: (seconds_since_start, ibi_interval).
        # Fall back to a single interval column if only one value is present.
        value = row[1] if len(row) >= 2 else row[0]
        intervals.append(float(value))

    offsets_s = np.array(intervals, dtype=float)
    return IBISeries(start_utc=start_utc, offsets_s=offsets_s)


def load_tags(path: str) -> list[float]:
    """Load a ``tags.csv`` file of button-press timestamps.

    Each entry is read as a button-press timestamp in epoch seconds
    (Requirement 1.3). The file has one timestamp per line and no header rows.
    """
    rows = _read_rows(path)
    return [float(row[0]) for row in rows]


def cohort_of(participant_id: str) -> str:
    """Map a participant identifier to its cohort label.

    A participant id beginning with ``"S"`` is Cohort_V1 (``"V1"``), and one
    beginning with ``"f"`` is Cohort_V2 (``"V2"``) (Requirements 1.4, 1.5).
    Matching is case-sensitive per the dataset's identifier convention
    (uppercase ``S``, lowercase ``f``).

    Raises ``ValueError`` for an empty id or one with an unrecognized prefix.
    """
    if not participant_id:
        raise ValueError("participant_id must be a non-empty string")
    prefix = participant_id[0]
    if prefix == "S":
        return COHORT_V1
    if prefix == "f":
        return COHORT_V2
    raise ValueError(
        f"participant_id {participant_id!r} must begin with 'S' (Cohort_V1) "
        f"or 'f' (Cohort_V2)"
    )


def read_cp1252(path: str) -> str:
    """Decode a dataset reference text file using Windows-1252 (cp1252).

    Used for ``data_constraints.txt`` and ``Data_Dictionary.csv``, which are
    cp1252-encoded (Requirement 1.6). Returns the decoded text as a ``str``.
    """
    with open(path, encoding="cp1252") as fh:
        return fh.read()


def load_participant(session_dir: str, pid: str) -> ParticipantSignals:
    """Assemble all signals for one participant/session, tolerating gaps.

    Loads ``HR.csv``, ``TEMP.csv``, ``ACC.csv``, ``IBI.csv`` and ``tags.csv``
    from ``<session_dir>/<pid>/`` into a :class:`ParticipantSignals` record.
    ``session_dir`` is the session folder (e.g.
    ``data/sessions/Wearable_Dataset_sessions/Wearable_Dataset/STRESS``) and
    ``pid`` is the participant subfolder (e.g. ``"S01"``). The session label on
    the record is taken from the basename of ``session_dir``.

    Missing-file tolerance (Requirement 1.7): if a required signal file is
    absent, its name is recorded in ``absent`` and loading continues with the
    remaining signals rather than raising. The cohort is derived from ``pid``
    via :func:`cohort_of`.
    """
    participant_dir = os.path.join(session_dir, pid)
    session = os.path.basename(os.path.normpath(session_dir))

    result = ParticipantSignals(
        person_id=pid,
        cohort=cohort_of(pid),
        session=session,
    )

    for signal_name, attr in _EXPECTED_SIGNALS:
        path = os.path.join(participant_dir, f"{signal_name}.csv")
        if not os.path.isfile(path):
            result.absent.append(signal_name)
            continue

        if signal_name == "IBI":
            loaded: object = load_ibi(path)
        elif signal_name == "tags":
            loaded = load_tags(path)
        else:
            loaded = load_signal(path, signal_name)

        setattr(result, attr, loaded)

    return result
