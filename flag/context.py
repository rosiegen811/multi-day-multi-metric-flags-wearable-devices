"""Movement_Check — flag/context.py: movement as a benign cause (Requirement 8).

This module sits immediately *after* the Flagger (``flag/tiers.py``) in the
pipeline. The Flagger produces a *pre-movement* Severity_Tier from the combined
deviation score (gate → median-of-3 tier → single-signal cap → hysteresis); the
Movement_Check then asks a different question: *is this shift just the body
working?* An exercise-driven rise in heart rate and skin temperature is a real
physiological change, but it is a benign, self-explanatory one, so when the
accelerometer shows the person was moving we drop the flag by one tier and label
it "explained by movement" (design "Flagger … and Movement_Check", Requirement 8).

Two primitives, one flow
------------------------
- :func:`movement_label` — reads the window's accelerometer samples and decides
  whether the person was moving, returning a :class:`~data.schema.MovementState`
  with two independent flags and an optional cadence:

  * **sustained** — the accelerometer magnitude is *elevated and variable* across
    the window (walking, jogging, fidgeting at the desk, any gross motion). This
    is the "obvious movement" case (AC 8.1).
  * **rhythmic** — a *dominant periodic component at a steady cadence* is present
    even when the gross motion is small. Pedalling on a stationary bike barely
    translates the wrist yet imprints a steady beat on the accelerometer; a
    cyclic cadence like that is still movement (AC 8.2). ``cadence_hz`` carries
    the detected beat frequency when ``rhythmic`` is true, else ``None``.

- :func:`apply_movement` — the tier adjustment: when the state shows sustained
  **or** rhythmic movement, reduce the tier by **exactly one** level, floored at
  :data:`~data.schema.TIER_NONE` (0), and return ``(new_tier, True)`` — the flag
  is "explained by movement". When neither is present, return ``(tier, False)``
  unchanged (AC 8.1–8.3, Property 20).

Detection approach
------------------
Both tests run on a short 1-D series of **accelerometer magnitude** samples
``sqrt(x^2 + y^2 + z^2)`` — one scalar per ACC sample. Working on the magnitude
rather than the three raw axes makes the check orientation-independent (it does
not matter how the watch is rotated) and lets the same code run from either raw
ACC (an ``(N, 3)`` array, reduced here) or an already-computed magnitude series.

*Sustained* is a pair of thresholds on that series:

* a **variability** threshold on the robust spread (scaled MAD) of the magnitude
  — steady motion makes the signal jump around far more than the sensor noise of
  a still wrist; and
* an **elevation** threshold on how far the mean magnitude sits above the
  resting 1 g (``64`` counts at 1/64 g) level.

Either condition alone is enough: a high-variance burst or a sustained lift of
the mean both mean "moving". The thresholds are named module constants so they
can be tuned against the dataset later.

*Rhythmic* is a **normalized autocorrelation peak** on the mean-removed
magnitude. A steady cadence makes the signal correlate with a lagged copy of
itself; we scan physiologically plausible cadence lags (roughly
:data:`CADENCE_MIN_HZ`–:data:`CADENCE_MAX_HZ`, covering slow pedalling through a
brisk run) and call the window rhythmic when the strongest peak in that band
exceeds :data:`RHYTHMIC_PEAK_MIN`. Autocorrelation (rather than a full FFT) keeps
the dependency surface to numpy and is robust on the short series a single window
provides; the detected cadence is ``sample_rate_hz / best_lag``. ``prev_state``
is accepted for symmetry with the design signature and to allow light hysteresis
across windows later; it is not required by the acceptance criteria and does not
change the current decision.

Why reduce by exactly one, floored at zero
------------------------------------------
Movement is evidence the shift is benign, not proof it is nothing — a Strong,
clearly-moving flag still becomes Notable, not None. Dropping by exactly one
level (never two, never to zero from Watch unless it was Watch) keeps the
adjustment proportional and is what Property 20 pins down; the floor at
``TIER_NONE`` keeps a Watch (1) flag from going negative — it becomes None (0).

Dependencies are numpy / stdlib plus the shared records in ``data/schema.py``
(``MovementState`` and the ``TIER_NONE … TIER_STRONG`` constants). scipy is not
needed — the autocorrelation is a short numpy correlation. Every function is
pure over its inputs.

Requirements covered
--------------------
- 8.1  sustained movement across a flagged window → drop one tier, "explained by
       movement" (:func:`movement_label` sustained branch, :func:`apply_movement`).
- 8.2  rhythmic motion at a steady cadence → drop one tier, "explained by
       movement" (:func:`movement_label` rhythmic branch, :func:`apply_movement`).
- 8.3  no sustained/rhythmic movement → tier and label unchanged
       (:func:`apply_movement`).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from data.schema import (
    TIER_NONE,
    MovementState,
)

__all__ = [
    "ACC_COUNTS_PER_G",
    "SUSTAINED_MAD_COUNTS",
    "SUSTAINED_ELEVATION_COUNTS",
    "RHYTHMIC_PEAK_MIN",
    "CADENCE_MIN_HZ",
    "CADENCE_MAX_HZ",
    "MIN_RHYTHMIC_SAMPLES",
    "MAD_SCALE",
    "acc_magnitude_series",
    "movement_label",
    "apply_movement",
]


# ---------------------------------------------------------------------------
# Tunable detection constants (documented so they can be calibrated later)
# ---------------------------------------------------------------------------

# The E4 accelerometer reports in units of 1/64 g, so 1 g (the resting magnitude
# of a still wrist under gravity) is 64 counts (dataset README / Accelerometer
# Magnitude glossary entry).
ACC_COUNTS_PER_G: float = 64.0

# Scale factor turning a raw median absolute deviation into a robust,
# normal-consistent standard-deviation estimate (same convention as the
# Baseline_Estimator's scaled MAD).
MAD_SCALE: float = 1.4826

# --- Sustained-movement thresholds (AC 8.1) --------------------------------
# A still wrist's magnitude barely varies (sensor noise is a few counts); steady
# motion makes the magnitude swing by a substantial fraction of 1 g. Flag
# "sustained" when the robust spread (scaled MAD) of the window's magnitude
# exceeds this many counts.
SUSTAINED_MAD_COUNTS: float = 12.0

# ...or when the mean magnitude sits this many counts *above* the resting 1 g
# level, i.e. the arm is being held/driven well away from rest for the window.
SUSTAINED_ELEVATION_COUNTS: float = 10.0

# --- Rhythmic-cadence thresholds (AC 8.2) ----------------------------------
# Normalized autocorrelation peak (0..1) required within the cadence band to
# call the window rhythmic. A steady cadence produces a sharp peak; noise does
# not.
RHYTHMIC_PEAK_MIN: float = 0.5

# Plausible human cadence band in Hz: ~0.8 Hz (slow pedalling / ~48 rpm) up to
# ~3.5 Hz (a brisk run / fast sprint cadence). Autocorrelation lags outside this
# band are ignored so slow drift and high-frequency sensor noise do not register
# as a cadence.
CADENCE_MIN_HZ: float = 0.8
CADENCE_MAX_HZ: float = 3.5

# Autocorrelation needs enough samples to see a few cycles; below this the
# rhythmic test is skipped (returns not-rhythmic) rather than firing on noise.
MIN_RHYTHMIC_SAMPLES: int = 16


# ---------------------------------------------------------------------------
# Accelerometer magnitude helper
# ---------------------------------------------------------------------------


def acc_magnitude_series(acc_window) -> np.ndarray:
    """Reduce a window's accelerometer samples to a 1-D magnitude series.

    ``acc_window`` is the per-window accelerometer samples in either accepted
    form:

    - **raw ACC** — a 2-D ``(N, 3)`` array of ``x, y, z`` triples (as produced by
      :class:`~data.physionet_load.SignalSeries` for ACC). Reduced per sample to
      the vector magnitude ``sqrt(x^2 + y^2 + z^2)``.
    - **magnitude series** — an already-1-D sequence of per-sample magnitudes.
      Returned as a float array unchanged.

    The ``Per_Window_Table`` ``acc_mag`` column is a *single aggregate* per
    window (the trailing-30 s mean), which is why the Movement_Check needs the
    underlying per-sample samples rather than that one column — cadence detection
    (:func:`movement_label`) has nothing to work with from a single number. Pass
    the raw per-window ACC samples (or their per-sample magnitudes) here.

    Returns an empty float array for empty input. Raises ``ValueError`` for an
    array that is neither 1-D nor ``(N, 3)``.
    """
    arr = np.asarray(acc_window, dtype=float)
    if arr.size == 0:
        return np.empty(0, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 3:
        return np.sqrt(np.sum(arr * arr, axis=1))
    raise ValueError(
        "acc_window must be a 1-D magnitude series or a 2-D (N, 3) raw ACC "
        f"array, got shape {arr.shape}"
    )


# ---------------------------------------------------------------------------
# Movement detection (AC 8.1, 8.2)
# ---------------------------------------------------------------------------


def _scaled_mad(x: np.ndarray) -> float:
    """Normal-consistent scaled median absolute deviation of ``x``."""
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(MAD_SCALE * mad)


def _detect_rhythmic(
    mag: np.ndarray, sample_rate_hz: float
) -> tuple[bool, float | None]:
    """Autocorrelation cadence test on a magnitude series.

    Returns ``(rhythmic, cadence_hz)``. The series is mean-removed, then its
    normalized autocorrelation is scanned over the lags corresponding to the
    :data:`CADENCE_MIN_HZ`–:data:`CADENCE_MAX_HZ` band. When the strongest peak
    in that band meets :data:`RHYTHMIC_PEAK_MIN`, the window is rhythmic at
    ``cadence_hz = sample_rate_hz / best_lag``.

    Returns ``(False, None)`` when there are too few samples, the sample rate is
    not usable, the signal is flat (zero variance), or no in-band lag clears the
    peak threshold.
    """
    n = mag.size
    if n < MIN_RHYTHMIC_SAMPLES:
        return False, None
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        return False, None

    centered = mag - np.mean(mag)
    var0 = float(np.dot(centered, centered))
    if var0 <= 0.0:
        # Perfectly flat signal — no cadence.
        return False, None

    # Full autocorrelation; keep non-negative lags and normalize so lag 0 == 1.
    full = np.correlate(centered, centered, mode="full")
    acf = full[full.size // 2:] / var0  # acf[0] == 1.0

    # Lag band from the cadence band: higher frequency -> smaller lag.
    min_lag = max(1, int(np.floor(sample_rate_hz / CADENCE_MAX_HZ)))
    max_lag = min(n - 1, int(np.ceil(sample_rate_hz / CADENCE_MIN_HZ)))
    if max_lag < min_lag:
        return False, None

    band = acf[min_lag : max_lag + 1]
    if band.size == 0:
        return False, None

    best_offset = int(np.argmax(band))
    best_peak = float(band[best_offset])
    best_lag = min_lag + best_offset

    if best_peak < RHYTHMIC_PEAK_MIN or best_lag <= 0:
        return False, None

    cadence_hz = float(sample_rate_hz) / float(best_lag)
    return True, cadence_hz


def movement_label(
    acc_window,
    prev_state: MovementState | None = None,
    *,
    sample_rate_hz: float = 32.0,
) -> MovementState:
    """Label a window's accelerometer as sustained and/or rhythmic movement.

    Detects the two benign-movement cases the Movement_Check acts on (AC 8.1,
    8.2) and packages them into a :class:`~data.schema.MovementState`.

    Parameters
    ----------
    acc_window:
        The window's accelerometer samples, either raw ACC as an ``(N, 3)`` array
        of ``x, y, z`` triples or an already-1-D per-sample magnitude series (see
        :func:`acc_magnitude_series`). The ``Per_Window_Table`` ``acc_mag`` column
        is a single aggregate per window, so cadence detection needs these
        underlying samples, not that column.
    prev_state:
        The previous window's :class:`MovementState`, accepted for symmetry with
        the design signature (and to allow cross-window hysteresis later). It does
        not affect the current decision — each window is labeled on its own
        samples.
    sample_rate_hz:
        ACC sampling rate in Hz, needed to turn autocorrelation lags into a
        cadence. Defaults to the E4's 32 Hz.

    Detection
    ---------
    - **sustained** (AC 8.1): true when the robust spread (scaled MAD) of the
      magnitude exceeds :data:`SUSTAINED_MAD_COUNTS`, **or** the mean magnitude
      sits more than :data:`SUSTAINED_ELEVATION_COUNTS` above the resting 1 g
      (:data:`ACC_COUNTS_PER_G`) level. Either an agitated or a persistently
      lifted arm counts as sustained movement.
    - **rhythmic** (AC 8.2): true when a steady cadence is detected in the
      :data:`CADENCE_MIN_HZ`–:data:`CADENCE_MAX_HZ` band (normalized
      autocorrelation peak ≥ :data:`RHYTHMIC_PEAK_MIN`); ``cadence_hz`` carries
      the detected beat frequency, else ``None``.

    An empty window (no samples) is labeled still: ``MovementState(False, False,
    None)``.

    Returns
    -------
    MovementState
        ``sustained`` / ``rhythmic`` flags and the ``cadence_hz`` when rhythmic.
    """
    mag = acc_magnitude_series(acc_window)
    if mag.size == 0:
        return MovementState(sustained=False, rhythmic=False, cadence_hz=None)

    finite = mag[np.isfinite(mag)]
    if finite.size == 0:
        return MovementState(sustained=False, rhythmic=False, cadence_hz=None)

    # --- Sustained: variability OR elevation above resting 1 g (AC 8.1) ---
    spread = _scaled_mad(finite)
    elevation = float(np.mean(finite)) - ACC_COUNTS_PER_G
    sustained = (
        spread > SUSTAINED_MAD_COUNTS or elevation > SUSTAINED_ELEVATION_COUNTS
    )

    # --- Rhythmic: steady cadence in the plausible band (AC 8.2) ---
    rhythmic, cadence_hz = _detect_rhythmic(finite, sample_rate_hz)

    return MovementState(
        sustained=bool(sustained),
        rhythmic=bool(rhythmic),
        cadence_hz=cadence_hz,
    )


# ---------------------------------------------------------------------------
# Tier adjustment (AC 8.1, 8.2, 8.3; Property 20)
# ---------------------------------------------------------------------------


def apply_movement(tier: int, mv: MovementState) -> tuple[int, bool]:
    """Adjust a pre-movement tier for a benign movement cause (AC 8.1–8.3).

    Given the Flagger's pre-movement ``tier`` and the window's
    :class:`~data.schema.MovementState`:

    - When ``mv`` shows **sustained or rhythmic** movement, reduce the tier by
      **exactly one** level, floored at :data:`~data.schema.TIER_NONE` (0), and
      return ``(new_tier, True)`` — the flag is "explained by movement" (AC 8.1,
      8.2). A Watch (1) flag with movement becomes None (0); it never goes
      negative.
    - When ``mv`` shows **neither**, return ``(tier, False)`` with the tier and
      label unchanged (AC 8.3).

    The reduction is exactly one level and floored at 0 (Property 20): the
    movement label is evidence the shift is benign, not proof it is absent, so a
    Strong flag with movement is demoted to Notable rather than erased.

    Returns
    -------
    tuple[int, bool]
        ``(adjusted_tier, movement_explained)``.
    """
    if mv.sustained or mv.rhythmic:
        new_tier = max(TIER_NONE, int(tier) - 1)
        return new_tier, True
    return int(tier), False
