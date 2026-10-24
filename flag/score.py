"""Scorer for the Wearable Flag Pipeline — signed z-scores and combined scores.

This module turns a single ``Per_Window_Table`` row into a per-signal *signed*
z-score against that person's resting baseline, then combines those z-scores into
one deviation value per window. "Combining" is the point of the whole pipeline:
several small shifts across heart rate, heart-rate variability and skin
temperature, each too small to notice alone, add up here into one larger signal
(design "Scorer — flag/score.py", Requirement 6).

It consumes the :class:`~data.schema.SignalBaseline` records produced by the
Baseline_Estimator (``flag/baseline.py``) — ``median`` (robust center), ``mad``
(scaled MAD spread) and ``shrunk_mad`` (empirical-Bayes-shrunk spread) — and the
three *scored* signals :data:`~flag.baseline.SCORED_SIGNALS` = ``("hr", "rmssd",
"skin_temp")``. The accelerometer magnitude is not scored here; it drives the
movement check (Task 9), not the deviation score.

Dependencies are numpy / pandas / scikit-learn (``sklearn.covariance.LedoitWolf``)
plus sibling records. Every function is pure over its inputs (no I/O, no mutation
of the passed row or baselines).

Sign convention (AC 6.1–6.3)
----------------------------
Each signal is signed so that *more physiological stress* reads as a more
*positive* z-score, which lets :func:`directional_sum` add them coherently:

- ``hr`` — an **increase above** baseline is **positive** (sign ``+1`` on the raw
  deviation ``value - median``). AC 6.1.
- ``rmssd`` — a **decrease below** baseline is **positive** (sign ``-1``, i.e. the
  raw deviation is negated): HRV typically *drops* under stress. AC 6.2.
- ``skin_temp`` — signed by the **direction of deviation** from baseline: positive
  above, negative below (the raw signed deviation, sign ``+1``). Skin temperature
  is genuinely bidirectional — it tends to *fall* under acute stress
  (vasoconstriction) yet *rises* with fever — so the design keeps its natural
  signed direction rather than forcing one polarity. AC 6.3.

Magnitude (AC 6.4)
------------------
The magnitude of every signed z equals ``|value - median| / spread`` — the
baseline-centered value divided by the scaled MAD spread. The design's z-score,
``z = s_m * (x - median) / (1.4826 * MAD)`` with ``s_m`` the sign above, is
realized here by dividing directly by the baseline ``spread``: the ``1.4826 *
MAD`` scaling is already baked into ``shrunk_mad`` / ``mad`` by
``flag/baseline.py`` (``MAD_SCALE = 1.4826``), so it must not be applied twice.

Combined scores (AC 6.5–6.6)
----------------------------
- :func:`directional_sum` — the simple, explainable baseline: the plain sum of the
  signed z-scores across the available signals. Permutation-invariant (Property
  12).
- :func:`mahalanobis_score` — a Mahalanobis distance in z-space using a
  **Ledoit–Wolf shrinkage covariance** (``sklearn.covariance.LedoitWolf``), which
  accounts for correlation between signals while staying well-conditioned for the
  short baselines this pipeline sees. Non-negative, and zero only at the baseline
  center (``z = 0``) (Property 13).

Unavailable signals (AC 6.7)
----------------------------
A window may be missing a signal (``NaN`` — e.g. S01 anaerobic RMSSD, AC 2.6), or
a signal may lack a baseline. :func:`combined_score` computes the score from
*exactly the available signals* for that window — an unavailable signal is simply
dropped from the combination — and always returns a finite value (Property 14).

Requirements covered
--------------------
- 6.1  HR increase above baseline → positive signed z (:func:`signed_z`).
- 6.2  RMSSD decrease below baseline → positive signed z (:func:`signed_z`).
- 6.3  Skin_Temperature signed by direction of deviation (:func:`signed_z`).
- 6.4  Signed z magnitude = baseline-centered value / scaled-MAD spread
       (:func:`signed_z`).
- 6.5  Directional-sum Combined_Score = sum of signed z-scores (:func:`directional_sum`).
- 6.6  Mahalanobis Combined_Score with Ledoit–Wolf covariance (:func:`mahalanobis_score`).
- 6.7  Combined_Score over the available signals only (:func:`combined_score`).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from data.schema import SignalBaseline
from flag.baseline import MAD_FLOOR, SCORED_SIGNALS

__all__ = [
    "SCORED_SIGNALS",
    "SIGNAL_SIGN",
    "METHOD_DIRECTIONAL_SUM",
    "METHOD_MAHALANOBIS",
    "signed_z",
    "directional_sum",
    "fit_ledoit_wolf",
    "mahalanobis_score",
    "window_signed_z",
    "combined_score",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sign applied to the raw deviation ``value - median`` for each scored signal so
# that "more stress" reads positive (AC 6.1–6.3). HR up is +; RMSSD down is + (so
# its raw deviation is negated); skin temperature keeps its natural signed
# direction (bidirectional — falls under acute stress, rises with fever).
SIGNAL_SIGN: Mapping[str, int] = {
    "hr": +1,
    "rmssd": -1,
    "skin_temp": +1,
}

# Combined_Score method selectors for :func:`combined_score` (AC 6.5, 6.6).
METHOD_DIRECTIONAL_SUM: str = "directional_sum"
METHOD_MAHALANOBIS: str = "mahalanobis"


# ---------------------------------------------------------------------------
# Signed z-score (AC 6.1–6.4)
# ---------------------------------------------------------------------------


def _baseline_spread(b: SignalBaseline) -> float:
    """Return the usable spread for ``b``: ``shrunk_mad`` if positive, else ``mad``.

    Downstream z-scoring uses the empirical-Bayes-shrunk spread (``shrunk_mad``);
    this falls back to the raw ``mad`` when ``shrunk_mad`` is missing / non-finite
    / non-positive, and finally to :data:`~flag.baseline.MAD_FLOOR` so the divide
    never produces a non-finite z (design Error Handling: zero-MAD baselines are
    guarded with a small floor so z-scores stay finite).
    """
    shrunk = float(b.shrunk_mad)
    if np.isfinite(shrunk) and shrunk > 0.0:
        return shrunk
    raw = float(b.mad)
    if np.isfinite(raw) and raw > 0.0:
        return raw
    return MAD_FLOOR


def signed_z(
    value: float,
    b: SignalBaseline,
    signal: str,
    direction: int | None = None,
) -> float:
    """Signed z-score of ``value`` against baseline ``b`` for ``signal`` (AC 6.1–6.4).

    The z-score is ``sign * (value - b.median) / spread`` where ``spread`` is the
    baseline's shrunk MAD (``b.shrunk_mad``, falling back to ``b.mad`` then
    :data:`~flag.baseline.MAD_FLOOR`; the ``1.4826`` scaling is already baked into
    those spreads by ``flag/baseline.py``). The magnitude therefore equals
    ``|value - median| / spread`` (AC 6.4), and ``sign`` orients the signal so
    more physiological stress is more positive (AC 6.1–6.3):

    - ``hr``: ``sign = +1`` — an increase above baseline is positive (AC 6.1).
    - ``rmssd``: ``sign = -1`` — a decrease below baseline is positive (AC 6.2).
    - ``skin_temp``: ``sign = +1`` — the raw signed direction of deviation,
      positive above and negative below baseline (AC 6.3). Temperature is
      bidirectional: it can fall under acute stress or rise with fever, so its
      natural signed direction is preserved rather than forced one way.

    Parameters
    ----------
    value:
        The window's signal value.
    b:
        The participant's :class:`~data.schema.SignalBaseline` for ``signal``.
    signal:
        One of :data:`~flag.baseline.SCORED_SIGNALS`; selects the default sign.
    direction:
        Optional explicit sign override (``+1`` / ``-1``). When ``None`` the sign
        is taken from :data:`SIGNAL_SIGN` for ``signal``. An unknown ``signal``
        with no override defaults to ``+1`` (raw signed deviation).

    Returns
    -------
    float
        The signed z-score, or ``nan`` when ``value`` or the baseline center is
        non-finite (an unavailable cell — the caller drops it, AC 6.7).
    """
    x = float(value)
    center = float(b.median)
    if not np.isfinite(x) or not np.isfinite(center):
        return float("nan")

    if direction is not None:
        sign = 1.0 if direction >= 0 else -1.0
    else:
        sign = float(SIGNAL_SIGN.get(signal, +1))

    spread = _baseline_spread(b)
    return sign * (x - center) / spread


# ---------------------------------------------------------------------------
# Directional-sum Combined_Score (AC 6.5)
# ---------------------------------------------------------------------------


def directional_sum(z: dict[str, float]) -> float:
    """Directional-sum Combined_Score: the sum of signed z-scores (AC 6.5).

    ``z`` maps ``signal_name -> signed z-score``. The result is the plain sum of
    the finite entries, so it is **permutation-invariant** — reordering the
    signals does not change the sum (Property 12). Non-finite entries (an
    unavailable signal that slipped through) are skipped so the sum stays finite;
    an empty mapping sums to ``0.0``.
    """
    total = 0.0
    for v in z.values():
        fv = float(v)
        if np.isfinite(fv):
            total += fv
    return total


# ---------------------------------------------------------------------------
# Mahalanobis Combined_Score with Ledoit–Wolf shrinkage covariance (AC 6.6)
# ---------------------------------------------------------------------------


def fit_ledoit_wolf(z_matrix: np.ndarray) -> LedoitWolf:
    """Fit a :class:`~sklearn.covariance.LedoitWolf` covariance on ``z_matrix``.

    ``z_matrix`` is a ``(n_windows, n_signals)`` array of signed z-vectors (one
    row per window). A Ledoit–Wolf estimator shrinks the sample covariance toward
    a structured target, staying well-conditioned even for the short baselines
    this pipeline sees (design "Combined scoring"). The returned fitted estimator
    is what :func:`mahalanobis_score` and :func:`combined_score` accept as ``cov``.

    ``assume_centered=True`` is used because the z-vectors are already
    baseline-centered: the Mahalanobis center is the origin ``0``, not the sample
    mean (AC 6.6 / Property 13 — zero only at the baseline center).
    """
    data = np.asarray(z_matrix, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return LedoitWolf(assume_centered=True).fit(data)


def _precision_from_cov(cov, n_dims: int) -> np.ndarray:
    """Resolve ``cov`` to an ``(n_dims, n_dims)`` precision (inverse-covariance).

    ``cov`` may be:

    - a fitted :class:`~sklearn.covariance.LedoitWolf` (its ``precision_`` /
      ``covariance_`` is used),
    - a square covariance matrix (``numpy`` array — inverted here), or
    - ``None`` (falls back to the identity, i.e. a plain Euclidean norm in
      z-space, which is still a valid non-negative distance).

    The matrix dimension must match ``n_dims`` (the number of available signals);
    a mismatch raises ``ValueError`` so a caller that restricted the z-vector to a
    subset of signals must pass a covariance over that same subset (AC 6.7).
    """
    if cov is None:
        return np.eye(n_dims)

    precision = getattr(cov, "precision_", None)
    if precision is not None:
        precision = np.asarray(precision, dtype=float)
    else:
        covariance = getattr(cov, "covariance_", None)
        if covariance is not None:
            matrix = np.asarray(covariance, dtype=float)
        else:
            matrix = np.asarray(cov, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "mahalanobis covariance must be a square matrix or a fitted "
                f"LedoitWolf, got shape {matrix.shape}"
            )
        precision = np.linalg.pinv(matrix)

    if precision.shape != (n_dims, n_dims):
        raise ValueError(
            f"covariance/precision has shape {precision.shape} but the z-vector "
            f"has {n_dims} available signal(s); the covariance must cover exactly "
            f"the available signals (AC 6.7)"
        )
    return precision


def mahalanobis_score(z_vec: np.ndarray, cov: LedoitWolf) -> float:
    """Mahalanobis Combined_Score of ``z_vec`` in z-space (AC 6.6).

    The distance is ``sqrt(z^T P z)`` where ``P`` is the precision (inverse
    covariance) resolved from ``cov`` via :func:`_precision_from_cov`. Because the
    z-vector is already baseline-centered, the Mahalanobis *center* is the origin
    ``0`` — so the score is ``0`` exactly when ``z_vec`` is the zero vector (the
    baseline center) and strictly positive otherwise (Property 13).

    ``cov`` is a fitted :class:`~sklearn.covariance.LedoitWolf` (see
    :func:`fit_ledoit_wolf`), a square covariance matrix, or ``None`` for a plain
    Euclidean norm. The distance is non-negative and finite; the quadratic form is
    clamped at ``0`` before the square root to absorb any tiny negative value from
    floating-point round-off on a near-singular precision.
    """
    z = np.asarray(z_vec, dtype=float).ravel()
    n = z.size
    if n == 0:
        return 0.0
    precision = _precision_from_cov(cov, n)
    quad = float(z @ precision @ z)
    if not np.isfinite(quad) or quad <= 0.0:
        return 0.0
    return float(np.sqrt(quad))


# ---------------------------------------------------------------------------
# Per-window combination over the available signals (AC 6.7)
# ---------------------------------------------------------------------------


def _row_value(window_row, signal: str) -> float:
    """Read ``signal`` from a pandas Series / mapping row as a float (``nan`` if absent)."""
    try:
        value = window_row[signal]
    except (KeyError, IndexError, TypeError):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def window_signed_z(
    window_row,
    baselines: Mapping[str, SignalBaseline],
) -> dict[str, float]:
    """Signed z-scores for the **available** scored signals of one window (AC 6.7).

    Iterates the scored signals (:data:`~flag.baseline.SCORED_SIGNALS`) and
    includes ``signal`` only when it has both a baseline in ``baselines`` and a
    finite value in ``window_row`` that yields a finite signed z. A signal that is
    ``NaN`` for the window, missing from the row, or missing a baseline is simply
    omitted — so the returned mapping names exactly the available signals for this
    window. ``window_row`` may be a pandas ``Series`` or any mapping with the
    signal keys; it is only read, never mutated.
    """
    z: dict[str, float] = {}
    for signal in SCORED_SIGNALS:
        b = baselines.get(signal)
        if b is None:
            continue
        value = _row_value(window_row, signal)
        if not np.isfinite(value):
            continue
        zi = signed_z(value, b, signal)
        if np.isfinite(zi):
            z[signal] = zi
    return z


def combined_score(
    window_row,
    baselines: Mapping[str, SignalBaseline],
    method: str = METHOD_DIRECTIONAL_SUM,
    *,
    cov: LedoitWolf | None = None,
) -> float:
    """Combined_Score for one window, over the available signals only (AC 6.5–6.7).

    Computes the per-signal signed z for each *available* scored signal of
    ``window_row`` via :func:`window_signed_z` (an unavailable signal — ``NaN`` in
    the row or no baseline — is dropped from the combination for that window, AC
    6.7) and combines them by ``method``:

    - ``"directional_sum"`` (:data:`METHOD_DIRECTIONAL_SUM`) — the sum of the
      signed z-scores (AC 6.5).
    - ``"mahalanobis"`` (:data:`METHOD_MAHALANOBIS`) — the Mahalanobis distance in
      z-space using the Ledoit–Wolf ``cov`` (AC 6.6). The z-vector is ordered by
      :data:`~flag.baseline.SCORED_SIGNALS` over the available signals; ``cov``
      must cover exactly those dimensions (or be ``None`` for a Euclidean norm).

    The result is always finite (Property 14): an empty window (no available
    signals) yields ``0.0``.

    Parameters
    ----------
    window_row:
        A pandas ``Series`` or mapping with ``hr`` / ``rmssd`` / ``skin_temp``
        keys (typically one ``Per_Window_Table`` row).
    baselines:
        The participant's ``{signal -> SignalBaseline}`` from
        :func:`~flag.baseline.person_baseline` /
        :func:`~flag.baseline.shrink_to_cohort`.
    method:
        ``"directional_sum"`` or ``"mahalanobis"``.
    cov:
        Fitted :class:`~sklearn.covariance.LedoitWolf` (or covariance matrix) for
        the Mahalanobis method; ignored by the directional sum.

    Raises
    ------
    ValueError
        If ``method`` is not a recognized Combined_Score method.
    """
    z = window_signed_z(window_row, baselines)

    if method == METHOD_DIRECTIONAL_SUM:
        return directional_sum(z)

    if method == METHOD_MAHALANOBIS:
        if not z:
            return 0.0
        # Order the z-vector by the canonical scored-signal order so it matches a
        # covariance fit over the same available-signal ordering.
        ordered = [z[s] for s in SCORED_SIGNALS if s in z]
        return mahalanobis_score(np.asarray(ordered, dtype=float), cov)

    raise ValueError(
        f"unknown combined_score method {method!r}; expected "
        f"{METHOD_DIRECTIONAL_SUM!r} or {METHOD_MAHALANOBIS!r}"
    )
