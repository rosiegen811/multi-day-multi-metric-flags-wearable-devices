"""FHIR exporter for the Wearable Flag Pipeline — a Tier 2/3 flag as a Bundle.

This module turns a fired :class:`~data.schema.Flag` into a FHIR ``Bundle`` — a
plain Python ``dict`` with the shape FHIR R4 expects, so it serializes to JSON
with the standard library and needs **no external FHIR dependency**. The Bundle
carries one ``Observation`` per metric (``hr``, ``rmssd``/HRV, ``skin_temp``) and
one *summary* ``Observation`` for the overall flag (design "Exporter —
export/fhir.py", Requirement 14; README "Stretch: exporting a flag").

Gate (AC 14.1)
--------------
Export is for high-severity flags only: a Bundle is emitted **only** when the
flag's tier is Tier 2 Notable or Tier 3 Strong (``tier >= TIER_NOTABLE``). A
sub-threshold flag (Tier 0/1) is **not exportable**: :func:`to_fhir_bundle`
returns ``None`` for it (it does not raise). Callers that require a Bundle can use
:func:`require_fhir_bundle`, which raises :class:`NotExportableError` instead —
the two entry points make the "return ``None``" vs "raise" choice explicit at the
call site rather than hidden in a flag.

LOINC codes (candidate — verify before clinical use)
----------------------------------------------------
The README lists *candidate* LOINC codes to verify. They are reproduced here as
candidates, not asserted as correct:

- ``hr`` — **40443-4** "Heart rate --resting" (candidate), unit ``bpm``.
- ``skin_temp`` — **39106-0** "Skin temperature" (candidate), unit ``Cel``.
- ``rmssd`` (HRV) — there is no exact resting-HRV/RMSSD LOINC in common use, so
  this Observation uses a **local code** (system ``CODE_SYSTEM_LOCAL``, code
  ``rmssd``) with a clear human display "Heart rate variability (RMSSD)" and the
  ``ms`` unit. (The README's third candidate, 9279-1, is *respiratory rate*,
  which this pipeline does not measure, so it is intentionally not used.)

Every code carries a ``display`` so a reader is never left guessing, and the
summary Observation documents that the metric codes are candidates to verify.

Non-diagnostic wording (AC 15.1, 15.2)
--------------------------------------
The summary Observation describes a *sustained change* from the person's resting
normal, never a diagnosis, and uses only the neutral tier label / wording from
:mod:`data.schema` — no color-coded severity terms. The flag explanation text
comes from the Explainer (:func:`flag.explain.explain`) so wording stays
consistent and neutral across the pipeline.

Dependencies are the standard library only (``json``) plus the sibling records in
:mod:`data.schema`, :mod:`flag.explain`, and :mod:`flag.score`. Every function is
pure over its inputs (no I/O beyond the explicit JSON string helpers, no
mutation of the inputs).

Requirements covered
--------------------
- 14.1 For a Tier 2/3 flag with export enabled, emit a FHIR Bundle with one
       Observation per metric plus one summary Observation for the flag
       (:func:`to_fhir_bundle`).
- 15.1 Flag described as a sustained change, never a diagnosis (summary wording).
- 15.2 Neutral tier wording, no color-coded ranges (summary wording).
"""

from __future__ import annotations

import json
from typing import Mapping

from data.schema import (
    Flag,
    SignalBaseline,
    TIER_NOTABLE,
    tier_label,
    tier_wording,
)
from flag.explain import Explanation, explain
from flag.score import SCORED_SIGNALS

__all__ = [
    "CODE_SYSTEM_LOINC",
    "CODE_SYSTEM_LOCAL",
    "METRIC_CODES",
    "SUMMARY_CODE",
    "NotExportableError",
    "is_exportable",
    "metric_observation",
    "summary_observation",
    "to_fhir_bundle",
    "require_fhir_bundle",
    "bundle_to_json",
]


# ---------------------------------------------------------------------------
# Coding systems and per-metric codes
# ---------------------------------------------------------------------------

#: The LOINC code system URI (FHIR canonical).
CODE_SYSTEM_LOINC: str = "http://loinc.org"

#: A local code system for metrics without a suitable standard code (RMSSD/HRV).
#: Local — not a registered terminology; the ``display`` carries the meaning.
CODE_SYSTEM_LOCAL: str = "https://wearable-flag-pipeline/local-codes"

#: FHIR Observation units are UCUM (http://unitsofmeasure.org).
UCUM_SYSTEM: str = "http://unitsofmeasure.org"

# Per-metric coding. Each entry: the coding ``system``, ``code``, human
# ``display``, the UCUM ``unit`` code, and whether the code is a *candidate*
# LOINC to verify (True) or a local stand-in (False). See the module docstring
# for the provenance of each code.
METRIC_CODES: Mapping[str, Mapping[str, object]] = {
    "hr": {
        "system": CODE_SYSTEM_LOINC,
        "code": "40443-4",
        "display": "Heart rate --resting",
        "unit": "bpm",
        "candidate": True,
    },
    "rmssd": {
        # No exact resting-HRV/RMSSD LOINC in common use: use a clear local code.
        "system": CODE_SYSTEM_LOCAL,
        "code": "rmssd",
        "display": "Heart rate variability (RMSSD)",
        "unit": "ms",
        "candidate": False,
    },
    "skin_temp": {
        "system": CODE_SYSTEM_LOINC,
        "code": "39106-0",
        "display": "Skin temperature",
        "unit": "Cel",
        "candidate": True,
    },
}

# The summary Observation's own code (a local code: this is a pipeline-specific
# composite flag, not a standard lab observation).
SUMMARY_CODE: Mapping[str, object] = {
    "system": CODE_SYSTEM_LOCAL,
    "code": "sustained-change-flag",
    "display": "Sustained multi-metric change flag",
}


class NotExportableError(ValueError):
    """Raised by :func:`require_fhir_bundle` for a sub-threshold (Tier 0/1) flag.

    Export is gated on Tier 2 Notable or Tier 3 Strong (AC 14.1); a lower tier is
    not exportable. :func:`to_fhir_bundle` signals this by returning ``None``;
    :func:`require_fhir_bundle` signals it by raising this error.
    """


# ---------------------------------------------------------------------------
# Export gate (AC 14.1)
# ---------------------------------------------------------------------------


def is_exportable(flag: Flag) -> bool:
    """Return ``True`` iff ``flag`` is Tier 2 Notable or higher (AC 14.1)."""
    return int(flag.tier) >= TIER_NOTABLE


# ---------------------------------------------------------------------------
# Observation builders
# ---------------------------------------------------------------------------


def _coding(system: str, code: str, display: str) -> dict:
    """A FHIR ``CodeableConcept`` with a single coding plus a text fallback."""
    return {
        "coding": [{"system": system, "code": code, "display": display}],
        "text": display,
    }


def _quantity(value: float | None, unit: str) -> dict:
    """A FHIR ``Quantity`` in UCUM. A non-finite/absent value omits ``value``."""
    q: dict = {"unit": unit, "system": UCUM_SYSTEM, "code": unit}
    if value is not None:
        v = float(value)
        if v == v and v not in (float("inf"), float("-inf")):  # finite check
            q["value"] = v
    return q


def metric_observation(
    signal: str,
    value: float | None,
    baseline: SignalBaseline | None,
    *,
    person_id: str,
    effective_t: float,
) -> dict:
    """Build one metric ``Observation`` for ``signal`` (part of AC 14.1).

    The Observation carries the metric's real-unit ``value`` as its
    ``valueQuantity`` and the participant's rest baseline center
    (``baseline.median``) as a ``referenceRange`` (the reference the shift is
    measured against). ``signal`` must be one of :data:`~flag.score.SCORED_SIGNALS`.
    """
    spec = METRIC_CODES.get(signal)
    if spec is None:
        raise ValueError(
            f"unknown metric {signal!r}; expected one of {list(METRIC_CODES)}"
        )

    unit = str(spec["unit"])
    obs: dict = {
        "resourceType": "Observation",
        "status": "final",
        "category": [
            _coding(
                "http://terminology.hl7.org/CodeSystem/observation-category",
                "vital-signs",
                "Vital Signs",
            )
        ],
        "code": _coding(str(spec["system"]), str(spec["code"]), str(spec["display"])),
        "subject": {"reference": f"Patient/{person_id}"},
        "effectiveDateTime": _effective(effective_t),
        "valueQuantity": _quantity(value, unit),
    }

    if spec.get("candidate"):
        obs.setdefault("_code_note", "candidate LOINC code; verify before clinical use")

    if baseline is not None:
        obs["referenceRange"] = [
            {
                "text": "participant resting baseline (median)",
                "low": _quantity(float(baseline.median), unit),
                "high": _quantity(float(baseline.median), unit),
            }
        ]

    return obs


def summary_observation(
    flag: Flag,
    explanation: Explanation,
    *,
    members: int,
) -> dict:
    """Build the one *summary* ``Observation`` for the overall flag (part of AC 14.1).

    Carries the assigned tier (neutral label + wording, never a diagnosis or a
    color-coded term — AC 15.1/15.2), the Combined_Score, the seconds the flag has
    persisted, whether movement explains it, and the plain-language explanation
    text. ``members`` is the count of per-metric Observations it summarizes (used
    only to size the ``hasMember`` references, filled by :func:`to_fhir_bundle`).
    """
    tier = int(flag.tier)
    components = [
        _component("combined-score", "Combined deviation score", float(flag.combined_score), "1"),
        _component("persisted-seconds", "Seconds flag has persisted", float(flag.persisted_s), "s"),
        _component_int("severity-tier", "Severity tier (0-3, neutral)", tier),
    ]

    obs: dict = {
        "resourceType": "Observation",
        "status": "final",
        "code": _coding(
            str(SUMMARY_CODE["system"]),
            str(SUMMARY_CODE["code"]),
            str(SUMMARY_CODE["display"]),
        ),
        "subject": {"reference": f"Patient/{flag.person_id}"},
        "effectiveDateTime": _effective(float(flag.start_t)),
        # Neutral tier label as the coded value; no color-coded terms (AC 15.2).
        "valueCodeableConcept": _coding(
            CODE_SYSTEM_LOCAL, f"tier-{tier}", tier_label(tier)
        ),
        "interpretation": [
            _coding(CODE_SYSTEM_LOCAL, "sustained-change", tier_wording(tier))
        ],
        "component": components,
        # Plain-language, non-diagnostic explanation (AC 15.1).
        "note": [{"text": explanation.text}],
        "_disclaimer": (
            "Sustained change from the participant's resting normal; not a "
            "diagnosis. Metric codes are candidate codes to verify."
        ),
        "_movement_explained": bool(flag.movement_explained),
    }
    return obs


def _component(code: str, display: str, value: float, unit: str) -> dict:
    """An Observation ``component`` carrying a numeric quantity."""
    return {
        "code": _coding(CODE_SYSTEM_LOCAL, code, display),
        "valueQuantity": _quantity(value, unit),
    }


def _component_int(code: str, display: str, value: int) -> dict:
    """An Observation ``component`` carrying an integer quantity."""
    return {
        "code": _coding(CODE_SYSTEM_LOCAL, code, display),
        "valueQuantity": {"value": int(value), "unit": "1", "system": UCUM_SYSTEM, "code": "1"},
    }


def _effective(t: float) -> str:
    """Render a window time (seconds) as a relative FHIR-ish marker.

    The pipeline works in seconds-from-session-start, not wall-clock, so this
    keeps the numeric offset explicit rather than fabricating a calendar
    timestamp. Stored as a string so the Bundle re-parses from JSON unchanged.
    """
    return f"PT{float(t):g}S"


# ---------------------------------------------------------------------------
# Bundle assembly (AC 14.1)
# ---------------------------------------------------------------------------


def to_fhir_bundle(
    flag: Flag,
    values: Mapping[str, float],
    baselines: Mapping[str, SignalBaseline],
    *,
    contributions: Mapping[str, float] | None = None,
    explanation: Explanation | None = None,
) -> dict | None:
    """Emit a FHIR ``Bundle`` for a Tier 2/3 ``flag``, else ``None`` (AC 14.1).

    When the flag is below Tier 2 Notable this returns ``None`` (the flag is not
    exportable); use :func:`require_fhir_bundle` to raise instead.

    The Bundle is a ``collection`` of ``Observation`` entries: one per metric in
    :data:`~flag.score.SCORED_SIGNALS` that has a value and/or a baseline, plus one
    summary Observation for the overall flag. The summary lists the metric
    Observations in its ``hasMember`` references.

    Parameters
    ----------
    flag:
        The fired :class:`~data.schema.Flag`.
    values:
        ``{signal -> raw real-unit value}`` for the flagged window (``hr`` bpm,
        ``rmssd`` ms, ``skin_temp`` C).
    baselines:
        ``{signal -> SignalBaseline}`` for the participant; ``median`` is the rest
        reference carried as each metric Observation's reference range.
    contributions:
        Optional ``{signal -> signed z}`` for the flagged window. When given (and
        no ``explanation`` is supplied) it drives the Explainer so the summary
        note's ranking matches the rest of the pipeline. Defaults to an empty
        mapping, which yields a tier-wording-only explanation.
    explanation:
        Optional precomputed :class:`~flag.explain.Explanation`. When supplied it
        is used verbatim and the Explainer is not re-run.

    Returns
    -------
    dict | None
        The Bundle as a JSON-serializable ``dict``, or ``None`` for a
        sub-threshold flag.
    """
    if not is_exportable(flag):
        return None

    expl = explanation if explanation is not None else explain(
        flag,
        dict(contributions or {}),
        baselines,
        values=values,
    )

    # One Observation per metric that has a value or a baseline, in canonical order.
    metric_obs: list[dict] = []
    for signal in SCORED_SIGNALS:
        value = values.get(signal) if values is not None else None
        baseline = baselines.get(signal) if baselines is not None else None
        if value is None and baseline is None:
            continue
        metric_obs.append(
            metric_observation(
                signal,
                value,
                baseline,
                person_id=str(flag.person_id),
                effective_t=float(flag.start_t),
            )
        )

    summary = summary_observation(flag, expl, members=len(metric_obs))

    # Give each entry a stable fullUrl and wire the summary's hasMember list.
    entries: list[dict] = []
    member_refs: list[dict] = []
    for i, obs in enumerate(metric_obs):
        full_url = f"urn:observation:metric:{i}"
        entries.append({"fullUrl": full_url, "resource": obs})
        member_refs.append({"reference": full_url})

    summary["hasMember"] = member_refs
    entries.append({"fullUrl": "urn:observation:summary", "resource": summary})

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries,
    }
    return bundle


def require_fhir_bundle(
    flag: Flag,
    values: Mapping[str, float],
    baselines: Mapping[str, SignalBaseline],
    *,
    contributions: Mapping[str, float] | None = None,
    explanation: Explanation | None = None,
) -> dict:
    """Like :func:`to_fhir_bundle` but raise :class:`NotExportableError` if Tier < 2.

    Use this at call sites that require a Bundle and want a sub-threshold flag to
    fail loudly rather than return ``None``.
    """
    bundle = to_fhir_bundle(
        flag,
        values,
        baselines,
        contributions=contributions,
        explanation=explanation,
    )
    if bundle is None:
        raise NotExportableError(
            f"flag for {flag.person_id!r} is Tier {int(flag.tier)} "
            f"(< Tier {TIER_NOTABLE} Notable): not exportable to FHIR"
        )
    return bundle


def bundle_to_json(bundle: dict, *, indent: int | None = 2) -> str:
    """Serialize a Bundle ``dict`` to a JSON string with the standard library.

    ``indent`` controls pretty-printing (default 2; pass ``None`` for compact).
    The result re-parses with :func:`json.loads` to an equal structure.
    """
    return json.dumps(bundle, indent=indent, sort_keys=False, ensure_ascii=False)
