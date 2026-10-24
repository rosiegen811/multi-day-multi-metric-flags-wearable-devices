"""CMS care-update exporter for the Wearable Flag Pipeline — a Tier 2/3 flag.

This module turns a fired :class:`~data.schema.Flag` into a CMS-style *care
update* document — a structured Python ``dict`` (JSON-serializable) following the
four-section CMS care-update template (design "Exporter — export/care_update.py",
Requirement 14; README "Stretch: exporting a flag"):

1. **Clinical Measures** — baseline vs most-recent value per metric (AC 14.2).
2. **Other relevant clinical data** — the plain-language flag explanation (AC 14.2).
3. **Assessment** — left **unpopulated** for a clinician (AC 14.3).
4. **Plan** — left **unpopulated** for a clinician (AC 14.3).

Gate (AC 14.2)
--------------
Export is for high-severity flags only: a care update is emitted **only** when the
flag is Tier 2 Notable or Tier 3 Strong (``tier >= TIER_NOTABLE``). A sub-threshold
flag (Tier 0/1) is **not exportable**: :func:`to_care_update` returns ``None`` for
it (it does not raise), mirroring :func:`export.fhir.to_fhir_bundle`. Callers that
require a document can use :func:`require_care_update`, which raises
:class:`~export.fhir.NotExportableError` instead.

Empty clinician sections (AC 14.3)
----------------------------------
The Assessment and Plan sections are always present in the document but always
**empty** — ``assessment`` and ``plan`` are the empty string ``""``. This module
never writes into them; they are the clinician's to fill. :func:`is_unpopulated`
is the predicate the care-update's "leave clinician sections empty" property
checks against.

Non-diagnostic wording (AC 15.1, 15.2)
--------------------------------------
"Other relevant clinical data" carries the Explainer's text
(:func:`flag.explain.explain`), which frames the flag as a *sustained change* from
the resting normal — never a diagnosis — using only neutral tier wording (no
color-coded severity terms). Because this module leaves Assessment/Plan empty, it
adds no clinical interpretation of its own.

Dependencies are the standard library only (``json``) plus the sibling records in
:mod:`data.schema`, :mod:`flag.explain`, :mod:`flag.score`, and the shared export
gate in :mod:`export.fhir`. Every function is pure over its inputs.

Requirements covered
--------------------
- 14.2 For a Tier 2/3 flag with export enabled, emit a CMS care-update populated
       with baseline and most-recent clinical measures and the flag explanation
       (:func:`to_care_update`).
- 14.3 Leave the care-update Assessment and Plan sections unpopulated
       (:func:`to_care_update`, :func:`is_unpopulated`).
- 15.1 Flag described as a sustained change, never a diagnosis (explanation text).
- 15.2 Neutral tier wording, no color-coded ranges.
"""

from __future__ import annotations

import json
from typing import Mapping

from data.schema import Flag, SignalBaseline, TIER_NOTABLE, tier_label, tier_wording
from export.fhir import NotExportableError, is_exportable
from flag.explain import (
    Explanation,
    explain,
    signal_display_name,
    signal_unit,
)
from flag.score import SCORED_SIGNALS

__all__ = [
    "to_care_update",
    "require_care_update",
    "clinical_measures",
    "is_unpopulated",
    "care_update_to_json",
    "render_text",
]


# ---------------------------------------------------------------------------
# Clinical Measures (AC 14.2)
# ---------------------------------------------------------------------------


def clinical_measures(
    values: Mapping[str, float],
    baselines: Mapping[str, SignalBaseline],
) -> list[dict]:
    """Build the Clinical Measures: baseline vs most-recent value per metric (AC 14.2).

    Returns one entry per scored metric (:data:`~flag.score.SCORED_SIGNALS`) that
    has a most-recent value and/or a baseline, in canonical signal order. Each
    entry carries the metric's display name and unit, its ``baseline`` (the rest
    reference ``SignalBaseline.median``), its ``most_recent`` value for the flagged
    window, and the real-unit ``delta`` (most-recent minus baseline), with ``None``
    for any part that is unavailable.
    """
    rows: list[dict] = []
    for signal in SCORED_SIGNALS:
        value = values.get(signal) if values is not None else None
        baseline = baselines.get(signal) if baselines is not None else None
        if value is None and baseline is None:
            continue

        recent = _finite_or_none(value)
        base = _finite_or_none(baseline.median) if baseline is not None else None
        delta = (recent - base) if (recent is not None and base is not None) else None

        rows.append(
            {
                "signal": signal,
                "measure": signal_display_name(signal),
                "unit": signal_unit(signal),
                "baseline": base,
                "most_recent": recent,
                "delta": delta,
            }
        )
    return rows


def _finite_or_none(value) -> float | None:
    """Return ``float(value)`` when finite, else ``None``."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


# ---------------------------------------------------------------------------
# Care-update document (AC 14.2, 14.3)
# ---------------------------------------------------------------------------


def to_care_update(
    flag: Flag,
    values: Mapping[str, float],
    baselines: Mapping[str, SignalBaseline],
    *,
    contributions: Mapping[str, float] | None = None,
    explanation: Explanation | None = None,
) -> dict | None:
    """Emit a CMS care-update for a Tier 2/3 ``flag``, else ``None`` (AC 14.2, 14.3).

    When the flag is below Tier 2 Notable this returns ``None`` (not exportable);
    use :func:`require_care_update` to raise instead.

    The returned ``dict`` has the four CMS sections. ``clinical_measures`` holds
    baseline-vs-most-recent per metric (AC 14.2); ``other_relevant_clinical_data``
    holds the flag explanation text (AC 14.2); ``assessment`` and ``plan`` are the
    empty string, left for a clinician (AC 14.3).

    Parameters
    ----------
    flag:
        The fired :class:`~data.schema.Flag`.
    values:
        ``{signal -> raw real-unit value}`` for the flagged window.
    baselines:
        ``{signal -> SignalBaseline}``; ``median`` is the rest baseline reference.
    contributions:
        Optional ``{signal -> signed z}`` for the flagged window; drives the
        Explainer when no ``explanation`` is supplied, so wording matches the rest
        of the pipeline.
    explanation:
        Optional precomputed :class:`~flag.explain.Explanation`, used verbatim.

    Returns
    -------
    dict | None
        The care-update document, or ``None`` for a sub-threshold flag.
    """
    if not is_exportable(flag):
        return None

    expl = explanation if explanation is not None else explain(
        flag,
        dict(contributions or {}),
        baselines,
        values=values,
    )

    tier = int(flag.tier)
    document = {
        "document_type": "CMS care update",
        "subject": {"person_id": str(flag.person_id)},
        "flag": {
            "severity_tier": tier,
            "severity_label": tier_label(tier),
            "severity_wording": tier_wording(tier),
            "combined_score": float(flag.combined_score),
            "start_t": float(flag.start_t),
            "persisted_s": float(flag.persisted_s),
            "movement_explained": bool(flag.movement_explained),
        },
        # Section 1 — Clinical Measures (AC 14.2).
        "clinical_measures": clinical_measures(values, baselines),
        # Section 2 — Other relevant clinical data: the flag explanation (AC 14.2).
        "other_relevant_clinical_data": expl.text,
        # Section 3 — Assessment: left unpopulated for a clinician (AC 14.3).
        "assessment": "",
        # Section 4 — Plan: left unpopulated for a clinician (AC 14.3).
        "plan": "",
        "_disclaimer": (
            "Sustained change from the participant's resting normal; not a "
            "diagnosis. Assessment and Plan are left for a clinician."
        ),
    }
    return document


def require_care_update(
    flag: Flag,
    values: Mapping[str, float],
    baselines: Mapping[str, SignalBaseline],
    *,
    contributions: Mapping[str, float] | None = None,
    explanation: Explanation | None = None,
) -> dict:
    """Like :func:`to_care_update` but raise :class:`~export.fhir.NotExportableError` if Tier < 2."""
    document = to_care_update(
        flag,
        values,
        baselines,
        contributions=contributions,
        explanation=explanation,
    )
    if document is None:
        raise NotExportableError(
            f"flag for {flag.person_id!r} is Tier {int(flag.tier)} "
            f"(< Tier {TIER_NOTABLE} Notable): not exportable to a care update"
        )
    return document


# ---------------------------------------------------------------------------
# Clinician-section predicate (AC 14.3)
# ---------------------------------------------------------------------------


def is_unpopulated(document: Mapping[str, object]) -> bool:
    """Return ``True`` iff the care-update's Assessment and Plan are both empty (AC 14.3).

    "Empty" means absent or an all-whitespace string. This is the predicate the
    "leave clinician sections empty" property checks.
    """
    for section in ("assessment", "plan"):
        value = document.get(section, "")
        if value is None:
            continue
        if str(value).strip() != "":
            return False
    return True


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def care_update_to_json(document: dict, *, indent: int | None = 2) -> str:
    """Serialize a care-update ``dict`` to a JSON string with the standard library."""
    return json.dumps(document, indent=indent, sort_keys=False, ensure_ascii=False)


def render_text(document: Mapping[str, object]) -> str:
    """Render the care-update as a plain-text, four-section clinician document.

    The Assessment and Plan sections appear with their headings but no content,
    so the clinician sees exactly where to write (AC 14.3).
    """
    subject = document.get("subject", {}) or {}
    flag = document.get("flag", {}) or {}
    person_id = subject.get("person_id", "") if isinstance(subject, Mapping) else ""

    lines: list[str] = []
    lines.append("CMS CARE UPDATE")
    lines.append(f"Participant: {person_id}")
    if isinstance(flag, Mapping):
        lines.append(
            f"Flag: {flag.get('severity_label', '')} "
            f"(tier {flag.get('severity_tier', '')}), "
            f"combined score {flag.get('combined_score', '')}, "
            f"persisted {flag.get('persisted_s', '')} s."
        )
    lines.append("")

    lines.append("CLINICAL MEASURES (baseline vs most recent)")
    measures = document.get("clinical_measures", []) or []
    if isinstance(measures, list) and measures:
        for m in measures:
            if not isinstance(m, Mapping):
                continue
            unit = m.get("unit", "")
            unit_sfx = f" {unit}" if unit else ""
            baseline = _fmt(m.get("baseline"))
            recent = _fmt(m.get("most_recent"))
            delta = _fmt(m.get("delta"), signed=True)
            lines.append(
                f"  - {m.get('measure', m.get('signal', ''))}: "
                f"baseline {baseline}{unit_sfx}, most recent {recent}{unit_sfx} "
                f"(delta {delta}{unit_sfx})"
            )
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("OTHER RELEVANT CLINICAL DATA")
    other = document.get("other_relevant_clinical_data", "") or ""
    lines.append(f"  {other}" if other else "  (none)")
    lines.append("")

    # Assessment / Plan intentionally left blank for a clinician (AC 14.3).
    lines.append("ASSESSMENT")
    lines.append("")
    lines.append("PLAN")
    lines.append("")

    return "\n".join(lines)


def _fmt(value, *, signed: bool = False) -> str:
    """Format a numeric measure value for text; ``None`` renders as ``n/a``."""
    v = _finite_or_none(value)
    if v is None:
        return "n/a"
    rounded = round(v, 1)
    if rounded == int(rounded):
        body = str(int(rounded))
    else:
        body = f"{rounded:.1f}"
    if signed and rounded >= 0:
        return f"+{body}"
    return body
