"""Pure verification logic. No I/O, no network — fully unit-testable (R7)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

# 27 CFR 16.21 — verbatim. Do not edit; see docs/spec/requirements.md.
WARNING_CANONICAL = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
    "drink alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
    "operate machinery, and may cause health problems."
)

MATCH, REVIEW, MISMATCH, MISSING = "MATCH", "REVIEW", "MISMATCH", "MISSING"


@dataclass
class FieldCheck:
    status: str
    detail: str
    expected: str | None = None
    found: str | None = None


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def check_warning(ocr_text: str) -> FieldCheck:
    text = _squash(ocr_text)
    anchor = re.search(r"government\s+warning\s*:", text, re.IGNORECASE)
    if not anchor:
        return FieldCheck(MISSING, "No government warning statement found on label.",
                          expected=WARNING_CANONICAL)
    # Candidate region: from the anchor, canonical length plus slack for OCR noise.
    region = text[anchor.start(): anchor.start() + len(WARNING_CANONICAL) + 40]
    body_ok = _squash(region).casefold().startswith(_squash(WARNING_CANONICAL).casefold())
    prefix_caps = region.startswith("GOVERNMENT WARNING")
    if body_ok and prefix_caps:
        return FieldCheck(MATCH, "Warning statement present, verbatim per 27 CFR 16.21. "
                          "Bold type and minimum type size require visual confirmation (16.22).",
                          found=region[: len(WARNING_CANONICAL)])
    if body_ok:
        return FieldCheck(REVIEW, "Warning text is verbatim but 'GOVERNMENT WARNING' is not in "
                          "capital letters — 27 CFR 16.22 requires capitals and bold. Agent review.",
                          expected="GOVERNMENT WARNING:", found=region[:20])
    sim = fuzz.ratio(_squash(region[: len(WARNING_CANONICAL)]).casefold(),
                     _squash(WARNING_CANONICAL).casefold())
    return FieldCheck(MISMATCH, f"Warning statement present but deviates from the required text "
                      f"({sim:.0f}% similar). 16.21 requires the exact statement.",
                      expected=WARNING_CANONICAL, found=region[: len(WARNING_CANONICAL)])
