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
    if sim >= 90:
        # This close, the tool cannot distinguish an applicant's wording error
        # from an OCR misread of correct text ("SURGEOH", "PREGHAVICY"). Either
        # way a human must look at the label — so advise, don't verdict (R10).
        return FieldCheck(REVIEW, f"Warning text differs slightly from the required statement "
                          f"({sim:.0f}% similar) — likely OCR misreading, possibly a wording "
                          f"error. Compare the highlighted text against the label image.",
                          expected=WARNING_CANONICAL, found=region[: len(WARNING_CANONICAL)])
    return FieldCheck(MISMATCH, f"Warning statement present but deviates from the required text "
                      f"({sim:.0f}% similar). 16.21 requires the exact statement.",
                      expected=WARNING_CANONICAL, found=region[: len(WARNING_CANONICAL)])


def _norm_brand(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'")
    return _squash(s).casefold()


def check_brand(expected: str, ocr_text: str) -> FieldCheck:
    """Line-oriented: the brand is one line of the label, compared as a unit (R6)."""
    want = _norm_brand(expected)
    lines = [_norm_brand(l) for l in ocr_text.splitlines() if l.strip()]
    best_line, best = "", 0.0
    whole_word = re.compile(rf"\b{re.escape(want)}\b")
    for line in lines:
        # Whole-word substring = the brand appears intact ("STONE'S THROW BREWING CO").
        # Plain ratio otherwise — partial_ratio would wrongly score "SUNSET ALES"
        # as a perfect match for "Sunset Ale".
        score = 100.0 if whole_word.search(line) else fuzz.ratio(want, line)
        if score > best:
            best, best_line = score, line
    if best >= 97:
        exact_case = any(expected in l for l in ocr_text.splitlines())
        note = "" if exact_case else " (capitalization differs from application — same brand)"
        return FieldCheck(MATCH, f"Brand name found on label{note}.", found=best_line)
    if best >= 85:
        return FieldCheck(REVIEW, f"Label text '{best_line}' is close to but not identical to "
                          f"application brand '{expected}' ({best:.0f}% similar). Agent judgment.",
                          expected=expected, found=best_line)
    return FieldCheck(MISSING, f"Brand '{expected}' not found on label.", expected=expected)


_ABV_RE = re.compile(
    r"(?:alc(?:ohol)?\.?\s*)?(\d{1,2}(?:\.\d{1,2})?)\s*%"
    r"(?:\s*(?:alc[./]?\s*)?(?:by\s+)?vol(?:ume)?\.?)?",
    re.IGNORECASE)


def parse_abv(text: str) -> float | None:
    t = _squash(text)
    for m in _ABV_RE.finditer(t):
        window = t[max(0, m.start() - 12): m.end() + 12].casefold()
        if "alc" in window or "vol" in window:
            return float(m.group(1))
    return None


def check_abv(expected: float, ocr_text: str) -> FieldCheck:
    found = parse_abv(ocr_text)
    if found is None:
        return FieldCheck(MISSING, "No alcohol content statement found on label.",
                          expected=f"{expected}%")
    if abs(found - expected) < 0.05:
        return FieldCheck(MATCH, f"ABV on label ({found}%) matches application.", found=f"{found}%")
    return FieldCheck(MISMATCH, f"Label states {found}% ABV; application states {expected}%.",
                      expected=f"{expected}%", found=f"{found}%")


def verify_label(app_data, ocr_text: str):
    """Aggregate per-field checks. Never auto-rejects: any problem -> NEEDS REVIEW (R10)."""
    from app.models import LabelResult
    fields: dict[str, FieldCheck] = {"brand_name": check_brand(app_data.brand_name, ocr_text)}
    if app_data.abv is not None:
        fields["abv"] = check_abv(app_data.abv, ocr_text)
    fields["warning"] = check_warning(ocr_text)
    overall = "PASS" if all(f.status == MATCH for f in fields.values()) else "NEEDS REVIEW"
    return LabelResult(application_id=app_data.application_id, overall=overall,
                       fields=fields, ocr_text=ocr_text)
