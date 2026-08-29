# TTB Label Verification Prototype — Distilled Requirements

Source: `take-home-instructions.md` (stakeholder interviews). The assessment's real test is
extracting hard requirements from soft narrative; this document is that extraction, with
every requirement traced to who said it.

## Hard requirements

| # | Requirement | Source | Verbatim anchor |
|---|---|---|---|
| R1 | Verify label image against application data: brand name, ABV, government warning | Sarah | "Brand name matches? Check. ABV is correct? Check. Government warning is there? Check." |
| R2 | Results in **≤ 5 seconds** per label | Sarah | "If we can't get results back in about 5 seconds, nobody's going to use it." |
| R3 | UI simple enough for a 73-year-old novice; "clean, obvious, no hunting for buttons" | Sarah | "something my mother could figure out" |
| R4 | **Batch upload** of 200–300 applications | Sarah | "handle batch uploads, that would be huge" |
| R5 | Warning statement must match **exactly, word-for-word**, with "GOVERNMENT WARNING:" in ALL CAPS (bold, per regulation) | Jenny + 27 CFR 16.21/16.22 | "It has to be exact… 'Government Warning' in title case instead of all caps. Rejected." |
| R6 | Brand matching needs **judgment**, not naive string equality: "STONE'S THROW" vs "Stone's Throw" is the same brand — flag, don't hard-fail | Dave | "Technically a mismatch? Sure. But it's obviously the same thing." |
| R7 | Core processing must work **without outbound calls to cloud ML endpoints** | Marcus | "our network blocks outbound traffic to a lot of domains… their firewall blocked connections to their ML endpoints" |
| R8 | Standalone prototype; **no COLA integration**; no sensitive data storage | Marcus | "not looking to integrate with COLA directly… not storing anything sensitive" |
| R9 | Tolerate imperfect images (angle, glare, lighting) as far as practical | Jenny | "handle images that aren't perfectly shot" (she marks it maybe-out-of-scope: treat as stretch, not core) |
| R10 | Tool advises; the **agent decides**. Output is review support, not auto-rejection | Dave | "You need judgment… don't make my life harder" |

## Regulatory facts (verified against eCFR, current through 2026-08)

**27 CFR §16.21 — exact required text (single continuous statement):**

> GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink
> alcoholic beverages during pregnancy because of the risk of birth defects. (2) Consumption
> of alcoholic beverages impairs your ability to drive a car or operate machinery, and may
> cause health problems.

**27 CFR §16.22 — presentation rules:**
- "GOVERNMENT WARNING" must be capital letters **and bold**; the remainder must NOT be bold
- Continuous statement, readily legible, contrasting background, not compressed
- Minimum type size by container: ≤237 ml → 1 mm; 237 ml–3 L → 2 mm; >3 L → 3 mm

OCR cannot measure boldness or millimeters from an uncontrolled photo. The checker
verifies text and capitalization deterministically; bold/type-size are reported as
"requires visual confirmation" line items (agent judgment, R10), with the optional AI
assist offering an advisory opinion.

## Field checks in scope

brand_name (fuzzy, R6) · abv (numeric parse + compare) · warning (exact, R5) ·
net_contents (fuzzy numeric, if provided) · class_type (fuzzy, if provided).
Bottler address / country of origin: out of scope for prototype, listed in APPROACH.md.

## Architecture decisions

- **Local-first OCR** (Tesseract + OpenCV preprocessing: grayscale, denoise, adaptive
  threshold, deskew) satisfies R2/R7. Measured target: < 3 s per image on 1 vCPU.
- **Optional AI assist** (Claude vision) — opt-in per request, used only when local OCR
  confidence is low or the user asks; degrades gracefully when no API key is configured.
  Satisfies R9 without violating R7 (off by default; a deployment behind Treasury's
  firewall simply leaves it disabled).
- **Verdicts are three-state per field:** MATCH / REVIEW (e.g., case-only difference,
  fuzzy-close) / MISMATCH — plus MISSING. REVIEW exists because of Dave (R6, R10).
- **Stack:** Python 3.12, FastAPI, server-rendered Jinja2 + vanilla JS (no build step),
  pytest, Docker. Deployed on Fly.io (always-on small VM — free-tier cold starts would
  violate R2 during evaluation); README documents the Azure Container Apps path since
  Treasury runs Azure.
- **No database.** Uploads processed in memory, results returned, nothing persisted (R8).

## Out of scope (documented, not silently dropped)

COLA integration, authentication, persistence/audit log, bottler-address & country
checks, true bold/type-size measurement, FedRAMP posture (addressed as a written
"path to production" section in APPROACH.md).
