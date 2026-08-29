# TTB Label Verification Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A deployed web app that checks an alcohol-label image against COLA application data (brand, ABV, government warning) in under 5 seconds, single and batch, with agent-in-control verdicts.

**Architecture:** FastAPI server-rendered app. Pure verification logic in `app/checks.py` (fully unit-tested, no I/O). Local Tesseract OCR with OpenCV preprocessing as the default path; optional Claude vision assist behind an env flag for hard images. No database; nothing persisted.

**Tech Stack:** Python 3.12, FastAPI, Jinja2, pytesseract + opencv-python-headless + Pillow, rapidfuzz, anthropic (optional path), pytest, Docker, Fly.io.

**Spec:** `docs/spec/requirements.md` (+ raw assignment in `docs/spec/take-home-instructions.md`)

## Global Constraints

- Local verify path must complete in **≤ 5 s** per image (R2); measure in Task 8/9.
- Core path makes **zero outbound network calls** (R7). AI assist is opt-in per request AND requires `ANTHROPIC_API_KEY`; absent key → feature hidden.
- Warning text must match 27 CFR §16.21 **verbatim** (constant in checks.py, copied from spec — never retype from memory).
- Field verdicts are exactly: `MATCH`, `REVIEW`, `MISMATCH`, `MISSING` (R6/R10).
- Resume-honesty rule: docs claim only what the code does.
- Every commit: `git -C ~/Forge/ttb-label-check ...`; run tests with `.venv/bin/python -m pytest`.
- Local deps once (Task 1): `brew install tesseract` (missing on this Mac); Docker image installs `tesseract-ocr` via apt.

---

### Task 1: Scaffold + warning-statement checker (the regulatory heart)

**Files:**
- Create: `requirements.txt`, `app/__init__.py`, `app/checks.py`
- Test: `tests/test_checks.py`

**Interfaces:**
- Produces: `WARNING_CANONICAL: str`; `check_warning(ocr_text: str) -> FieldCheck` where `FieldCheck = dataclass(status: str, detail: str, expected: str|None, found: str|None)` (statuses per Global Constraints)

- [ ] **Step 1: venv + deps + failing tests**

```bash
cd ~/Forge/ttb-label-check && python3 -m venv .venv && .venv/bin/pip install fastapi uvicorn jinja2 python-multipart pytesseract opencv-python-headless pillow rapidfuzz anthropic pytest httpx2
brew list tesseract >/dev/null 2>&1 || brew install tesseract
printf 'fastapi\nuvicorn\njinja2\npython-multipart\npytesseract\nopencv-python-headless\npillow\nrapidfuzz\nanthropic\npytest\n' > requirements.txt
```

`tests/test_checks.py`:

```python
from app.checks import WARNING_CANONICAL, check_warning

GOOD = ("GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
        "drink alcoholic beverages during pregnancy because of the risk of birth defects. "
        "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
        "operate machinery, and may cause health problems.")

def test_canonical_matches_cfr_16_21():
    assert WARNING_CANONICAL == GOOD

def test_exact_warning_matches():
    r = check_warning("SUNSET ALE 5.9% ALC/VOL\n" + GOOD)
    assert r.status == "MATCH"

def test_ocr_linebreaks_and_double_spaces_still_match():
    mangled = GOOD.replace("Surgeon General,", "Surgeon  General,\n")
    assert check_warning(mangled).status == "MATCH"

def test_title_case_prefix_is_review_not_match():
    # Jenny's real rejection: right words, wrong capitalization (16.22 violation)
    r = check_warning(GOOD.replace("GOVERNMENT WARNING:", "Government Warning:"))
    assert r.status == "REVIEW"
    assert "capital" in r.detail.lower()

def test_reworded_warning_is_mismatch():
    r = check_warning(GOOD.replace("birth defects", "health issues"))
    assert r.status == "MISMATCH"

def test_absent_warning_is_missing():
    assert check_warning("SUNSET ALE 5.9% ALC/VOL 12 FL OZ").status == "MISSING"
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest tests/test_checks.py -q` → FAIL (no module app.checks)

- [ ] **Step 3: Implement `app/checks.py`**

```python
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
    # Candidate region: from the anchor, as many chars as the canonical text (+slack for OCR noise)
    region = text[anchor.start(): anchor.start() + len(WARNING_CANONICAL) + 40]
    body_ok = _squash(region).casefold().startswith(_squash(WARNING_CANONICAL).casefold())
    prefix_caps = region[: anchor.end() - anchor.start()].startswith("GOVERNMENT WARNING")
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
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_checks.py -q` → 6 passed
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: warning-statement checker per 27 CFR 16.21/16.22 (TDD)"`

---

### Task 2: Brand fuzzy match + ABV parse/compare

**Files:**
- Modify: `app/checks.py` (append)
- Test: `tests/test_checks.py` (append)

**Interfaces:**
- Produces: `check_brand(expected: str, ocr_text: str) -> FieldCheck`; `parse_abv(text: str) -> float | None`; `check_abv(expected: float, ocr_text: str) -> FieldCheck`

- [ ] **Step 1: Failing tests** (append to `tests/test_checks.py`)

```python
from app.checks import check_brand, parse_abv, check_abv

def test_brand_exact():
    assert check_brand("Sunset Ale", "SUNSET ALE\nIPA 5.9% ALC/VOL").status == "MATCH"

def test_brand_case_only_difference_is_match_with_note():
    # Dave's STONE'S THROW case: same brand, different case — judgment, not rejection
    r = check_brand("Stone's Throw", "STONE'S THROW BREWING CO")
    assert r.status == "MATCH" and "case" in r.detail.lower()

def test_brand_curly_apostrophe_ocr_variant_is_match():
    assert check_brand("Stone's Throw", "STONE’S THROW").status == "MATCH"

def test_brand_close_but_not_equal_is_review():
    assert check_brand("Sunset Ale", "SUNSET ALES").status == "REVIEW"

def test_brand_absent_is_missing():
    assert check_brand("Sunset Ale", "MOONRISE LAGER 4.5% ALC/VOL").status == "MISSING"

def test_parse_abv_formats():
    assert parse_abv("5.9% ALC/VOL") == 5.9
    assert parse_abv("ALC. 5.9% BY VOL.") == 5.9
    assert parse_abv("ALCOHOL 13% BY VOLUME") == 13.0
    assert parse_abv("12 FL OZ") is None

def test_abv_match_and_mismatch():
    assert check_abv(5.9, "SUNSET ALE 5.9% ALC/VOL").status == "MATCH"
    r = check_abv(5.9, "SUNSET ALE 6.2% ALC/VOL")
    assert r.status == "MISMATCH" and "6.2" in r.detail

def test_abv_absent_is_missing():
    assert check_abv(5.9, "SUNSET ALE 12 FL OZ").status == "MISSING"
```

- [ ] **Step 2: Run** → new tests FAIL (names undefined)
- [ ] **Step 3: Implement** (append to `app/checks.py`)

```python
def _norm_brand(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'")
    return _squash(s).casefold()

def check_brand(expected: str, ocr_text: str) -> FieldCheck:
    """Line-oriented: the brand is one line of the label, compared as a unit (R6)."""
    want = _norm_brand(expected)
    lines = [_norm_brand(l) for l in ocr_text.splitlines() if l.strip()]
    best_line, best = "", 0.0
    for line in lines:
        score = max(fuzz.ratio(want, line), fuzz.partial_ratio(want, line))
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
    r"(?:alc(?:ohol)?\.?\s*)?(\d{1,2}(?:\.\d{1,2})?)\s*%(?:\s*(?:alc[./]?\s*)?(?:by\s+)?vol(?:ume)?\.?)?",
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
```

- [ ] **Step 4: Run** — `.venv/bin/python -m pytest tests/test_checks.py -q` → all pass
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: brand fuzzy match (Dave rule) and ABV parse/compare"`

---

### Task 3: Aggregate verdict + result models

**Files:**
- Create: `app/models.py`
- Modify: `app/checks.py` (append `verify_label`)
- Test: `tests/test_checks.py` (append)

**Interfaces:**
- Produces: `ApplicationData` (pydantic: `brand_name: str`, `abv: float | None`, `application_id: str = ""`); `verify_label(app_data: ApplicationData, ocr_text: str) -> LabelResult` with `LabelResult.fields: dict[str, FieldCheck]`, `.overall: str` ("PASS" all MATCH / "NEEDS REVIEW" any REVIEW-or-MISSING-or-MISMATCH), `.ocr_text: str`

- [ ] **Step 1: Failing tests**

```python
from app.checks import verify_label
from app.models import ApplicationData

def test_verify_label_all_match_is_pass():
    r = verify_label(ApplicationData(brand_name="Sunset Ale", abv=5.9),
                     "SUNSET ALE\n5.9% ALC/VOL\n" + GOOD)
    assert r.overall == "PASS"
    assert set(r.fields) == {"brand_name", "abv", "warning"}

def test_any_problem_is_needs_review_never_autoreject():
    r = verify_label(ApplicationData(brand_name="Sunset Ale", abv=5.9),
                     "SUNSET ALE\n6.2% ALC/VOL\n" + GOOD)
    assert r.overall == "NEEDS REVIEW"  # Dave (R10): tool advises, agent decides

def test_abv_omitted_from_application_skips_abv_check():
    r = verify_label(ApplicationData(brand_name="Sunset Ale", abv=None),
                     "SUNSET ALE\n" + GOOD)
    assert "abv" not in r.fields and r.overall == "PASS"
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement**

`app/models.py`:

```python
from pydantic import BaseModel
from app.checks import FieldCheck

class ApplicationData(BaseModel):
    application_id: str = ""
    brand_name: str
    abv: float | None = None

class LabelResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    application_id: str = ""
    overall: str
    fields: dict[str, FieldCheck]
    ocr_text: str = ""
    elapsed_ms: int = 0
    ai_assist_used: bool = False
```

Append to `app/checks.py`:

```python
def verify_label(app_data, ocr_text: str):
    from app.models import LabelResult
    fields: dict[str, FieldCheck] = {"brand_name": check_brand(app_data.brand_name, ocr_text)}
    if app_data.abv is not None:
        fields["abv"] = check_abv(app_data.abv, ocr_text)
    fields["warning"] = check_warning(ocr_text)
    overall = "PASS" if all(f.status == MATCH for f in fields.values()) else "NEEDS REVIEW"
    return LabelResult(application_id=app_data.application_id, overall=overall,
                       fields=fields, ocr_text=ocr_text)
```

- [ ] **Step 4: Run** → pass · **Step 5: Commit** — `"feat: aggregate label verdict (PASS / NEEDS REVIEW)"`

---

### Task 4: OCR pipeline + sample label generator

**Files:**
- Create: `app/ocr.py`, `scripts/make_samples.py`
- Test: `tests/test_ocr.py`

**Interfaces:**
- Produces: `extract_text(image_bytes: bytes) -> str` (preprocess + tesseract); `samples/good_label.png`, `samples/case_violation.png`, `samples/abv_mismatch.png` on disk

- [ ] **Step 1: Sample generator** — `scripts/make_samples.py`:

```python
"""Generate synthetic label PNGs for tests and the demo. PIL only, deterministic."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from app.checks import WARNING_CANONICAL
import textwrap

OUT = Path(__file__).resolve().parent.parent / "samples"; OUT.mkdir(exist_ok=True)
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"  # Docker: DejaVuSans (fallback below)
try:    big, small = ImageFont.truetype(FONT, 64), ImageFont.truetype(FONT, 28)
except OSError:
    from PIL import ImageFont as F
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    big, small = F.truetype(p, 64), F.truetype(p, 28)

def label(name: str, brand: str, abv: str, warning: str):
    img = Image.new("RGB", (1200, 900), "white"); d = ImageDraw.Draw(img)
    d.text((60, 60), brand, font=big, fill="black")
    d.text((60, 180), f"{abv} ALC/VOL  ·  12 FL OZ", font=small, fill="black")
    y = 620
    for line in textwrap.wrap(warning, width=70):
        d.text((60, y), line, font=small, fill="black"); y += 40
    img.save(OUT / name)

label("good_label.png", "SUNSET ALE", "5.9%", WARNING_CANONICAL)
label("case_violation.png", "SUNSET ALE", "5.9%",
      WARNING_CANONICAL.replace("GOVERNMENT WARNING:", "Government Warning:"))
label("abv_mismatch.png", "SUNSET ALE", "6.2%", WARNING_CANONICAL)
print("samples written to", OUT)
```

Run: `.venv/bin/python scripts/make_samples.py`

- [ ] **Step 2: Failing test** — `tests/test_ocr.py`:

```python
import time
from pathlib import Path
from app.ocr import extract_text
from app.checks import verify_label
from app.models import ApplicationData

SAMPLES = Path(__file__).resolve().parent.parent / "samples"

def test_ocr_reads_good_label_and_pipeline_passes():
    start = time.monotonic()
    text = extract_text((SAMPLES / "good_label.png").read_bytes())
    elapsed = time.monotonic() - start
    r = verify_label(ApplicationData(brand_name="Sunset Ale", abv=5.9), text)
    assert r.overall == "PASS", r.fields
    assert elapsed < 5.0  # R2 (Sarah): the whole point

def test_ocr_catches_case_violation():
    text = extract_text((SAMPLES / "case_violation.png").read_bytes())
    r = verify_label(ApplicationData(brand_name="Sunset Ale", abv=5.9), text)
    assert r.fields["warning"].status == "REVIEW"

def test_ocr_catches_abv_mismatch():
    text = extract_text((SAMPLES / "abv_mismatch.png").read_bytes())
    r = verify_label(ApplicationData(brand_name="Sunset Ale", abv=5.9), text)
    assert r.fields["abv"].status == "MISMATCH"
```

- [ ] **Step 3: Implement** — `app/ocr.py`:

```python
"""Local OCR: OpenCV preprocess + Tesseract. Zero network (R7); target < 3 s (R2)."""
import cv2, numpy as np, pytesseract

def _preprocess(image_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Not a decodable image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Upscale small images: tesseract wants >= ~30px x-height
    h, w = gray.shape
    if max(h, w) < 1000:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=15)
    # Otsu threshold handles uneven lighting/glare better than a fixed cut (R9, partial)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_text(image_bytes: bytes) -> str:
    return pytesseract.image_to_string(_preprocess(image_bytes), config="--psm 3")
```

- [ ] **Step 4: Run** — `.venv/bin/python -m pytest tests/ -q` → all pass (tune `label()` font sizes if OCR flubs; keep tests strict)
- [ ] **Step 5: Commit** — `"feat: local OCR pipeline + synthetic sample labels; <5s verified in test"`

---

### Task 5: Optional AI assist (Claude vision)

**Files:**
- Create: `app/ai.py`
- Test: `tests/test_ai.py`

**Interfaces:**
- Produces: `ai_available() -> bool`; `ai_extract_text(image_bytes: bytes, media_type: str) -> str` (raises `AIUnavailable` if no key)

- [ ] **Step 1: Failing tests** (mock the SDK; no live calls in CI):

```python
import pytest
from unittest.mock import patch, MagicMock
import app.ai as ai

def test_ai_available_false_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert ai.ai_available() is False

def test_ai_extract_uses_vision_block(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake = MagicMock(); fake.content = [MagicMock(type="text", text="SUNSET ALE 5.9%")]
    with patch.object(ai, "_client") as c:
        c.return_value.messages.create.return_value = fake
        out = ai.ai_extract_text(b"png-bytes", "image/png")
    assert out == "SUNSET ALE 5.9%"
    kwargs = c.return_value.messages.create.call_args.kwargs
    blocks = kwargs["messages"][0]["content"]
    assert blocks[0]["type"] == "image" and blocks[0]["source"]["type"] == "base64"
```

- [ ] **Step 2: Implement** — `app/ai.py`:

```python
"""Opt-in AI assist for hard images (glare, angle — Jenny, R9).

Off by default and hidden when no key is configured, so a deployment behind
Treasury's firewall (Marcus, R7) simply never calls out.
"""
import base64, os
import anthropic

AI_MODEL = os.environ.get("AI_MODEL", "claude-opus-5")

class AIUnavailable(RuntimeError): ...

def ai_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))

def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()

def ai_extract_text(image_bytes: bytes, media_type: str) -> str:
    if not ai_available():
        raise AIUnavailable("ANTHROPIC_API_KEY not configured")
    data = base64.standard_b64encode(image_bytes).decode("utf-8")
    response = _client().with_options(timeout=30.0).messages.create(
        model=AI_MODEL,
        max_tokens=2048,
        output_config={"effort": "low"},  # transcription task; latency matters (R2)
        system=("Transcribe ALL text visible on this alcohol beverage label exactly as printed, "
                "preserving capitalization and line breaks. Output only the transcription."),
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": media_type, "data": data}},
            {"type": "text", "text": "Transcribe this label."},
        ]}],
    )
    return "".join(b.text for b in response.content if b.type == "text")
```

- [ ] **Step 3: Run** → pass · **Step 4: Commit** — `"feat: opt-in Claude vision assist, hidden when unconfigured"`
- [ ] **Step 5 (manual, once):** with a real key, `.venv/bin/python -c "from app.ai import ai_extract_text; print(ai_extract_text(open('samples/good_label.png','rb').read(),'image/png')[:120])"` — verify a live transcription.

---

### Task 6: Web app — single label verify

**Files:**
- Create: `app/main.py`, `app/templates/index.html`, `app/templates/_result.html`, `app/static/style.css`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `GET /` (form), `POST /verify` (multipart: `label_image`, `brand_name`, `abv` optional, `use_ai` optional checkbox) → HTML result; `GET /health` → `{"status":"ok"}`; `app.main:app` for uvicorn

- [ ] **Step 1: Failing tests** — `tests/test_api.py`:

```python
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
SAMPLES = Path(__file__).resolve().parent.parent / "samples"

def test_health():
    assert client.get("/health").json() == {"status": "ok"}

def test_index_serves_form():
    html = client.get("/").text
    assert "Verify" in html and "label_image" in html

def test_verify_good_label_passes():
    r = client.post("/verify", data={"brand_name": "Sunset Ale", "abv": "5.9"},
                    files={"label_image": ("l.png", (SAMPLES/"good_label.png").read_bytes(), "image/png")})
    assert r.status_code == 200 and "PASS" in r.text

def test_verify_bad_abv_needs_review():
    r = client.post("/verify", data={"brand_name": "Sunset Ale", "abv": "5.9"},
                    files={"label_image": ("l.png", (SAMPLES/"abv_mismatch.png").read_bytes(), "image/png")})
    assert "NEEDS REVIEW" in r.text

def test_verify_rejects_non_image():
    r = client.post("/verify", data={"brand_name": "X"},
                    files={"label_image": ("l.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
```

- [ ] **Step 2: Implement** — `app/main.py`:

```python
import time
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.models import ApplicationData
from app.checks import verify_label
from app.ocr import extract_text
from app import ai

BASE = Path(__file__).parent
app = FastAPI(title="TTB Label Check")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")
ALLOWED = {"image/png", "image/jpeg", "image/webp"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"ai_available": ai.ai_available()})

def _run_one(image: bytes, media_type: str, data: ApplicationData, use_ai: bool):
    start = time.monotonic()
    used_ai = False
    text = extract_text(image)
    if use_ai and ai.ai_available() and len(text.strip()) < 40:  # local OCR came up near-empty
        text, used_ai = ai.ai_extract_text(image, media_type), True
    result = verify_label(data, text)
    result.elapsed_ms = int((time.monotonic() - start) * 1000)
    result.ai_assist_used = used_ai
    return result

@app.post("/verify", response_class=HTMLResponse)
async def verify(request: Request, label_image: UploadFile = File(...),
                 brand_name: str = Form(...), abv: str = Form(""), use_ai: bool = Form(False)):
    if label_image.content_type not in ALLOWED:
        raise HTTPException(400, "Upload a PNG, JPEG, or WebP image")
    data = ApplicationData(brand_name=brand_name.strip(),
                           abv=float(abv) if abv.strip() else None)
    result = _run_one(await label_image.read(), label_image.content_type, data, use_ai)
    return templates.TemplateResponse(request, "_result.html", {"results": [result]})
```

`app/templates/index.html` — one screen, three inputs, one giant button (Sarah's mother, R3):

```html
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label Check</title><link rel="stylesheet" href="/static/style.css"></head><body>
<main>
  <h1>Label Check</h1>
  <p class="tagline">Upload a label. Enter what the application says. We compare them.</p>
  <form action="/verify" method="post" enctype="multipart/form-data">
    <label>Label image <input type="file" name="label_image" accept="image/*" required></label>
    <label>Brand name on application <input type="text" name="brand_name" required placeholder="e.g., Sunset Ale"></label>
    <label>Alcohol content (%) <input type="text" name="abv" inputmode="decimal" placeholder="e.g., 5.9 (optional)"></label>
    {% if ai_available %}<label class="row"><input type="checkbox" name="use_ai" value="true"> Use AI assist for hard-to-read photos (slower)</label>{% endif %}
    <button type="submit">Check this label</button>
  </form>
  <p><a href="/batch">Checking many labels at once? Batch upload →</a></p>
</main></body></html>
```

`app/templates/_result.html`:

```html
{% for r in results %}
<section class="result {{ 'pass' if r.overall == 'PASS' else 'review' }}">
  <h2>{{ r.application_id or 'Result' }}: {{ r.overall }}</h2>
  <table>
    {% for name, f in r.fields.items() %}
    <tr class="{{ f.status|lower }}">
      <th>{{ {'brand_name':'Brand name','abv':'Alcohol content','warning':'Government warning'}[name] }}</th>
      <td class="status">{{ f.status }}</td><td>{{ f.detail }}</td>
    </tr>
    {% endfor %}
  </table>
  <p class="meta">Checked in {{ r.elapsed_ms }} ms{{ ' · AI assist used' if r.ai_assist_used }}.
     This tool advises — the reviewing agent makes the final call.</p>
</section>
{% endfor %}
<p><a href="/">← Check another label</a></p>
```

`app/static/style.css` — large type, high contrast, obvious states:

```css
:root { font-size: 18px; }
body { font-family: system-ui, sans-serif; margin: 0; background: #f5f7f9; color: #1a2530; }
main { max-width: 720px; margin: 3rem auto; padding: 0 1rem; }
h1 { font-size: 2.2rem; } .tagline { color: #44586b; }
form { background: #fff; border: 1px solid #d5dee6; border-radius: 12px; padding: 1.5rem; display: grid; gap: 1.1rem; }
label { display: grid; gap: .4rem; font-weight: 600; }
label.row { grid-auto-flow: column; justify-content: start; align-items: center; gap: .6rem; }
input[type=text], input[type=file] { font-size: 1rem; padding: .7rem; border: 1px solid #b9c6d2; border-radius: 8px; }
button { font-size: 1.25rem; font-weight: 700; padding: 1rem; border: 0; border-radius: 10px; background: #0b5fff; color: #fff; cursor: pointer; }
button:hover { background: #0a4fd4; }
.result { background:#fff; border-radius:12px; padding:1.2rem 1.5rem; margin:1.2rem 0; border-left: 10px solid; }
.result.pass { border-color:#1d8a45; } .result.review { border-color:#d97706; }
.result h2 { margin-top: 0; }
table { border-collapse: collapse; width: 100%; } th, td { text-align:left; padding:.5rem .6rem; vertical-align: top; }
tr.match .status { color:#1d8a45; font-weight:700; } tr.review .status,{}
tr.review .status { color:#d97706; font-weight:700; }
tr.mismatch .status, tr.missing .status { color:#c22525; font-weight:700; }
.meta { color:#44586b; font-size:.9rem; }
```

- [ ] **Step 3: Run** — `.venv/bin/python -m pytest tests/ -q` → all pass; eyeball with `.venv/bin/uvicorn app.main:app --port 8080`
- [ ] **Step 4: Commit** — `"feat: single-label verify web UI"`

---

### Task 7: Batch mode (Sarah's 200–300 dump)

**Files:**
- Modify: `app/main.py` (add `GET /batch`, `POST /batch`)
- Create: `app/templates/batch.html`, `samples/batch.csv`
- Test: `tests/test_api.py` (append)

**Interfaces:**
- Produces: `POST /batch` (multipart: `csv_file` with header `application_id,brand_name,abv`; `images` = many files, each filename stem == application_id) → results table sorted problems-first

- [ ] **Step 1: Failing tests**

```python
def test_batch_two_labels():
    csv_bytes = b"application_id,brand_name,abv\ngood_label,Sunset Ale,5.9\nabv_mismatch,Sunset Ale,5.9\n"
    files = [("images", ("good_label.png", (SAMPLES/"good_label.png").read_bytes(), "image/png")),
             ("images", ("abv_mismatch.png", (SAMPLES/"abv_mismatch.png").read_bytes(), "image/png"))]
    r = client.post("/batch", files=[("csv_file", ("apps.csv", csv_bytes, "text/csv"))] + files)
    assert r.status_code == 200
    assert "NEEDS REVIEW" in r.text and "PASS" in r.text

def test_batch_image_without_csv_row_reported():
    files = [("images", ("mystery.png", (SAMPLES/"good_label.png").read_bytes(), "image/png"))]
    r = client.post("/batch", files=[("csv_file", ("apps.csv", b"application_id,brand_name,abv\n", "text/csv"))] + files)
    assert "no application row" in r.text.lower()
```

- [ ] **Step 2: Implement** (append to `app/main.py`):

```python
import csv, io
from fastapi import UploadFile

@app.get("/batch", response_class=HTMLResponse)
def batch_form(request: Request):
    return templates.TemplateResponse(request, "batch.html", {"ai_available": ai.ai_available()})

@app.post("/batch", response_class=HTMLResponse)
async def batch(request: Request, csv_file: UploadFile = File(...),
                images: list[UploadFile] = File(...), use_ai: bool = Form(False)):
    rows = {r["application_id"].strip(): r for r in
            csv.DictReader(io.StringIO((await csv_file.read()).decode("utf-8-sig")))}
    results, orphans = [], []
    for img in images:
        stem = Path(img.filename).stem
        row = rows.get(stem)
        if row is None:
            orphans.append(img.filename); continue
        data = ApplicationData(application_id=stem, brand_name=row["brand_name"].strip(),
                               abv=float(row["abv"]) if row.get("abv", "").strip() else None)
        results.append(_run_one(await img.read(), img.content_type or "image/png", data, use_ai))
    results.sort(key=lambda r: r.overall == "PASS")  # problems first — agents triage top-down
    return templates.TemplateResponse(request, "_result.html",
                                      {"results": results, "orphans": orphans})
```

`app/templates/batch.html` mirrors index.html: CSV input + multi-file input + same button styling. Prepend to `_result.html`:

```html
{% if orphans %}<section class="result review"><h2>Unmatched images</h2>
<p>These files had no application row in the CSV (match by filename): {{ orphans|join(', ') }}</p></section>{% endif %}
```

- [ ] **Step 3: Run** → pass · **Step 4: Commit** — `"feat: batch verify — CSV + images matched by filename, problems sorted first"`

---

### Task 8: Docker + container smoke test

**Files:**
- Create: `Dockerfile`, `.dockerignore`

- [ ] **Step 1: Dockerfile**

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY samples ./samples
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

`.dockerignore`: `.venv`, `.git`, `docs`, `tests`, `__pycache__`

- [ ] **Step 2: Build + smoke + timing**

```bash
docker build -t ttb-label-check . && docker run -d -p 8080:8080 --name ttb ttb-label-check
sleep 3 && curl -s localhost:8080/health
time curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" -F label_image=@samples/good_label.png -F brand_name="Sunset Ale" -F abv=5.9 localhost:8080/verify
docker rm -f ttb
```

Expected: `{"status":"ok"}`, then `200` in **< 5 s** (record the number for APPROACH.md).

- [ ] **Step 3: Commit** — `"build: Dockerfile (tesseract baked in), smoke-tested under 5s"`

---

### Task 9: Deploy to Fly.io

**Files:**
- Create: `fly.toml`

- [ ] **Step 1:** `brew install flyctl` (if missing) → `fly auth login` (Marley does this interactively: suggest he run `! fly auth login`)
- [ ] **Step 2:** `fly launch --no-deploy --name ttb-label-check --region den` (accept generated fly.toml; set `min_machines_running = 1` and `auto_stop_machines = false` in `[http_service]` — cold starts would break R2 during Treasury's test)
- [ ] **Step 3:** `fly deploy` → `curl -s https://ttb-label-check.fly.dev/health` → `{"status":"ok"}`; repeat the timed `/verify` curl against the live URL, confirm < 5 s
- [ ] **Step 4:** (optional AI assist live) `fly secrets set ANTHROPIC_API_KEY=...` — Marley's decision; app is fully functional without it
- [ ] **Step 5: Commit** — `"deploy: fly.io config, always-on machine (no cold starts)"`

---

### Task 10: README + APPROACH — the graded documents

**Files:**
- Create: `README.md`, `docs/APPROACH.md`

- [ ] **Step 1: README.md** must contain, in order: one-paragraph what-it-is; live URL; quickstart (`docker build/run`, or venv + `brew install tesseract` + `uvicorn`); how to run tests; batch CSV format with example; **Tools used** section disclosing AI-assisted development (Claude) and the in-app optional Claude vision assist — honest and specific; screenshot of a result.
- [ ] **Step 2: docs/APPROACH.md** must contain: requirements table mapping each feature to the stakeholder who asked (lift from `docs/spec/requirements.md`); architecture and why local-first (Marcus's firewall, Sarah's 5 s — include the measured timing from Tasks 8–9); the three-state verdict design and why auto-reject is wrong (Dave); the 16.21/16.22 verbatim-text + capitals design and what OCR cannot verify (bold, mm type size) with the visual-confirmation fallback; assumptions (synthetic samples, filename↔CSV matching, English labels, no auth/persistence); path to production (Azure Container Apps since Treasury runs Azure, FedRAMP considerations, COLA integration out of scope per Marcus).
- [ ] **Step 3:** Create public GitHub repo `saltxd/ttb-label-check`, push, verify README renders and the clone-quickstart works.
- [ ] **Step 4: Commit + push** — `"docs: README + approach writeup"`

---

## Self-Review (completed)

- **Spec coverage:** R1→T1–3, R2→T4/T8/T9 (measured), R3→T6 CSS/copy, R4→T7, R5→T1, R6→T2, R7→T4+T5 gating, R8→no persistence anywhere, R9→T4 preprocessing + T5, R10→T3 overall-verdict design + result copy. Gatsby-era lesson: nothing in the plan claims what the code doesn't do.
- **Placeholder scan:** none — all code inline.
- **Type consistency:** `FieldCheck`/`ApplicationData`/`LabelResult` names and fields consistent across T1/T3/T6/T7; `_run_one` signature consistent T6/T7.
