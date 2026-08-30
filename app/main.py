import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import ai
from app.checks import verify_label
from app.models import ApplicationData
from app.ocr import extract_text

BASE = Path(__file__).parent
app = FastAPI(title="TTB Label Check")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")
ALLOWED = {"image/png", "image/jpeg", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # public endpoint: bound memory per upload
MAX_BATCH_IMAGES = 400              # Sarah's 200-300 dump, with headroom


def _parse_abv_field(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise HTTPException(400, "Alcohol content must be a number, e.g. 5.9")


async def _read_image(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large (10 MB max)")
    return data


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html",
                                      {"ai_available": ai.ai_available()})


def _run_one(image: bytes, media_type: str, data: ApplicationData, use_ai: bool):
    start = time.monotonic()
    used_ai = False
    text = extract_text(image)
    # AI assist only when asked AND local OCR came up near-empty (hard photo).
    if use_ai and ai.ai_available() and len(text.strip()) < 40:
        text, used_ai = ai.ai_extract_text(image, media_type), True
    result = verify_label(data, text)
    result.elapsed_ms = int((time.monotonic() - start) * 1000)
    result.ai_assist_used = used_ai
    return result


@app.get("/batch", response_class=HTMLResponse)
def batch_form(request: Request):
    return templates.TemplateResponse(request, "batch.html",
                                      {"ai_available": ai.ai_available()})


@app.post("/batch", response_class=HTMLResponse)
async def batch(request: Request, csv_file: UploadFile = File(...),
                images: list[UploadFile] = File(...), use_ai: bool = Form(False)):
    import csv
    import io
    if len(images) > MAX_BATCH_IMAGES:
        raise HTTPException(413, f"Too many images (max {MAX_BATCH_IMAGES} per batch)")
    rows = {r["application_id"].strip(): r for r in
            csv.DictReader(io.StringIO((await csv_file.read()).decode("utf-8-sig")))}
    results, orphans = [], []
    for img in images:
        stem = Path(img.filename).stem
        row = rows.get(stem)
        if row is None:
            orphans.append(img.filename)
            continue
        data = ApplicationData(application_id=stem, brand_name=row["brand_name"].strip(),
                               abv=_parse_abv_field(row.get("abv", "")))
        results.append(_run_one(await _read_image(img), img.content_type or "image/png",
                                data, use_ai))
    results.sort(key=lambda r: r.overall == "PASS")  # problems first — agents triage top-down
    return templates.TemplateResponse(request, "_result.html",
                                      {"results": results, "orphans": orphans})


@app.post("/verify", response_class=HTMLResponse)
async def verify(request: Request, label_image: UploadFile = File(...),
                 brand_name: str = Form(...), abv: str = Form(""),
                 use_ai: bool = Form(False)):
    if label_image.content_type not in ALLOWED:
        raise HTTPException(400, "Upload a PNG, JPEG, or WebP image")
    data = ApplicationData(brand_name=brand_name.strip(), abv=_parse_abv_field(abv))
    result = _run_one(await _read_image(label_image), label_image.content_type, data, use_ai)
    return templates.TemplateResponse(request, "_result.html", {"results": [result]})
