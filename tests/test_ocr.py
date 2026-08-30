import time
from pathlib import Path

from app.checks import verify_label
from app.models import ApplicationData
from app.ocr import extract_text

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
    assert r.fields["warning"].status == "REVIEW", r.fields["warning"]


def test_ladder_recovers_rotated_label():
    """5-degree rotation defeats single-pass OCR; the ladder's deskew rung recovers it."""
    import cv2
    import numpy as np
    from app.main import _run_one
    img = cv2.imread(str(SAMPLES / "good_label.png"))
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 5, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))
    ok, buf = cv2.imencode(".png", rotated)
    r = _run_one(buf.tobytes(), "image/png",
                 ApplicationData(brand_name="Sunset Ale", abv=5.9), use_ai=False)
    assert r.overall == "PASS", r.fields


def test_ocr_catches_abv_mismatch():
    text = extract_text((SAMPLES / "abv_mismatch.png").read_bytes())
    r = verify_label(ApplicationData(brand_name="Sunset Ale", abv=5.9), text)
    assert r.fields["abv"].status == "MISMATCH", r.fields["abv"]
