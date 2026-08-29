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


def test_ocr_catches_abv_mismatch():
    text = extract_text((SAMPLES / "abv_mismatch.png").read_bytes())
    r = verify_label(ApplicationData(brand_name="Sunset Ale", abv=5.9), text)
    assert r.fields["abv"].status == "MISMATCH", r.fields["abv"]
