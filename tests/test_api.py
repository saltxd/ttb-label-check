from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_index_serves_form():
    html = client.get("/").text
    assert "Check this label" in html and "label_image" in html


def test_verify_good_label_passes():
    r = client.post("/verify", data={"brand_name": "Sunset Ale", "abv": "5.9"},
                    files={"label_image": ("l.png", (SAMPLES / "good_label.png").read_bytes(), "image/png")})
    assert r.status_code == 200 and "PASS" in r.text


def test_verify_bad_abv_needs_review():
    r = client.post("/verify", data={"brand_name": "Sunset Ale", "abv": "5.9"},
                    files={"label_image": ("l.png", (SAMPLES / "abv_mismatch.png").read_bytes(), "image/png")})
    assert "NEEDS REVIEW" in r.text


def test_verify_rejects_non_image():
    r = client.post("/verify", data={"brand_name": "X"},
                    files={"label_image": ("l.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
