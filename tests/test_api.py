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


def test_demo_buttons_on_home_page():
    html = client.get("/").text
    assert "/demo/pass" in html and "/demo/warning-case" in html and "/demo/abv-mismatch" in html


def test_demo_routes_run_end_to_end():
    assert "PASS" in client.post("/demo/pass").text
    assert "NEEDS REVIEW" in client.post("/demo/warning-case").text
    assert "NEEDS REVIEW" in client.post("/demo/abv-mismatch").text
    assert client.post("/demo/nope").status_code == 404


def test_result_shows_expected_vs_found_on_mismatch():
    r = client.post("/demo/abv-mismatch").text
    assert "Application:" in r and "Label:" in r and "5.9%" in r and "6.2%" in r


def test_ai_assist_triggers_on_garbage_ocr(monkeypatch):
    from app import main as m
    monkeypatch.setattr(m, "extract_text", lambda b: "x" * 200)  # junk, brand+warning MISSING
    monkeypatch.setattr(m.ai, "ai_available", lambda: True)
    calls = {}
    def fake_ai(image, media_type):
        calls["hit"] = True
        return "SUNSET ALE\n5.9% ALC/VOL"
    monkeypatch.setattr(m.ai, "ai_extract_text", fake_ai)
    from app.models import ApplicationData
    result = m._run_one(b"img", "image/png", ApplicationData(brand_name="Sunset Ale", abv=5.9), True)
    assert calls.get("hit") and result.ai_assist_used


def test_batch_form_serves():
    assert "Check all labels" in client.get("/batch").text


def test_batch_two_labels():
    csv_bytes = (b"application_id,brand_name,abv\n"
                 b"good_label,Sunset Ale,5.9\nabv_mismatch,Sunset Ale,5.9\n")
    files = [("images", ("good_label.png", (SAMPLES / "good_label.png").read_bytes(), "image/png")),
             ("images", ("abv_mismatch.png", (SAMPLES / "abv_mismatch.png").read_bytes(), "image/png"))]
    r = client.post("/batch",
                    files=[("csv_file", ("apps.csv", csv_bytes, "text/csv"))] + files)
    assert r.status_code == 200
    assert "NEEDS REVIEW" in r.text and "PASS" in r.text
    # problems sorted first
    assert r.text.index("NEEDS REVIEW") < r.text.index("PASS")


def test_verify_rejects_oversize_image():
    big = b"\x89PNG" + b"\x00" * (10 * 1024 * 1024 + 10)
    r = client.post("/verify", data={"brand_name": "X"},
                    files={"label_image": ("l.png", big, "image/png")})
    assert r.status_code == 413


def test_verify_bad_abv_is_400_not_500():
    r = client.post("/verify", data={"brand_name": "X", "abv": "five point nine"},
                    files={"label_image": ("l.png", (SAMPLES / "good_label.png").read_bytes(), "image/png")})
    assert r.status_code == 400


def test_batch_image_without_csv_row_reported():
    files = [("images", ("mystery.png", (SAMPLES / "good_label.png").read_bytes(), "image/png"))]
    r = client.post("/batch",
                    files=[("csv_file", ("apps.csv", b"application_id,brand_name,abv\n", "text/csv"))] + files)
    assert "no application row" in r.text.lower()
