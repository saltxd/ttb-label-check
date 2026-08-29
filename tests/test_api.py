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


def test_batch_image_without_csv_row_reported():
    files = [("images", ("mystery.png", (SAMPLES / "good_label.png").read_bytes(), "image/png"))]
    r = client.post("/batch",
                    files=[("csv_file", ("apps.csv", b"application_id,brand_name,abv\n", "text/csv"))] + files)
    assert "no application row" in r.text.lower()
