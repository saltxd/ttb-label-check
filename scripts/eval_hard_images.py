"""Robustness evaluation: run the full pipeline against degraded label images.

Ground truth is known (degradations of samples/good_label.png, which contains a
correct warning, brand 'SUNSET ALE', 5.9% ABV), so every wrong verdict is a
measured failure, not a guess. Usage:

    python3 scripts/eval_hard_images.py [extra_image.png ...]

Extra images (e.g. real COLA label scans) are run through OCR + checks and
reported without ground-truth judgment.
"""
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.checks import verify_label          # noqa: E402
from app.models import ApplicationData       # noqa: E402
from app.ocr import extract_text             # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
GOOD = cv2.imread(str(SAMPLES / "good_label.png"))


def encode(img, ext=".png", q=95):
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, q] if ext == ".jpg" else [])
    assert ok
    return buf.tobytes()


def rotate(img, deg):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))


def glare(img):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    spot = np.exp(-(((xx - w * 0.65) ** 2 + (yy - h * 0.7) ** 2) / (2 * (w * 0.25) ** 2)))
    out = img.astype(np.float32) + (spot[..., None] * 180)
    return np.clip(out, 0, 255).astype(np.uint8)


def noise(img, sigma=18):
    n = np.random.default_rng(7).normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


CASES = [
    ("baseline",        encode(GOOD)),
    ("rotated 2deg",    encode(rotate(GOOD, 2))),
    ("rotated 5deg",    encode(rotate(GOOD, 5))),
    ("jpeg q25",        encode(GOOD, ".jpg", 25)),
    ("half resolution", encode(cv2.resize(GOOD, None, fx=0.5, fy=0.5))),
    ("gaussian noise",  encode(noise(GOOD))),
    ("glare spot",      encode(glare(GOOD))),
    ("glare + rot 3",   encode(glare(rotate(GOOD, 3)))),
]

APP = ApplicationData(brand_name="Sunset Ale", abv=5.9)

from app.main import _run_one  # noqa: E402  (evaluate the shipped path, ladder included)

print(f"{'case':<18} {'ms':>6}  {'overall':<13} brand/abv/warning")
fails = 0
for name, data in CASES:
    t0 = time.monotonic()
    result = _run_one(data, "image/png", APP, use_ai=False)
    ms = int((time.monotonic() - t0) * 1000)
    statuses = "/".join(result.fields[k].status for k in ("brand_name", "abv", "warning"))
    ok = result.overall == "PASS"
    fails += 0 if ok else 1
    flag = "" if ok else "   <-- degradation defeated OCR"
    print(f"{name:<18} {ms:>6}  {result.overall:<13} {statuses}{flag}")

print(f"\n{len(CASES) - fails}/{len(CASES)} degraded-but-correct labels still PASS")

for extra in sys.argv[1:]:
    data = Path(extra).read_bytes()
    t0 = time.monotonic()
    text = extract_text(data)
    ms = int((time.monotonic() - t0) * 1000)
    result = verify_label(APP, text)
    print(f"\n=== {extra} ({ms} ms) — no ground truth, showing raw outcome ===")
    for k, f in result.fields.items():
        print(f"  {k}: {f.status} — {f.detail[:110]}")
