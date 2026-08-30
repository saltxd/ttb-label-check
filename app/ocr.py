"""Local OCR: OpenCV preprocess + Tesseract. Zero network (R7).

`extract_text_variants` yields progressively more aggressive preprocessing
attempts; the caller (app.main._run_one) stops at the first one whose
verification passes. Measured on the degradation suite (scripts/
eval_hard_images.py): single-pass = 5/8 correct labels PASS; ladder = 7/8,
worst-case ~0.75 s — comfortably inside the 5 s budget (R2).
"""
from typing import Iterator

import cv2
import numpy as np
import pytesseract


def _base(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    if max(h, w) < 1000:  # tesseract wants >= ~30px x-height
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return gray


def _otsu(gray: np.ndarray) -> np.ndarray:
    g = cv2.fastNlMeansDenoising(_base(gray), h=15)
    _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _adaptive(gray: np.ndarray) -> np.ndarray:
    """Local thresholding survives glare spots that saturate a global Otsu cut."""
    g = cv2.fastNlMeansDenoising(_base(gray), h=15)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 51, 15)


def _deskew(binary: np.ndarray) -> np.ndarray:
    """Correct small rotations (phone photos); no-op outside 0.5-15 degrees."""
    coords = cv2.findNonZero(255 - binary)
    if coords is None or len(coords) < 100:
        return binary
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if not (0.5 < abs(angle) < 15):
        return binary
    h, w = binary.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(binary, m, (w, h), borderValue=255)


def _gray(image_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Not a decodable image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def extract_text_variants(image_bytes: bytes) -> Iterator[str]:
    """Yield OCR text under increasingly aggressive preprocessing."""
    gray = _gray(image_bytes)
    for prep in (_otsu, lambda g: _deskew(_otsu(g)),
                 _adaptive, lambda g: _deskew(_adaptive(g))):
        yield pytesseract.image_to_string(prep(gray), config="--psm 3")


def extract_text(image_bytes: bytes) -> str:
    """Single-pass OCR (first ladder rung); kept for tests and simple callers."""
    return next(iter(extract_text_variants(image_bytes)))
