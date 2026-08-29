"""Local OCR: OpenCV preprocess + Tesseract. Zero network (R7); target < 3 s (R2)."""
import cv2
import numpy as np
import pytesseract


def _preprocess(image_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Not a decodable image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Upscale small images: tesseract wants >= ~30px x-height.
    h, w = gray.shape
    if max(h, w) < 1000:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=15)
    # Otsu threshold handles uneven lighting/glare better than a fixed cut (R9, partial).
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def extract_text(image_bytes: bytes) -> str:
    return pytesseract.image_to_string(_preprocess(image_bytes), config="--psm 3")
