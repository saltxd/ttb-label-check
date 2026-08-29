"""Generate synthetic label PNGs for tests and the demo. PIL only, deterministic."""
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.checks import WARNING_CANONICAL  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "samples"
OUT.mkdir(exist_ok=True)

_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",            # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",         # Debian (Docker)
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise RuntimeError("No usable TrueType font found")


def label(name: str, brand: str, abv: str, warning: str) -> None:
    img = Image.new("RGB", (1200, 900), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 60), brand, font=_font(64), fill="black")
    d.text((60, 180), f"{abv} ALC/VOL  -  12 FL OZ", font=_font(28), fill="black")
    y = 560
    for line in textwrap.wrap(warning, width=70):
        d.text((60, y), line, font=_font(28), fill="black")
        y += 42
    img.save(OUT / name)


label("good_label.png", "SUNSET ALE", "5.9%", WARNING_CANONICAL)
label("case_violation.png", "SUNSET ALE", "5.9%",
      WARNING_CANONICAL.replace("GOVERNMENT WARNING:", "Government Warning:"))
label("abv_mismatch.png", "SUNSET ALE", "6.2%", WARNING_CANONICAL)
print("samples written to", OUT)
