from __future__ import annotations

import os
from PIL import Image
import pytesseract

from ..settings import settings


def extract_png(path: str) -> str:
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    image = Image.open(path)
    return pytesseract.image_to_string(image)

