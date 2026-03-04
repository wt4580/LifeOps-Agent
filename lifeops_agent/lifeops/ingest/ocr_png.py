from __future__ import annotations

"""lifeops.ingest.ocr_png

PNG / 截图类图片的 OCR 提取。

为什么要做预处理：
- 资源管理器/网页截图常见字体小、抗锯齿、背景不纯，直接 OCR 很容易只识别出日期/数字，丢失中文文件夹名。
- 预处理（放大、灰度、对比度增强、二值化、锐化）能显著提高识别率。

默认参数（可在 .env 调整）：
- OCR_LANG=chi_sim+eng
- OCR_PSM=6  （适合“块状/列表”文本）
- OCR_OEM=3
- OCR_DEBUG=1 保存预处理图到 ./data/ocr_debug 方便肉眼检查
"""

import os
from datetime import datetime

from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

from ..settings import settings


def _preprocess(img: Image.Image) -> Image.Image:
    # 1) 放大：小字识别的关键（Windows 截图很常见）
    scale = 2
    img = img.convert("RGB")
    img = img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)

    # 2) 灰度
    img = img.convert("L")

    # 3) 对比度增强
    img = ImageEnhance.Contrast(img).enhance(2.0)

    # 4) 二值化（阈值可按需调）
    threshold = 170
    img = img.point(lambda p: 255 if p > threshold else 0)

    # 5) 轻微锐化
    img = img.filter(ImageFilter.SHARPEN)
    return img


def extract_png(path: str) -> str:
    # 指定 tesseract 路径（如果用户在 .env 里设置了）
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    image = Image.open(path)
    image = _preprocess(image)

    # OCR 参数：对列表/截图更友好
    config = f"--oem {settings.ocr_oem} --psm {settings.ocr_psm}"

    if settings.ocr_debug:
        os.makedirs("data/ocr_debug", exist_ok=True)
        base = os.path.basename(path)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = os.path.join("data", "ocr_debug", f"{stamp}_{base}")
        try:
            image.save(debug_path)
        except Exception:
            pass

    return pytesseract.image_to_string(image, lang=settings.ocr_lang, config=config)
