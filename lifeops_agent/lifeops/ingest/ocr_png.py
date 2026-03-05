from __future__ import annotations

"""lifeops.ingest.ocr_png

这个模块负责把 PNG 图片（例如截图）中的文字提取出来。

初学者可把它理解为两步：
1) 图像预处理：把“人眼看得清、机器未必看得清”的截图先加工成更适合 OCR 的样子。
2) OCR 识别：调用 Tesseract 把处理后的图片转换成文本。

为什么这里要做预处理：
- 截图中的字体可能很小，直接识别容易漏字。
- 背景、抗锯齿、颜色噪声会降低识别率。
- 通过放大、灰度、增强对比、二值化、锐化，通常可以明显提升识别质量。

默认参数（可在 .env 调整）：
- OCR_LANG=chi_sim+eng
- OCR_PSM=6  （适合“块状/列表”文本）
- OCR_OEM=3
- OCR_DEBUG=1 保存预处理图到 ./data/ocr_debug，方便人工排查
"""

import os
from datetime import datetime

from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

from ..settings import settings


def _preprocess(img: Image.Image) -> Image.Image:
    """对输入图片做 OCR 友好的预处理。

    预处理顺序：
    1) 放大
    2) 灰度
    3) 对比度增强
    4) 二值化
    5) 锐化

    说明：
    - 这个顺序来自常见 OCR 工程实践，不是唯一最佳方案。
    - 如果你的数据类型变化较大，可按样本调节 scale/threshold。
    """

    # 1) 放大：小字识别的关键步骤之一。
    # 很多截图在原分辨率下文字笔画太细，先放大能提升 OCR 稳定性。
    scale = 2

    # 转成 RGB 后再缩放，兼容更多输入模式（如带透明通道的图片）。
    img = img.convert("RGB")

    # 使用 LANCZOS 重采样，缩放后的边缘更平滑，细节保留较好。
    img = img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)

    # 2) 转灰度：减少颜色维度，让后续阈值处理更直接。
    img = img.convert("L")

    # 3) 对比度增强：拉大“字 vs 背景”的灰度差。
    img = ImageEnhance.Contrast(img).enhance(2.0)

    # 4) 二值化：把像素压缩为黑/白两类，减轻背景噪声干扰。
    # threshold 越高，保留的亮部越多；越低，保留的暗部越多。
    threshold = 170
    img = img.point(lambda p: 255 if p > threshold else 0)

    # 5) 轻微锐化：增强边缘，帮助识别细笔画。
    img = img.filter(ImageFilter.SHARPEN)

    return img


def extract_png(path: str) -> str:
    """读取 PNG 并执行 OCR，返回识别文本。

    参数：
    - path: PNG 文件路径。

    返回：
    - OCR 识别得到的字符串。

    注意：
    - 如果配置了 TESSERACT_CMD，会优先使用该路径。
    - OCR_DEBUG 开启时会保存预处理结果图，方便你肉眼排查“为什么识别差”。
    """

    # 指定 tesseract 可执行文件路径（当用户在 .env 中显式配置时）。
    # 这样可以避免 PATH 未生效导致的 “TesseractNotFoundError”。
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    # 打开原始图片。
    image = Image.open(path)

    # 先做预处理，再交给 OCR。
    image = _preprocess(image)

    # 组装 Tesseract 配置参数：
    # - oem: OCR 引擎模式
    # - psm: 页面分割模式
    config = f"--oem {settings.ocr_oem} --psm {settings.ocr_psm}"

    # 若开启调试，把预处理后的图片落盘，便于复盘识别效果。
    if settings.ocr_debug:
        # 确保调试目录存在。
        os.makedirs("data/ocr_debug", exist_ok=True)

        # 生成带时间戳的文件名，避免覆盖历史调试图。
        base = os.path.basename(path)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = os.path.join("data", "ocr_debug", f"{stamp}_{base}")

        # 调试保存是“附加能力”，不应影响主流程。
        # 因此这里做宽松异常处理：保存失败直接忽略。
        try:
            image.save(debug_path)
        except Exception:
            pass

    # 调用 Tesseract 识别并返回文本。
    # lang 可配置为中英混合（chi_sim+eng）。
    return pytesseract.image_to_string(image, lang=settings.ocr_lang, config=config)
