from __future__ import annotations

from dataclasses import dataclass
from pypdf import PdfReader


@dataclass
class PageText:
    """表示 PDF 中“单页文本”的结构化结果。"""

    # page: 页码（从 1 开始，便于和阅读器页面编号对齐）。
    page: int
    # text: 该页提取到的纯文本内容。
    text: str


def extract_pdf(path: str) -> list[PageText]:
    """读取 PDF，按页提取文本。

    参数：
    - path: PDF 文件路径。

    返回：
    - list[PageText]，列表中每个元素对应 PDF 的一页。

    关键设计：
    - 我们保留“页码 -> 文本”的映射，后续做引用展示时可以告诉用户答案来自第几页。
    - 即便某页提取不到文本，也会返回空字符串（而不是报错），保证处理链条稳定。
    """

    # 创建 PDF 读取器。pypdf 会解析文件结构并暴露 pages 列表。
    reader = PdfReader(path)

    # 用于累积每一页的提取结果。
    pages: list[PageText] = []

    # enumerate(..., start=1) 让页码从 1 开始，更符合用户阅读习惯。
    for idx, page in enumerate(reader.pages, start=1):
        # extract_text() 可能返回 None（例如扫描版 PDF 或该页无可提取文字）。
        # 这里统一兜底为空字符串，避免后续代码处理 None 分支。
        text = page.extract_text() or ""

        # 把“页码 + 文本”封装成结构化对象，加入结果列表。
        pages.append(PageText(page=idx, text=text))

    # 返回整份 PDF 的逐页文本结果。
    return pages
