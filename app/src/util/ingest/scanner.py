from __future__ import annotations

"""lifeops.ingest.scanner

这个模块是 ingest 层的“总调度器”
- 负责递归扫描目录
- 按文件类型把任务分发给对应提取器（PDF/TXT/PNG）
- 把提取结果统一包装成 Document / TextChunk 结构，供上层索引逻辑使用

为什么需要这个模块
- 上层只想知道“有哪些文档、每个文档有哪些文本块”，不想关心底层文件解析细节
- 因此 scanner 把不同来源的文本统一成同一接口，简化后续处理
"""

import os
from dataclasses import dataclass

from .pdf_text import extract_pdf
from .txt_text import extract_txt
from .ocr_png import extract_png


@dataclass
class TextChunk:
    """统一的文本块结构。"""

    # text: 该块的文本内容。
    text: str
    # page: 来源页码（仅 PDF 常见）；TXT/PNG 通常为 None。
    page: int | None = None


@dataclass
class Document:
    """统一的文档结构：一个路径 + 多个文本块。"""

    # path: 原始文件路径，用于引用可追溯。
    path: str
    # chunks: 文档被切成的多个 TextChunk。
    chunks: list[TextChunk]


def _chunk_text(text: str, size: int = 800) -> list[str]:
    """把长文本按固定长度切块（无重叠）。

    参数：
    - text: 原始长文本。
    - size: 每块最大字符数，默认 800。

    返回：
    - list[str]，每个元素是一段切分后的文本。

    说明：
    - 这是 MVP 版本的简单切分策略，优点是实现直观、性能稳定。
    - 缺点是语义边界不一定优雅（可能切在句子中间）。
      后续可升级成带 overlap 的递归切分器。
    """

    # 去掉首尾空白，避免产生只有空格的 chunk。
    text = text.strip()

    # 空文本直接返回空列表，避免进入后续循环。
    if not text:
        return []

    # parts 用来收集切分结果。
    parts = []

    # start 是当前切片起始下标。
    start = 0

    # 每次取 [start:start+size] 这一段，然后把 start 向后推进 size。
    while start < len(text):
        parts.append(text[start : start + size])
        start += size

    return parts


def scan_documents(root_dir: str) -> list[Document]:
    """递归扫描目录并提取支持格式的文本。

    参数：
    - root_dir: 知识库根目录，会递归扫描其所有子目录。

    返回：
    - list[Document]：每个文件对应一个 Document，内部包含多个 TextChunk。

    处理规则：
    - PDF：按页提取，保留 page。
    - TXT：全文读取后按固定长度分块。
    - PNG：OCR 后按固定长度分块。
    - 其他格式：跳过。
    """

    # 最终返回给上层的文档集合。
    documents: list[Document] = []

    # os.walk 会递归遍历 root_dir 下所有子目录。
    # dirpath: 当前目录；filenames: 当前目录下文件名列表。
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 拼出文件完整路径。
            path = os.path.join(dirpath, filename)

            # 统一把扩展名转小写，避免 .PDF / .Pdf 这类大小写差异。
            ext = os.path.splitext(filename)[1].lower()

            if ext == ".pdf":
                # PDF：逐页提取，天然带页码。
                pages = extract_pdf(path)
                # 过滤掉空页文本（p.text 为空字符串时不入块）。
                chunks = [TextChunk(text=p.text, page=p.page) for p in pages if p.text]
            elif ext == ".txt":
                # TXT：读取全文后做固定长度分块。
                text = extract_txt(path)
                chunks = [TextChunk(text=chunk) for chunk in _chunk_text(text)]
            elif ext == ".png":
                # PNG：先 OCR 成文本，再做固定长度分块。
                text = extract_png(path)
                chunks = [TextChunk(text=chunk) for chunk in _chunk_text(text)]
            else:
                # 不支持的格式直接跳过。
                continue

            # 把当前文件的结构化结果加入总列表。
            documents.append(Document(path=path, chunks=chunks))

    return documents

