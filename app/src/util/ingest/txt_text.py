from __future__ import annotations


def extract_txt(path: str) -> str:
    """读取 TXT 文件并返回完整文本。

    参数：
    - path: TXT 文件的绝对路径或相对路径。

    返回：
    - 文件全部文本（字符串）。

    设计说明：
    - 使用 UTF-8 作为默认编码，兼容大多数中文文本。
    - errors="ignore" 表示遇到坏字节时跳过，避免因为少量编码问题导致整份文件读取失败。
      这个策略更偏“鲁棒优先”（能读尽量读），适合知识库批处理场景。
    """

    # 以“文本模式 + UTF-8 编码”打开文件。
    # 如果文件里有个别非法字节，errors="ignore" 会跳过它们，继续读取后续内容。
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # 一次性读取全文并返回给上层。
        return f.read()
