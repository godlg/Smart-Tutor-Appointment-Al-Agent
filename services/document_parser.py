"""
文档解析器 (P1-13)

把上传的 PDF / Markdown 文件字节流解析成纯文本,作为下游分块器的输入。

设计要点:
- markitdown 在 parse_pdf 函数体内 lazy import —— 模块顶部不强制拉入
  pymupdf / mammoth 等重依赖,服务启动不受影响。
- ParsedDocument 是 frozen dataclass,不可变。
- Markdown 解码用 utf-8-sig (兼容 BOM);首行 `# heading` 抽作 title。
- detect_kind 用扩展名优先, content-type 兜底;不支持的类型抛 UnsupportedFileType。
- 空文本抛 EmptyDocument,让 API 层映射到 400。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from typing import Literal, Optional


class UnsupportedFileType(Exception):
    """文件扩展名或 content-type 不在白名单。"""
    pass


class EmptyDocument(Exception):
    """解析结果为空 (PDF 是图片型扫描件 / md 文件全是空白)。"""
    pass


class DocumentParseError(Exception):
    """PDF 解析过程出错 (损坏 / 加密 / OCR 缺失等)。"""
    pass


SUPPORTED_EXTENSIONS = frozenset({".pdf", ".md", ".markdown"})

# content-type 白名单 (兜底,扩展名优先)
_CONTENT_TYPE_MAP = {
    "application/pdf": "pdf",
    "text/markdown": "markdown",
    "text/x-markdown": "markdown",
}


@dataclass(frozen=True)
class ParsedDocument:
    """解析后的纯文档。"""
    text: str
    title: Optional[str] = None
    page_count: Optional[int] = None


# ----------------------------------------------------------------------
# 类型识别
# ----------------------------------------------------------------------

def detect_kind(
    filename: str, content_type: Optional[str] = None,
) -> Literal["pdf", "markdown"]:
    """根据文件名扩展名(优先)和 content-type(兜底)识别文件类型。

    Args:
        filename: 原始文件名,例如 "notes.pdf"
        content_type: HTTP 上传的 Content-Type,例如 "text/markdown"

    Returns:
        "pdf" 或 "markdown"

    Raises:
        UnsupportedFileType: 无法识别
    """
    if filename:
        lower = filename.lower()
        for ext in SUPPORTED_EXTENSIONS:
            if lower.endswith(ext):
                return "pdf" if ext == ".pdf" else "markdown"
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        if ct in _CONTENT_TYPE_MAP:
            return _CONTENT_TYPE_MAP[ct]
    raise UnsupportedFileType(
        f"不支持的文件类型: filename={filename!r}, content_type={content_type!r}"
    )


# ----------------------------------------------------------------------
# 顶层入口
# ----------------------------------------------------------------------

def parse(
    file_bytes: bytes,
    filename: str,
    content_type: Optional[str] = None,
) -> tuple[Literal["pdf", "markdown"], ParsedDocument]:
    """顶层入口:detect_kind → 分发 → 返回 (kind, parsed)。"""
    kind = detect_kind(filename, content_type)
    if kind == "pdf":
        parsed = parse_pdf(file_bytes)
    else:
        parsed = parse_markdown(file_bytes)
    return kind, parsed


# ----------------------------------------------------------------------
# Markdown
# ----------------------------------------------------------------------

_HEADING_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


def _extract_first_heading(text: str) -> Optional[str]:
    """取第一行 `# heading` 的标题文字,无则 None。"""
    m = _HEADING_RE.search(text)
    return m.group(1).strip() if m else None


def parse_markdown(file_bytes: bytes) -> ParsedDocument:
    """解码 markdown 字节流,返回纯文本与(可选)title。

    Raises:
        EmptyDocument: 解码后内容为空
    """
    try:
        text = file_bytes.decode("utf-8-sig")  # 兼容 BOM
    except UnicodeDecodeError:
        # 兜底:用 gbk 解码中文 markdown(部分 Windows 工具会这样写)
        try:
            text = file_bytes.decode("gbk")
        except UnicodeDecodeError as e:
            raise DocumentParseError(f"markdown 解码失败: {e}")
    if not text or not text.strip():
        raise EmptyDocument("markdown 内容为空")
    return ParsedDocument(
        text=text,
        title=_extract_first_heading(text),
        page_count=None,
    )


# ----------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------

def parse_pdf(file_bytes: bytes) -> ParsedDocument:
    """用 markitdown 把 PDF 字节流转成 markdown。

    Raises:
        DocumentParseError: markitdown 抛错 (损坏 PDF / 加密 / OCR 缺失)
        EmptyDocument: 解析结果为空 (扫描件) 或输入为空
    """
    if not file_bytes:
        raise EmptyDocument("PDF 内容为空")
    try:
        # lazy import: 模块顶部不强制拉入 pymupdf 等重包
        from markitdown import MarkItDown
    except ImportError as e:
        raise DocumentParseError(
            f"markitdown 未安装,无法解析 PDF: {e}"
        )

    try:
        m = MarkItDown()
        # markitdown 接受 file-like 或 path,BytesIO 是最简方式
        result = m.convert_stream(BytesIO(file_bytes))
    except Exception as e:
        raise DocumentParseError(f"PDF 解析失败: {e}")

    text = (getattr(result, "markdown_content", None) or "").strip()
    if not text:
        raise EmptyDocument("PDF 解析结果为空 (可能为扫描件)")
    title = _extract_first_heading(text)
    return ParsedDocument(text=text, title=title, page_count=None)