from __future__ import annotations

from pathlib import Path

from orchestration.reporting.generate_pdf_report import convert_markdown_to_pdf


def debug_markdown_to_pdf(
    markdown_path: str | Path,
    output_pdf_path: str | Path | None = None,
    resource_dir: str | Path | None = None,
) -> Path:
    """
    调试用：单独执行 Markdown -> PDF 转换，不依赖推理、分析和上传流程。

    Args:
        markdown_path: 输入 markdown 文件路径
        output_pdf_path: 输出 PDF 路径，默认与 markdown 同目录同名 .pdf
        resource_dir: pandoc 资源目录，默认 markdown 所在目录

    Returns:
        生成的 PDF 绝对路径
    """
    resolved_markdown_path = Path(markdown_path).expanduser().resolve()
    if not resolved_markdown_path.is_file():
        raise FileNotFoundError(f"markdown 文件不存在: {resolved_markdown_path}")

    if output_pdf_path is None:
        resolved_output_pdf_path = resolved_markdown_path.with_suffix(".pdf")
    else:
        resolved_output_pdf_path = Path(output_pdf_path).expanduser().resolve()
    resolved_output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if resource_dir is None:
        resolved_resource_dir = resolved_markdown_path.parent
    else:
        resolved_resource_dir = Path(resource_dir).expanduser().resolve()
    if not resolved_resource_dir.is_dir():
        raise NotADirectoryError(f"resource_dir 不是有效目录: {resolved_resource_dir}")

    convert_markdown_to_pdf(
        markdown_path=resolved_markdown_path,
        output_pdf=resolved_output_pdf_path,
        work_dir=resolved_resource_dir,
    )
    return resolved_output_pdf_path
