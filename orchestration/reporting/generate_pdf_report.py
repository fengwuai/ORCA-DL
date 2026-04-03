from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from orchestration.xiamen_uploader import upload_file_to_xiamen


ROOT = Path(__file__).resolve().parents[2]
TMP_BASE_DIR = ROOT / "tmp"
DEFAULT_REPORT_OUTPUT_DIR = ROOT / "output" / "reports"
REPORT_ASSETS_DIR = ROOT / "orchestration" / "reporting" / "assets"
REPORT_TEMPLATE_PATH = REPORT_ASSETS_DIR / "report_template.md"
REPORT_ANALYZER_PATH = REPORT_ASSETS_DIR / "analyzer.py"
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_MODEL_NAME = "doubao-seed-2-0-lite-260215"
PDF_MARGIN = "16mm"
PDF_FONT_CANDIDATES = (
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
)
REQUIRED_HEADING_SEQUENCE: list[tuple[int, str]] = [
    (2, "1. 摘要"),
    (2, "2. 气候趋势分析：ENSO 演变"),
    (3, "2.1 Nino 3.4 指数时间序列"),
    (3, "2.2 阶段划分"),
    (2, "3. 海洋环境可视化"),
    (3, "3.1 全球海温分布"),
    (3, "3.2 表面洋流强度"),
    (2, "4. 对航运行业的影响分析"),
    (3, "4.1 航运风险与建议清单"),
    (2, "5. 结论"),
]
PHASE_TABLE_HEADER = ["时间段", "状态", "距平特征", "依据"]
REPORT_IMAGE_FILES = [
    "nino34_timeseries.png",
    "sst_map_0.png",
    "sst_map_12.png",
    "sst_map_23.png",
    "mean_current_speed.png",
]
PLACEHOLDER_PATTERN = re.compile(r"{{\s*[^{}]+\s*}}")
ALIGNMENT_CELL_PATTERN = re.compile(r"^:?-{3,}:?$")


def load_root_env() -> None:
    env_path = ROOT / ".env"
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)


def parse_target_month(target_month: str) -> str:
    try:
        dt = datetime.strptime(target_month, "%Y-%m")
    except ValueError as exc:
        raise ValueError(f"target_month 格式错误：{target_month}，应为 YYYY-MM（如 2026-02）") from exc
    if dt.strftime("%Y-%m") != target_month:
        raise ValueError(f"target_month 格式错误：{target_month}，应为 YYYY-MM（如 2026-02）")
    return target_month


def resolve_local_pdf_path(target_month: str, output_dir: Path) -> Path:
    return output_dir / f"{target_month}.pdf"


def save_pdf_to_local_dir(pdf_path: Path, target_month: str, output_dir: Path) -> Path:
    local_pdf_path = resolve_local_pdf_path(target_month, output_dir=output_dir)
    local_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, local_pdf_path)
    return local_pdf_path


def validate_report_assets() -> None:
    if not REPORT_TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"报告模板不存在: {REPORT_TEMPLATE_PATH}")
    if not REPORT_ANALYZER_PATH.is_file():
        raise FileNotFoundError(f"报告分析脚本不存在: {REPORT_ANALYZER_PATH}")


def run_analyzer(analyzer_script: Path, input_netcdf: Path, asset_dir: Path) -> tuple[Path, Path]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            str(analyzer_script),
            "--input",
            str(input_netcdf),
            "--output-dir",
            str(asset_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "analyzer 执行失败\n"
            f"stdout:\n{result.stdout.strip()}\n"
            f"stderr:\n{result.stderr.strip()}"
        )

    stats_path = asset_dir / "stats_summary.txt"
    stats_json_path = asset_dir / "stats_summary.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"analyzer 未生成统计文件: {stats_path}")
    if not stats_json_path.is_file():
        raise FileNotFoundError(f"analyzer 未生成结构化统计文件: {stats_json_path}")

    for filename in REPORT_IMAGE_FILES:
        image_file = asset_dir / filename
        if not image_file.is_file():
            raise FileNotFoundError(f"报告图片缺失: {image_file}")

    return stats_path, stats_json_path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"结构化摘要格式错误，期望 JSON object: {path}")
    return loaded


def format_month_label(month_text: str, suffix: str) -> str:
    try:
        dt = datetime.strptime(month_text, "%Y-%m")
        return f"{dt.year}年{dt.month}月 ({suffix})"
    except ValueError:
        return f"{month_text} ({suffix})"


def format_month_compact(month_text: str) -> str:
    if re.fullmatch(r"\d{4}-\d{2}", month_text):
        dt = datetime.strptime(month_text, "%Y-%m")
        return f"{dt.year}.{dt.month}"
    return month_text


def build_template_fixed_values(summary_data: dict[str, Any]) -> dict[str, str]:
    dataset = summary_data.get("dataset", {})
    if not isinstance(dataset, dict):
        dataset = {}
    map_months = summary_data.get("map_months", {})
    if not isinstance(map_months, dict):
        map_months = {}

    period_start = str(dataset.get("time_start", "[NOT_AVAILABLE]"))
    period_end = str(dataset.get("time_end", "[NOT_AVAILABLE]"))

    report_title = (
        f"{format_month_compact(period_start)}-{format_month_compact(period_end)}"
        "全球海洋气候趋势与航运影响分析报告"
    )
    data_source = f"ORCA-DL Ocean State Predictions ({period_start} - {period_end})"

    return {
        "report_title": report_title,
        "report_date": datetime.now().strftime("%Y年%m月%d日"),
        "data_source": data_source,
        "period_start": period_start,
        "period_end": period_end,
        "sst_start_label": format_month_label(str(map_months.get("sst_map_0", "[NOT_AVAILABLE]")), "初期"),
        "sst_mid_label": format_month_label(str(map_months.get("sst_map_12", "[NOT_AVAILABLE]")), "中期"),
        "sst_end_label": format_month_label(str(map_months.get("sst_map_23", "[NOT_AVAILABLE]")), "末期"),
    }


def apply_template_fixed_values(template_markdown: str, fixed_values: dict[str, str]) -> str:
    rendered = template_markdown
    for key, value in fixed_values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def build_prompt(
    template_markdown: str,
    stats_summary_text: str,
    stats_summary_json: dict[str, Any],
    image_links: dict[str, str],
) -> list[dict[str, str]]:
    system_prompt = (
        "你是专业海洋气候与航运分析师。"
        "你必须遵守模板结构并且只使用输入数据。"
        "任何无法确定的信息统一填 [NOT_AVAILABLE]。"
    )

    user_prompt = f"""
请根据模板与新数据生成 markdown 报告。

硬性要求：
1. 严格保持模板的标题层级与顺序，不得新增或删除章节。
2. 必须填充模板中的占位符，最终输出不得出现任何 {{{{...}}}}。
3. 仅允许使用 <new_data_summary_json> 与 <new_data_summary_text> 可推断的信息，禁止臆造数值。
4. 图片链接必须逐字使用以下路径：
   - Nino 3.4 Time Series: {image_links['nino34']}
   - SST Map Start: {image_links['sst0']}
   - SST Map Mid: {image_links['sst12']}
   - SST Map End: {image_links['sst23']}
   - Mean Current Speed: {image_links['current']}
5. 2.2 阶段划分必须是 4 列表格（时间段/状态/距平特征/依据），至少 2 条数据行。
6. 4.1 航运风险与建议清单必须是编号列表，至少 3 条，每条必须包含以下字段：
   - 风险信号
   - 影响区域/航线
   - 时间窗
   - 运营影响
   - 建议动作
   - 置信度（仅可用 高/中/低）
7. 输出必须是纯 markdown，不要输出解释文字或代码块围栏。

<template>
{template_markdown}
</template>

<new_data_summary_json>
{json.dumps(stats_summary_json, ensure_ascii=False, indent=2)}
</new_data_summary_json>

<new_data_summary_text>
{stats_summary_text}
</new_data_summary_text>
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_report_with_ark(messages: list[dict[str, str]]) -> str:
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 ARK_API_KEY，请在仓库根目录 .env 中配置")

    client = OpenAI(api_key=api_key, base_url=ARK_BASE_URL)
    response = client.chat.completions.create(
        model=ARK_MODEL_NAME,
        messages=messages,
        temperature=0.2,
    )

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("火山 API 返回空内容")
    if not isinstance(content, str):
        content = str(content)
    return strip_markdown_fence(content)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def inlines_to_text(inlines: list[Any]) -> str:
    parts: list[str] = []

    def walk_inline(node: Any) -> None:
        if not isinstance(node, dict):
            return
        node_type = node.get("t")
        node_content = node.get("c")
        if node_type == "Str" and isinstance(node_content, str):
            parts.append(node_content)
            return
        if node_type in {"Space", "SoftBreak", "LineBreak"}:
            parts.append(" ")
            return
        if node_type == "Code" and isinstance(node_content, list) and len(node_content) > 1:
            parts.append(str(node_content[1]))
            return
        if node_type in {"Emph", "Strong", "Strikeout", "Underline", "SmallCaps", "Superscript", "Subscript"}:
            if isinstance(node_content, list):
                for item in node_content:
                    walk_inline(item)
            return
        if node_type in {"Link", "Image"} and isinstance(node_content, list) and len(node_content) > 1:
            inline_items = node_content[1]
            if isinstance(inline_items, list):
                for item in inline_items:
                    walk_inline(item)
            return
        if node_type == "Quoted" and isinstance(node_content, list) and len(node_content) > 1:
            inline_items = node_content[1]
            if isinstance(inline_items, list):
                for item in inline_items:
                    walk_inline(item)

    for inline in inlines:
        walk_inline(inline)

    return normalize_whitespace("".join(parts))


def parse_markdown_ast(report_text: str) -> dict[str, Any]:
    result = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "json"],
        input=report_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Markdown AST 解析失败\n"
            f"stdout:\n{result.stdout.strip()}\n"
            f"stderr:\n{result.stderr.strip()}"
        )
    parsed = json.loads(result.stdout)
    if not isinstance(parsed, dict):
        raise RuntimeError("Markdown AST 解析结果无效")
    return parsed


def collect_headings(ast_doc: dict[str, Any]) -> list[tuple[int, str]]:
    blocks = ast_doc.get("blocks")
    if not isinstance(blocks, list):
        return []

    headings: list[tuple[int, str]] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("t") != "Header":
            continue
        content = block.get("c")
        if not isinstance(content, list) or len(content) < 3:
            continue
        level = content[0]
        inlines = content[2]
        if not isinstance(level, int) or not isinstance(inlines, list):
            continue
        headings.append((level, inlines_to_text(inlines)))
    return headings


def collect_image_paths(ast_doc: dict[str, Any]) -> set[str]:
    paths: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("t") == "Image":
                content = node.get("c")
                if isinstance(content, list) and len(content) >= 3:
                    target = content[2]
                    if isinstance(target, list) and target:
                        paths.add(str(target[0]))
            for value in node.values():
                walk(value)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(ast_doc)
    return paths


def validate_heading_sequence(headings: list[tuple[int, str]]) -> None:
    cursor = 0
    missing: list[str] = []
    normalized_headings = [(level, normalize_whitespace(text)) for level, text in headings]
    for required_level, required_title in REQUIRED_HEADING_SEQUENCE:
        found = False
        normalized_required_title = normalize_whitespace(required_title)
        while cursor < len(normalized_headings):
            level, title = normalized_headings[cursor]
            cursor += 1
            if level == required_level and title == normalized_required_title:
                found = True
                break
        if not found:
            missing.append(f"{'#' * required_level} {required_title}")
    if missing:
        raise RuntimeError(f"报告格式校验失败，缺失或顺序错误的章节: {missing}")


def extract_section_text(report_text: str, heading_line: str) -> str:
    lines = report_text.splitlines()
    start_idx: int | None = None
    for idx, line in enumerate(lines):
        if line.strip() == heading_line:
            start_idx = idx + 1
            break
    if start_idx is None:
        raise RuntimeError(f"报告格式校验失败，未找到章节: {heading_line}")

    end_idx = len(lines)
    for idx in range(start_idx, len(lines)):
        if re.match(r"^#{1,6}\s+", lines[idx].strip()):
            end_idx = idx
            break
    return "\n".join(lines[start_idx:end_idx]).strip()


def parse_table_row(row_line: str) -> list[str]:
    return [cell.strip() for cell in row_line.strip().strip("|").split("|")]


def is_alignment_row(cells: list[str]) -> bool:
    return bool(cells) and all(ALIGNMENT_CELL_PATTERN.fullmatch(cell.replace(" ", "")) for cell in cells)


def parse_first_markdown_table(section_text: str) -> tuple[list[str], list[list[str]]]:
    lines = [line.rstrip() for line in section_text.splitlines()]
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if idx + 1 >= len(lines):
            continue
        separator = lines[idx + 1].strip()
        if not separator.startswith("|"):
            continue

        header_cells = parse_table_row(stripped)
        separator_cells = parse_table_row(separator)
        if not is_alignment_row(separator_cells):
            continue

        data_rows: list[list[str]] = []
        row_idx = idx + 2
        while row_idx < len(lines):
            candidate = lines[row_idx].strip()
            if not candidate.startswith("|"):
                break
            data_rows.append(parse_table_row(candidate))
            row_idx += 1
        return header_cells, data_rows

    raise RuntimeError("报告格式校验失败，未找到合法 markdown 表格")


def validate_table_structure(
    section_text: str,
    expected_header: list[str],
    min_rows: int,
    section_name: str,
) -> None:
    header, rows = parse_first_markdown_table(section_text)
    if header != expected_header:
        raise RuntimeError(
            f"报告格式校验失败，{section_name} 表头不匹配，期望: {expected_header}，实际: {header}"
        )
    if len(rows) < min_rows:
        raise RuntimeError(
            f"报告格式校验失败，{section_name} 数据行不足，至少需要 {min_rows} 行，实际 {len(rows)} 行"
        )
    expected_col_count = len(expected_header)
    for row in rows:
        if len(row) != expected_col_count:
            raise RuntimeError(
                f"报告格式校验失败，{section_name} 存在列数错误行，期望 {expected_col_count} 列，实际 {len(row)} 列"
            )


def parse_risk_list_items(section_text: str) -> list[list[str]]:
    item_start_pattern = re.compile(r"^\s*\d+\.\s+\*\*风险信号\*\*：\s*\S+")
    lines = section_text.splitlines()
    items: list[list[str]] = []
    current_item: list[str] | None = None

    for line in lines:
        if item_start_pattern.match(line):
            if current_item is not None:
                items.append(current_item)
            current_item = [line]
            continue
        if current_item is not None:
            current_item.append(line)

    if current_item is not None:
        items.append(current_item)
    return items


def validate_risk_list_structure(section_text: str, min_items: int = 3) -> None:
    items = parse_risk_list_items(section_text)
    if len(items) < min_items:
        raise RuntimeError(f"报告格式校验失败，4.1 航运风险与建议清单条目不足，至少需要 {min_items} 条")

    required_field_patterns = {
        "影响区域/航线": re.compile(r"^\s*-\s+\*\*影响区域/航线\*\*：\s*\S+"),
        "时间窗": re.compile(r"^\s*-\s+\*\*时间窗\*\*：\s*\S+"),
        "运营影响": re.compile(r"^\s*-\s+\*\*运营影响\*\*：\s*\S+"),
        "建议动作": re.compile(r"^\s*-\s+\*\*建议动作\*\*：\s*\S+"),
        "置信度": re.compile(r"^\s*-\s+\*\*置信度\*\*：\s*(高|中|低)\s*$"),
    }

    risk_signal_pattern = re.compile(r"^\s*\d+\.\s+\*\*风险信号\*\*：\s*\S+")
    for idx, item_lines in enumerate(items, start=1):
        if not item_lines or not risk_signal_pattern.match(item_lines[0]):
            raise RuntimeError(f"报告格式校验失败，4.1 第 {idx} 条缺少“风险信号”字段")
        for field_name, field_pattern in required_field_patterns.items():
            if not any(field_pattern.match(line) for line in item_lines):
                raise RuntimeError(f"报告格式校验失败，4.1 第 {idx} 条缺少或非法字段: {field_name}")


def validate_report_markdown(report_text: str, image_links: dict[str, str]) -> None:
    if PLACEHOLDER_PATTERN.search(report_text):
        raise RuntimeError("报告格式校验失败，存在未替换的模板占位符")
    if "data/output/" in report_text:
        raise RuntimeError("报告格式校验失败，检测到旧路径 data/output/，应改为 assets/")

    ast_doc = parse_markdown_ast(report_text)
    headings = collect_headings(ast_doc)
    validate_heading_sequence(headings)

    image_paths = collect_image_paths(ast_doc)
    missing_images = [path for path in image_links.values() if path not in image_paths]
    if missing_images:
        raise RuntimeError(f"报告格式校验失败，缺失图片链接: {missing_images}")

    phase_section = extract_section_text(report_text, "### 2.2 阶段划分")
    validate_table_structure(
        section_text=phase_section,
        expected_header=PHASE_TABLE_HEADER,
        min_rows=2,
        section_name="2.2 阶段划分",
    )

    risk_section = extract_section_text(report_text, "### 4.1 航运风险与建议清单")
    validate_risk_list_structure(risk_section, min_items=3)


def convert_markdown_to_pdf(markdown_path: Path, output_pdf: Path, work_dir: Path) -> None:
    font_list_result = subprocess.run(
        ["typst", "fonts"],
        capture_output=True,
        text=True,
        check=False,
    )
    if font_list_result.returncode != 0:
        raise RuntimeError(
            "无法获取 typst 字体列表\n"
            f"stdout:\n{font_list_result.stdout.strip()}\n"
            f"stderr:\n{font_list_result.stderr.strip()}"
        )

    available_fonts = {line.strip().lower() for line in font_list_result.stdout.splitlines() if line.strip()}
    pdf_main_font = next(
        (font for font in PDF_FONT_CANDIDATES if font.lower() in available_fonts),
        None,
    )
    if pdf_main_font is None:
        raise RuntimeError(
            "未找到可用中文字体，请在运行环境安装 CJK 字体（建议 fonts-noto-cjk），"
            f"候选字体: {', '.join(PDF_FONT_CANDIDATES)}"
        )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".typ",
        dir=work_dir,
        delete=False,
    ) as margin_override_file:
        margin_override_file.write(f"#set page(margin: (x: {PDF_MARGIN}, y: {PDF_MARGIN}))\n")
        margin_override_path = Path(margin_override_file.name)

    try:
        result = subprocess.run(
            [
                "pandoc",
                str(markdown_path),
                "--pdf-engine=typst",
                "-V",
                f"mainfont={pdf_main_font}",
                "--include-before-body",
                str(margin_override_path),
                "--resource-path",
                str(work_dir),
                "-o",
                str(output_pdf),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        margin_override_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(
            "pandoc 转换 PDF 失败\n"
            f"stdout:\n{result.stdout.strip()}\n"
            f"stderr:\n{result.stderr.strip()}"
        )


def upload_pdf_to_xiamen(pdf_path: Path, target_month: str) -> str:
    return upload_file_to_xiamen(pdf_path, object_name=f"{target_month}.pdf")


def generate_pdf_report(
    target_month: str,
    input_netcdf_path: str | Path,
    output_dir: str | Path = DEFAULT_REPORT_OUTPUT_DIR,
) -> str:
    resolved_month = parse_target_month(target_month)
    load_root_env()
    resolved_input_netcdf = Path(input_netcdf_path).expanduser().resolve()
    if not resolved_input_netcdf.is_file():
        raise FileNotFoundError(f"预测文件不存在: {resolved_input_netcdf}")
    resolved_output_dir = Path(output_dir).expanduser().resolve()

    validate_report_assets()

    TMP_BASE_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=TMP_BASE_DIR, prefix="orca_report_") as tmp_dir:
        temp_dir = Path(tmp_dir)
        temp_asset_dir = temp_dir / "assets"
        temp_markdown_path = temp_dir / "report.md"

        stats_path, stats_json_path = run_analyzer(
            analyzer_script=REPORT_ANALYZER_PATH,
            input_netcdf=resolved_input_netcdf,
            asset_dir=temp_asset_dir,
        )

        image_links = {
            "nino34": "assets/nino34_timeseries.png",
            "sst0": "assets/sst_map_0.png",
            "sst12": "assets/sst_map_12.png",
            "sst23": "assets/sst_map_23.png",
            "current": "assets/mean_current_speed.png",
        }

        stats_summary_json = read_json(stats_json_path)
        template_markdown = apply_template_fixed_values(
            template_markdown=read_text(REPORT_TEMPLATE_PATH),
            fixed_values=build_template_fixed_values(stats_summary_json),
        )

        messages = build_prompt(
            template_markdown=template_markdown,
            stats_summary_text=read_text(stats_path),
            stats_summary_json=stats_summary_json,
            image_links=image_links,
        )
        markdown_text = generate_report_with_ark(messages)
        validate_report_markdown(markdown_text, image_links)

        temp_markdown_path.write_text(markdown_text + "\n", encoding="utf-8")
        temp_pdf_path = temp_dir / f"{resolved_month}.pdf"
        convert_markdown_to_pdf(
            markdown_path=temp_markdown_path,
            output_pdf=temp_pdf_path,
            work_dir=temp_dir,
        )
        save_pdf_to_local_dir(temp_pdf_path, resolved_month, output_dir=resolved_output_dir)
        return upload_pdf_to_xiamen(temp_pdf_path, resolved_month)

    raise RuntimeError("报告上传失败")
