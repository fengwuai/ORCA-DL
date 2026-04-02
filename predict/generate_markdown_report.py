from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
TMP_BASE_DIR = ROOT / "tmp"
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_MODEL_NAME = "doubao-seed-2-0-lite-260215"
REQUIRED_SECTIONS = [
    "## 1. 摘要",
    "## 2. 气候趋势分析：ENSO 演变",
    "### 2.1 Nino 3.4 指数时间序列",
    "### 2.2 阶段划分",
    "## 3. 海洋环境可视化",
    "### 3.1 全球海温分布",
    "### 3.2 表面洋流强度",
    "## 4. 对航运行业的影响分析",
    "## 5. 结论",
]


def load_root_env() -> None:
    env_path = ROOT / ".env"
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)


def find_demo_dir() -> Path:
    demo_root = ROOT / "demo"
    candidates = sorted(demo_root.glob("*自动化海洋报告生成"))
    if not candidates:
        candidates = sorted(path for path in demo_root.iterdir() if path.is_dir())

    for path in candidates:
        if (path / "analyzer.py").is_file() and (path / "report.md").is_file():
            return path

    raise FileNotFoundError("未找到可用的 demo 报告目录（需包含 analyzer.py 和 report.md）")


def run_analyzer(analyzer_script: Path, input_netcdf: Path, asset_dir: Path) -> Path:
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
    if not stats_path.is_file():
        raise FileNotFoundError(f"analyzer 未生成统计文件: {stats_path}")
    return stats_path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def to_posix_relative(path: Path, base: Path) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:markdown|md)?\\s*", "", cleaned)
    cleaned = re.sub(r"\\s*```$", "", cleaned)
    return cleaned.strip()


def build_prompt(
    template_markdown: str,
    stats_summary: str,
    image_links: dict[str, str],
) -> list[dict[str, str]]:
    system_prompt = (
        "你是专业海洋气候与航运分析师。"
        "请严格复用参考报告的章节结构、标题层级、表格样式与写作风格，"
        "仅替换为新数据对应的内容。"
    )

    user_prompt = f"""
请根据给定模板与新数据摘要，生成新的 markdown 报告。

要求：
1. 一级标题到四级标题、章节顺序、表格结构与模板一致。
2. 只使用 <new_data_summary> 中可推断的数据，不臆造来源外数值。
3. 图片链接必须严格使用以下路径（不要改名）：
   - Nino 3.4 Time Series: {image_links['nino34']}
   - SST Map Start: {image_links['sst0']}
   - SST Map Mid: {image_links['sst12']}
   - SST Map End: {image_links['sst23']}
   - Mean Current Speed: {image_links['current']}
4. 输出必须是纯 markdown，不要输出解释文字或代码块围栏。

<reference_report>
{template_markdown}
</reference_report>

<new_data_summary>
{stats_summary}
</new_data_summary>
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


def validate_report_markdown(report_text: str, image_links: dict[str, str]) -> None:
    missing_sections = [section for section in REQUIRED_SECTIONS if section not in report_text]
    if missing_sections:
        raise RuntimeError(f"报告格式校验失败，缺失章节: {missing_sections}")

    missing_images = [path for path in image_links.values() if path not in report_text]
    if missing_images:
        raise RuntimeError(f"报告格式校验失败，缺失图片链接: {missing_images}")


def generate_report(input_netcdf: Path, output_markdown: Path) -> Path:
    if not input_netcdf.is_file():
        raise FileNotFoundError(f"预测文件不存在: {input_netcdf}")

    load_root_env()

    demo_dir = find_demo_dir()
    template_path = demo_dir / "report.md"
    analyzer_script = demo_dir / "analyzer.py"
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    final_asset_dir = output_markdown.parent / f"{output_markdown.stem}_assets"
    final_asset_dir.mkdir(parents=True, exist_ok=True)

    TMP_BASE_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=TMP_BASE_DIR, prefix="orca_report_") as tmp_dir:
        temp_asset_dir = Path(tmp_dir) / "assets"
        stats_path = run_analyzer(
            analyzer_script=analyzer_script,
            input_netcdf=input_netcdf,
            asset_dir=temp_asset_dir,
        )

        image_links = {
            "nino34": to_posix_relative(final_asset_dir / "nino34_timeseries.png", output_markdown.parent),
            "sst0": to_posix_relative(final_asset_dir / "sst_map_0.png", output_markdown.parent),
            "sst12": to_posix_relative(final_asset_dir / "sst_map_12.png", output_markdown.parent),
            "sst23": to_posix_relative(final_asset_dir / "sst_map_23.png", output_markdown.parent),
            "current": to_posix_relative(final_asset_dir / "mean_current_speed.png", output_markdown.parent),
        }

        messages = build_prompt(
            template_markdown=read_text(template_path),
            stats_summary=read_text(stats_path),
            image_links=image_links,
        )
        markdown_text = generate_report_with_ark(messages)
        validate_report_markdown(markdown_text, image_links)

        for filename in [
            "nino34_timeseries.png",
            "sst_map_0.png",
            "sst_map_12.png",
            "sst_map_23.png",
            "mean_current_speed.png",
        ]:
            src_file = temp_asset_dir / filename
            if not src_file.is_file():
                raise FileNotFoundError(f"报告图片缺失: {src_file}")
            shutil.copy2(src_file, final_asset_dir / filename)

    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(markdown_text + "\n", encoding="utf-8")
    return output_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成海洋预测 markdown 报告")
    parser.add_argument("--input-netcdf", required=True, help="预测 NetCDF 文件路径")
    parser.add_argument("--output-markdown", required=True, help="输出 markdown 文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = generate_report(
        input_netcdf=Path(args.input_netcdf).resolve(),
        output_markdown=Path(args.output_markdown).resolve(),
    )
    print(f"报告生成完成: {output_path}")


if __name__ == "__main__":
    main()
