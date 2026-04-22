"""
ORCA-DL 数据预处理 CLI（下载 + 预处理）。

使用方式：
    pixi run -e model preprocess 2026-02
    pixi run -e model preprocess 2026-02 --source psl
    pixi run -e model preprocess 2026-02 --processed-dir ./output/preprocessed/custom
"""

import argparse
import sys

from inference import run_data_preprocess_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORCA-DL 数据预处理（下载 + 预处理）")
    parser.add_argument(
        "target_month",
        help="目标月份，格式 YYYY-MM（如 2026-02）",
    )
    parser.add_argument(
        "--source",
        choices=["cpc", "psl"],
        default="cpc",
        help="数据源：cpc（默认）或 psl",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="原始数据目录（可选，不传则按 source 使用默认值）",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="预处理输出目录（可选，不传则默认 ./output/preprocessed/{YYYY-MM}/{source}）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = run_data_preprocess_pipeline(
        target_month=args.target_month,
        source=args.source,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )
    print(f"预处理完成，输出目录：{processed_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误：{e}", file=sys.stderr)
        sys.exit(1)
