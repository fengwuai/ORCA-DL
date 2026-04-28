#!/usr/bin/env python3
"""批量执行仅推理流程，输出 NetCDF 文件。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


INFERENCE_CMD = ["pixi", "run", "-e", "model", "inference"]
DEFAULT_OUTPUT_DIR = "./output/models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量执行仅推理流程（仅输出 NC）")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="要运行的年份列表，如：2014 2015 2016",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        type=int,
        default=list(range(1, 13)),
        help="要运行的月份列表（1-12），默认全部月份",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"NC 输出目录，默认 {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="遇到错误时立即停止（默认遇错继续）",
    )
    return parser.parse_args()


def validate_months(months: list[int]) -> list[int]:
    invalid_months = [month for month in months if month < 1 or month > 12]
    if invalid_months:
        raise ValueError(f"月份仅支持 1-12，当前非法值：{sorted(set(invalid_months))}")
    return sorted(set(months))


def validate_years(years: list[int]) -> list[int]:
    invalid_years = [year for year in years if year <= 0]
    if invalid_years:
        raise ValueError(f"年份必须为正整数，当前非法值：{sorted(set(invalid_years))}")
    return sorted(set(years))


def build_target_months(years: list[int], months: list[int]) -> list[str]:
    return [f"{year}-{month:02d}" for year in years for month in months]


def run_inference(target_month: str, output_dir: Path) -> subprocess.CompletedProcess[str]:
    command = [
        *INFERENCE_CMD,
        target_month,
        "--source",
        "cpc",
        "--output-dir",
        str(output_dir),
    ]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def main() -> int:
    args = parse_args()
    years = validate_years(args.years)
    months = validate_months(args.months)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    continue_on_error = not args.stop_on_error

    target_months = build_target_months(years=years, months=months)

    print("=" * 72)
    print("批量仅推理任务（仅输出 NC）")
    print("=" * 72)
    print(f"年份: {years}")
    print(f"月份: {months}")
    print("数据源: cpc（固定）")
    print(f"输出目录: {output_dir}")
    print(f"执行模式: {'遇错继续' if continue_on_error else '遇错停止'}")
    print(f"总任务数: {len(target_months)}")
    print("=" * 72)

    skip_count = 0
    success_count = 0
    failed_items: list[tuple[str, int, str]] = []

    for index, target_month in enumerate(target_months, start=1):
        nc_path = output_dir / f"{target_month}.nc"
        prefix = f"[{index}/{len(target_months)}] {target_month}"

        if nc_path.exists():
            print(f"{prefix} [SKIP] 已存在: {nc_path}")
            skip_count += 1
            continue

        print(f"{prefix} [RUN ] 开始推理")
        result = run_inference(target_month=target_month, output_dir=output_dir)
        if result.returncode == 0:
            print(f"{prefix} [OK  ] 完成: {nc_path}")
            success_count += 1
            continue

        stderr = (result.stderr or "").strip()
        first_line = stderr.splitlines()[0] if stderr else "未知错误"
        print(f"{prefix} [FAIL] 退出码={result.returncode} 错误={first_line}")
        failed_items.append((target_month, result.returncode, first_line))

        if not continue_on_error:
            break

    print("\n" + "=" * 72)
    print("批量任务汇总")
    print("=" * 72)
    print(f"总任务数: {len(target_months)}")
    print(f"跳过: {skip_count}")
    print(f"成功: {success_count}")
    print(f"失败: {len(failed_items)}")

    if failed_items:
        print("\n失败月份清单:")
        for month, code, error in failed_items:
            print(f"- {month}: exit={code}, error={error}")

    return 1 if failed_items else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        sys.exit(1)
