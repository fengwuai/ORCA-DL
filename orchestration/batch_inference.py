#!/usr/bin/env python3
"""批量运行多年份的推理和报告生成"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestration.prefect_monthly_inference import run_monthly_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行多年份的推理和报告生成")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="要运行的年份列表，如：2014 2015 2016 2022 2023 2024",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        type=int,
        default=list(range(1, 13)),
        help="要运行的月份列表（1-12），默认全部月份",
    )
    parser.add_argument(
        "--source",
        choices=["cpc", "psl"],
        default="cpc",
        help="数据源：cpc（默认）或 psl",
    )
    parser.add_argument(
        "--output-dir",
        default="./output/reports",
        help="报告输出目录",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="仅本地模式，不上传到远程",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="遇到错误时继续运行后续任务",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("批量推理和报告生成")
    print("=" * 60)
    print(f"年份: {args.years}")
    print(f"月份: {args.months}")
    print(f"数据源: {args.source}")
    print(f"输出目录: {args.output_dir}")
    print(f"本地模式: {args.local_only}")
    print(f"遇错继续: {args.continue_on_error}")
    print("=" * 60)

    # 生成所有 target_month
    target_months = []
    for year in sorted(args.years):
        for month in sorted(args.months):
            target_months.append(f"{year}-{month:02d}")

    print(f"\n总共需要运行 {len(target_months)} 个月份")
    print(f"预计时间: 约 {len(target_months) * 5} 分钟（假设每个月 5 分钟）\n")

    # 确认
    response = input("是否继续？(y/N): ")
    if response.lower() != "y":
        print("已取消")
        return

    # 批量运行
    success_count = 0
    failed_months = []

    for i, target_month in enumerate(target_months, 1):
        print(f"\n[{i}/{len(target_months)}] 处理 {target_month}...")
        print("-" * 60)

        try:
            report_uri = run_monthly_pipeline(
                target_month=target_month,
                source=args.source,
                output_dir=args.output_dir,
                local_only=args.local_only,
            )
            print(f"✓ {target_month} 完成: {report_uri}")
            success_count += 1

        except Exception as exc:
            print(f"✗ {target_month} 失败: {exc}")
            failed_months.append((target_month, str(exc)))

            if not args.continue_on_error:
                print("\n遇到错误，停止运行")
                break

    # 总结
    print("\n" + "=" * 60)
    print("批量运行完成")
    print("=" * 60)
    print(f"成功: {success_count}/{len(target_months)}")
    print(f"失败: {len(failed_months)}/{len(target_months)}")

    if failed_months:
        print("\n失败的月份:")
        for month, error in failed_months:
            print(f"  - {month}: {error}")


if __name__ == "__main__":
    main()
