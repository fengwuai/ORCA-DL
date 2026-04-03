from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from prefect import flow, get_run_logger
from prefect.client.schemas.schedules import CronSchedule

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestration.reporting.generate_pdf_report import generate_pdf_report

NAME = "海洋模型预测"
TIMEZONE = "Asia/Shanghai"
CRON = "0 2 20 * *"
INFERENCE_CMD = ["pixi", "run", "-e", "model", "inference"]
TMP_BASE_DIR = ROOT / "tmp"
DEFAULT_REPORT_OUTPUT_DIR = "./output/reports"

load_dotenv(dotenv_path=ROOT / ".env", override=False)


def resolve_target_month(target_month: str | None) -> str:
    if target_month:
        try:
            dt = datetime.strptime(target_month, "%Y-%m")
        except ValueError as exc:
            raise ValueError(f"target_month 格式错误：{target_month}，应为 YYYY-MM（如 2026-02）") from exc
        if dt.strftime("%Y-%m") != target_month:
            raise ValueError(f"target_month 格式错误：{target_month}，应为 YYYY-MM（如 2026-02）")
        return target_month

    now = datetime.now(ZoneInfo(TIMEZONE))
    previous_month_last_day = now.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    ) - timedelta(days=1)
    return previous_month_last_day.strftime("%Y-%m")


def validate_source(source: str) -> str:
    resolved_source = source.lower().strip()
    if resolved_source not in {"cpc", "psl"}:
        raise ValueError(f"source 仅支持 cpc/psl，当前: {source}")
    return resolved_source


def resolve_report_output_dir(output_dir: str) -> Path:
    resolved = Path(output_dir).expanduser()
    if not resolved.is_absolute():
        resolved = (ROOT / resolved).resolve()
    else:
        resolved = resolved.resolve()
    return resolved


def resolve_logger() -> logging.Logger:
    try:
        return get_run_logger()
    except Exception:
        return logging.getLogger(__name__)


def run_command(command: list[str], task_name: str) -> None:
    logger = resolve_logger()
    logger.info("[%s] 执行命令: %s", task_name, " ".join(command))

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.stdout:
        logger.info("[%s] stdout:\n%s", task_name, result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            f"{task_name} 失败，退出码: {result.returncode}\n"
            f"命令: {' '.join(command)}\n"
            f"stderr:\n{result.stderr.strip()}"
        )

    if result.stderr:
        logger.warning("[%s] stderr:\n%s", task_name, result.stderr)


def run_monthly_pipeline(
    target_month: str | None = None,
    source: str = "cpc",
    output_dir: str = DEFAULT_REPORT_OUTPUT_DIR,
) -> str:
    logger = resolve_logger()
    resolved_month = resolve_target_month(target_month=target_month)
    resolved_source = validate_source(source)
    resolved_output_dir = resolve_report_output_dir(output_dir)

    logger.info("流程入参 target_month=%s", target_month)
    logger.info("流程入参 source=%s", source)
    logger.info("流程入参 output_dir=%s", output_dir)
    logger.info("流程最终执行月份=%s", resolved_month)
    logger.info("流程最终数据源=%s", resolved_source)
    logger.info("流程最终报告目录=%s", resolved_output_dir)

    TMP_BASE_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=TMP_BASE_DIR, prefix="pipeline_") as pipeline_tmp_dir:
        temp_model_dir = Path(pipeline_tmp_dir) / "models"
        temp_model_dir.mkdir(parents=True, exist_ok=True)

        run_command(
            [
                *INFERENCE_CMD,
                resolved_month,
                "--source",
                resolved_source,
                "--output-dir",
                str(temp_model_dir),
            ],
            task_name="执行推理流程",
        )

        prediction_path = temp_model_dir / f"{resolved_month}.nc"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"推理结果未生成: {prediction_path}")

        report_uri = generate_pdf_report(
            target_month=resolved_month,
            input_netcdf_path=prediction_path,
            output_dir=resolved_output_dir,
        )

        logger.info("流程完成，报告路径=%s", report_uri)
        return report_uri


@flow(name=NAME, log_prints=True)
def monthly_inference_flow(
    target_month: str | None = None,
    source: str = "cpc",
    output_dir: str = DEFAULT_REPORT_OUTPUT_DIR,
) -> str:
    return run_monthly_pipeline(target_month=target_month, source=source, output_dir=output_dir)


def serve_deployment() -> None:
    monthly_inference_flow.serve(
        name=NAME,
        schedule=CronSchedule(cron=CRON, timezone=TIMEZONE),
        parameters={"source": "cpc", "output_dir": DEFAULT_REPORT_OUTPUT_DIR},
        tags=["orca", "inference", "monthly"],
        pause_on_shutdown=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORCA-DL 月度流程")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="执行完整流程（推理 -> 生成报告 -> 上传）")
    run_parser.add_argument(
        "target_month",
        nargs="?",
        help="目标月份，格式 YYYY-MM（如 2026-02）；默认上个月",
    )
    run_parser.add_argument(
        "--output-dir",
        default=DEFAULT_REPORT_OUTPUT_DIR,
        help=f"报告输出目录（默认 {DEFAULT_REPORT_OUTPUT_DIR}）",
    )
    run_parser.add_argument(
        "--source",
        choices=["cpc", "psl"],
        default="cpc",
        help="数据源：cpc（默认）或 psl",
    )

    subparsers.add_parser("serve", help="启动 Prefect 部署服务")

    parser.set_defaults(command="run")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    if args.command == "serve":
        serve_deployment()
        return

    report_uri = run_monthly_pipeline(
        target_month=args.target_month,
        source=args.source,
        output_dir=args.output_dir,
    )
    print(f"报告上传完成: {report_uri}")


if __name__ == "__main__":
    main()
