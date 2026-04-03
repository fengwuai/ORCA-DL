from __future__ import annotations

import os
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from prefect import flow, get_run_logger, task
from prefect.client.schemas.schedules import CronSchedule
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt, wait_exponential


ROOT = Path(__file__).resolve().parents[1]
NAME = "海洋模型预测"
TIMEZONE = "Asia/Shanghai"
CRON = "0 2 20 * *"
INFERENCE_CMD = ["pixi", "run", "-e", "model", "inference"]
REPORT_CMD = ["pixi", "run", "-e", "orchestrator", "report"]
TMP_BASE_DIR = ROOT / "tmp"
CPC_CACHE_DIR = TMP_BASE_DIR / "cpc_cache"
S3_BUCKET = "szcx-ds-wthr-public"
S3_PREFIX = "ocean_report"
TASK_RETRIES = 1
TASK_RETRY_DELAY_SECONDS = 30
DOWNLOAD_CMD_RETRY_ATTEMPTS = 3
DOWNLOAD_CMD_RETRY_WAIT_MIN_SECONDS = 2
DOWNLOAD_CMD_RETRY_WAIT_MAX_SECONDS = 15

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


def resolve_raw_dir(pipeline_tmp_dir: str) -> Path:
    return Path(pipeline_tmp_dir) / "raw"


def resolve_download_dir(source: str, pipeline_tmp_dir: str) -> Path:
    if source == "cpc":
        return CPC_CACHE_DIR
    return resolve_raw_dir(pipeline_tmp_dir)


def resolve_processed_dir(pipeline_tmp_dir: str) -> Path:
    return Path(pipeline_tmp_dir) / "processed"


def resolve_preds_path(pipeline_tmp_dir: str) -> Path:
    return Path(pipeline_tmp_dir) / "preds.npy"


def resolve_prediction_path(target_month: str, pipeline_tmp_dir: str) -> Path:
    year, month = target_month.split("-")
    return Path(pipeline_tmp_dir) / f"orca_dl_prediction_{year}_{month}_24months.nc"


def resolve_report_uri(target_month: str) -> str:
    return f"s3://{S3_BUCKET}/{S3_PREFIX}/{target_month}.pdf"


def run_command(command: list[str], task_name: str, env: dict[str, str] | None = None) -> None:
    logger = get_run_logger()
    logger.info("[%s] 执行命令: %s", task_name, " ".join(command))

    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
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


def _is_retryable_download_command_error(exc: BaseException) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    non_retryable_signals = (
        "数据不存在",
        "不支持的 source",
        "变量名不支持",
        "格式错误",
    )
    return not any(signal in str(exc) for signal in non_retryable_signals)


@retry(
    reraise=True,
    stop=stop_after_attempt(DOWNLOAD_CMD_RETRY_ATTEMPTS),
    wait=wait_exponential(
        multiplier=1,
        min=DOWNLOAD_CMD_RETRY_WAIT_MIN_SECONDS,
        max=DOWNLOAD_CMD_RETRY_WAIT_MAX_SECONDS,
    ),
    retry=retry_if_exception(_is_retryable_download_command_error),
    before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
)
def run_download_command_with_retry(command: list[str], task_name: str) -> None:
    run_command(command=command, task_name=task_name)


@task(name="检查推理依赖", retries=TASK_RETRIES, retry_delay_seconds=TASK_RETRY_DELAY_SECONDS)
def check_inference_dependencies_task(dry_run: bool = False) -> None:
    logger = get_run_logger()
    if dry_run:
        logger.warning("dry_run=True，跳过依赖检查")
        return

    run_command(
        [*INFERENCE_CMD, "--stage", "check", "--source", "cpc"],
        task_name="检查推理依赖",
    )


@task(name="下载海洋数据", retries=TASK_RETRIES, retry_delay_seconds=TASK_RETRY_DELAY_SECONDS)
def download_data_task(
    target_month: str,
    pipeline_tmp_dir: str,
    source: str = "cpc",
    dry_run: bool = False,
) -> str:
    logger = get_run_logger()
    raw_dir = resolve_download_dir(source, pipeline_tmp_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.warning("dry_run=True，跳过下载，source=%s", source)
        return str(raw_dir)

    run_download_command_with_retry(
        [
            *INFERENCE_CMD,
            target_month,
            "--source",
            source,
            "--stage",
            "download",
            "--raw-dir",
            str(raw_dir),
        ],
        task_name=f"下载海洋数据（source={source}）",
    )

    logger.info("数据下载完成，目录: %s", raw_dir)
    return str(raw_dir)


@task(name="预处理海洋数据", retries=TASK_RETRIES, retry_delay_seconds=TASK_RETRY_DELAY_SECONDS)
def preprocess_data_task(
    target_month: str,
    pipeline_tmp_dir: str,
    source: str = "cpc",
    dry_run: bool = False,
) -> str:
    logger = get_run_logger()
    raw_dir = resolve_download_dir(source, pipeline_tmp_dir)
    processed_dir = resolve_processed_dir(pipeline_tmp_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.warning("dry_run=True，跳过预处理")
        return str(processed_dir)

    run_command(
        [
            *INFERENCE_CMD,
            target_month,
            "--source",
            source,
            "--stage",
            "preprocess",
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
        ],
        task_name="预处理海洋数据",
    )

    return str(processed_dir)


@task(name="执行模型推理", retries=TASK_RETRIES, retry_delay_seconds=TASK_RETRY_DELAY_SECONDS)
def run_inference_task(target_month: str, pipeline_tmp_dir: str, dry_run: bool = False) -> str:
    logger = get_run_logger()
    processed_dir = resolve_processed_dir(pipeline_tmp_dir)
    preds_path = resolve_preds_path(pipeline_tmp_dir)

    if dry_run:
        logger.warning("dry_run=True，跳过模型推理")
        return str(preds_path)

    run_command(
        [
            *INFERENCE_CMD,
            target_month,
            "--stage",
            "infer",
            "--processed-dir",
            str(processed_dir),
            "--preds-path",
            str(preds_path),
        ],
        task_name="执行模型推理",
    )

    if not preds_path.is_file():
        raise FileNotFoundError(f"推理中间结果未生成: {preds_path}")

    return str(preds_path)


@task(name="转换预测结果", retries=TASK_RETRIES, retry_delay_seconds=TASK_RETRY_DELAY_SECONDS)
def convert_prediction_task(target_month: str, pipeline_tmp_dir: str, dry_run: bool = False) -> str:
    logger = get_run_logger()
    preds_path = resolve_preds_path(pipeline_tmp_dir)
    prediction_path = resolve_prediction_path(target_month, pipeline_tmp_dir)

    if dry_run:
        logger.warning("dry_run=True，跳过结果转换")
        return str(prediction_path)

    run_command(
        [
            *INFERENCE_CMD,
            target_month,
            "--stage",
            "convert",
            "--preds-path",
            str(preds_path),
            "--output",
            str(prediction_path),
        ],
        task_name="转换预测结果",
    )

    if not prediction_path.is_file():
        raise FileNotFoundError(f"预测文件未生成: {prediction_path}")

    return str(prediction_path)


@task(name="生成海洋模型报告", retries=TASK_RETRIES, retry_delay_seconds=TASK_RETRY_DELAY_SECONDS)
def run_report_task(prediction_path: str, target_month: str, dry_run: bool = False) -> str:
    logger = get_run_logger()
    output_uri = resolve_report_uri(target_month)

    if dry_run:
        logger.warning("dry_run=True，跳过报告生成")
        return output_uri

    env = os.environ.copy()
    env["ORCA_INFERENCE_OUTPUT_NC"] = prediction_path

    run_command(
        [*REPORT_CMD, target_month],
        task_name="生成海洋模型报告",
        env=env,
    )

    logger.info("报告上传成功: %s", output_uri)
    return output_uri


@flow(name=NAME, log_prints=True)
def monthly_inference_flow(
    target_month: str | None = None,
    source: str = "cpc",
    dry_run: bool = False,
) -> str:
    logger = get_run_logger()
    resolved_month = resolve_target_month(target_month=target_month)
    resolved_source = source.lower().strip()
    if resolved_source not in {"cpc", "psl"}:
        raise ValueError(f"source 仅支持 cpc/psl，当前: {source}")

    logger.info("流程入参 target_month=%s", target_month)
    logger.info("流程入参 source=%s", source)
    logger.info("流程最终执行月份=%s", resolved_month)
    logger.info("流程最终数据源=%s", resolved_source)

    TMP_BASE_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=TMP_BASE_DIR, prefix="pipeline_") as pipeline_tmp_dir:
        check_inference_dependencies_task(dry_run=dry_run)
        download_data_task(
            target_month=resolved_month,
            pipeline_tmp_dir=pipeline_tmp_dir,
            source=resolved_source,
            dry_run=dry_run,
        )

        preprocess_data_task(
            target_month=resolved_month,
            pipeline_tmp_dir=pipeline_tmp_dir,
            source=resolved_source,
            dry_run=dry_run,
        )

        run_inference_task(
            target_month=resolved_month,
            pipeline_tmp_dir=pipeline_tmp_dir,
            dry_run=dry_run,
        )

        prediction_path = convert_prediction_task(
            target_month=resolved_month,
            pipeline_tmp_dir=pipeline_tmp_dir,
            dry_run=dry_run,
        )

        report_uri = run_report_task(
            prediction_path=prediction_path,
            target_month=resolved_month,
            dry_run=dry_run,
        )

        logger.info("流程完成，报告路径=%s", report_uri)
        return report_uri


def serve_deployment() -> None:
    monthly_inference_flow.serve(
        name=NAME,
        schedule=CronSchedule(cron=CRON, timezone=TIMEZONE),
        parameters={"source": "cpc", "dry_run": False},
        tags=["orca", "inference", "monthly"],
        pause_on_shutdown=False,
    )


if __name__ == "__main__":
    serve_deployment()
