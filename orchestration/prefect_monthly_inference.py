from __future__ import annotations

import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from prefect import flow, get_run_logger, task
from prefect.client.schemas.schedules import CronSchedule


ROOT = Path(__file__).resolve().parents[1]
NAME = "海洋模型预测"
TIMEZONE = "Asia/Shanghai"
CRON = "0 2 1 * *"
INFERENCE_CMD = ["pixi", "run", "-e", "model", "python", "predict/inference.py"]


def load_dotenv_if_exists() -> None:
    from dotenv import load_dotenv

    env_path = ROOT / ".env"
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)


load_dotenv_if_exists()


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


@task(name="run-orca-inference")
def run_inference_task(target_month: str, dry_run: bool = False) -> None:
    logger = get_run_logger()
    command = [*INFERENCE_CMD, target_month]

    logger.info("执行命令: %s", " ".join(command))

    if dry_run:
        logger.warning("dry_run=True，跳过实际推理执行")
        return

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.stdout:
        logger.info("stdout:\n%s", result.stdout)

    if result.returncode != 0:
        error_message = (
            f"推理命令执行失败，退出码: {result.returncode}\n"
            f"命令: {' '.join(command)}\n"
            f"stderr:\n{result.stderr.strip()}"
        )
        raise RuntimeError(error_message)

    if result.stderr:
        logger.warning("stderr:\n%s", result.stderr)

    logger.info("推理命令执行成功")


@flow(name=NAME, log_prints=True)
def monthly_inference_flow(
    target_month: str | None = None,
    dry_run: bool = False,
) -> str:
    logger = get_run_logger()
    resolved_month = resolve_target_month(target_month=target_month)

    logger.info("流程入参 target_month=%s", target_month)
    logger.info("流程最终执行月份=%s", resolved_month)
    run_inference_task(target_month=resolved_month, dry_run=dry_run)
    return resolved_month


def serve_deployment() -> None:
    monthly_inference_flow.serve(
        name=NAME,
        schedule=CronSchedule(cron=CRON, timezone=TIMEZONE),
        parameters={"dry_run": False},
        tags=["orca", "inference", "monthly"],
        pause_on_shutdown=False,
    )


if __name__ == "__main__":
    serve_deployment()
