from __future__ import annotations

import logging
import os
from pathlib import Path

import boto3
import requests
import urllib3
from botocore.config import Config
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, ConnectTimeout, ReadTimeout, RequestException
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential


# Disable SSL warnings for proxy upload endpoint.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("xiamen_uploader")

# Only AK/SK are read from environment.
ENV_AK = "XIAMEN_S3_AK"
ENV_SK = "XIAMEN_S3_SK"

# Fixed endpoint and bucket settings.
XIAMEN_S3_INTERNAL_IP = "100.64.46.202"
XIAMEN_S3_PUBLIC_DOMAIN = "qx.app.xmschain.com"
XIAMEN_S3_BUCKET = "szcx-ds-wthr-public"
XIAMEN_S3_REGION = "fjxm1"
XIAMEN_S3_PATH = "ocean_report"

PRESIGNED_EXPIRES_SECONDS = 7200
CONNECT_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 10
MAX_RETRY_ATTEMPTS = 8


def _require_credential(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少环境变量: {name}")
    return value


def _normalize_object_key(file_path: Path, object_name: str | None) -> str:
    name = (object_name or file_path.name).strip().strip("/")
    if not name:
        raise ValueError("object_name 不能为空")
    prefix = XIAMEN_S3_PATH.strip("/")
    return f"{prefix}/{name}" if prefix else name


def _create_presigned_upload_url(object_key: str, ak: str, sk: str) -> str:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        endpoint_url=f"http://{XIAMEN_S3_INTERNAL_IP}",
        region_name=XIAMEN_S3_REGION,
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
    )
    signed_url = s3_client.generate_presigned_url(
        "put_object",
        Params={"Bucket": XIAMEN_S3_BUCKET, "Key": object_key},
        ExpiresIn=PRESIGNED_EXPIRES_SECONDS,
    )
    return signed_url.replace(
        f"http://{XIAMEN_S3_INTERNAL_IP}/{XIAMEN_S3_BUCKET}",
        f"https://{XIAMEN_S3_PUBLIC_DOMAIN}/public",
    )


def _build_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=0, pool_block=False)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@retry(
    reraise=True,
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type((RuntimeError, ConnectTimeout, ReadTimeout, ConnectionError, RequestException)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _upload_with_retry(upload_url: str, file_path: Path) -> None:
    headers = {
        "Connection": "keep-alive",
        "Content-Type": "application/octet-stream",
        "Content-Length": str(file_path.stat().st_size),
    }
    session = _build_session()
    try:
        with file_path.open("rb") as file_obj:
            response = session.put(
                upload_url,
                data=file_obj,
                headers=headers,
                verify=False,
                proxies={"http": None, "https": None},
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            )
        if response.status_code not in (200, 201):
            body_preview = (response.text or "")[:500]
            raise RuntimeError(f"上传失败，状态码={response.status_code}，响应={body_preview}")
    finally:
        session.close()


def upload_file_to_xiamen(file_path: Path | str, object_name: str | None = None) -> str:
    """
    Upload a local file to Xiamen S3 path and return S3 URI.

    Args:
        file_path: Local file path.
        object_name: Optional object name under fixed prefix ocean_report.

    Returns:
        Uploaded object URI, e.g. s3://szcx-ds-wthr-public/ocean_report/2026-02.pdf
    """
    resolved_file_path = Path(file_path).expanduser().resolve()
    if not resolved_file_path.is_file():
        raise FileNotFoundError(f"文件不存在: {resolved_file_path}")

    ak = _require_credential(ENV_AK)
    sk = _require_credential(ENV_SK)

    object_key = _normalize_object_key(resolved_file_path, object_name=object_name)
    upload_url = _create_presigned_upload_url(object_key=object_key, ak=ak, sk=sk)

    logger.info("开始上传到厦门: %s", object_key)
    _upload_with_retry(upload_url=upload_url, file_path=resolved_file_path)
    logger.info("上传完成: %s", object_key)

    return f"s3://{XIAMEN_S3_BUCKET}/{object_key}"
