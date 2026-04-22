"""
ORCA-DL GODAS 数据推理 CLI（仅推理流程）

功能：
    根据指定月份下载 GODAS 数据、执行预处理与模型推理，
    最终输出 NetCDF 文件到指定目录。

使用方式：
    pixi run -e model inference
    pixi run -e model inference 2026-02 --source psl --output-dir ./output/models

参数：
    target_month: 初始化月份，格式 YYYY-MM（如 2025-12）
                  不传时默认上个月（Asia/Shanghai）
    --output-dir: NetCDF 输出目录，默认 ./output/models
    --source:   数据源（默认 cpc）
                cpc: CPC 单月 GRIB 文件
                psl: PSL 按变量分年 NetCDF 文件

输出：
    输出文件为 {output_dir}/{target_month}.nc
    预测时间从初始化月份的下个月开始（例如 2025-12 初始化 -> 2026-01 首报）
    推理中间文件统一使用 TemporaryDirectory 管理并自动清理。

    包含以下变量（24 个月 × 6 个变量）：
    - so: 盐度 (salinity) [g/kg], shape: (24, 16, 128, 360)
    - thetao: 位温 (potential temperature) [°C], shape: (24, 16, 128, 360)
    - tos: 海表温度 (sea surface temperature) [°C], shape: (24, 128, 360)
    - uo: 纬向流速 (zonal current) [m/s], shape: (24, 16, 128, 360)
    - vo: 经向流速 (meridional current) [m/s], shape: (24, 16, 128, 360)
    - zos: 海表高度 (sea surface height) [m], shape: (24, 128, 360)

依赖：
    - pixi 环境 'model': 包含 PyTorch 和相关 Python 包
    - pixi 环境 'exec': 包含 CDO 工具
    - 模型文件: ./ckpt/seed_1.bin, ./model_config.json
    - 统计文件: ./stat/mean/*.npy, ./stat/std/*.npy

注意事项：
    1. 确保 GODAS 数据已更新到指定月份
    2. 当前流程固定使用 CPU 进行推理
    3. 临时文件会自动清理（位于 ./tmp 目录）
"""

import os
import sys
import argparse
import json
import subprocess
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import torch
import xarray as xr
import pandas as pd
from urllib.request import urlretrieve
from urllib.error import ContentTooShortError, HTTPError, URLError
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt, wait_exponential

# ============ 配置参数 ============
# 模型配置
MODEL_CONFIG_PATH = "./model_config.json"
MODEL_CKPT_PATH = "./ckpt/seed_1.bin"
STAT_DIR = "./stat"

# 数据配置
GRID_FILE = "./grid"
ZAXIS_FILE = "./zaxis.txt"
GODAS_BASE_URL = "https://downloads.psl.noaa.gov/Datasets/godas"
GODAS_CPC_URL = "https://ftp.cpc.ncep.noaa.gov/godas/monthly"

# CPC GRIB1 code 映射
GRIB_CODES = {
    "pottmp": 13,
    "salt": 88,
    "ucur": 49,
    "vcur": 50,
    "sshg": 198,
    "uflx": 124,
    "vflx": 125,
}
CPC_SST_PROXY_CODE = 13
CPC_SST_PROXY_LEVEL = 5

# 推理配置
PREDICT_STEPS = 24   # 预测月数
INPUT_STEPS = 1     # 输入时间步数
BATCH_SIZE = 1      # 推理批次大小

# 变量配置（GODAS 命名）
# 顺序需与 demo.ipynb 严格一致：salt, pottmp, sst, ucur, vcur, sshg
GODAS_OCEAN_VARS_ORDER = ['salt', 'pottmp', 'sst', 'ucur', 'vcur', 'sshg']
GODAS_VARS_3D = ['salt', 'pottmp', 'ucur', 'vcur']  # 3D 变量（16 层）
GODAS_VARS_2D = ['sst', 'sshg']                      # 2D 变量（1 层）
GODAS_VARS_2D_FROM_GRIB = ['sshg']                   # CPC GRIB 中可直接提取的 2D 变量
GODAS_VARS_ATMO = ['uflx', 'vflx']                   # 大气强迫变量
ALL_GODAS_VARS = GODAS_OCEAN_VARS_ORDER + GODAS_VARS_ATMO

# 模型变量映射（GODAS -> 模型）
VAR_MAPPING = {
    'salt': 'so',
    'pottmp': 'thetao',
    'sst': 'tos',
    'ucur': 'uo',
    'vcur': 'vo',
    'sshg': 'zos'
}

# 输出配置
TMP_BASE_DIR = "./tmp"  # 临时文件基础目录（统一通过 TemporaryDirectory 管理）
CPC_CACHE_DIR = os.path.join(TMP_BASE_DIR, "cpc_cache")
DEFAULT_OUTPUT_DIR = "./output/models"
TIMEZONE = "Asia/Shanghai"

# 深度层级（米）
DEPTH_LEVELS = [10, 15, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800, 1000]

# 下载重试配置
DOWNLOAD_RETRY_ATTEMPTS = 5
DOWNLOAD_RETRY_WAIT_MIN_SECONDS = 2
DOWNLOAD_RETRY_WAIT_MAX_SECONDS = 20

download_logger = logging.getLogger("predict.inference.download")


# ============ 工具函数 ============

def parse_date(date_str: str) -> Tuple[int, int]:
    """
    解析日期字符串

    Args:
        date_str: 格式为 YYYY-MM 的日期字符串

    Returns:
        (year, month) 元组

    Raises:
        ValueError: 日期格式错误
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m")
    except ValueError as exc:
        raise ValueError(f"日期格式错误：{date_str}，应为 YYYY-MM 格式（如 2025-12）") from exc
    if dt.strftime("%Y-%m") != date_str:
        raise ValueError(f"日期格式错误：{date_str}，应为 YYYY-MM 格式（如 2025-12）")
    return dt.year, dt.month


def shift_year_month(year: int, month: int, month_offset: int) -> Tuple[int, int]:
    """在指定年月基础上平移 month_offset 个月。"""
    total = year * 12 + (month - 1) + month_offset
    shifted_year = total // 12
    shifted_month = (total % 12) + 1
    return shifted_year, shifted_month


def resolve_first_forecast_month(init_year: int, init_month: int) -> Tuple[int, int]:
    """根据初始化年月计算首报年月（初始化月后 1 个月）。"""
    return shift_year_month(init_year, init_month, 1)


def resolve_target_month(target_month: str | None) -> str:
    if target_month:
        parse_date(target_month)
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


def check_dependencies():
    """检查必要的文件和工具是否存在"""
    # 检查模型文件
    if not os.path.exists(MODEL_CONFIG_PATH):
        raise FileNotFoundError(f"模型配置文件不存在：{MODEL_CONFIG_PATH}")
    if not os.path.exists(MODEL_CKPT_PATH):
        raise FileNotFoundError(f"模型权重文件不存在：{MODEL_CKPT_PATH}")

    # 检查网格文件
    if not os.path.exists(GRID_FILE):
        raise FileNotFoundError(f"网格文件不存在：{GRID_FILE}")
    if not os.path.exists(ZAXIS_FILE):
        raise FileNotFoundError(f"垂直轴文件不存在：{ZAXIS_FILE}")

    # 检查统计文件
    for var in ALL_GODAS_VARS:
        mean_file = os.path.join(STAT_DIR, "mean", f"{var}.npy")
        std_file = os.path.join(STAT_DIR, "std", f"{var}.npy")
        if not os.path.exists(mean_file):
            raise FileNotFoundError(f"均值统计文件不存在：{mean_file}")
        if not os.path.exists(std_file):
            raise FileNotFoundError(f"标准差统计文件不存在：{std_file}")

    # 检查 CDO 工具
    try:
        subprocess.run(
            ["pixi", "run", "-e", "exec", "cdo", "--version"],
            capture_output=True,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("CDO 工具不可用，请检查 pixi exec 环境配置")

    print("✓ 依赖检查通过")


# ============ 数据下载 ============

def _is_retryable_download_exception(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError)


@retry(
    reraise=True,
    stop=stop_after_attempt(DOWNLOAD_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=DOWNLOAD_RETRY_WAIT_MIN_SECONDS, max=DOWNLOAD_RETRY_WAIT_MAX_SECONDS),
    retry=retry_if_exception(_is_retryable_download_exception),
    before_sleep=before_sleep_log(download_logger, logging.WARNING),
)
def _download_file_with_retry(url: str, output_file: str) -> str:
    if os.path.exists(output_file):
        os.remove(output_file)
    try:
        urlretrieve(url, output_file)
    except HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(f"GODAS 数据不存在：{url}") from e
        raise RuntimeError(f"下载失败：{url}，HTTP 错误码：{e.code}") from e
    except (URLError, ContentTooShortError, TimeoutError) as e:
        raise RuntimeError(f"下载失败：{url}，网络错误：{e}") from e
    return output_file


def download_godas_data(year: int, month: int, var_name: str, output_dir: str) -> str:
    """
    下载 GODAS 数据

    Args:
        year: 年份
        month: 月份（1-12）
        var_name: 变量名（GODAS 命名）
        output_dir: 输出目录

    Returns:
        下载文件的路径

    Raises:
        HTTPError: 数据不存在或下载失败
    """
    # GODAS 数据按年份存储，URL 格式：https://downloads.psl.noaa.gov/Datasets/godas/pottmp.2025.nc
    url = f"{GODAS_BASE_URL}/{var_name}.{year}.nc"
    output_file = os.path.join(output_dir, f"{var_name}.{year}.nc")

    if os.path.exists(output_file):
        print(f"  - {var_name}: 文件已存在，跳过下载")
        return output_file

    print(f"  - {var_name}: 正在下载... ", end="", flush=True)
    try:
        _download_file_with_retry(url=url, output_file=output_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n"
            f"请确认 {year} 年的数据已发布"
        ) from e
    print("完成")
    return output_file


def download_all_variables(year: int, month: int, output_dir: str) -> Dict[str, str]:
    """
    下载所有需要的 GODAS 变量

    Args:
        year: 年份
        month: 月份
        output_dir: 输出目录

    Returns:
        变量名到文件路径的映射
    """
    os.makedirs(output_dir, exist_ok=True)

    all_vars = GODAS_VARS_3D + GODAS_VARS_2D + GODAS_VARS_ATMO
    file_paths = {}

    for var in all_vars:
        file_paths[var] = download_godas_data(year, month, var, output_dir)

    return file_paths


def resolve_cpc_grib_path(year: int, month: int, output_dir: str) -> str:
    return os.path.join(output_dir, f"godas.M.{year}{month:02d}.grb")


def download_godas_grib(year: int, month: int, output_dir: str) -> str:
    """
    下载 CPC GODAS 单月 GRIB 文件

    Args:
        year: 年份
        month: 月份（1-12）
        output_dir: 输出目录

    Returns:
        GRIB 文件路径
    """
    filename = f"godas.M.{year}{month:02d}.grb"
    url = f"{GODAS_CPC_URL}/{filename}"
    output_file = resolve_cpc_grib_path(year, month, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_file):
        print(f"  - {filename}: 文件已存在，跳过下载")
        return output_file

    print(f"  - {filename}: 正在下载... ", end="", flush=True)
    try:
        _download_file_with_retry(url=url, output_file=output_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n"
            f"请确认 {year}-{month:02d} 的数据已发布"
        ) from e
    print("完成")
    return output_file


# ============ 数据预处理（CDO）============

def run_cdo_command(cmd: list, description: str):
    """
    执行 CDO 命令

    Args:
        cmd: CDO 命令列表
        description: 命令描述（用于错误提示）

    Raises:
        RuntimeError: CDO 执行失败
    """
    try:
        result = subprocess.run(
            ["pixi", "run", "-e", "exec"] + cmd,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"CDO 执行失败：{description}\n"
            f"命令：{' '.join(cmd)}\n"
            f"错误信息：{e.stderr}"
        )


def preprocess_2d_variable(input_nc: str, output_nc: str, year: int, month: int, var_name: str):
    """
    预处理 2D 变量（水平插值）

    Args:
        input_nc: 输入 NetCDF 文件
        output_nc: 输出 NetCDF 文件
        year: 年份
        month: 月份
        var_name: 变量名
    """
    # 选择指定月份，然后进行水平插值
    # CDO 命令：cdo -b f64 remapbil,grid -selmon,MM -selyear,YYYY input.nc output.nc
    cmd = [
        "cdo", "-b", "f64",
        "remapbil," + GRID_FILE,
        "-selmon," + str(month),
        "-selyear," + str(year),
        input_nc,
        output_nc
    ]
    run_cdo_command(cmd, f"2D 变量插值：{var_name}")


def preprocess_3d_variable(input_nc: str, output_nc: str, year: int, month: int, var_name: str):
    """
    预处理 3D 变量（水平+垂直插值）

    Args:
        input_nc: 输入 NetCDF 文件
        output_nc: 输出 NetCDF 文件
        year: 年份
        month: 月份
        var_name: 变量名
    """
    # 选择指定月份，垂直插值到标准层级，设置垂直轴，然后水平插值
    # CDO 命令：cdo -b f64 remapbil,grid -setzaxis,zaxis -intlevel,10,15,...,1000 -selmon,MM -selyear,YYYY input.nc output.nc
    levels = ",".join(map(str, DEPTH_LEVELS))
    cmd = [
        "cdo", "-b", "f64",
        "remapbil," + GRID_FILE,
        "-setzaxis," + ZAXIS_FILE,
        "-intlevel," + levels,
        "-selmon," + str(month),
        "-selyear," + str(year),
        input_nc,
        output_nc
    ]
    run_cdo_command(cmd, f"3D 变量插值：{var_name}")


def preprocess_all_variables(raw_dir: str, processed_dir: str, year: int, month: int):
    """
    预处理所有变量

    Args:
        raw_dir: 原始数据目录
        processed_dir: 预处理输出目录
        year: 年份
        month: 月份
    """
    os.makedirs(processed_dir, exist_ok=True)

    # 处理 3D 变量
    for var in GODAS_VARS_3D:
        input_file = os.path.join(raw_dir, f"{var}.{year}.nc")
        output_file = os.path.join(processed_dir, f"{var}.nc")
        print(f"  - 处理 3D 变量：{var}")
        preprocess_3d_variable(input_file, output_file, year, month, var)

    # 处理 2D 变量
    for var in GODAS_VARS_2D + GODAS_VARS_ATMO:
        input_file = os.path.join(raw_dir, f"{var}.{year}.nc")
        output_file = os.path.join(processed_dir, f"{var}.nc")
        print(f"  - 处理 2D 变量：{var}")
        preprocess_2d_variable(input_file, output_file, year, month, var)


def preprocess_3d_from_grib(grib_file: str, output_nc: str, var_name: str):
    """从 GRIB 提取并预处理 3D 变量。"""
    code = GRIB_CODES[var_name]
    levels = ",".join(map(str, DEPTH_LEVELS))
    cmd = [
        "cdo", "-f", "nc4", "-b", "f64",
        f"setname,{var_name}",
        "-remapbil," + GRID_FILE,
        "-setzaxis," + ZAXIS_FILE,
        "-intlevel," + levels,
        f"-selcode,{code}",
        grib_file,
        output_nc,
    ]
    run_cdo_command(cmd, f"GRIB 3D 变量插值：{var_name}")


def preprocess_2d_from_grib(grib_file: str, output_nc: str, var_name: str):
    """从 GRIB 提取并预处理 2D 变量。"""
    code = GRIB_CODES[var_name]
    cmd = [
        "cdo", "-f", "nc4", "-b", "f64",
        f"setname,{var_name}",
        "-remapbil," + GRID_FILE,
        f"-selcode,{code}",
        grib_file,
        output_nc,
    ]
    run_cdo_command(cmd, f"GRIB 2D 变量插值：{var_name}")


def preprocess_sst_proxy_from_grib(grib_file: str, output_nc: str):
    """从 GRIB 的温度场提取 5m 层作为 SST 代理，并完成水平插值。"""
    cmd = [
        "cdo", "-f", "nc4", "-b", "f64",
        "setname,sst",
        "-remapbil," + GRID_FILE,
        f"-sellevel,{CPC_SST_PROXY_LEVEL}",
        f"-selcode,{CPC_SST_PROXY_CODE}",
        grib_file,
        output_nc,
    ]
    run_cdo_command(cmd, f"GRIB SST 代理提取（code={CPC_SST_PROXY_CODE}, level={CPC_SST_PROXY_LEVEL}m）")


def preprocess_all_from_grib(grib_file: str, processed_dir: str):
    """从单个 GRIB 文件预处理所有变量。"""
    os.makedirs(processed_dir, exist_ok=True)

    for var in GODAS_VARS_3D:
        output_file = os.path.join(processed_dir, f"{var}.nc")
        print(f"  - 处理 3D 变量：{var}")
        preprocess_3d_from_grib(grib_file, output_file, var)

    sst_output_file = os.path.join(processed_dir, "sst.nc")
    print("  - 处理 2D 变量：sst（5m 代理）")
    preprocess_sst_proxy_from_grib(grib_file, sst_output_file)

    for var in GODAS_VARS_2D_FROM_GRIB + GODAS_VARS_ATMO:
        output_file = os.path.join(processed_dir, f"{var}.nc")
        print(f"  - 处理 2D 变量：{var}")
        preprocess_2d_from_grib(grib_file, output_file, var)


def align_preprocessed_units_to_example(processed_dir: str) -> None:
    """将预处理结果的单位体系统一到与 example/stat 一致。"""
    target_units = {
        "salt": "g/kg",
        "pottmp": "degC",
        "sst": "degC",
    }

    for var, unit in target_units.items():
        nc_file = Path(processed_dir) / f"{var}.nc"
        if not nc_file.is_file():
            continue

        with xr.open_dataset(nc_file) as ds:
            ds_aligned = ds.load()

        da = ds_aligned[var]
        changed = []

        # example 中 sst 为 2D（time, lat, lon）；若 depth 仅 1 层则压缩
        if var == "sst" and "depth" in da.dims and da.sizes.get("depth") == 1:
            ds_aligned = ds_aligned.squeeze("depth", drop=True)
            da = ds_aligned[var]
            changed.append("squeeze_depth")

        data = da.values
        finite = np.isfinite(data)
        if finite.any():
            median_abs = float(np.nanmedian(np.abs(data[finite])))
            median_val = float(np.nanmedian(data[finite]))

            # 温度：K -> degC
            if var in ("pottmp", "sst") and median_val > 100:
                data = data - 273.15
                ds_aligned[var].data = data
                changed.append("K_to_degC")

            # 盐度：kg/kg -> g/kg
            if var == "salt" and median_abs < 1:
                data = data * 1000.0
                ds_aligned[var].data = data
                changed.append("kgkg_to_gkg")

        ds_aligned[var].attrs["units"] = unit

        tmp_file = nc_file.with_suffix(".nc.tmp")
        ds_aligned.to_netcdf(tmp_file, format="NETCDF4", engine="netcdf4")
        ds_aligned.close()
        os.replace(tmp_file, nc_file)

        if changed:
            print(f"[单位对齐] {var}: {', '.join(changed)} -> {unit}")
        else:
            print(f"[单位对齐] {var}: already aligned -> {unit}")


# ============ 数据归一化 ============

def load_statistics(stat_dir: str, var_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载变量的统计量

    Args:
        stat_dir: 统计文件目录
        var_name: 变量名（GODAS 命名）

    Returns:
        (mean, std) 元组，shape: (12, ...) 表示 12 个月的统计量
    """
    mean_file = os.path.join(stat_dir, "mean", f"{var_name}.npy")
    std_file = os.path.join(stat_dir, "std", f"{var_name}.npy")

    mean = np.load(mean_file)
    std = np.load(std_file)

    return mean, std


def normalize_data(data: np.ndarray, mean: np.ndarray, std: np.ndarray, month: int) -> np.ndarray:
    """
    执行 Z-score 归一化

    Args:
        data: 输入数据
        mean: 均值统计，shape: (12, ...)
        std: 标准差统计，shape: (12, ...)
        month: 月份（1-12）

    Returns:
        归一化后的数据
    """
    # 月份索引（0-11）
    month_idx = month - 1

    # Z-score 归一化
    normalized = (data - mean[month_idx]) / (std[month_idx] + 1e-8)

    # 处理 NaN 值（陆地掩码）
    normalized = np.nan_to_num(normalized, nan=0.0)

    return normalized


# ============ 模型推理 ============

def load_model(config_path: str, ckpt_path: str, device: torch.device):
    """
    加载 ORCA-DL 模型

    Args:
        config_path: 模型配置文件路径
        ckpt_path: 模型权重文件路径
        device: 计算设备

    Returns:
        加载好的模型
    """
    # 导入模型类
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.orca_dl import ORCADLModel, ORCADLConfig

    # 加载配置
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # 过滤掉不需要的参数（如 architectures, transformers_version 等）
    # 这些是 Hugging Face 格式的元数据，不是模型参数
    exclude_keys = ['architectures', 'transformers_version', 'torch_dtype']
    model_config_dict = {k: v for k, v in config_dict.items() if k not in exclude_keys}

    # 创建配置对象
    config = ORCADLConfig(**model_config_dict)

    # 创建模型
    model = ORCADLModel(config)

    # 加载权重
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    # 移至设备并设置为评估模式
    model = model.to(device)
    model.eval()

    return model


def prepare_model_input(processed_dir: str, month: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    准备模型输入

    Args:
        processed_dir: 预处理数据目录
        month: 月份（1-12）

    Returns:
        (ocean_vars, atmo_vars) 元组
        - ocean_vars: shape (1, 66, 128, 360) = 16×4 + 1×2
        - atmo_vars: shape (1, 2, 128, 360)
    """
    ocean_channels = []
    atmo_channels = []

    # 海洋变量顺序严格对齐 demo.ipynb：salt, pottmp, sst, ucur, vcur, sshg
    for var in GODAS_OCEAN_VARS_ORDER:
        nc_file = os.path.join(processed_dir, f"{var}.nc")
        with xr.open_dataset(nc_file) as ds:
            data = ds[var].values

        if var in GODAS_VARS_3D:
            # 目标格式：(16, 128, 360)
            if data.ndim == 4:  # (time, level, lat, lon)
                data = data[0]
            if data.ndim != 3:
                raise ValueError(f"{var} 维度不符合预期，期望 3D，实际 shape={data.shape}")

            mean, std = load_statistics(STAT_DIR, var)
            normalized = normalize_data(data, mean, std, month)
            ocean_channels.extend(normalized[level] for level in range(16))
            continue

        # 2D 变量：sst / sshg
        if data.ndim == 4:      # (time, depth=1, lat, lon)
            data = data[0, 0]
        elif data.ndim == 3:    # (time, lat, lon)
            data = data[0]
        elif data.ndim != 2:    # (lat, lon)
            raise ValueError(f"{var} 维度不符合预期，期望 2D，实际 shape={data.shape}")

        mean, std = load_statistics(STAT_DIR, var)
        normalized = normalize_data(data, mean, std, month)
        ocean_channels.append(normalized)

    # 读取并归一化大气强迫变量
    for var in GODAS_VARS_ATMO:
        nc_file = os.path.join(processed_dir, f"{var}.nc")
        with xr.open_dataset(nc_file) as ds:
            data = ds[var].values
        if data.ndim == 3:      # (time, lat, lon)
            data = data[0]
        elif data.ndim != 2:
            raise ValueError(f"{var} 维度不符合预期，期望 2D，实际 shape={data.shape}")

        mean, std = load_statistics(STAT_DIR, var)
        normalized = normalize_data(data, mean, std, month)

        atmo_channels.append(normalized)

    # 拼接为张量
    ocean_vars = np.stack(ocean_channels, axis=0)  # (66, 128, 360)
    atmo_vars = np.stack(atmo_channels, axis=0)    # (2, 128, 360)

    # 添加批次维度
    ocean_vars = torch.from_numpy(ocean_vars).float().unsqueeze(0)  # (1, 66, 128, 360)
    atmo_vars = torch.from_numpy(atmo_vars).float().unsqueeze(0)    # (1, 2, 128, 360)

    return ocean_vars, atmo_vars


# ============ 后处理与保存 ============

def denormalize_predictions(
    preds: np.ndarray,
    stat_dir: str,
    start_month: int
) -> Dict[str, np.ndarray]:
    """
    反归一化预测结果

    Args:
        preds: 模型预测输出，shape: (1, 24, 66, 128, 360)
        stat_dir: 统计文件目录
        start_month: 初始化月份（1-12）

    Returns:
        变量名到反归一化数据的映射
        - so: (24, 16, 128, 360)
        - thetao: (24, 16, 128, 360)
        - tos: (24, 1, 128, 360)
        - uo: (24, 16, 128, 360)
        - vo: (24, 16, 128, 360)
        - zos: (24, 1, 128, 360)
    """
    # 移除批次维度
    preds = preds[0]  # (24, 66, 128, 360)

    # 根据 out_chans: [16, 16, 1, 16, 16, 1] 分割通道
    split_indices = [16, 32, 33, 49, 65]
    split_preds = np.split(preds, split_indices, axis=1)

    # 变量顺序与 demo 输入顺序一致
    var_names = GODAS_OCEAN_VARS_ORDER
    model_var_names = [VAR_MAPPING[v] for v in var_names]

    results = {}

    for i, (godas_var, model_var, pred) in enumerate(zip(var_names, model_var_names, split_preds)):
        # 加载统计量
        mean, std = load_statistics(stat_dir, godas_var)

        # 对每个时间步进行反归一化
        denormed_steps = []
        for step in range(PREDICT_STEPS):
            # 计算预测月份（首报为初始化月 + 1，循环 12 个月）
            pred_month = ((start_month + step) % 12)  # 0-11

            # 反归一化：pred * std + mean
            denormed = pred[step] * std[pred_month] + mean[pred_month]

            denormed_steps.append(denormed)

        # 拼接所有时间步
        results[model_var] = np.stack(denormed_steps, axis=0)

    return results


def save_to_netcdf(
    predictions: Dict[str, np.ndarray],
    output_path: str,
    start_year: int,
    start_month: int
):
    """
    保存预测结果为 NetCDF 文件

    Args:
        predictions: 变量名到数据的映射
        output_path: 输出文件路径
        start_year: 初始化年份
        start_month: 初始化月份
    """
    forecast_start_year, forecast_start_month = resolve_first_forecast_month(start_year, start_month)
    forecast_end_year, forecast_end_month = shift_year_month(start_year, start_month, PREDICT_STEPS)

    # 创建时间坐标（24 个月）
    time_coord = pd.date_range(
        start=f"{forecast_start_year}-{forecast_start_month:02d}",
        periods=PREDICT_STEPS,
        freq='MS'  # Month Start
    )

    # 创建空间坐标
    depth_coord = DEPTH_LEVELS
    lat_coord = np.linspace(-63.5, 63.5, 128)
    lon_coord = np.linspace(0.5, 359.5, 360)

    # 创建数据变量
    data_vars = {}

    # 3D 变量（有深度维度）
    for var in ['so', 'thetao', 'uo', 'vo']:
        data_vars[var] = (
            ['time', 'depth', 'lat', 'lon'],
            predictions[var],
            {
                'long_name': get_var_long_name(var),
                'units': get_var_units(var),
                'description': get_var_description(var)
            }
        )

    # 2D 变量（无深度维度）
    data_vars['tos'] = (
        ['time', 'lat', 'lon'],
        predictions['tos'][:, 0, :, :],  # 移除深度维度
        {
            'long_name': get_var_long_name('tos'),
            'units': get_var_units('tos'),
            'description': get_var_description('tos')
        }
    )

    data_vars['zos'] = (
        ['time', 'lat', 'lon'],
        predictions['zos'][:, 0, :, :],  # 移除深度维度
        {
            'long_name': get_var_long_name('zos'),
            'units': get_var_units('zos'),
            'description': get_var_description('zos')
        }
    )

    # 创建 Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'time': time_coord,
            'depth': depth_coord,
            'lat': lat_coord,
            'lon': lon_coord
        },
        attrs={
            'title': 'ORCA-DL Ocean State Predictions',
            'institution': 'ORCA-DL Model',
            'source': 'ORCA-DL deep learning model trained on GODAS data',
            'history': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'initialization_date': f'{start_year}-{start_month:02d}',
            'forecast_start_date': f'{forecast_start_year}-{forecast_start_month:02d}',
            'forecast_end_date': f'{forecast_end_year}-{forecast_end_month:02d}',
            'forecast_months': PREDICT_STEPS,
            'model_checkpoint': MODEL_CKPT_PATH,
            'conventions': 'CF-1.8'
        }
    )

    # 为坐标添加属性
    ds['lat'].attrs['long_name'] = 'Latitude'
    ds['lat'].attrs['units'] = 'degrees_north'
    ds['lat'].attrs['standard_name'] = 'latitude'

    ds['lon'].attrs['long_name'] = 'Longitude'
    ds['lon'].attrs['units'] = 'degrees_east'
    ds['lon'].attrs['standard_name'] = 'longitude'

    ds['depth'].attrs['long_name'] = 'Depth'
    ds['depth'].attrs['units'] = 'm'
    ds['depth'].attrs['positive'] = 'down'
    ds['depth'].attrs['standard_name'] = 'depth'

    ds['time'].attrs['long_name'] = 'Time'
    ds['time'].attrs['standard_name'] = 'time'

    # 保存为 NetCDF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.to_netcdf(output_path, format='NETCDF4', engine='netcdf4')
    ds.close()

    print(f"✓ 预测结果已保存：{output_path}")


def get_var_long_name(var: str) -> str:
    """获取变量的长名称"""
    names = {
        'so': 'Sea Water Salinity',
        'thetao': 'Sea Water Potential Temperature',
        'tos': 'Sea Surface Temperature',
        'uo': 'Eastward Sea Water Velocity',
        'vo': 'Northward Sea Water Velocity',
        'zos': 'Sea Surface Height Above Geoid'
    }
    return names.get(var, var)


def get_var_units(var: str) -> str:
    """获取变量的单位"""
    units = {
        'so': 'g/kg',
        'thetao': 'degC',
        'tos': 'degC',
        'uo': 'm/s',
        'vo': 'm/s',
        'zos': 'm'
    }
    return units.get(var, '')


def get_var_description(var: str) -> str:
    """获取变量的描述"""
    descriptions = {
        'so': 'Salinity of sea water at 16 depth levels',
        'thetao': 'Potential temperature of sea water at 16 depth levels',
        'tos': 'Temperature of sea water at the surface',
        'uo': 'Eastward component of ocean current velocity at 16 depth levels',
        'vo': 'Northward component of ocean current velocity at 16 depth levels',
        'zos': 'Sea surface height anomaly relative to geoid'
    }
    return descriptions.get(var, '')


# ============ 主流程 ============

def resolve_device() -> torch.device:
    """固定使用 CPU。"""
    device = torch.device("cpu")
    print("⚠ 固定使用 CPU 推理")
    return device


def run_model_inference(processed_dir: str, month: int, preds_output_path: str) -> str:
    """执行模型推理并保存原始预测张量为 NPY。"""
    print("\n[推理] 准备模型输入（归一化）...")
    ocean_vars, atmo_vars = prepare_model_input(processed_dir, month)
    print(f"✓ 海洋变量形状：{ocean_vars.shape}")
    print(f"✓ 大气变量形状：{atmo_vars.shape}")

    print("\n[推理] 加载模型...")
    print(f"  模型配置：{MODEL_CONFIG_PATH}")
    print(f"  模型权重：{MODEL_CKPT_PATH}")
    device = resolve_device()
    model = load_model(MODEL_CONFIG_PATH, MODEL_CKPT_PATH, device)
    print("✓ 模型已加载")

    print(f"\n[推理] 开始预测未来 {PREDICT_STEPS} 个月...")
    with torch.no_grad():
        output = model(
            ocean_vars=ocean_vars.to(device),
            atmo_vars=atmo_vars.to(device),
            predict_time_steps=PREDICT_STEPS,
        )

    preds = output.preds.cpu().numpy()
    preds_path = Path(preds_output_path)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(preds_path, preds)
    print(f"✓ 推理完成，输出形状：{preds.shape}")
    print(f"✓ 中间结果已保存：{preds_path}")
    return str(preds_path)


def convert_predictions_to_netcdf(preds_path: str, output_path: str, year: int, month: int) -> str:
    """将模型预测张量转换为最终 NetCDF 文件。"""
    print("\n[转换] 读取推理结果并反归一化...")
    preds = np.load(preds_path)
    predictions = denormalize_predictions(preds, STAT_DIR, month)

    print(f"[转换] 保存 NetCDF：{output_path}")
    save_to_netcdf(predictions, output_path, year, month)
    return output_path


def run_download_step(target_month: str, raw_dir: str, source: str = "cpc") -> str:
    """下载数据（按 source 选择 CPC 或 PSL）。"""
    year, month = parse_date(target_month)

    if source == "cpc":
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        print(f"[下载] source={source}, 月份={target_month}")
        print(f"[下载] CPC 缓存目录：{raw_dir}")
        return download_godas_grib(year, month, raw_dir)

    if source == "psl":
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        print(f"[下载] source={source}, 月份={target_month}")
        download_all_variables(year, month, raw_dir)
        return raw_dir

    raise ValueError(f"不支持的 source: {source}")


def validate_preprocessed_outputs(processed_dir: str) -> None:
    """校验预处理产物是否齐全。"""
    required_vars = GODAS_OCEAN_VARS_ORDER + GODAS_VARS_ATMO
    missing = [
        var for var in required_vars
        if not os.path.isfile(os.path.join(processed_dir, f"{var}.nc"))
    ]
    if missing:
        raise FileNotFoundError(
            "预处理结果缺失变量文件："
            + ", ".join(f"{var}.nc" for var in missing)
        )


def run_preprocess_step(target_month: str, raw_dir: str, processed_dir: str, source: str = "cpc") -> str:
    """执行预处理（按 source 选择 CPC 或 PSL）。"""
    year, month = parse_date(target_month)
    effective_raw_dir = raw_dir
    print(f"[预处理] source={source}, 月份={target_month}")
    if source == "cpc":
        grib_file = resolve_cpc_grib_path(year, month, effective_raw_dir)
        if not os.path.exists(grib_file):
            raise FileNotFoundError(f"CPC GRIB 文件不存在：{grib_file}")
        preprocess_all_from_grib(grib_file, processed_dir)
        align_preprocessed_units_to_example(processed_dir)
        validate_preprocessed_outputs(processed_dir)
        return processed_dir

    if source == "psl":
        preprocess_all_variables(effective_raw_dir, processed_dir, year, month)
        align_preprocessed_units_to_example(processed_dir)
        validate_preprocessed_outputs(processed_dir)
        return processed_dir

    raise ValueError(f"不支持的 source: {source}")


def validate_source(source: str) -> str:
    if source not in ("cpc", "psl"):
        raise ValueError(f"不支持的 source: {source}，可选 cpc/psl")
    return source


def run_data_preprocess_pipeline(
    target_month: str,
    source: str = "cpc",
    raw_dir: str | None = None,
    processed_dir: str | None = None,
) -> str:
    """按日期执行下载与预处理，返回预处理目录绝对路径。"""
    parse_date(target_month)
    resolved_source = validate_source(source)

    if raw_dir is None:
        if resolved_source == "cpc":
            raw_dir = CPC_CACHE_DIR
        else:
            raw_dir = os.path.join(TMP_BASE_DIR, "preprocess_raw", target_month)

    if processed_dir is None:
        processed_dir = os.path.join(
            "./output/preprocessed",
            target_month,
            resolved_source,
        )

    raw_dir_path = Path(raw_dir).expanduser().resolve()
    processed_dir_path = Path(processed_dir).expanduser().resolve()
    raw_dir_path.mkdir(parents=True, exist_ok=True)
    processed_dir_path.mkdir(parents=True, exist_ok=True)

    run_download_step(
        target_month=target_month,
        raw_dir=str(raw_dir_path),
        source=resolved_source,
    )
    run_preprocess_step(
        target_month=target_month,
        raw_dir=str(raw_dir_path),
        processed_dir=str(processed_dir_path),
        source=resolved_source,
    )
    return str(processed_dir_path)


def run_inference_only_pipeline(
    target_month: str | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    source: str = "cpc",
) -> str:
    """执行仅推理流程并将 NetCDF 落盘到 output_dir。"""
    resolved_month = resolve_target_month(target_month)
    resolved_source = validate_source(source)
    year, month = parse_date(resolved_month)
    forecast_start_year, forecast_start_month = resolve_first_forecast_month(year, month)
    forecast_end_year, forecast_end_month = shift_year_month(year, month, PREDICT_STEPS)
    output_path = Path(output_dir).expanduser().resolve() / f"{resolved_month}.nc"

    print("=" * 60)
    print("ORCA-DL 海洋状态预测系统（仅推理）")
    print("=" * 60)
    print(f"\n[1/4] 解析输入日期：{resolved_month}")
    print(f"✓ 初始化日期：{year} 年 {month} 月")
    print(f"✓ 首报日期：{forecast_start_year} 年 {forecast_start_month} 月")
    print(f"✓ 数据源：{resolved_source.upper()}")
    print(f"✓ 输出目录：{output_path.parent}")

    print("\n[2/4] 检查依赖与设备...")
    check_dependencies()
    resolve_device()

    print(f"\n[3/4] 下载 GODAS 数据（{year}-{month:02d}）并执行预处理...")
    os.makedirs(TMP_BASE_DIR, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=TMP_BASE_DIR, prefix="orca_infer_") as tmp_dir:
        raw_dir = CPC_CACHE_DIR if resolved_source == "cpc" else os.path.join(tmp_dir, "raw")
        processed_dir = os.path.join(tmp_dir, "processed")
        preds_path = os.path.join(tmp_dir, "preds.npy")
        processed_dir = run_data_preprocess_pipeline(
            target_month=resolved_month,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            source=resolved_source,
        )

        print("\n[4/4] 执行模型推理与结果转换...")
        run_model_inference(
            processed_dir=processed_dir,
            month=month,
            preds_output_path=preds_path,
        )
        convert_predictions_to_netcdf(
            preds_path=preds_path,
            output_path=str(output_path),
            year=year,
            month=month,
        )

    print("\n" + "=" * 60)
    print("推理完成！")
    print("=" * 60)
    print(f"\n输出文件：{output_path}")
    print(f"预测时间范围：{forecast_start_year}-{forecast_start_month:02d} 至 {forecast_end_year}-{forecast_end_month:02d}")
    print("包含变量：so, thetao, tos, uo, vo, zos")
    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORCA-DL 海洋状态预测（仅推理）")
    parser.add_argument(
        "target_month",
        nargs="?",
        help="初始化月份，格式 YYYY-MM（如 2025-12）；默认上个月",
    )
    parser.add_argument(
        "--source",
        choices=["cpc", "psl"],
        default="cpc",
        help="数据源：cpc（默认）或 psl",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认 {DEFAULT_OUTPUT_DIR}）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference_only_pipeline(
        target_month=args.target_month,
        output_dir=args.output_dir,
        source=args.source,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误：{e}", file=sys.stderr)
        sys.exit(1)
