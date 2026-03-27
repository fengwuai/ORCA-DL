"""
ORCA-DL GODAS 数据推理脚本

功能：
    根据指定的年月下载 GODAS 数据，进行预处理和归一化，
    使用训练好的 ORCA-DL 模型进行 24 个月的海洋状态预测，
    并将结果保存为 NetCDF 文件。

使用方式：
    pixi run -e model python predict/inference.py 2025-12
    pixi run -e model python predict/inference.py 2025-12 --source psl

参数：
    input_date: 初始化月份，格式为 YYYY-MM（如 2025-12）
                该月份的 GODAS 数据必须已发布
    --source:   数据源选择（默认 cpc）
                cpc - CPC FTP 单月 GRIB 文件（推荐，下载更快）
                psl - PSL 按变量分年 NetCDF 文件

输出：
    ./output/predictions/orca_dl_prediction_2025_12_24months.nc

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
    2. 推理需要约 12 GB GPU 显存
    3. 临时文件会自动清理（位于 ./tmp 目录）
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Tuple
from datetime import datetime

import numpy as np
import torch
import xarray as xr
import pandas as pd
from urllib.request import urlretrieve
from urllib.error import HTTPError

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

# GRIB1 变量代码映射（CPC 数据源用）
GRIB_CODES = {
    'pottmp': 13,
    'salt': 88,
    'ucur': 49,
    'vcur': 50,
    'sshg': 198,
    'uflx': 124,
    'vflx': 125,
}

# 推理配置
PREDICT_STEPS = 24   # 预测月数
INPUT_STEPS = 1     # 输入时间步数
BATCH_SIZE = 1      # 推理批次大小

# 变量配置（GODAS 命名）
GODAS_VARS_3D = ['pottmp', 'salt', 'ucur', 'vcur']  # 3D 变量（16层）
GODAS_VARS_2D = ['sshg']                            # 2D 变量（1层）
GODAS_VARS_ATMO = ['uflx', 'vflx']                  # 大气强迫变量

# 注意：sst 不需要下载，会从 pottmp 的第一层提取

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
OUTPUT_DIR = "./output/predictions"
TMP_BASE_DIR = "./tmp"  # 临时文件基础目录
GODAS_RAW_DIR = "./tmp/GODAS_raw"  # GODAS 原始数据目录（持久化）

# 深度层级（米）
DEPTH_LEVELS = [10, 15, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800, 1000]


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
        return dt.year, dt.month
    except ValueError:
        raise ValueError(f"日期格式错误：{date_str}，应为 YYYY-MM 格式（如 2025-12）")


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
    all_vars = GODAS_VARS_3D + GODAS_VARS_2D + GODAS_VARS_ATMO
    for var in all_vars:
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

    try:
        print(f"  - {var_name}: 正在下载... ", end="", flush=True)
        urlretrieve(url, output_file)
        print("完成")
        return output_file
    except HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(
                f"GODAS 数据不存在：{url}\n"
                f"请确认 {year} 年的数据已发布"
            )
        else:
            raise RuntimeError(f"下载失败：{url}，错误码：{e.code}")


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


def download_godas_grib(year: int, month: int, output_dir: str) -> str:
    """
    从 CPC 下载单月 GODAS GRIB 文件

    Args:
        year: 年份
        month: 月份（1-12）
        output_dir: 输出目录

    Returns:
        GRIB 文件路径
    """
    filename = f"godas.M.{year}{month:02d}.grb"
    url = f"{GODAS_CPC_URL}/{filename}"
    output_file = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_file):
        print(f"  - {filename}: 文件已存在，跳过下载")
        return output_file

    try:
        print(f"  - {filename}: 正在下载... ", end="", flush=True)
        urlretrieve(url, output_file)
        print("完成")
        return output_file
    except HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(
                f"GODAS GRIB 数据不存在：{url}\n"
                f"请确认 {year}-{month:02d} 的数据已发布"
            )
        else:
            raise RuntimeError(f"下载失败：{url}，错误码：{e.code}")

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
    """
    从 GRIB 文件预处理 3D 变量（按 code 提取 + 垂直插值 + 水平插值）
    """
    levels = ",".join(map(str, DEPTH_LEVELS))
    code = GRIB_CODES[var_name]
    # -f nc4: GRIB 输入默认输出 GRIB，强制输出 NetCDF4
    # setname: GRIB 中变量名为 varXX，重命名以匹配 prepare_model_input 中的读取逻辑
    cmd = [
        "cdo", "-f", "nc4", "-b", "f64",
        f"setname,{var_name}",
        "-remapbil," + GRID_FILE,
        "-setzaxis," + ZAXIS_FILE,
        "-intlevel," + levels,
        f"-selcode,{code}",
        grib_file,
        output_nc
    ]
    run_cdo_command(cmd, f"GRIB 3D 变量插值：{var_name}")


def preprocess_2d_from_grib(grib_file: str, output_nc: str, var_name: str):
    """
    从 GRIB 文件预处理 2D 变量（按 code 提取 + 水平插值）
    """
    code = GRIB_CODES[var_name]
    cmd = [
        "cdo", "-f", "nc4", "-b", "f64",
        f"setname,{var_name}",
        "-remapbil," + GRID_FILE,
        f"-selcode,{code}",
        grib_file,
        output_nc
    ]
    run_cdo_command(cmd, f"GRIB 2D 变量插值：{var_name}")


def preprocess_all_from_grib(grib_file: str, processed_dir: str):
    """
    从单个 GRIB 文件预处理所有变量

    Args:
        grib_file: GRIB 文件路径
        processed_dir: 预处理输出目录
    """
    os.makedirs(processed_dir, exist_ok=True)

    for var in GODAS_VARS_3D:
        output_file = os.path.join(processed_dir, f"{var}.nc")
        print(f"  - 处理 3D 变量：{var}")
        preprocess_3d_from_grib(grib_file, output_file, var)

    for var in GODAS_VARS_2D + GODAS_VARS_ATMO:
        output_file = os.path.join(processed_dir, f"{var}.nc")
        print(f"  - 处理 2D 变量：{var}")
        preprocess_2d_from_grib(grib_file, output_file, var)


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

    # 读取并归一化 3D 海洋变量（每个 16 层）
    pottmp_data = None  # 保存 pottmp 数据用于提取 SST

    for var in GODAS_VARS_3D:
        nc_file = os.path.join(processed_dir, f"{var}.nc")
        ds = xr.open_dataset(nc_file)

        # 获取数据并去掉时间维度
        # CDO 输出格式: (1, 16, 128, 360) -> 需要取 [0] 得到 (16, 128, 360)
        data = ds[var].values
        if data.ndim == 4:  # (time, level, lat, lon)
            data = data[0]  # 取第一个时间步，得到 (16, 128, 360)

        # 保存 pottmp 数据用于后续提取 SST
        if var == 'pottmp':
            pottmp_data = data.copy()

        # 加载统计量并归一化
        mean, std = load_statistics(STAT_DIR, var)
        normalized = normalize_data(data, mean, std, month)

        # 添加到通道列表（16 个通道）
        for level in range(16):
            ocean_channels.append(normalized[level])

        ds.close()

    # 从 pottmp 第一层提取 SST 并归一化
    if pottmp_data is not None:
        sst_data = pottmp_data[0, :, :]  # 提取第一层 (128, 360)
        mean, std = load_statistics(STAT_DIR, 'sst')
        sst_normalized = normalize_data(sst_data, mean, std, month)
        ocean_channels.append(sst_normalized)

    # 读取并归一化 2D 海洋变量（sshg）
    for var in GODAS_VARS_2D:
        nc_file = os.path.join(processed_dir, f"{var}.nc")
        ds = xr.open_dataset(nc_file)

        # 获取数据并去掉时间维度
        data = ds[var].values
        if data.ndim == 3:  # (time, lat, lon)
            data = data[0]  # 取第一个时间步，得到 (128, 360)

        mean, std = load_statistics(STAT_DIR, var)
        normalized = normalize_data(data, mean, std, month)

        ocean_channels.append(normalized)
        ds.close()

    # 读取并归一化大气强迫变量
    for var in GODAS_VARS_ATMO:
        nc_file = os.path.join(processed_dir, f"{var}.nc")
        ds = xr.open_dataset(nc_file)

        # 获取数据并去掉时间维度
        data = ds[var].values
        if data.ndim == 3:  # (time, lat, lon)
            data = data[0]  # 取第一个时间步，得到 (128, 360)

        mean, std = load_statistics(STAT_DIR, var)
        normalized = normalize_data(data, mean, std, month)

        atmo_channels.append(normalized)
        ds.close()

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
        start_month: 起始月份（1-12）

    Returns:
        变量名到反归一化数据的映射
        - so: (24, 16, 128, 360)
        - thetao: (24, 16, 128, 360)
        - tos: (24, 1, 128, 360)  # 临时保留深度维度，后续会提取
        - uo: (24, 16, 128, 360)
        - vo: (24, 16, 128, 360)
        - zos: (24, 1, 128, 360)
    """
    # 移除批次维度
    preds = preds[0]  # (24, 66, 128, 360)

    # 根据 out_chans: [16, 16, 1, 16, 16, 1] 分割通道
    split_indices = [16, 32, 33, 49, 65]
    split_preds = np.split(preds, split_indices, axis=1)

    # 变量顺序：so, thetao, tos, uo, vo, zos
    var_names = ['salt', 'pottmp', 'sst', 'ucur', 'vcur', 'sshg']
    model_var_names = ['so', 'thetao', 'tos', 'uo', 'vo', 'zos']

    results = {}

    for i, (godas_var, model_var, pred) in enumerate(zip(var_names, model_var_names, split_preds)):
        # 加载统计量
        mean, std = load_statistics(stat_dir, godas_var)

        # 对每个时间步进行反归一化
        denormed_steps = []
        for step in range(PREDICT_STEPS):
            # 计算预测月份（循环 12 个月）
            pred_month = ((start_month - 1 + step) % 12)  # 0-11

            # 反归一化：pred * std + mean
            denormed = pred[step] * std[pred_month] + mean[pred_month]

            denormed_steps.append(denormed)

        # 拼接所有时间步
        results[model_var] = np.stack(denormed_steps, axis=0)

    return results


def extract_sst_from_pottmp(pottmp_data: np.ndarray) -> np.ndarray:
    """
    从位温数据中提取海表温度（第 1 层）

    Args:
        pottmp_data: 位温数据，shape: (24, 16, 128, 360)

    Returns:
        海表温度数据，shape: (24, 128, 360)
    """
    # 提取第 1 层（索引 0）
    return pottmp_data[:, 0, :, :]


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
        start_year: 起始年份
        start_month: 起始月份
    """
    # 创建时间坐标（24 个月）
    time_coord = pd.date_range(
        start=f"{start_year}-{start_month:02d}",
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
        predictions['tos'],
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
        'tos': 'Temperature of sea water at the surface (extracted from first level of potential temperature)',
        'uo': 'Eastward component of ocean current velocity at 16 depth levels',
        'vo': 'Northward component of ocean current velocity at 16 depth levels',
        'zos': 'Sea surface height anomaly relative to geoid'
    }
    return descriptions.get(var, '')


# ============ 主流程 ============

def main(input_date: str, source: str = "cpc"):
    """
    主推理流程

    Args:
        input_date: 格式为 YYYY-MM，如 "2025-12"
        source: 数据源，"cpc"（CPC GRIB，默认）或 "psl"（PSL NetCDF）
    """
    print("=" * 60)
    print("ORCA-DL 海洋状态预测系统")
    print("=" * 60)

    # 1. 解析输入日期
    print(f"\n[1/8] 解析输入日期：{input_date}")
    year, month = parse_date(input_date)
    print(f"✓ 起始日期：{year} 年 {month} 月")
    print(f"✓ 数据源：{source.upper()}")

    # 2. 检查依赖
    print("\n[2/8] 检查依赖项...")
    check_dependencies()

    # 3. 检查 GPU 可用性
    print("\n[3/8] 检查计算设备...")
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print(f"✓ 使用 GPU：{torch.cuda.get_device_name(0)}")
    #     print(f"  显存：{torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    # else:
    device = torch.device("cpu")
    print("⚠ GPU 不可用，使用 CPU（推理速度会较慢）")

    # 4. 下载数据
    print(f"\n[4/8] 下载 GODAS 数据（{year}-{month:02d}）...")
    os.makedirs(GODAS_RAW_DIR, exist_ok=True)

    if source == "cpc":
        grib_file = download_godas_grib(year, month, GODAS_RAW_DIR)
    else:
        download_all_variables(year, month, GODAS_RAW_DIR)

    with TemporaryDirectory(dir=TMP_BASE_DIR, prefix="orca_dl_") as tmp_dir:
        processed_dir = os.path.join(tmp_dir, "processed")

        # 5. 预处理数据
        print(f"\n[5/8] 预处理数据（CDO 插值）...")
        if source == "cpc":
            preprocess_all_from_grib(grib_file, processed_dir)
        else:
            preprocess_all_variables(GODAS_RAW_DIR, processed_dir, year, month)

        # 6. 准备模型输入
        print(f"\n[6/8] 准备模型输入（归一化）...")
        ocean_vars, atmo_vars = prepare_model_input(processed_dir, month)
        print(f"✓ 海洋变量形状：{ocean_vars.shape}")
        print(f"✓ 大气变量形状：{atmo_vars.shape}")

        # 7. 加载模型并推理
        print("\n[7/8] 加载模型并推理...")
        print(f"  模型配置：{MODEL_CONFIG_PATH}")
        print(f"  模型权重：{MODEL_CKPT_PATH}")
        model = load_model(MODEL_CONFIG_PATH, MODEL_CKPT_PATH, device)
        print("✓ 模型已加载")

        print(f"\n  正在推理（预测未来 {PREDICT_STEPS} 个月）...")
        with torch.no_grad():
            output = model(
                ocean_vars=ocean_vars.to(device),
                atmo_vars=atmo_vars.to(device),
                predict_time_steps=PREDICT_STEPS
            )

        print(f"✓ 推理完成，输出形状：{output.preds.shape}")

        # 8. 后处理并保存
        print("\n[8/8] 后处理并保存结果...")
        print("  正在反归一化...")
        predictions = denormalize_predictions(
            output.preds.cpu().numpy(),
            STAT_DIR,
            month
        )

        # 提取 SST（从 pottmp 第 1 层）
        print("  正在提取海表温度...")
        predictions['tos'] = extract_sst_from_pottmp(predictions['thetao'])

        # 保存结果
        output_file = os.path.join(
            OUTPUT_DIR,
            f"orca_dl_prediction_{year}_{month:02d}_24months.nc"
        )
        print(f"  正在保存到：{output_file}")
        save_to_netcdf(predictions, output_file, year, month)

    print("\n" + "=" * 60)
    print("推理完成！")
    print("=" * 60)
    print(f"\n输出文件：{output_file}")
    print(f"预测时间范围：{year}-{month:02d} 至 {year + (month + PREDICT_STEPS - 1) // 12}-{((month + PREDICT_STEPS - 1) % 12) + 1:02d}")
    print("包含变量：so, thetao, tos, uo, vo, zos")
    print("\n验证命令：")
    print(f"  pixi run -e exec ncdump -h {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ORCA-DL 海洋状态预测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  pixi run -e model python predict/inference.py 2025-12\n"
            "  pixi run -e model python predict/inference.py 2025-12 --source psl"
        ),
    )
    parser.add_argument("input_date", help="初始化月份，格式 YYYY-MM（如 2025-12）")
    parser.add_argument(
        "--source",
        choices=["cpc", "psl"],
        default="cpc",
        help="数据源：cpc（CPC GRIB，默认）或 psl（PSL NetCDF）",
    )
    args = parser.parse_args()

    try:
        main(args.input_date, source=args.source)
    except Exception as e:
        print(f"\n错误：{e}", file=sys.stderr)
        sys.exit(1)

