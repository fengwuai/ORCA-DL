from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class MonthlyAnomaly:
    month: str
    sst_c: float
    baseline_c: float
    anomaly_c: float
    baseline_source: str


@dataclass(frozen=True)
class ONISeason:
    season: str
    start_month: str
    end_month: str
    center_month: str
    oni_c: float
    phase: str


SEASON_LABEL_BY_START_MONTH: dict[int, str] = {
    1: "JFM",
    2: "FMA",
    3: "MAM",
    4: "AMJ",
    5: "MJJ",
    6: "JJA",
    7: "JAS",
    8: "ASO",
    9: "SON",
    10: "OND",
    11: "NDJ",
    12: "DJF",
}


def to_month(value: Any) -> str:
    return pd.to_datetime(value).strftime("%Y-%m")


def classify_oni_phase(oni_c: float) -> str:
    if oni_c >= 1.5:
        return "Strong El Nino"
    if oni_c >= 0.5:
        return "El Nino"
    if oni_c <= -1.5:
        return "Strong La Nina"
    if oni_c <= -0.5:
        return "La Nina"
    return "Neutral"


def load_climatology(clim_path: Path) -> xr.Dataset:
    """加载气候态文件并校验维度"""
    if not clim_path.is_file():
        raise FileNotFoundError(f"气候态文件不存在: {clim_path}")

    clim_ds = xr.open_dataset(clim_path)
    clim = clim_ds["clim"]

    expected_shape = (24, 12, 128, 360)
    if clim.shape != expected_shape:
        raise ValueError(f"气候态维度错误: {clim.shape}, 期望 {expected_shape}")

    return clim_ds


def extract_initialization_month(ds: xr.Dataset) -> int:
    """从 NetCDF 全局属性提取起报月份（1-12）"""
    init_date = ds.attrs.get("initialization_date", "")
    if not init_date:
        raise ValueError("NetCDF 缺少 initialization_date 属性")

    try:
        year, month = init_date.split("-")
        month = int(month)
        if not 1 <= month <= 12:
            raise ValueError(f"月份超出范围: {month}")

        # 计算预测开始月份（初始化月份的下一个月）
        forecast_start_month = (month % 12) + 1
        return forecast_start_month
    except (IndexError, ValueError) as exc:
        raise ValueError(f"无法解析 initialization_date: {init_date}") from exc


def compute_ssta_from_climatology(
    sst: xr.DataArray,
    clim_ds: xr.Dataset,
    start_month: int
) -> np.ndarray:
    """使用气候态计算 Nino3.4 SSTA 时间序列"""
    clim = clim_ds["clim"]
    lat = sst.lat.values
    lon = sst.lon.values

    lat_mask = (lat >= -5) & (lat <= 5)
    lon_mask = (lon >= 190) & (lon <= 240)

    n_time = sst.shape[0]
    ssta_series = np.zeros(n_time)

    month_idx = start_month - 1

    for t in range(n_time):
        sst_region = sst.values[t, lat_mask, :][:, lon_mask]
        clim_region = clim.values[t, month_idx, lat_mask, :][:, lon_mask]

        ssta = sst_region - clim_region
        with np.errstate(all="ignore"):
            ssta_series[t] = np.nanmean(ssta)

    return ssta_series


def discover_member_files(input_file: Path) -> list[Path] | None:
    """自动发现多 seed 成员文件"""
    target_month = input_file.stem
    members_dir = input_file.parent / "members" / target_month

    if not members_dir.is_dir():
        return None

    member_files = sorted(members_dir.glob("*.nc"))
    return member_files if member_files else None


def build_oni_seasons(monthly_anomalies: list[MonthlyAnomaly]) -> list[ONISeason]:
    if len(monthly_anomalies) < 3:
        return []

    oni_seasons: list[ONISeason] = []
    for idx in range(len(monthly_anomalies) - 2):
        window = monthly_anomalies[idx : idx + 3]
        oni_c = float(np.mean([item.anomaly_c for item in window]))
        start_month_num = int(window[0].month.split("-")[1])
        season_label = SEASON_LABEL_BY_START_MONTH[start_month_num]
        center_month = window[1].month
        center_year = center_month.split("-")[0]
        oni_seasons.append(
            ONISeason(
                season=f"{center_year}-{season_label}",
                start_month=window[0].month,
                end_month=window[2].month,
                center_month=center_month,
                oni_c=round(oni_c, 2),
                phase=classify_oni_phase(oni_c),
            )
        )
    return oni_seasons


def build_oni_events(oni_seasons: list[ONISeason]) -> list[dict[str, str | float | int]]:
    events: list[dict[str, str | float | int]] = []
    if not oni_seasons:
        return events

    def sign(value: float) -> int:
        if value >= 0.5:
            return 1
        if value <= -0.5:
            return -1
        return 0

    run_start: int | None = None
    run_sign = 0

    def flush_run(end_idx_exclusive: int) -> None:
        nonlocal run_start, run_sign
        if run_start is None or run_sign == 0:
            run_start = None
            run_sign = 0
            return

        run = oni_seasons[run_start:end_idx_exclusive]
        if len(run) < 5:
            run_start = None
            run_sign = 0
            return

        oni_values = [item.oni_c for item in run]
        peak_idx = int(np.argmax(np.abs(oni_values)))
        peak = run[peak_idx]
        events.append(
            {
                "event_type": "El Nino" if run_sign > 0 else "La Nina",
                "start_season": run[0].season,
                "end_season": run[-1].season,
                "start_center_month": run[0].center_month,
                "end_center_month": run[-1].center_month,
                "duration_seasons": len(run),
                "oni_min_c": round(float(np.min(oni_values)), 2),
                "oni_max_c": round(float(np.max(oni_values)), 2),
                "oni_mean_c": round(float(np.mean(oni_values)), 2),
                "peak_season": peak.season,
                "peak_oni_c": peak.oni_c,
            }
        )
        run_start = None
        run_sign = 0

    for idx, season in enumerate(oni_seasons):
        current_sign = sign(season.oni_c)
        if current_sign == 0:
            flush_run(idx)
            continue

        if run_start is None:
            run_start = idx
            run_sign = current_sign
            continue

        if current_sign != run_sign:
            flush_run(idx)
            run_start = idx
            run_sign = current_sign

    flush_run(len(oni_seasons))
    return events


def analyze(input_file: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file_path = output_dir / "stats_summary.txt"
    stats_json_path = output_dir / "stats_summary.json"

    with stats_file_path.open("w", encoding="utf-8") as stats_file:

        def log(text: str = "") -> None:
            print(text)
            stats_file.write(text + "\n")

        log("Loading data...")
        CLIM_PATH = Path(__file__).resolve().parent.parent.parent.parent / "clim" / "orca_clim_198002_202201.nc"
        clim_ds = load_climatology(CLIM_PATH)

        with xr.open_dataset(input_file) as ds:
            if int(ds.sizes.get("time", 0)) == 0:
                raise ValueError("Dataset has no time dimension")

            start_month = extract_initialization_month(ds)
            log(f"Initialization month: {start_month}")

            member_files = discover_member_files(input_file)
            if member_files:
                log(f"Found {len(member_files)} ensemble members")

            log("Analyzing ENSO with climatology baseline...")
            sst = ds["tos"]
            nino34_sst = sst.sel(lat=slice(-5, 5), lon=slice(190, 240)).mean(dim=["lat", "lon"])
            nino34_values = np.asarray(nino34_sst.values, dtype=float)

            member_ssta_list = []
            if member_files:
                log("Computing SSTA for each ensemble member...")
                for member_file in member_files:
                    with xr.open_dataset(member_file) as member_ds:
                        member_sst = member_ds["tos"]
                        member_ssta = compute_ssta_from_climatology(member_sst, clim_ds, start_month)
                        member_ssta_list.append(member_ssta)

                member_ssta_array = np.array(member_ssta_list)
                mme_ssta = np.mean(member_ssta_array, axis=0)
                spread_std = np.std(member_ssta_array, axis=0)

                log("Using MME SSTA computed from ensemble members")
                ssta_values = mme_ssta
            else:
                log("Computing SSTA from single MME file...")
                ssta_values = compute_ssta_from_climatology(sst, clim_ds, start_month)

            monthly_anomalies: list[MonthlyAnomaly] = []
            for t, (timestamp, sst_val, anom_val) in enumerate(
                zip(nino34_sst.time.values, nino34_values, ssta_values)
            ):
                baseline_val = float(sst_val - anom_val)
                monthly_anomalies.append(
                    MonthlyAnomaly(
                        month=to_month(timestamp),
                        sst_c=float(sst_val),
                        baseline_c=baseline_val,
                        anomaly_c=float(anom_val),
                        baseline_source=f"climatology:lead_{t}_month_{start_month}",
                    )
                )

            anomaly_values = np.asarray([item.anomaly_c for item in monthly_anomalies], dtype=float)

            plt.figure(figsize=(10, 6))
            if member_files:
                for i, member_ssta in enumerate(member_ssta_list):
                    plt.plot(
                        nino34_sst["time"].values,
                        member_ssta,
                        color="gray",
                        alpha=0.3,
                        linewidth=1,
                        label="Member" if i == 0 else None,
                    )
                plt.plot(
                    nino34_sst["time"].values,
                    ssta_values,
                    color="black",
                    linewidth=2.5,
                    marker="s",
                    label="MME",
                )
            else:
                plt.plot(
                    nino34_sst["time"].values,
                    ssta_values,
                    marker="o",
                    label="Nino 3.4 SSTA",
                )

            plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
            plt.title("Nino 3.4 SSTA Prediction")
            plt.xlabel("Date")
            plt.ylabel("SSTA (°C)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "nino34_timeseries.png")
            plt.close()

            max_idx = int(np.nanargmax(nino34_values))
            min_idx = int(np.nanargmin(nino34_values))
            max_month = to_month(nino34_sst.time.values[max_idx])
            min_month = to_month(nino34_sst.time.values[min_idx])
            max_sst = float(nino34_values[max_idx])
            min_sst = float(nino34_values[min_idx])

            max_anom_idx = int(np.nanargmax(anomaly_values))
            min_anom_idx = int(np.nanargmin(anomaly_values))

            log("\n--- Nino 3.4 Monthly Stats ---")
            log(f"Max SST: {max_sst:.2f} C at {max_month}")
            log(f"Min SST: {min_sst:.2f} C at {min_month}")
            log(
                f"Max Monthly Anomaly: {anomaly_values[max_anom_idx]:.2f} C at "
                f"{monthly_anomalies[max_anom_idx].month}"
            )
            log(
                f"Min Monthly Anomaly: {anomaly_values[min_anom_idx]:.2f} C at "
                f"{monthly_anomalies[min_anom_idx].month}"
            )

            oni_seasons = build_oni_seasons(monthly_anomalies)
            oni_events = build_oni_events(oni_seasons)
            log("\nONI Seasons (3-month running mean):")
            for item in oni_seasons:
                log(f"{item.season}: {item.oni_c:.2f} C -> {item.phase}")

            log("\nDetected ONI Events (>= 5 overlapping seasons):")
            if not oni_events:
                log("None")
            for event in oni_events:
                log(
                    f"{event['event_type']}: {event['start_season']} -> {event['end_season']}, "
                    f"peak {event['peak_oni_c']:.2f} C at {event['peak_season']}"
                )

            time_size = int(ds.sizes["time"])
            sst_map_indices = {
                "sst_map_0.png": 0,
                "sst_map_12.png": min(12, time_size - 1),
                "sst_map_23.png": min(23, time_size - 1),
            }

            log("\nGenerating SST Maps...")
            sst_map_months: dict[str, str] = {}
            for file_name, t_idx in sst_map_indices.items():
                plt.figure(figsize=(12, 6))
                data_slice = sst.isel(time=t_idx)
                data_slice.plot(cmap="RdBu_r", vmin=-2, vmax=32)
                month_value = to_month(ds.time.values[t_idx])
                sst_map_months[file_name] = month_value
                plt.title(f"Global SST Prediction: {month_value}")
                plt.tight_layout()
                plt.savefig(output_dir / file_name)
                plt.close()

            log("\nAnalyzing Currents...")
            u_surf = ds["uo"].isel(depth=0)
            v_surf = ds["vo"].isel(depth=0)
            current_speed = np.sqrt(u_surf**2 + v_surf**2)
            mean_speed = current_speed.mean(dim="time")
            current_speed_values = np.asarray(current_speed.values, dtype=float)
            mean_speed_value = float(np.nanmean(current_speed_values))
            p90_speed_value = float(np.nanpercentile(current_speed_values, 90))
            max_speed_value = float(np.nanmax(current_speed_values))

            plt.figure(figsize=(12, 6))
            mean_speed.plot(cmap="viridis", vmax=1.5)
            plt.title("Predicted Mean Surface Current Speed")
            plt.tight_layout()
            plt.savefig(output_dir / "mean_current_speed.png")
            plt.close()

            summary_payload: dict[str, Any] = {
                "dataset": {
                    "time_start": to_month(ds.time.values[0]),
                    "time_end": to_month(ds.time.values[time_size - 1]),
                    "time_count": time_size,
                },
                "nino34": {
                    "max_sst_c": round(max_sst, 2),
                    "max_month": max_month,
                    "min_sst_c": round(min_sst, 2),
                    "min_month": min_month,
                    "clim_file": str(CLIM_PATH),
                    "baseline_note": "ORCA climatology (1980-2022) by lead time and month",
                },
                "nino34_anomaly_monthly": [
                    {
                        "month": item.month,
                        "sst_c": round(item.sst_c, 2),
                        "baseline_c": round(item.baseline_c, 2),
                        "anomaly_c": round(item.anomaly_c, 2),
                        "baseline_source": item.baseline_source,
                    }
                    for item in monthly_anomalies
                ],
                "oni": [
                    {
                        "season": item.season,
                        "start_month": item.start_month,
                        "end_month": item.end_month,
                        "center_month": item.center_month,
                        "oni_c": item.oni_c,
                        "phase": item.phase,
                    }
                    for item in oni_seasons
                ],
                "oni_events": oni_events,
                "oni_thresholds": {
                    "warm_threshold_c": 0.5,
                    "cold_threshold_c": -0.5,
                    "minimum_consecutive_overlapping_seasons": 5,
                },
                "map_months": {
                    "sst_map_0": sst_map_months["sst_map_0.png"],
                    "sst_map_12": sst_map_months["sst_map_12.png"],
                    "sst_map_23": sst_map_months["sst_map_23.png"],
                },
                "current_speed": {
                    "mean_mps": round(mean_speed_value, 3),
                    "p90_mps": round(p90_speed_value, 3),
                    "max_mps": round(max_speed_value, 3),
                },
            }

            if member_files:
                summary_payload["ensemble_members"] = {
                    "count": len(member_files),
                    "member_ssta": [m.tolist() for m in member_ssta_list],
                    "mme_ssta": ssta_values.tolist(),
                    "spread_std": spread_std.tolist(),
                }

            stats_json_path.write_text(
                json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        log("\nAnalysis Complete.")
        log(f"Structured summary saved: {stats_json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ocean prediction data.")
    parser.add_argument("--input", required=True, help="Input NetCDF file path.")
    parser.add_argument("--output-dir", required=True, help="Output directory path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(input_file=Path(args.input), output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
