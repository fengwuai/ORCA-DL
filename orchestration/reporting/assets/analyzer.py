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
class CPCBaselineRow:
    year: int
    month: int
    total_c: float
    clim_adjust_c: float
    anom_c: float


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


CPC_SNAPSHOT_PATH = Path(__file__).resolve().parent / "cpc" / "detrend.nino34.ascii.txt"
ROLLING_BASELINE_YEARS = 30
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


def parse_cpc_snapshot(snapshot_path: Path) -> list[CPCBaselineRow]:
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"CPC 基准快照不存在: {snapshot_path}")

    rows: list[CPCBaselineRow] = []
    for raw_line in snapshot_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("YR"):
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        year, month = int(parts[0]), int(parts[1])
        total_c, clim_adjust_c, anom_c = float(parts[2]), float(parts[3]), float(parts[4])
        rows.append(
            CPCBaselineRow(
                year=year,
                month=month,
                total_c=total_c,
                clim_adjust_c=clim_adjust_c,
                anom_c=anom_c,
            )
        )

    if not rows:
        raise ValueError(f"CPC 基准快照为空: {snapshot_path}")
    return rows


def build_total_lookup(
    rows: list[CPCBaselineRow],
) -> tuple[dict[tuple[int, int], float], int, int, dict[int, int], dict[int, int]]:
    total_lookup: dict[tuple[int, int], float] = {}
    years: set[int] = set()
    available_months: set[int] = set()
    min_year_by_month: dict[int, int] = {}
    max_year_by_month: dict[int, int] = {}

    for row in rows:
        total_lookup[(row.year, row.month)] = row.total_c
        years.add(row.year)
        available_months.add(row.month)
        current_min = min_year_by_month.get(row.month)
        current_max = max_year_by_month.get(row.month)
        min_year_by_month[row.month] = row.year if current_min is None else min(current_min, row.year)
        max_year_by_month[row.month] = row.year if current_max is None else max(current_max, row.year)

    missing_months = [month for month in range(1, 13) if month not in available_months]
    if missing_months:
        raise ValueError(f"CPC 快照缺少月份: {missing_months}")
    if not years:
        raise ValueError("CPC 快照中没有可用年份")
    return total_lookup, min(years), max(years), min_year_by_month, max_year_by_month


def resolve_rolling_30y_baseline_c(
    year: int,
    month: int,
    total_lookup: dict[tuple[int, int], float],
    month_min_year: int,
    month_max_year: int,
) -> tuple[float, str]:
    target_end_year = year - 1
    baseline_end_year = min(target_end_year, month_max_year)
    baseline_start_year = baseline_end_year - (ROLLING_BASELINE_YEARS - 1)

    if baseline_start_year < month_min_year:
        raise ValueError(
            "CPC 快照覆盖不足，无法计算过去30年基准："
            f"target={year}-{month:02d}, window={baseline_start_year}-{baseline_end_year}, "
            f"month_range={month_min_year}-{month_max_year}"
        )

    baseline_values: list[float] = []
    missing_years: list[int] = []
    for baseline_year in range(baseline_start_year, baseline_end_year + 1):
        value = total_lookup.get((baseline_year, month))
        if value is None:
            missing_years.append(baseline_year)
            continue
        baseline_values.append(value)

    if missing_years:
        raise ValueError(
            "CPC 快照缺失过去30年窗口数据："
            f"target={year}-{month:02d}, month={month:02d}, "
            f"missing_years={missing_years[:6]}{'...' if len(missing_years) > 6 else ''}"
        )

    baseline_c = float(np.mean(baseline_values))
    baseline_source = f"rolling_30y_total:{baseline_start_year}-{baseline_end_year}"
    if target_end_year > month_max_year:
        baseline_source += "(future_fixed_window)"
    return baseline_c, baseline_source


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
        cpc_rows = parse_cpc_snapshot(CPC_SNAPSHOT_PATH)
        (
            total_lookup,
            cpc_min_year,
            cpc_max_year,
            min_year_by_month,
            max_year_by_month,
        ) = build_total_lookup(cpc_rows)

        with xr.open_dataset(input_file) as ds:
            if int(ds.sizes.get("time", 0)) == 0:
                raise ValueError("Dataset has no time dimension")

            log("Analyzing ENSO with CPC baseline...")
            sst = ds["tos"]
            nino34_sst = sst.sel(lat=slice(-5, 5), lon=slice(190, 240)).mean(dim=["lat", "lon"])
            nino34_values = np.asarray(nino34_sst.values, dtype=float)
            monthly_anomalies: list[MonthlyAnomaly] = []
            future_fixed_window_count = 0

            for timestamp, sst_value in zip(nino34_sst.time.values, nino34_values):
                dt = pd.to_datetime(timestamp)
                baseline_c, baseline_source = resolve_rolling_30y_baseline_c(
                    year=int(dt.year),
                    month=int(dt.month),
                    total_lookup=total_lookup,
                    month_min_year=min_year_by_month[int(dt.month)],
                    month_max_year=max_year_by_month[int(dt.month)],
                )
                anomaly_c = float(sst_value - baseline_c)
                if "(future_fixed_window)" in baseline_source:
                    future_fixed_window_count += 1
                monthly_anomalies.append(
                    MonthlyAnomaly(
                        month=to_month(timestamp),
                        sst_c=float(sst_value),
                        baseline_c=float(baseline_c),
                        anomaly_c=anomaly_c,
                        baseline_source=baseline_source,
                    )
                )

            anomaly_values = np.asarray([item.anomaly_c for item in monthly_anomalies], dtype=float)
            plt.figure(figsize=(10, 6))
            plt.plot(
                nino34_sst["time"].values,
                anomaly_values,
                marker="o",
                label="Nino 3.4 SSTA",
            )
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
            log(f"Future fixed-window baseline months: {future_fixed_window_count}")

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
                    "cpc_snapshot_path": str(CPC_SNAPSHOT_PATH),
                    "baseline_note": "Past-30-year monthly mean from CPC TOTAL column",
                    "baseline_window_years": ROLLING_BASELINE_YEARS,
                    "cpc_data_year_range": {
                        "start_year": cpc_min_year,
                        "end_year": cpc_max_year,
                    },
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
