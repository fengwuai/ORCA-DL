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
class MonthlyPhase:
    month: str
    anomaly_c: float
    phase: str


def to_month(value: Any) -> str:
    return pd.to_datetime(value).strftime("%Y-%m")


def classify_phase(anomaly_c: float) -> str:
    if anomaly_c >= 1.0:
        return "Strong El Nino"
    if anomaly_c >= 0.5:
        return "Weak El Nino"
    if anomaly_c <= -1.0:
        return "Strong La Nina"
    if anomaly_c <= -0.5:
        return "La Nina"
    return "Neutral"


def build_phase_windows(monthly_phases: list[MonthlyPhase]) -> list[dict[str, str | float]]:
    if not monthly_phases:
        return []

    windows: list[dict[str, str | float]] = []
    window_start = monthly_phases[0].month
    window_end = monthly_phases[0].month
    window_phase = monthly_phases[0].phase
    window_anomalies: list[float] = [monthly_phases[0].anomaly_c]

    for item in monthly_phases[1:]:
        if item.phase == window_phase:
            window_end = item.month
            window_anomalies.append(item.anomaly_c)
            continue

        windows.append(
            {
                "start_month": window_start,
                "end_month": window_end,
                "phase": window_phase,
                "anomaly_min_c": round(min(window_anomalies), 2),
                "anomaly_max_c": round(max(window_anomalies), 2),
                "anomaly_mean_c": round(float(np.mean(window_anomalies)), 2),
            }
        )
        window_start = item.month
        window_end = item.month
        window_phase = item.phase
        window_anomalies = [item.anomaly_c]

    windows.append(
        {
            "start_month": window_start,
            "end_month": window_end,
            "phase": window_phase,
            "anomaly_min_c": round(min(window_anomalies), 2),
            "anomaly_max_c": round(max(window_anomalies), 2),
            "anomaly_mean_c": round(float(np.mean(window_anomalies)), 2),
        }
    )
    return windows


def analyze(input_file: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file_path = output_dir / "stats_summary.txt"
    stats_json_path = output_dir / "stats_summary.json"

    with stats_file_path.open("w", encoding="utf-8") as stats_file:

        def log(text: str = "") -> None:
            print(text)
            stats_file.write(text + "\n")

        log("Loading data...")
        with xr.open_dataset(input_file) as ds:
            if int(ds.sizes.get("time", 0)) == 0:
                raise ValueError("Dataset has no time dimension")

            log("Analyzing El Nino...")
            sst = ds["tos"]
            nino34_sst = sst.sel(lat=slice(-5, 5), lon=slice(190, 240)).mean(dim=["lat", "lon"])
            nino34_values = np.asarray(nino34_sst.values, dtype=float)
            nino34_mean = float(np.asarray(nino34_sst.mean().values).item())
            nino34_anom_values = nino34_values - nino34_mean

            plt.figure(figsize=(10, 6))
            plt.plot(nino34_sst["time"].values, nino34_values, marker="o", label="Nino 3.4 SST (Absolute)")
            plt.axhline(y=nino34_mean, color="r", linestyle="--", label="24-month Mean")
            plt.title("Nino 3.4 Region Sea Surface Temperature Prediction")
            plt.xlabel("Date")
            plt.ylabel("Temperature (°C)")
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

            log("\n--- Nino 3.4 Stats ---")
            log(f"Mean SST: {nino34_mean:.2f} C")
            log(f"Max SST: {max_sst:.2f} C at {max_month}")
            log(f"Min SST: {min_sst:.2f} C at {min_month}")

            monthly_phases: list[MonthlyPhase] = []
            log("\nPossible Phases (Relative to 2-year mean):")
            for t, v in zip(nino34_sst.time.values, nino34_anom_values):
                month = to_month(t)
                anomaly_c = float(v)
                phase = classify_phase(anomaly_c)
                monthly_phases.append(MonthlyPhase(month=month, anomaly_c=anomaly_c, phase=phase))
                log(f"{month}: {anomaly_c:.2f} C Anomaly -> {phase}")

            phase_windows = build_phase_windows(monthly_phases)

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
                    "mean_sst_c": round(nino34_mean, 2),
                    "max_sst_c": round(max_sst, 2),
                    "max_month": max_month,
                    "min_sst_c": round(min_sst, 2),
                    "min_month": min_month,
                    "warm_threshold_c": 0.5,
                    "cold_threshold_c": -0.5,
                },
                "monthly_phases": [
                    {
                        "month": item.month,
                        "anomaly_c": round(item.anomaly_c, 2),
                        "phase": item.phase,
                    }
                    for item in monthly_phases
                ],
                "phase_windows": phase_windows,
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
