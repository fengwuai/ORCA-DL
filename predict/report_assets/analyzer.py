from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def analyze(input_file: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file_path = output_dir / "stats_summary.txt"

    with stats_file_path.open("w", encoding="utf-8") as stats_file:

        def log(text: str = "") -> None:
            print(text)
            stats_file.write(text + "\n")

        log("Loading data...")
        ds = xr.open_dataset(input_file)

        log("Analyzing El Nino...")
        nino34_lat_slice = slice(-5, 5)
        nino34_lon_slice = slice(190, 240)

        sst = ds["tos"]
        nino34_sst = sst.sel(lat=nino34_lat_slice, lon=nino34_lon_slice).mean(
            dim=["lat", "lon"]
        )
        nino34_mean = nino34_sst.mean()
        nino34_anom = nino34_sst - nino34_mean

        plt.figure(figsize=(10, 6))
        plt.plot(
            nino34_sst["time"],
            nino34_sst,
            marker="o",
            label="Nino 3.4 SST (Absolute)",
        )
        plt.axhline(y=nino34_mean, color="r", linestyle="--", label="24-month Mean")
        plt.title("Nino 3.4 Region Sea Surface Temperature Prediction")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "nino34_timeseries.png")
        plt.close()

        log("\n--- Nino 3.4 Stats ---")
        log(f"Mean SST: {nino34_mean.values:.2f} C")
        log(f"Max SST: {nino34_sst.max().values:.2f} C at {nino34_sst.idxmax().values}")
        log(f"Min SST: {nino34_sst.min().values:.2f} C at {nino34_sst.idxmin().values}")

        warm_phase = nino34_anom > 0.5
        cold_phase = nino34_anom < -0.5
        log("\nPossible Phases (Relative to 2-year mean):")
        for t, w, c, v in zip(
            nino34_sst.time.values,
            warm_phase.values,
            cold_phase.values,
            nino34_anom.values,
        ):
            phase = "Neutral"
            if w:
                phase = "Warm (El Nino-like)"
            if c:
                phase = "Cold (La Nina-like)"
            log(f"{pd.to_datetime(t).strftime('%Y-%m')}: {v:.2f} C Anomaly -> {phase}")

        log("\nGenerating SST Maps...")
        for t_idx in [0, 12, 23]:
            plt.figure(figsize=(12, 6))
            data_slice = sst.isel(time=t_idx)
            data_slice.plot(cmap="RdBu_r", vmin=-2, vmax=32)
            time_str = pd.to_datetime(ds.time[t_idx].values).strftime("%Y-%m")
            plt.title(f"Global SST Prediction: {time_str}")
            plt.tight_layout()
            plt.savefig(output_dir / f"sst_map_{t_idx}.png")
            plt.close()

        log("\nAnalyzing Currents...")
        u_surf = ds["uo"].isel(depth=0)
        v_surf = ds["vo"].isel(depth=0)
        current_speed = np.sqrt(u_surf**2 + v_surf**2)
        mean_speed = current_speed.mean(dim="time")

        plt.figure(figsize=(12, 6))
        mean_speed.plot(cmap="viridis", vmax=1.5)
        plt.title("Predicted Mean Surface Current Speed")
        plt.tight_layout()
        plt.savefig(output_dir / "mean_current_speed.png")
        plt.close()

        log("\nAnalysis Complete.")


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
