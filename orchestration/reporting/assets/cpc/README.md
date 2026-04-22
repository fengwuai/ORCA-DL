# CPC Niño3.4 baseline snapshot

This directory stores local snapshots from NOAA CPC used by `analyzer.py`.

## File

- `detrend.nino34.ascii.txt`
  - Source: https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt
  - Columns: `YR MON TOTAL ClimAdjust ANOM`
  - Usage in this repo:
    - `TOTAL` is used to compute monthly past-30-year rolling baseline (same calendar month).
    - `ClimAdjust` and `ANOM` are retained as reference columns from CPC snapshot.

## Update

Refresh this file periodically from CPC to keep baseline consistent with CPC updates.
