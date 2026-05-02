"""Create a starter AHU dataset template CSV for manual filling.

Usage:
  python scripts/prepare_dataset_template.py --out data/ahu_template.csv --rows 1000
"""

from __future__ import annotations

import argparse
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/ahu_template.csv")
    ap.add_argument("--rows", type=int, default=1000)
    args = ap.parse_args()

    t = pd.date_range("2025-01-01", periods=args.rows, freq="5min")
    cols = {
        "timestamp": t,
        "supply_air_temp": 0.0,
        "return_air_temp": 0.0,
        "outdoor_air_temp": 0.0,
        "supply_air_humidity": 0.0,
        "fan_speed": 0.0,
        "damper_position": 0.0,
        "cooling_valve": 0.0,
        "heating_valve": 0.0,
        "static_pressure": 0.0,
        "fault_label": 0,
    }
    df = pd.DataFrame(cols)
    df.to_csv(args.out, index=False)
    print(f"Template written to: {args.out}")


if __name__ == "__main__":
    main()
