#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

def main():
    pairs = [
        ("base",  Path("out/equity_metrics.csv")),
        ("r20",   Path("out/equity_metrics_r20.csv")),
        ("bin30", Path("out/equity_metrics_bin30.csv")),
    ]
    frames = []
    for name, p in pairs:
        if p.exists():
            df = pd.read_csv(p)
            df["scenario"] = name
            frames.append(df)
            print(f"[OK] read {p} rows={len(df)}")
        else:
            print(f"[WARN] missing {p}")

    if not frames:
        raise SystemExit("No metrics files found in out/")

    out = pd.concat(frames, ignore_index=True)
    out_path = Path("out/equity_metrics_all.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} rows={len(out)}")
    print(out.head().to_string(index=False))

if __name__ == "__main__":
    main()