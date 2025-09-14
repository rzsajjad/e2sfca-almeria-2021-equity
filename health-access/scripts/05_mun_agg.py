#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import geopandas as gpd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lau", default="data/LAU_RG_01M_2021_3035.gpkg")
    ap.add_argument("--lau-name", default="LAU_NAME",
                    help="column in LAU with municipality name")
    ap.add_argument("--centroids", default="out/centroids.gpkg")
    ap.add_argument("--centroids-layer", default="centroids")
    ap.add_argument("--results", default="out/e2sfca_results.csv")
    ap.add_argument("--out", default="out/mun_accessibility.csv")
    args = ap.parse_args()

    cent = gpd.read_file(args.centroids, layer=args.centroids_layer)[["cell_id","geometry"]]
    lau  = gpd.read_file(args.lau)
    lau  = lau[lau.geometry.geom_type.isin(["Polygon","MultiPolygon"])].copy()
    lau  = lau.to_crs(cent.crs)

    if args.lau_name not in lau.columns:
        raise SystemExit(f"LAU name column '{args.lau_name}' not found. Available: {list(lau.columns)}")

    sj = gpd.sjoin(
        cent,
        lau[[args.lau_name, "geometry"]].rename(columns={args.lau_name: "mun"}),
        how="left",
        predicate="within",
    )[["cell_id","mun"]]

    res = pd.read_csv(args.results)[["cell_id","A_E2SFCA","pop"]]
    m = sj.merge(res, on="cell_id", how="left").dropna(subset=["mun"])
    g = m.groupby("mun", dropna=True)

    def wmean(s, w):
        w = w.fillna(0)
        return (s.fillna(0) * w).sum() / w.sum() if w.sum() > 0 else 0.0

    out = pd.DataFrame({
        "cells": g.size(),
        "pop":   g["pop"].sum(),
        "A_mean": g["A_E2SFCA"].mean(),
        "A_popw": g.apply(lambda d: wmean(d["A_E2SFCA"], d["pop"])),
    }).sort_values("A_popw", ascending=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} rows={len(out)}")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()