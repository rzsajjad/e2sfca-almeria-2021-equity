#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

# ---------- utils ----------
def pick_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def gini_weighted(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(x) == 0 or np.nansum(w) <= 0:
        return float("nan")
    order = np.argsort(x)
    x = x[order]; w = w[order]
    cw = np.cumsum(w)
    y = np.cumsum(x * w) / np.nansum(x * w) if np.nansum(x * w) > 0 else np.zeros_like(x)
    xq = cw / cw[-1]
    area = np.trapz(y, xq)
    return float(1.0 - 2.0 * area)

def print_head(df: pd.DataFrame, n=8, title=""):
    if title:
        print(title)
    print(df.head(n).to_string(index=False))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centroids", default="out/centroids.gpkg")
    ap.add_argument("--results",   default="out/e2sfca_results.csv")
    ap.add_argument("--gpkg",      default="out/accessibility.gpkg", help="accessibility gpkg (layer=centroids). Optional.")
    ap.add_argument("--metrics",   default="out/equity_metrics.csv", help="metrics csv. Optional.")
    ap.add_argument("--grid-preview", default="out/grid_preview.csv", help="optional")
    ap.add_argument("--boundary",  default="data/almeria_boundary.gpkg", help="optional")
    ap.add_argument("--out-enriched", default="out/e2sfca_results_enriched.csv")
    args = ap.parse_args()

    # 1) read core inputs
    e = pd.read_csv(args.results)
    if "cell_id" not in e.columns or "A_E2SFCA" not in e.columns:
        raise SystemExit("results csv must contain 'cell_id' and 'A_E2SFCA'")

    pop_col = "pop" if "pop" in e.columns else None
    if pop_col is None:
        print("[WARN] 'pop' not found in results; will use equal weights")
        e["pop"] = 1.0
        pop_col = "pop"

    # enrich if needed
    if ("A_norm01" not in e.columns) or ("A_pct" not in e.columns):
        x = e["A_E2SFCA"].to_numpy()
        span = np.nanmax(x) - np.nanmin(x)
        e["A_norm01"] = (x - np.nanmin(x)) / span if span > 0 else 1.0
        e["A_pct"] = e["A_E2SFCA"].rank(pct=True) * 100
        e.to_csv(args.out_enriched, index=False)
        print(f"[wrote] {args.out_enriched}")

    # 2) centroids gpkg checks (optional but recommended)
    try:
        g = gpd.read_file(args.centroids, layer="centroids")
        if g.crs is None or g.crs.to_epsg() != 3035:
            print(f"[WARN] centroids CRS is {g.crs}; expected EPSG:3035")
        n_match = (len(g) == len(e))
        set_match = (set(g["cell_id"]) == set(e["cell_id"]))
        print(f"[rows] gpkg: {len(g)}  csv: {len(e)}  EPSG:{g.crs.to_epsg() if g.crs else 'None'}  ids_match: {set_match}")
        if not n_match or not set_match:
            missing_in_csv = sorted(set(g["cell_id"]) - set(e["cell_id"]))[:5]
            missing_in_gpkg = sorted(set(e["cell_id"]) - set(g["cell_id"]))[:5]
            print("[WARN] id mismatch. samples:",
                  {"only_in_gpkg": missing_in_csv, "only_in_csv": missing_in_gpkg})
    except Exception as ex:
        print(f"[SKIP] centroids check ({ex})")

    # 3) basic stats on results
    na_a = int(e["A_E2SFCA"].isna().sum())
    na_pop = int(e[pop_col].isna().sum())
    dups = int(e["cell_id"].duplicated().sum())
    share_a0 = float((e["A_E2SFCA"] == 0).mean())
    share_a0_pop = float((e.loc[e["A_E2SFCA"] == 0, pop_col].sum() / e[pop_col].sum())) if e[pop_col].sum() > 0 else float("nan")
    q = e["A_E2SFCA"].quantile([0, .25, .5, .75, .95, 1.0]).to_dict()

    print(f"rows={len(e)} | NA in A_E2SFCA: {na_a} | NA in pop: {na_pop} | dup cell_id: {dups}")
    print(f"A==0 share (unweighted): {share_a0:.3f} | pop-weighted: {share_a0_pop:.3f}")
    print("A_E2SFCA quantiles:", {k: float(v) for k, v in q.items()})

    # top/bottom 10 (non-zero)
    nz = e[e["A_E2SFCA"] > 0].copy()
    if len(nz) > 0:
        top10 = nz.nlargest(10, "A_E2SFCA")[["cell_id", "A_E2SFCA", pop_col]]
        bot10 = nz.nsmallest(10, "A_E2SFCA")[["cell_id", "A_E2SFCA", pop_col]]
        print_head(top10, title="TOP 10 (A_E2SFCA)")
        print_head(bot10, title="BOTTOM 10 (A_E2SFCA nonzero)")
    else:
        print("[WARN] all A_E2SFCA are zero")

    # 4) compare with accessibility.gpkg (optional)
    try:
        ag = gpd.read_file(args.gpkg, layer="centroids")
        if "A_E2SFCA" in ag.columns:
            merged = e.merge(ag[["cell_id", "A_E2SFCA"]].rename(columns={"A_E2SFCA": "A_gpkg"}),
                             on="cell_id", how="inner")
            diff = float(np.nanmax(np.abs(merged["A_E2SFCA"] - merged["A_gpkg"])))
            print(f"[gpkg vs csv] max |diff| = {diff:.6g} on {len(merged)} cells")
        else:
            print("[SKIP] accessibility.gpkg has no 'A_E2SFCA' column")
    except Exception as ex:
        print(f"[SKIP] gpkg comparison ({ex})")

    # 5) boundary coverage (optional)
    try:
        b = gpd.read_file(args.boundary)
        bpoly = b.unary_union
        g = gpd.read_file(args.centroids, layer="centroids")
        inside = g.within(bpoly).mean()
        print(f"[within boundary] share = {inside:.3f}")
    except Exception as ex:
        print(f"[SKIP] boundary check ({ex})")

    # 6) metrics cross-check (optional)
    try:
        m = pd.read_csv(args.metrics)
        # pick columns by name flexibly
        amean_col = pick_col(m.columns, ["A_mean", "A_mean(pop-weighted)", "A_popw", "A_mean_pop"])
        gini_col  = pick_col(m.columns, ["Gini", "gini"])
        if amean_col is not None and gini_col is not None:
            # recompute from results
            a = e["A_E2SFCA"].to_numpy()
            w = e[pop_col].to_numpy()
            ame = float(np.average(a, weights=w)) if np.nansum(w) > 0 else float("nan")
            gin = gini_weighted(a, w)
            print(f"[metrics check] A_mean(popw) file={float(m[amean_col].iloc[0]):.6g} | recomputed={ame:.6g}")
            print(f"[metrics check] Gini           file={float(m[gini_col].iloc[0]):.6g} | recomputed={gin:.6g}")
        else:
            print("[SKIP] metrics csv columns not found")
    except Exception as ex:
        print(f"[SKIP] metrics check ({ex})")

    # 7) optional: preview-based mun/prov summary if available
    try:
        prev = pd.read_csv(args.grid_preview)
        needed = {"cell_id", "lon", "lat", "mun", "prov"}
        if needed.issubset(prev.columns):
            # simple weighted mean by municipality
            res = e.set_index("cell_id")[["A_E2SFCA", pop_col]]
            prev = prev.set_index("cell_id")
            jj = prev.join(res, how="inner").reset_index()
            mun_mean = (jj.groupby("mun")
                          .apply(lambda d: pd.Series({
                              "cells": len(d),
                              "A_popw": float(np.average(d["A_E2SFCA"], weights=d[pop_col]))
                          }), include_groups=False)
                        .reset_index())
            print_head(mun_mean.sort_values("A_popw").head(10), title="[preview] bottom 10 municipalities by weighted accessibility")
        else:
            print("[SKIP] grid_preview: required columns not present")
    except FileNotFoundError:
        print("[SKIP] no grid_preview.csv (ok)")
    except Exception as ex:
        print(f"[SKIP] grid_preview check ({ex})")

    print("\nALL GOOD (sanity checks finished)")

if __name__ == "__main__":
    main()