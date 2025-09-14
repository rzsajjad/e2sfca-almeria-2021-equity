#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, unicodedata, re
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

# ---------- small utils ----------
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s

def pick_col(cols, tokens_all=(), tokens_any=()):
    """Pick a column whose normalized name contains tokens."""
    cand = []
    ncols = [norm(c) for c in cols]
    for i, c in enumerate(ncols):
        ok_all = all(t in c for t in tokens_all) if tokens_all else True
        ok_any = any(t in c for t in tokens_any) if tokens_any else True
        if ok_all and ok_any:
            cand.append((len(c), cols[i]))
    cand.sort()
    return cand[0][1] if cand else None

def gini_weighted(x: np.ndarray, w: np.ndarray) -> float:
    # remove non-positive weights and nans
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    if len(x) == 0 or w.sum() == 0:
        return float("nan")
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    # trapezoid integral between cumulative weight vs cumulative weighted value
    vx = np.cumsum(x * w)
    vx = vx / vx[-1] if vx[-1] != 0 else vx  # avoid div0
    area = np.trapz(vx, cw)
    # Gini = 1 - 2*area  (Lorenz curve area)
    return float(1.0 - 2.0 * area)

# ---------- IO helpers ----------
def read_centroids(p: Path) -> gpd.GeoDataFrame:
    g = gpd.read_file(p, layer="centroids") if p.suffix.lower()==".gpkg" else gpd.read_file(p)
    if "cell_id" not in g.columns:
        # try a safe fallback
        cid = pick_col(g.columns, tokens_all=("cell",), tokens_any=("id","idx","code"))
        if not cid:
            raise SystemExit("FATAL: could not find 'cell_id' column in centroids.")
        g = g.rename(columns={cid: "cell_id"})
    return g[["cell_id","geometry"]].copy()

def read_results(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # standardize required columns
    a = "A_E2SFCA" if "A_E2SFCA" in df.columns else pick_col(df.columns, tokens_all=("a",), tokens_any=("e2sfca","access"))
    pop = "pop" if "pop" in df.columns else pick_col(df.columns, tokens_all=("pop",), tokens_any=("population","people"))
    if not a or not pop:
        raise SystemExit("FATAL: results CSV needs columns for accessibility (A_E2SFCA) and population (pop).")
    out = df.rename(columns={a: "A_E2SFCA", pop: "pop"})
    return out[["cell_id","A_E2SFCA","pop"]].copy()

def read_lau(lau_path: Path) -> gpd.GeoDataFrame:
    lau = gpd.read_file(lau_path)
    # pick id/name robustly
    id_col = None
    name_col = None
    # explicit common names first
    for c in lau.columns:
        nc = norm(c)
        if nc in ("lau_id","lau_code","lauid","lau_code_2021","lau_id_2021","code"):
            id_col = c if id_col is None else id_col
        if nc in ("lau_name","name","lau_label","lau_name_latn"):
            name_col = c if name_col is None else name_col
    # heuristic fallback
    if id_col is None:
        id_col = pick_col(lau.columns, tokens_any=("lau","code","id"))
    if name_col is None:
        name_col = pick_col(lau.columns, tokens_any=("name","label","lau"))
    if id_col is None or name_col is None:
        raise SystemExit(f"FATAL: could not detect LAU id/name columns. Columns are: {list(lau.columns)}")
    lau = lau.rename(columns={id_col:"LAU_ID", name_col:"LAU_NAME"})
    # keep only polygons/multipolygons
    mask_poly = lau.geometry.geom_type.isin(["Polygon","MultiPolygon"])
    lau = lau.loc[mask_poly, ["LAU_ID","LAU_NAME","geometry"]].reset_index(drop=True)
    return lau

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Aggregate E2SFCA results to municipalities (LAU).")
    ap.add_argument("--centroids", required=True, help="GPKG/GeoPackage with layer=centroids (EPSG:3035)")
    ap.add_argument("--results",   required=True, help="CSV with columns: cell_id, A_E2SFCA, pop")
    ap.add_argument("--lau",       required=True, help="LAU polygons (e.g., LAU_RG_01M_2021_3035.gpkg)")
    ap.add_argument("--out",       required=True, help="Output CSV path")
    ap.add_argument("--lau-id-col", default=None, help="(optional) override LAU id column")
    ap.add_argument("--lau-name-col", default=None, help="(optional) override LAU name column")
    args = ap.parse_args()

    cent = read_centroids(Path(args.centroids))
    res  = read_results(Path(args.results))

    lau = gpd.read_file(args.lau)
    if args.lau_id_col and args.lau_id_col in lau.columns:
        lau = lau.rename(columns={args.lau_id_col:"LAU_ID"})
    if args.lau_name_col and args.lau_name_col in lau.columns:
        lau = lau.rename(columns={args.lau_name_col:"LAU_NAME"})
    # if still not standardized, autodetect
    if not {"LAU_ID","LAU_NAME"}.issubset(lau.columns):
        lau = read_lau(Path(args.lau))

    # CRS safe
    if hasattr(cent, "crs") and hasattr(lau, "crs") and cent.crs != lau.crs:
        lau = lau.to_crs(cent.crs)

    # spatial join (within), then nearest for leftovers
    j = gpd.sjoin(cent, lau[["LAU_ID","LAU_NAME","geometry"]], how="left", predicate="within").drop(columns=["index_right"])
    missing = j["LAU_ID"].isna()
    if missing.any():
        near = gpd.sjoin_nearest(cent.loc[missing], lau[["LAU_ID","LAU_NAME","geometry"]], how="left", distance_col="dist")
        j.loc[missing, ["LAU_ID","LAU_NAME"]] = near[["LAU_ID","LAU_NAME"]].values

    # join results
    j = j.merge(res, on="cell_id", how="left")
    # replace missing A with 0 per convention; pop NaN -> 0
    j["A_E2SFCA"] = j["A_E2SFCA"].fillna(0.0)
    j["pop"]      = j["pop"].fillna(0.0)

    # aggregate by municipality
    def agg(df):
        pop = df["pop"].to_numpy()
        a   = df["A_E2SFCA"].to_numpy()
        wmean = np.average(a, weights=pop) if pop.sum() > 0 else float(a.mean() if len(a) else np.nan)
        p05, p95 = (np.quantile(a, 0.05), np.quantile(a, 0.95)) if len(a) else (np.nan, np.nan)
        gini = gini_weighted(a, pop)
        return pd.Series({
            "cells": len(df),
            "pop": float(pop.sum()),
            "A_mean": float(wmean),
            "A_p05": float(p05),
            "A_p95": float(p95),
            "Gini": float(gini),
        })

    out = j.groupby("LAU_NAME", dropna=False, as_index=True).apply(agg).reset_index()
    out = out.rename(columns={"LAU_NAME":"mun"})
    out = out.sort_values(["mun"]).reset_index(drop=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # quick summary in console
    print(f"[OK] wrote {args.out} rows={len(out)}")
    print(out.head(12).to_string(index=False))

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if str(e):
            print(str(e), file=sys.stderr)
        raise