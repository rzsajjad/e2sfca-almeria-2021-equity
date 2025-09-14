#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# -------------------------
# helpers
# -------------------------

def decay_weight(d: np.ndarray, R: float, kind: str) -> np.ndarray:
    """
    Distance decay weights for distances (meters) in d.
    kind: 'gaussian' | 'linear' | 'step'
    """
    d = np.asarray(d, dtype="float64")
    if kind == "gaussian":
        # smooth; ~99.7% mass within R when sigma=R/3
        sigma = R / 3.0
        # prevent divide-by-zero if R is tiny
        if sigma <= 0:
            return (d <= R).astype("float64")
        return np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    elif kind == "linear":
        w = 1.0 - (d / R)
        w[d > R] = 0.0
        w[w < 0.0] = 0.0
        return w
    else:  # 'step'
        return (d <= R).astype("float64")


def gini_weighted(x: np.ndarray, w: np.ndarray) -> float:
    """
    Weighted Gini of non-negative values x with weights w (>=0).
    """
    x = np.asarray(x, dtype="float64")
    w = np.asarray(w, dtype="float64")
    mask = (w > 0) & np.isfinite(x)
    x = x[mask]
    w = w[mask]
    if x.size == 0 or w.sum() == 0:
        return float("nan")

    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cw = np.cumsum(w)
    cx = np.cumsum(x * w)
    cw = cw / cw[-1]
    cx = cx / cx[-1] if cx[-1] != 0 else cx  # avoid 0/0 if all x=0

    # numpy>=2.0: trapezoid (np.trapz is deprecated)
    B = np.trapezoid(cx, cw)
    gini = 1.0 - 2.0 * B
    return float(gini)


def read_centroids(path_gpkg: Path, layer: str = "centroids") -> gpd.GeoDataFrame:
    g = gpd.read_file(path_gpkg, layer=layer)
    if "cell_id" not in g.columns:
        raise SystemExit("centroids layer must contain 'cell_id' column")
    if g.crs is None:
        raise SystemExit("centroids has no CRS. It must be a projected CRS in meters (e.g., EPSG:3035).")
    return g


def read_facilities_csv(path_csv: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path_csv)
    need = {"lon", "lat", "name", "type", "mun", "prov"}
    missing = sorted(list(need - set(df.columns)))
    if missing:
        raise SystemExit(f"facilities CSV missing columns: {missing}")

    # keep only rows with coordinates
    df = df[~df["lon"].isna() & ~df["lat"].isna()].copy()
    df.reset_index(drop=True, inplace=True)

    g = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"].astype(float), df["lat"].astype(float), crs="EPSG:4326"),
    )
    return g


def pairs_within_radius(left: gpd.GeoDataFrame, right_points: gpd.GeoDataFrame, R: float) -> pd.DataFrame:
    """
    Return pairs (i,j) of left.index and right.index within radius R (meters).
    Uses buffering of right_points and sjoin for efficiency.
    """
    # buffer the facilities; we keep the original geometries for distance calc
    right_buf = right_points.copy()
    right_buf["geometry"] = right_buf.geometry.buffer(R)

    # spatial join: all left points inside any right buffer -> pairs
    # keep index mapping to facilities via 'index_right'
    joined = gpd.sjoin(left[["geometry"]], right_buf[["geometry"]], predicate="within", how="left")
    joined = joined.dropna(subset=["index_right"]).copy()
    joined["i"] = joined.index.values.astype(int)
    joined["j"] = joined["index_right"].astype(int)
    joined = joined[["i", "j"]].reset_index(drop=True)
    return joined


def distances_for_pairs(left_geom: gpd.GeoSeries, right_geom: gpd.GeoSeries, pairs: pd.DataFrame) -> np.ndarray:
    """
    Vectorized distance lookup for matched pairs (i,j).
    """
    # take geometries by position, then compute distance
    gi = left_geom.values.take(pairs["i"].to_numpy())
    gj = right_geom.values.take(pairs["j"].to_numpy())

    # shapely 2.0 has vectorized distance via array interface
    # but geopandas exposes as numpy object arrays; use list-comp then cast to np.array
    d = np.fromiter((a.distance(b) for a, b in zip(gi, gj)), dtype="float64", count=len(pairs))
    return d


# -------------------------
# main E2SFCA
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="E2SFCA accessibility on grid centroids")
    ap.add_argument("--centroids", required=True, help="GPKG with centroids layer")
    ap.add_argument("--facilities", required=True, help="CSV facilities with lon/lat/name/type/mun/prov")
    ap.add_argument("--out-gpkg", required=True, help="output GPKG path (will write layer=centroids)")
    ap.add_argument("--out-csv", required=True, help="output CSV with [cell_id, A_E2SFCA, A_per_10k]")
    ap.add_argument("--out-metrics", required=True, help="output CSV with summary metrics")
    ap.add_argument("--pop-col", default="T", help="population column name in centroids (default: T)")
    ap.add_argument("--radius-km", type=float, default=30.0, help="catchment radius in km (default: 30)")
    ap.add_argument("--decay", choices=["gaussian", "linear", "step"], default="gaussian",
                    help="distance decay function (default: gaussian)")
    args = ap.parse_args()

    cent_path = Path(args.centroids)
    fac_path = Path(args.facilities)
    out_gpkg = Path(args.out_gpkg)
    out_csv = Path(args.out_csv)
    out_metrics = Path(args.out_metrics)
    R = float(args.radius_km) * 1000.0

    print(f"[INFO] reading centroids: {cent_path}")
    g = read_centroids(cent_path, layer="centroids")
    print(f"rows(g)={len(g)}, crs={g.crs}")

    # population
    if args.pop_col not in g.columns:
        raise SystemExit(f"centroids missing population column '{args.pop_col}'")
    pop = g[args.pop_col].astype("float64").fillna(0.0).to_numpy()

    # facilities
    print(f"[INFO] reading facilities: {fac_path}")
    f_wgs = read_facilities_csv(fac_path)
    # project facilities to grid CRS (meters)
    f = f_wgs.to_crs(g.crs)
    f["supply"] = 1.0  # simple proxy (1 facility unit)

    if len(f) == 0:
        raise SystemExit("no facilities with coordinates found in CSV")

    print(f"[INFO] building pairs within {args.radius_km} km ...")
    # pairs for population around each facility and for accessibility at each centroid (same pairs)
    pairs = pairs_within_radius(g, f, R)
    if pairs.empty:
        print("[WARN] no centroid-facility pairs within radius; all A will be zero.")
        Ai = np.zeros(len(g), dtype="float64")
    else:
        # distances + weights
        d = distances_for_pairs(g.geometry, f.geometry, pairs)
        w = decay_weight(d, R, args.decay)

        # 1) Facility catchment population Pj = sum_i pop_i * w(d_ij)
        pairs_fac = pairs.copy()
        pairs_fac["w"] = w
        pairs_fac["pop_w"] = pop[pairs_fac["i"].to_numpy()] * pairs_fac["w"].to_numpy()
        Pj = pairs_fac.groupby("j", sort=False)["pop_w"].sum()
        # Avoid div-by-zero: facilities with no covered pop -> Rj=0
        Rj = pd.Series(0.0, index=f.index)
        nz = Pj > 0
        Rj.loc[nz.index[nz]] = (f.loc[nz.index[nz], "supply"] / Pj.loc[nz]).to_numpy()

        # 2) Accessibility at centroid: Ai = sum_j Rj * w(d_ij)
        pairs_acc = pairs.copy()
        pairs_acc["w"] = w
        pairs_acc["Rj"] = Rj.loc[pairs_acc["j"].to_numpy()].to_numpy()
        pairs_acc["Rw"] = pairs_acc["Rj"] * pairs_acc["w"]
        Ai = pairs_acc.groupby("i", sort=False)["Rw"].sum()
        # back to full length (zeros where no facilities nearby)
        Ai = Ai.reindex(range(len(g)), fill_value=0.0).to_numpy(dtype="float64")

    # -------------------------
    # metrics
    # -------------------------
    pct_pos = float((Ai > 0).mean()) * 100.0
    pop_w_mean = float(np.average(Ai, weights=pop)) if pop.sum() > 0 else float("nan")
    p05 = float(np.quantile(Ai, 0.05)) if len(Ai) else float("nan")
    p95 = float(np.quantile(Ai, 0.95)) if len(Ai) else float("nan")
    gini = gini_weighted(Ai, pop)

    print(f"A>0: {pct_pos:.1f}% | pop-weighted mean: {pop_w_mean:.10f} | p05: {p05:.10f} | Gini: {gini:.3f}")

    # -------------------------
    # write outputs
    # -------------------------
    out_df = pd.DataFrame({
        "cell_id": g["cell_id"].to_numpy(),
        "A_E2SFCA": Ai,
        "A_per_10k": Ai * 10000.0,  # scaled for readability
        "pop": pop
    })
    out_df.to_csv(out_csv, index=False, float_format="%.10g")
    print(f"[OK] wrote {out_csv}")

    g_out = g[["cell_id", "geometry"]].copy()
    g_out["A_E2SFCA"] = Ai
    g_out["A_per_10k"] = Ai * 10000.0
    g_out.to_file(out_gpkg, layer="centroids", driver="GPKG")
    print(f"[OK] wrote {out_gpkg} (layer=centroids)")

    met = pd.DataFrame([{
        "cells": int(len(g)),
        "total_pop": float(pop.sum()),
        "pct_A_gt_0": round(pct_pos, 1),
        "A_mean": float(np.average(Ai, weights=pop)) if pop.sum() > 0 else float("nan"),
        "A_p05": float(np.quantile(Ai, 0.05)) if len(Ai) else float("nan"),
        "A_p95": float(np.quantile(Ai, 0.95)) if len(Ai) else float("nan"),
        "Gini": gini_weighted(Ai, pop)
    }])
    met.to_csv(out_metrics, index=False)
    print(f"[OK] wrote {out_metrics}")

    # quick sanity echo
    print("rows gpkg:", len(g_out), "rows csv:", len(out_df), "crs:", g.crs)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)