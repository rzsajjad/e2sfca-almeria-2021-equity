#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick, publication-ready plots for E2SFCA outputs.
Outputs -> out/plots/*.png
- Histograms (A_E2SFCA, A_norm01)
- Cells map (centroids colored by A_E2SFCA)
- Municipal choropleth (A_mean by LAU)
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ---------- defaults ----------
DEFAULT_RESULTS   = "out/e2sfca_results_enriched.csv"
DEFAULT_ACCESS    = "out/accessibility.gpkg"            # layer=centroids
DEFAULT_LAU       = "data/LAU_RG_01M_2021_3035.gpkg"
DEFAULT_BOUNDARY  = "data/almeria_boundary.gpkg"
DEFAULT_MUNMET    = "out/mun_accessibility.csv"
DEFAULT_OUTDIR    = "out/plots"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 12, "axes.titlesize": 14,
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.facecolor": "white", "savefig.facecolor": "white",
})

def ensure_outdir(p: Path): p.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def _guess_lau_name_col(cols):
    prefs = ["LAU_NAME","LAU_NAME_EN","NAME_LATN","NAME_ENGL","NAME","name","NOMBRE"]
    for c in prefs:
        if c in cols: return c
    for c in cols:
        if "name" in c.lower(): return c
    return None

def _guess_df_name_col(cols):
    prefs = ["LAU_NAME","lau_name","mun","municipio","municipality","NAME","name"]
    for c in prefs:
        if c in cols: return c
    for c in cols:
        s = c.lower()
        if ("lau" in s and "name" in s) or ("mun" in s and "name" in s): return c
    return None

def _pick_metric(cols, preferred=("A_E2SFCA","A_mean","A_norm01")):
    for p in preferred:
        if p in cols: return p
    # fallback: first column that شبیه A_* است
    for c in cols:
        if c.lower().startswith("a_"): return c
    return None

# ---------- 1) histograms ----------
def plot_cells_hist(results_csv: Path, out_dir: Path, dpi: int = 300):
    df = pd.read_csv(results_csv)
    metric = "A_E2SFCA" if "A_E2SFCA" in df.columns else _pick_metric(df.columns)
    normcol = "A_norm01" if "A_norm01" in df.columns else None
    popcol  = "pop" if "pop" in df.columns else None

    # unweighted
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(df[metric].fillna(0), bins=60)
    ax.set_title("Histogram of A_E2SFCA (unweighted)")
    ax.set_xlabel(metric); ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(out_dir/"cells_hist_unweighted.png", dpi=dpi); plt.close(fig)

    # population-weighted
    if popcol:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(df[metric].fillna(0), bins=60, weights=df[popcol].fillna(0))
        ax.set_title("Histogram of A_E2SFCA (population-weighted)")
        ax.set_xlabel(metric); ax.set_ylabel("population-weighted count")
        fig.tight_layout(); fig.savefig(out_dir/"cells_hist_popweighted.png", dpi=dpi); plt.close(fig)

    # A_norm01 log-x
    if normcol:
        x = df[normcol].clip(lower=1e-5).fillna(1e-5)
        fig, ax = plt.subplots(figsize=(10,6))
        ax.hist(x, bins=80); ax.set_xscale("log")
        ax.set_title("Histogram of A_norm01 (log-x)")
        ax.set_xlabel(normcol); ax.set_ylabel("count")
        a0_share = float((df[normcol] <= 0).mean())
        ax.text(0.02, 0.95, f"A==0 share: {a0_share:.3f}", transform=ax.transAxes,
                ha="left", va="top")
        fig.tight_layout(); fig.savefig(out_dir/"cells_hist_A_norm01_log.png", dpi=dpi); plt.close(fig)

# ---------- 2) cells map ----------
def plot_cells_map(access_gpkg: Path, boundary_gpkg: Path, out_dir: Path, dpi: int = 300):
    g = gpd.read_file(access_gpkg, layer="centroids")
    b = gpd.read_file(boundary_gpkg)
    if g.crs != b.crs: g = g.to_crs(b.crs)
    bpoly = b.unary_union
    g = g[g.within(bpoly)]

    metric = "A_E2SFCA" if "A_E2SFCA" in g.columns else _pick_metric(g.columns)
    v = g[metric].fillna(0)
    vmin, vmax = np.quantile(v, [0.00, 0.99])
    norm = Normalize(vmin=float(vmin), vmax=float(vmax))

    fig, ax = plt.subplots(figsize=(8,8))
    b.plot(ax=ax, color="white", edgecolor="black", linewidth=0.8)
    g.plot(ax=ax, column=metric, cmap="viridis", markersize=6, alpha=0.9,
           norm=norm, legend=True)
    ax.set_title("Cells map – A_E2SFCA")
    ax.set_axis_off()
    fig.tight_layout(); fig.savefig(out_dir/"cells_map_points.png", dpi=dpi); plt.close(fig)

# ---------- 3) municipal map ----------
def plot_mun_map(mun_metrics_csv: Path, lau_gpkg: Path, boundary_gpkg: Path,
                 out_dir: Path, lau_name_col: str | None = None,
                 metric_col: str | None = None, dpi: int = 300):

    df  = pd.read_csv(mun_metrics_csv)
    lau = gpd.read_file(lau_gpkg)
    b   = gpd.read_file(boundary_gpkg)
    if lau.crs != b.crs: lau = lau.to_crs(b.crs)
    bpoly = b.unary_union

    if not lau_name_col or lau_name_col not in lau.columns:
        lau_name_col = _guess_lau_name_col(lau.columns)
        if not lau_name_col:
            raise SystemExit(f"[ERR] LAU name column not found in LAU file. Available: {list(lau.columns)}")

    df_name_col = _guess_df_name_col(df.columns)
    if not df_name_col:
        raise SystemExit(f"[ERR] Name column not found in municipal metrics CSV. Available: {list(df.columns)}")

    if not metric_col or metric_col not in df.columns:
        # رایج‌ترین‌ها: A_mean یا هر ستونی که با A_ شروع شود
        metric_col = "A_mean" if "A_mean" in df.columns else _pick_metric(df.columns)
        if not metric_col:
            raise SystemExit(f"[ERR] Metric column not found. Available: {list(df.columns)}")

    # join با کلید استاندارد شده
    left_key  = lau[lau_name_col].astype(str).str.strip().str.lower()
    right_key = df[df_name_col].astype(str).str.strip().str.lower()
    joined = lau.merge(df.assign(_key=right_key), left_on=left_key, right_on="_key", how="left")
    joined = joined[joined.within(bpoly)]

    fig, ax = plt.subplots(figsize=(8,8))
    b.plot(ax=ax, color="white", edgecolor="black", linewidth=0.8)
    joined.plot(ax=ax, column=metric_col, cmap="viridis", linewidth=0.3,
                edgecolor="grey", legend=True,
                legend_kwds={"label": metric_col, "shrink": 0.8})
    ax.set_title(f"Municipal accessibility ({metric_col})")
    ax.set_axis_off()
    fig.tight_layout(); fig.savefig(out_dir/"mun_map_A_mean.png", dpi=dpi); plt.close(fig)

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Quick plots for E2SFCA outputs")
    p.add_argument("--plot", choices=["all","cells_hist","cells_map","mun_map"], default="all")
    p.add_argument("--results", default=DEFAULT_RESULTS)
    p.add_argument("--accessibility", default=DEFAULT_ACCESS)
    p.add_argument("--boundary", default=DEFAULT_BOUNDARY)
    p.add_argument("--lau", default=DEFAULT_LAU)
    p.add_argument("--mun-metrics", default=DEFAULT_MUNMET)
    p.add_argument("--lau-name-col", default=None)
    p.add_argument("--metric-col", default=None)
    p.add_argument("--out-dir", default=DEFAULT_OUTDIR)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    out_dir = Path(args.out_dir); ensure_outdir(out_dir)

    if args.plot in ("all","cells_hist"):
        print("[PLOT] histograms …"); plot_cells_hist(Path(args.results), out_dir, dpi=args.dpi)
    if args.plot in ("all","cells_map"):
        print("[PLOT] cells map …"); plot_cells_map(Path(args.accessibility), Path(args.boundary), out_dir, dpi=args.dpi)
    if args.plot in ("all","mun_map"):
        print("[PLOT] municipal map …")
        plot_mun_map(Path(args.mun_metrics), Path(args.lau), Path(args.boundary),
                     out_dir, lau_name_col=args.lau_name_col, metric_col=args.metric_col, dpi=args.dpi)

    print(f"[OK] plots saved in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()