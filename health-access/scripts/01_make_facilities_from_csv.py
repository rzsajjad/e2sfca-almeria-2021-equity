# scripts/01_make_facilities_from_csv.py
# Create a clean facilities.csv with columns: name,type,mun,prov,lon,lat
# Robust to column names, encodings, and removes rows with missing name/lon/lat.

from __future__ import annotations
import argparse
import sys
import unicodedata
import re
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import geopandas as gpd
import fiona

# -------------------------
# helpers
# -------------------------
def norm_text(s: str) -> str:
    if pd.isna(s): 
        return ""
    s = str(s).strip().lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s

def pick_col_by_tokens(columns: List[str], tokens_all=(), tokens_any=()) -> Optional[str]:
    cols = list(columns)
    best = None
    best_len = 10**9
    for c in cols:
        key = norm_text(c)
        ok_all = all(t in key for t in tokens_all) if tokens_all else True
        ok_any = any(t in key for t in tokens_any) if tokens_any else True
        if ok_all and ok_any:
            if len(key) < best_len:
                best = c
                best_len = len(key)
    return best

def first_polygon_layer(gpkg_path: Path) -> Tuple[str, int]:
    layers = fiona.listlayers(str(gpkg_path))
    # Return first layer that has polygonal geometry (by name hint), otherwise the first layer
    for lyr in layers:
        key = norm_text(lyr)
        if any(t in key for t in ["poly", "bound", "mun", "lau"]):
            return lyr, len(layers)
    return layers[0], len(layers)

def classify_type(raw: str) -> str:
    s = norm_text(raw)
    if "hospital" in s or "especial" in s:
        return "hospital"
    if "centro de salud" in s or "health cen" in s:
        return "health_center"
    if "clin" in s:
        return "clinic"
    return "clinic"

# -------------------------
# I/O helpers
# -------------------------
def read_csv_smart(path: Path) -> pd.DataFrame:
    # try utf-8-sig → cp1252 → latin1
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # final fallback
    return pd.read_csv(path, encoding_errors="ignore")

def read_facilities_csv(path: Path) -> pd.DataFrame:
    df = read_csv_smart(path)

    # choose columns
    name_col = (pick_col_by_tokens(df.columns, tokens_any=("name","nombre","denomin")) or
                pick_col_by_tokens(df.columns, tokens_any=("centro","centres","centros")))

    type_col = (pick_col_by_tokens(df.columns, tokens_any=("type","tipo","clase")) or
                pick_col_by_tokens(df.columns, tokens_any=("nivel","nivelasist")))
    
    mun_col  = (pick_col_by_tokens(df.columns, tokens_any=("mun","municip","ayto","entidad")) or
                pick_col_by_tokens(df.columns, tokens_any=("localidad","pueblo","city","ciudad")))

    prov_col = (pick_col_by_tokens(df.columns, tokens_any=("prov","province","provincia")))

    if not name_col:
        print(f"[ERROR] Could not find a name column in {path}. Available: {list(df.columns)}", file=sys.stderr)
        sys.exit(2)

    # Normalize and select
    out = pd.DataFrame({
        "name": df[name_col].astype(str).str.strip(),
        "type_raw": df[type_col].astype(str).str.strip() if type_col else "",
        "mun": df[mun_col].astype(str).str.strip() if mun_col else "",
        "prov": df[prov_col].astype(str).str.strip() if prov_col else "",
    })

    # classify type
    out["type"] = out["type_raw"].map(classify_type)

    # drop blank names now (defensive)
    out = out[out["name"].astype(str).str.strip().ne("")]
    out = out.reset_index(drop=True)
    return out

def read_lau_and_boundary(lau_path: Path, boundary_path: Path) -> gpd.GeoDataFrame:
    lyr, n = first_polygon_layer(lau_path)
    lau = gpd.read_file(lau_path, layer=lyr)
    b_lyr, _ = first_polygon_layer(boundary_path)
    boundary = gpd.read_file(boundary_path, layer=b_lyr)

    # Choose municipal name column in LAU
    mun_col = (pick_col_by_tokens(lau.columns, tokens_all=("lau","name")) or
               pick_col_by_tokens(lau.columns, tokens_any=("name","mun","municip")))
    if not mun_col:
        print(f"[ERROR] Could not find municipal name column in LAU. Available: {list(lau.columns)}", file=sys.stderr)
        sys.exit(2)

    # clip to boundary (if CRS differ, align)
    if lau.crs != boundary.crs:
        boundary = boundary.to_crs(lau.crs)

    # keep only polygons that intersect; clip to be neat
    lau = lau[lau.geometry.notna()].copy()
    lau = gpd.clip(lau, boundary.unary_union)

    # Keep only necessary fields
    lau = lau[[mun_col, "geometry"]].rename(columns={mun_col: "mun"})
    lau = lau.drop_duplicates(subset=["mun"]).reset_index(drop=True)

    # robust centroids: compute in projected CRS then to WGS84
    if lau.crs is None:
        # assume EU LAEA if missing (EPSG:3035); adjust if needed
        lau.set_crs(3035, inplace=True)
    lau_proj = lau.to_crs(3035)
    lau_wgs  = lau_proj.to_crs(4326)
    lau_wgs["lon"] = lau_proj.geometry.centroid.to_crs(4326).x
    lau_wgs["lat"] = lau_proj.geometry.centroid.to_crs(4326).y
    return lau_wgs[["mun","geometry","lon","lat"]]

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fac", required=True, help="path to input raw facilities CSV")
    ap.add_argument("--lau", required=True, help="path to LAU GPKG")
    ap.add_argument("--boundary", required=True, help="path to boundary GPKG")
    ap.add_argument("--out", required=True, help="path to output facilities CSV")
    args = ap.parse_args()

    fac = read_facilities_csv(Path(args.fac))
    lau = read_lau_and_boundary(Path(args.lau), Path(args.boundary))

    # left join facilities to LAU by municipality name (string-normalized)
    key_fac = fac["mun"].map(norm_text)
    key_lau = lau["mun"].map(norm_text)
    fac = fac.assign(__key=key_fac)
    lau = lau.assign(__key=key_lau)

    joined = fac.merge(lau[["__key","lon","lat"]], on="__key", how="left")

    # final selection
    out = joined.drop(columns=["__key"])[["name","type","mun","prov","lon","lat"]]

    # HARD filter: remove anything without name or lon/lat (this kills the last NaN row)
    before = len(out)
    out = out[out["name"].astype(str).str.strip().ne("")]
    out = out.dropna(subset=["lon","lat"])
    after = len(out)

    # log unmatched (if you want to see which were removed)
    removed = before - after
    if removed > 0:
        print(f"[INFO] removed rows with missing name/lon/lat: {removed}")

    # write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # quick summary
    print(f"[OK] wrote {out_path} rows={len(out)}")
    print("nulls:", out.isna().sum().to_dict())
    print("sample:")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()