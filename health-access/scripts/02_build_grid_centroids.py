#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Optional, List

import geopandas as gpd
from shapely.geometry import Point
import fiona


def list_layers(path: Path) -> List[str]:
    try:
        return list(fiona.listlayers(path))
    except Exception:
        return []


def read_poly_layer(path: Path, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    if layer:
        g = gpd.read_file(path, layer=layer)
        if g.geom_type.str.contains("Polygon", case=False, na=False).any():
            return g
    # try first polygon-like layer
    for lyr in list_layers(path):
        gi = gpd.read_file(path, layer=lyr)
        if gi.geom_type.str.contains("Polygon", case=False, na=False).any():
            return gi
    raise SystemExit(f"No polygon layer found in {path}")


def read_point_layer(path: Path, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    if layer:
        g = gpd.read_file(path, layer=layer)
        if g.geom_type.str.contains("Point", case=False, na=False).any():
            return g
    for lyr in list_layers(path):
        gi = gpd.read_file(path, layer=lyr)
        if gi.geom_type.str.contains("Point", case=False, na=False).any():
            return gi
    raise SystemExit(f"No point layer found in {path}")


def read_grid_any(path: Path) -> gpd.GeoDataFrame:
    """
    Accepts a grid file that may be polygons or points.
    Returns a GeoDataFrame of POINT centroids in EPSG:3035.
    """
    # prefer an existing point layer if present
    try:
        pts = read_point_layer(path, layer=None)
        cent = pts.copy()
    except SystemExit:
        polys = read_poly_layer(path, layer=None)
        cent = polys.copy()
        # if not projected, project first (centroid on geographic CRS can be misleading)
        if cent.crs is None:
            raise SystemExit("Grid CRS is None; please set CRS on the grid file.")
        if cent.crs.to_epsg() != 3035:
            cent = cent.to_crs(3035)
        cent["geometry"] = cent.geometry.centroid

    # ensure EPSG:3035
    if cent.crs is None or cent.crs.to_epsg() != 3035:
        cent = cent.to_crs(3035)
    cent = cent.reset_index(drop=True)
    return cent


def main() -> None:
    p = argparse.ArgumentParser(description="Build centroids (points) and preview CSV")
    p.add_argument("--grid", required=True, type=Path, help="ESTAT grid (gpkg) path")
    p.add_argument("--boundary", required=True, type=Path, help="Boundary (gpkg) path")
    p.add_argument("--out-centroids", required=True, type=Path, help="Output GPKG")
    p.add_argument("--out-grid-preview", required=True, type=Path, help="Preview CSV")
    args = p.parse_args()

    grid_path: Path = args.grid
    bnd_path: Path = args.boundary
    out_gpkg: Path = args.out_centroids
    out_prev: Path = args.out_grid_preview

    # read boundary polygon (single/multi ok)
    boundary = read_poly_layer(bnd_path, layer=None)
    if boundary.crs is None:
        raise SystemExit("Boundary CRS is None; please set CRS on boundary.")
    # project boundary to 3035 to match grid processing
    if boundary.crs.to_epsg() != 3035:
        boundary = boundary.to_crs(3035)

    # read grid as points
    cent = read_grid_any(grid_path)

    # keep points inside boundary
    mask = cent.within(boundary.unary_union)
    cent = cent.loc[mask].copy()

    # add sequential id
    cent = cent.reset_index(drop=True)
    cent["cell_id"] = cent.index.astype("int64")

    # save GPKG
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    cent.to_file(out_gpkg, layer="centroids", driver="GPKG")

    # preview CSV in WGS84
    wgs = cent.to_crs(4326)
    prev = (
        gpd.GeoDataFrame(wgs, geometry="geometry", crs=4326)
        .assign(lon=wgs.geometry.x, lat=wgs.geometry.y)
        .loc[:, ["cell_id", "lon", "lat"]]
    )
    prev.to_csv(out_prev, index=False)

    print(f"[OK] wrote: {out_gpkg} (layer=centroids)  rows={len(cent)}")
    print(f"[OK] wrote: {out_prev}  rows={len(prev)}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # friendly exit if output is piped
        sys.exit(0)