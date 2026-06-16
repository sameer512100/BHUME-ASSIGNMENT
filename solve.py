#!/usr/bin/env python3
"""
BhuMe boundary correction.

Method:
  1. Global offset  — FFT cross-correlation on sampled plots → village-wide (dx, dy)
  2. Per-plot local — FFT on a PAD_M patch centred on the globally-shifted polygon
  3. Signal         — 0.6 * boundary hints + 0.4 * imagery edges
  4. Neighbour smooth — weighted-average nearby shifts, penalise outliers
  5. Confidence     — peak sharpness (peak / search-region mean) × improvement flag
  6. Flag           — area ratio outside [0.5, 2.0]

Run:
    uv run solve.py data/34855_vadnerbhairav_chandavad_nashik
    uv run solve.py data/malatavadi_folder
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from shapely.affinity import translate
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from scipy.ndimage import gaussian_filter, sobel
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from PIL import Image as PILImage

from bhume import load, write_predictions, score

warnings.filterwarnings("ignore")

# ── Tuning ────────────────────────────────────────────────────────────────────
GLOBAL_SAMPLE     = 80
GLOBAL_RADIUS_M   = 50.0    # search radius for global step
LOCAL_RADIUS_M    = 25.0    # search radius for local refinement
PAD_M             = 60.0    # patch padding (must be > search radius)
HINT_WEIGHT       = 0.6
AREA_RATIO_LO     = 0.5
AREA_RATIO_HI     = 2.0
NEIGHBOR_K        = 8
NEIGHBOR_RADIUS_M = 300.0

# ── CRS ───────────────────────────────────────────────────────────────────────

def _transformer(src, dst):
    return Transformer.from_crs(src, dst, always_xy=True)

def to_crs(geom: BaseGeometry, src_crs, dst_crs) -> BaseGeometry:
    t = _transformer(src_crs, dst_crs)
    return shp_transform(lambda xs, ys, z=None: t.transform(xs, ys), geom)

def utm_crs(geom: BaseGeometry) -> str:
    lon = geom.centroid.x
    return f"EPSG:{32600 + int((lon + 180) // 6) + 1}"

def shift_geom(geom_4326: BaseGeometry, dx_m: float, dy_m: float, utmc: str) -> BaseGeometry:
    """Translate geom by (dx_m east, dy_m north) in metres, return in 4326."""
    g = to_crs(geom_4326, "EPSG:4326", utmc)
    return to_crs(translate(g, dx_m, dy_m), utmc, "EPSG:4326")

# ── Raster patch ──────────────────────────────────────────────────────────────

def get_patch(src, geom_4326: BaseGeometry, pad: float, bands=None):
    """
    Extract raster patch around geom_4326 with padding.
    Returns (array C×H×W, window_transform, res_m) or (None, None, None).
    """
    g = to_crs(geom_4326, "EPSG:4326", str(src.crs))
    l, b, r, t = g.bounds
    l, b, r, t = l-pad, b-pad, r+pad, t+pad
    dl, db, dr, dt = src.bounds
    l, b, r, t = max(l,dl), max(b,db), min(r,dr), min(t,dt)
    if r <= l or t <= b:
        return None, None, None
    win = from_bounds(l, b, r, t, transform=src.transform)
    bds = bands or list(range(1, src.count+1))
    return src.read(bds, window=win), src.window_transform(win), src.res[0]

# ── Signals ───────────────────────────────────────────────────────────────────

def imagery_edges(arr) -> np.ndarray:
    gray = arr.mean(axis=0).astype(np.float32)
    gray = gaussian_filter(gray, sigma=1.0)
    e = np.hypot(sobel(gray, axis=1), sobel(gray, axis=0))
    mx = e.max()
    return (e/mx).astype(np.float32) if mx > 0 else e

def blended_signal(geom_4326, bsrc, isrc, pad):
    """
    Returns (signal H×W, window_transform, res_m) in imagery CRS.
    Blends boundary hints and imagery edges.
    """
    iarr, iwt, ires = get_patch(isrc, geom_4326, pad)
    if iarr is None:
        return None, None, None
    img_e = imagery_edges(iarr)

    if bsrc is not None:
        barr, _, _ = get_patch(bsrc, geom_4326, pad, bands=[1])
        if barr is not None and barr[0].max() > 0:
            h = barr[0].astype(np.float32)
            h = h / h.max()
            if h.shape != img_e.shape:
                h = np.array(PILImage.fromarray(h).resize(
                    (img_e.shape[1], img_e.shape[0]), PILImage.BILINEAR))
            signal = HINT_WEIGHT * h + (1-HINT_WEIGHT) * img_e
        else:
            signal = img_e
    else:
        signal = img_e

    return signal.astype(np.float32), iwt, ires

def polygon_edge_mask(geom_4326: BaseGeometry, wt, shape_hw, img_crs: str) -> np.ndarray:
    """Thin ring mask around polygon boundary, in imagery CRS pixels."""
    g = to_crs(geom_4326, "EPSG:4326", img_crs)
    ring = g.buffer(3).difference(g.buffer(-3))
    return rasterize(
        [(mapping(ring), 1)], out_shape=shape_hw,
        transform=wt, fill=0, dtype=np.uint8,
    ).astype(np.float32)

# ── Core FFT alignment ────────────────────────────────────────────────────────

def fft_best_shift(signal: np.ndarray, mask: np.ndarray,
                   res_m: float, radius_m: float):
    """
    Find the translation (dx_m east, dy_m north) that best aligns mask onto signal.

    The correlation peak at (row_offset, col_offset) from centre means:
      - col_offset > 0 → mask should move right  → dx positive (east)
      - row_offset > 0 → mask should move down   → dy negative (south, because map y flips)

    Returns (dx_m, dy_m, peak_sharpness, improved: bool)
    """
    corr = fftconvolve(signal, mask[::-1, ::-1], mode="same")
    H, W = corr.shape
    cy, cx = H // 2, W // 2

    r_px = max(1, int(radius_m / res_m))
    r0, r1 = max(0, cy-r_px), min(H, cy+r_px)
    c0, c1 = max(0, cx-r_px), min(W, cx+r_px)
    sub = corr[r0:r1, c0:c1]

    pi, pj = np.unravel_index(sub.argmax(), sub.shape)
    row_off = (pi + r0) - cy   # positive = peak is below centre = mask moves down = south
    col_off = (pj + c0) - cx   # positive = peak is right of centre = mask moves right = east

    best  = float(sub.max())
    base  = float(corr[cy, cx])
    smean = float(sub.mean()) if sub.size > 0 else 1.0

    peak_sharpness = best / (smean + 1e-6)
    improved = best > base

    dx_m =  col_off * res_m        # east
    dy_m = -row_off * res_m        # north (negate because image row↓ = south)

    return dx_m, dy_m, peak_sharpness, improved

# ── Confidence ────────────────────────────────────────────────────────────────

def confidence(peak_sharpness: float, improved: bool, signal_mean: float) -> float:
    if signal_mean < 1e-4 or not improved:
        return 0.1
    # sharpness=1 → flat (no clear snap), sharpness=6+ → clear peak
    c = np.clip((peak_sharpness - 1.0) / 5.0, 0.0, 1.0)
    return float(np.clip(c, 0.1, 0.95))

# ── Area flag ─────────────────────────────────────────────────────────────────

def area_flag(row) -> tuple[bool, str]:
    drawn    = row.get("map_area_sqm")
    recorded = row.get("recorded_area_sqm")
    pot      = row.get("pot_kharaba_ha")
    if not recorded or float(recorded) <= 0 or not drawn or float(drawn) <= 0:
        return False, ""
    total = float(recorded)
    if pot and not np.isnan(float(pot)):
        total += float(pot) * 10_000
    ratio = float(drawn) / total
    if ratio < AREA_RATIO_LO:
        return True, f"ratio={ratio:.2f} too small"
    if ratio > AREA_RATIO_HI:
        return True, f"ratio={ratio:.2f} too large"
    return False, ""

# ── Global offset ─────────────────────────────────────────────────────────────

def estimate_global(village, bsrc, isrc, utmc) -> tuple[float, float]:
    """
    Sample plots near median size, run FFT on each, return median (dx, dy).
    Using median-size plots avoids the large-plot bias that breaks dense villages.
    """
    plots = village.plots[village.plots.geometry.area > 0].copy()
    med = plots["map_area_sqm"].median()
    plots["_d"] = (plots["map_area_sqm"] - med).abs()
    sample = plots.sort_values("_d").iloc[:GLOBAL_SAMPLE]

    dxs, dys = [], []
    for _, row in sample.iterrows():
        geom = row.geometry
        try:
            sig, wt, res = blended_signal(geom, bsrc, isrc, PAD_M)
            if sig is None or sig.mean() < 1e-3:
                continue
            mask = polygon_edge_mask(geom, wt, sig.shape, str(isrc.crs))
            if mask.sum() < 5:
                continue
            dx, dy, sharpness, improved = fft_best_shift(sig, mask, res, GLOBAL_RADIUS_M)
            if improved:
                dxs.append(dx)
                dys.append(dy)
        except Exception:
            continue

    if not dxs:
        print("  [global] no signal — (0, 0)")
        return 0.0, 0.0

    gdx, gdy = float(np.median(dxs)), float(np.median(dys))
    print(f"  [global] {len(dxs)}/{len(sample)} plots → dx={gdx:.1f}m dy={gdy:.1f}m")
    return gdx, gdy

# ── Neighbour smoothing ───────────────────────────────────────────────────────

def smooth_shifts(pns, centroids, dx_arr, dy_arr, conf_arr):
    tree = cKDTree(centroids)
    sdx, sdy, sconf = dx_arr.copy(), dy_arr.copy(), conf_arr.copy()

    for i in range(len(pns)):
        dists, idxs = tree.query(centroids[i], k=NEIGHBOR_K+1,
                                 distance_upper_bound=NEIGHBOR_RADIUS_M)
        nbrs = [(d,j) for d,j in zip(dists[1:], idxs[1:])
                if d < NEIGHBOR_RADIUS_M and j < len(dx_arr)]
        if not nbrs:
            continue
        js = [j for _,j in nbrs]
        ws = conf_arr[js]
        if ws.sum() < 1e-6:
            continue
        ndx = float(np.average(dx_arr[js], weights=ws))
        ndy = float(np.average(dy_arr[js], weights=ws))
        dist_from_nbrs = np.hypot(dx_arr[i]-ndx, dy_arr[i]-ndy)

        if dist_from_nbrs > 20:   # outlier
            sdx[i]   = 0.25*dx_arr[i] + 0.75*ndx
            sdy[i]   = 0.25*dy_arr[i] + 0.75*ndy
            sconf[i] = conf_arr[i] * 0.4
        else:                      # consistent
            sdx[i]   = 0.5*dx_arr[i] + 0.5*ndx
            sdy[i]   = 0.5*dy_arr[i] + 0.5*ndy
            sconf[i] = min(0.95, conf_arr[i] * 1.15)

    return sdx, sdy, sconf

# ── Main ──────────────────────────────────────────────────────────────────────

def solve(village_dir: str):
    print(f"\n{'='*60}\n{village_dir}")
    village = load(village_dir)
    plots   = village.plots
    n       = len(plots)
    utmc    = utm_crs(plots.geometry.iloc[0])
    print(f"  {n} plots · utm={utmc}")

    with rasterio.open(village.imagery_path) as isrc:
        bsrc = rasterio.open(village.boundaries_path) if village.boundaries_path else None
        try:

            # 1. Global offset
            print("\n[1] Global offset...")
            gdx, gdy = estimate_global(village, bsrc, isrc, utmc)

            # 2. Per-plot local refinement
            print("\n[2] Per-plot alignment...")
            pns, dxs, dys, confs = [], [], [], []
            flags = {}

            for i, (pn, row) in enumerate(plots.iterrows()):
                if i % 400 == 0:
                    print(f"  {i}/{n}")
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue

                flagged, reason = area_flag(row)
                if flagged:
                    flags[str(pn)] = reason
                    continue

                try:
                    # Apply global shift first, then search locally around that position
                    geom_shifted = shift_geom(geom, gdx, gdy, utmc)

                    sig, wt, res = blended_signal(geom_shifted, bsrc, isrc, PAD_M)
                    if sig is None:
                        pns.append(str(pn)); dxs.append(gdx); dys.append(gdy); confs.append(0.15)
                        continue

                    mask = polygon_edge_mask(geom_shifted, wt, sig.shape, str(isrc.crs))
                    if mask.sum() < 3:
                        pns.append(str(pn)); dxs.append(gdx); dys.append(gdy); confs.append(0.15)
                        continue

                    # Scale local radius to plot size
                    diam = float(np.sqrt(float(row.get("map_area_sqm") or 5000)))
                    local_r = float(np.clip(diam * 0.25, 5.0, LOCAL_RADIUS_M))

                    ldx, ldy, sharpness, improved = fft_best_shift(sig, mask, res, local_r)

                    total_dx = gdx + ldx
                    total_dy = gdy + ldy
                    conf = confidence(sharpness, improved, float(sig.mean()))

                except Exception as e:
                    total_dx, total_dy, conf = gdx, gdy, 0.15

                pns.append(str(pn))
                dxs.append(total_dx)
                dys.append(total_dy)
                confs.append(conf)

            print(f"  conf spread: min={min(confs):.3f} max={max(confs):.3f} std={np.std(confs):.3f}")
            # 3. Neighbour smoothing
            print("\n[3] Neighbour smoothing...")
            utm_plots  = plots.to_crs(utmc)
            centroids  = np.array([
                [utm_plots.loc[p, "geometry"].centroid.x,
                 utm_plots.loc[p, "geometry"].centroid.y]
                for p in pns
            ])
            dx_arr   = np.array(dxs)
            dy_arr   = np.array(dys)
            conf_arr = np.array(confs)
            dx_s, dy_s, conf_s = smooth_shifts(pns, centroids, dx_arr, dy_arr, conf_arr)

            # 4. Build records
            records = []
            for pn, dx, dy, conf in zip(pns, dx_s, dy_s, conf_s):
                geom = plots.loc[pn, "geometry"]
                try:
                    new_geom = shift_geom(geom, float(dx), float(dy), utmc)
                except Exception:
                    new_geom = geom; conf = 0.1
                records.append({
                    "plot_number": pn,
                    "status":      "corrected",
                    "confidence":  round(float(np.clip(conf, 0.0, 1.0)), 3),
                    "method_note": f"dx={dx:.1f}m dy={dy:.1f}m",
                    "geometry":    new_geom,
                })
            for pn, reason in flags.items():
                records.append({
                    "plot_number": pn,
                    "status":      "flagged",
                    "confidence":  None,
                    "method_note": f"area: {reason}",
                    "geometry":    plots.loc[pn, "geometry"],
                })

        finally:
            if bsrc: bsrc.close()

    print(f"\n  {sum(r['status']=='corrected' for r in records)} corrected · "
          f"{sum(r['status']=='flagged' for r in records)} flagged")

    preds = gpd.GeoDataFrame(records, crs="EPSG:4326")
    out   = Path(village_dir) / "predictions.geojson"
    write_predictions(out, preds)
    print(f"[5] Written → {out}")

    if village.example_truths is not None:
        print("\n[6] Self-score:")
        print(score(preds, village))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run solve.py data/<village_folder>")
        sys.exit(1)
    solve(sys.argv[1])