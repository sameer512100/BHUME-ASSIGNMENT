#!/usr/bin/env python3
"""
BhuMe boundary correction.

Method:
  1. Global offset  — FFT cross-correlation across sampled plots → village-wide (dx, dy)
  2. Per-plot       — local FFT cross-correlation around global estimate → refined (dx, dy)
  3. Signal         — blended: 0.6 * boundary hints + 0.4 * imagery edges (robust to noisy hints)
  4. Neighbor smooth — spatial median of nearby shifts weighted by confidence (catches outliers)
  5. Confidence     — absolute match quality × relative improvement (calibrated, not flat)
  6. Flag           — area ratio outside [0.5, 2.0] → geometry wrong, translation won't help

Run:
    uv run solve.py data/34855_vadnerbhairav_chandavad_nashik
    uv run solve.py data/malatavadi_folder
"""

from __future__ import annotations

import sys
import warnings
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

from bhume import load, write_predictions, score
from bhume.geo import open_imagery

warnings.filterwarnings("ignore")

# ── Tuning ────────────────────────────────────────────────────────────────────
GLOBAL_SAMPLE     = 80       # plots to sample for global offset
SEARCH_RADIUS_M   = 40.0     # local FFT search radius (metres)
PAD_M             = 45.0     # patch padding around plot
HINT_WEIGHT       = 0.6      # boundary hint blend weight (rest = imagery edges)
AREA_RATIO_LO     = 0.5      # flag if drawn/recorded < this
AREA_RATIO_HI     = 2.0      # flag if drawn/recorded > this
NEIGHBOR_K        = 8        # neighbours for consistency smoothing
NEIGHBOR_RADIUS_M = 300.0    # max radius for neighbour search (metres)

# ── CRS helpers ───────────────────────────────────────────────────────────────

def _tf(src_crs, dst_crs):
    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)

def to_crs(geom: BaseGeometry, src, dst) -> BaseGeometry:
    t = _tf(src, dst)
    return shp_transform(lambda xs, ys, z=None: t.transform(xs, ys), geom)

def utm(geom: BaseGeometry) -> str:
    lon = geom.centroid.x
    return f"EPSG:{32600 + int((lon + 180) // 6) + 1}"

# ── Raster patch ──────────────────────────────────────────────────────────────

def _patch(src, geom_4326: BaseGeometry, pad: float, bands=None):
    """Return (array C×H×W, window_transform, resolution_m) or (None, None, None)."""
    g = to_crs(geom_4326, "EPSG:4326", str(src.crs))
    l, b, r, t = g.bounds
    l, b, r, t = l - pad, b - pad, r + pad, t + pad
    dl, db, dr, dt = src.bounds
    l, b, r, t = max(l, dl), max(b, db), min(r, dr), min(t, dt)
    if r <= l or t <= b:
        return None, None, None
    w = from_bounds(l, b, r, t, transform=src.transform)
    bds = bands or list(range(1, src.count + 1))
    arr = src.read(bds, window=w)
    return arr, src.window_transform(w), src.res[0]

# ── Signal maps ───────────────────────────────────────────────────────────────

def _img_edges(arr) -> np.ndarray:
    """Sobel edge map from RGB array (3×H×W) → float32 (H×W) in [0,1]."""
    gray = arr.mean(axis=0).astype(np.float32)
    gray = gaussian_filter(gray, sigma=1.0)
    e = np.hypot(sobel(gray, axis=1), sobel(gray, axis=0))
    mx = e.max()
    return (e / mx).astype(np.float32) if mx > 0 else e

def _hint_signal(arr) -> np.ndarray:
    """Normalise single-band boundary hint → float32 (H×W) in [0,1]."""
    h = arr[0].astype(np.float32)
    mx = h.max()
    return (h / mx) if mx > 0 else h

def _blended_signal(geom_4326, bsrc, isrc, pad) -> tuple[np.ndarray | None, object, float]:
    """
    Extract blended signal: HINT_WEIGHT * hints + (1-HINT_WEIGHT) * imagery_edges.
    Falls back to imagery-only if hints unavailable or empty.
    Returns (signal H×W, window_transform, res_m).
    """
    # imagery edges (always available)
    iarr, iwt, ires = _patch(isrc, geom_4326, pad)
    if iarr is None:
        return None, None, 1.0
    img_e = _img_edges(iarr)

    if bsrc is not None:
        barr, bwt, bres = _patch(bsrc, geom_4326, pad, bands=[1])
        if barr is not None and barr[0].max() > 0:
            # resize hint to match imagery patch if needed
            from PIL import Image as PILImage
            hint = _hint_signal(barr)
            if hint.shape != img_e.shape:
                hint = np.array(
                    PILImage.fromarray(hint).resize(
                        (img_e.shape[1], img_e.shape[0]), PILImage.BILINEAR
                    )
                )
            signal = HINT_WEIGHT * hint + (1 - HINT_WEIGHT) * img_e
        else:
            signal = img_e
    else:
        signal = img_e

    return signal.astype(np.float32), iwt, ires

# ── Polygon edge mask ─────────────────────────────────────────────────────────

def _edge_mask(geom_4326: BaseGeometry, wt, shape_hw, crs: str) -> np.ndarray:
    """Binary mask of the polygon boundary ring (thin border)."""
    g = to_crs(geom_4326, "EPSG:4326", crs)
    ring = g.buffer(3).difference(g.buffer(-3))
    mask = rasterize(
        [(mapping(ring), 1)],
        out_shape=shape_hw,
        transform=wt,
        fill=0,
        dtype=np.uint8,
    ).astype(np.float32)
    return mask

# ── FFT alignment ─────────────────────────────────────────────────────────────

def _fft_offset(signal: np.ndarray, mask: np.ndarray, res_m: float,
                radius_m: float) -> tuple[float, float, float, float, float]:
    """
    Cross-correlate mask against signal via FFT.
    Returns (dx_m, dy_m, best_score, baseline_score, peak_sharpness).
    dx positive = shift right, dy positive = shift up (in map coords).
    peak_sharpness = peak / mean_of_search_region (how isolated the peak is).
    """
    corr = fftconvolve(signal, mask[::-1, ::-1], mode="same")
    H, W = corr.shape
    cy, cx = H // 2, W // 2

    # restrict to search radius
    r_px = max(1, int(radius_m / res_m))
    r0, r1 = max(0, cy - r_px), min(H, cy + r_px)
    c0, c1 = max(0, cx - r_px), min(W, cx + r_px)
    sub = corr[r0:r1, c0:c1]

    peak = np.unravel_index(sub.argmax(), sub.shape)
    dy_px = peak[0] + r0 - cy   # row offset (positive = down in image = south)
    dx_px = peak[1] + c0 - cx   # col offset (positive = right = east)

    best_score = float(sub.max())
    baseline_score = float(corr[cy, cx])
    sub_mean = float(sub.mean()) if sub.size > 0 else 1.0
    peak_sharpness = best_score / (sub_mean + 1e-6)

    norm = float(mask.sum())
    if norm > 0:
        best_score /= norm
        baseline_score /= norm

    return dx_px * res_m, -dy_px * res_m, best_score, baseline_score, peak_sharpness
    # note: image row↓ = map south, so negate dy

# ── Confidence ────────────────────────────────────────────────────────────────

def _confidence(best_score: float, baseline_score: float,
                signal_mean: float, peak_sharpness: float) -> float:
    """
    Peak sharpness is the primary signal: a sharp isolated correlation peak means
    the polygon clearly snapped to one edge, not a noisy smear.
    We also require the peak to beat the no-shift baseline.
    """
    if signal_mean < 1e-4:
        return 0.15

    # sharpness: how much the peak stands above the search-region mean
    # sharpness=1 → peak == mean (flat, no clear snap) → low confidence
    # sharpness=5+ → clear isolated peak → high confidence
    sharpness_conf = np.clip((peak_sharpness - 1.0) / 6.0, 0.0, 1.0)

    # improvement: peak must beat no-shift baseline
    improved = 1.0 if best_score > baseline_score else 0.0

    conf = sharpness_conf * improved
    return float(np.clip(conf + 0.1, 0.1, 0.95))  # floor 0.1

# ── Neighbour consistency ─────────────────────────────────────────────────────

def _smooth_shifts(
    plot_numbers: list[str],
    centroids_utm: np.ndarray,   # (N, 2) in UTM metres
    dx_arr: np.ndarray,          # (N,)
    dy_arr: np.ndarray,          # (N,)
    conf_arr: np.ndarray,        # (N,)
    k: int,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each plot, compute a weighted median of its K nearest neighbours' shifts.
    Weight = neighbour confidence. Blend: 0.5 * own + 0.5 * neighbour_median.
    Returns (smoothed_dx, smoothed_dy, smoothed_conf).
    """
    tree = cKDTree(centroids_utm)
    sdx = dx_arr.copy()
    sdy = dy_arr.copy()
    sconf = conf_arr.copy()

    for i in range(len(plot_numbers)):
        dists, idxs = tree.query(centroids_utm[i], k=k + 1, distance_upper_bound=radius_m)
        # exclude self (index 0 is always self)
        valid = [(d, j) for d, j in zip(dists[1:], idxs[1:]) if d < radius_m and j < len(dx_arr)]
        if not valid:
            continue
        js = [j for _, j in valid]
        ws = conf_arr[js]
        if ws.sum() < 1e-6:
            continue
        nbr_dx = float(np.average(dx_arr[js], weights=ws))
        nbr_dy = float(np.average(dy_arr[js], weights=ws))

        # detect outlier: if own shift is far from neighbours, reduce confidence
        own_dist = np.hypot(dx_arr[i] - nbr_dx, dy_arr[i] - nbr_dy)
        if own_dist > 20:  # metres
            # trust neighbours more, own shift is suspicious
            sdx[i] = 0.3 * dx_arr[i] + 0.7 * nbr_dx
            sdy[i] = 0.3 * dy_arr[i] + 0.7 * nbr_dy
            sconf[i] = conf_arr[i] * 0.5
        else:
            sdx[i] = 0.5 * dx_arr[i] + 0.5 * nbr_dx
            sdy[i] = 0.5 * dy_arr[i] + 0.5 * nbr_dy
            # consistent with neighbours → boost confidence slightly
            sconf[i] = min(0.95, conf_arr[i] * 1.1)

    return sdx, sdy, sconf

# ── Area ratio flag ───────────────────────────────────────────────────────────

def _area_flag(row) -> tuple[bool, str]:
    drawn = row.get("map_area_sqm")
    recorded = row.get("recorded_area_sqm")
    pot = row.get("pot_kharaba_ha")
    if not recorded or float(recorded) <= 0:
        return False, ""
    total = float(recorded)
    if pot and not np.isnan(float(pot)):
        total += float(pot) * 10_000
    if not drawn or float(drawn) <= 0:
        return False, ""
    ratio = float(drawn) / total
    if ratio < AREA_RATIO_LO:
        return True, f"drawn/recorded={ratio:.2f} (too small)"
    if ratio > AREA_RATIO_HI:
        return True, f"drawn/recorded={ratio:.2f} (too large)"
    return False, ""

# ── Apply shift ───────────────────────────────────────────────────────────────

def _apply(geom_4326: BaseGeometry, dx_m: float, dy_m: float, utm_crs: str) -> BaseGeometry:
    g = to_crs(geom_4326, "EPSG:4326", utm_crs)
    return to_crs(translate(g, dx_m, dy_m), utm_crs, "EPSG:4326")

# ── Global offset ─────────────────────────────────────────────────────────────

def global_offset(village, bsrc, isrc) -> tuple[float, float]:
    plots = village.plots
    sample = plots[plots.geometry.area > 0].copy()
    # Use plots near the median size — avoids large-plot bias that breaks dense villages
    med = sample["map_area_sqm"].median()
    sample["_size_dist"] = (sample["map_area_sqm"] - med).abs()
    sample = sample.sort_values("_size_dist").iloc[:GLOBAL_SAMPLE]

    utm_crs = utm(sample.geometry.iloc[0])
    dxs, dys = [], []

    for pn, row in sample.iterrows():
        geom = row.geometry
        try:
            signal, wt, res = _blended_signal(geom, bsrc, isrc, PAD_M)
            if signal is None or signal.mean() < 1e-3:
                continue
            mask = _edge_mask(geom, wt, signal.shape, str(isrc.crs))
            if mask.sum() < 5:
                continue
            dx, dy, bs, bl = _fft_offset(signal, mask, res, SEARCH_RADIUS_M)
            if bs > bl:   # only count if we actually improved
                dxs.append(dx)
                dys.append(dy)
        except Exception:
            continue

    if not dxs:
        print("  [global] no signal — using (0, 0)")
        return 0.0, 0.0

    gdx, gdy = float(np.median(dxs)), float(np.median(dys))
    print(f"  [global] {len(dxs)} plots → dx={gdx:.1f}m dy={gdy:.1f}m")
    return gdx, gdy

# ── Main ──────────────────────────────────────────────────────────────────────

def solve(village_dir: str) -> None:
    print(f"\n{'='*60}\nVillage: {village_dir}")
    village = load(village_dir)
    plots = village.plots
    n = len(plots)
    print(f"  {n} plots · "
          f"{len(village.example_truths) if village.example_truths is not None else 0} example truths")

    utm_crs = utm(plots.geometry.iloc[0])

    with rasterio.open(village.imagery_path) as isrc:
        bsrc_obj = rasterio.open(village.boundaries_path) if village.boundaries_path else None
        try:
            # ── 1. Global offset ──────────────────────────────────────────
            print("\n[1] Global offset...")
            gdx, gdy = global_offset(village, bsrc_obj, isrc)

            # ── 2. Per-plot FFT alignment ─────────────────────────────────
            print("\n[2] Per-plot alignment...")
            pn_list, dx_list, dy_list, conf_list = [], [], [], []
            flag_list = {}   # pn → reason

            for i, (pn, row) in enumerate(plots.iterrows()):
                if i % 300 == 0:
                    print(f"  {i}/{n}...")
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue

                flagged, reason = _area_flag(row)
                if flagged:
                    flag_list[str(pn)] = reason
                    continue

                try:
                    signal, wt, res = _blended_signal(geom, bsrc_obj, isrc, PAD_M)
                    if signal is None:
                        pn_list.append(str(pn)); dx_list.append(gdx); dy_list.append(gdy); conf_list.append(0.2)
                        continue

                    mask = _edge_mask(geom, wt, signal.shape, str(isrc.crs))
                    if mask.sum() < 3:
                        pn_list.append(str(pn)); dx_list.append(gdx); dy_list.append(gdy); conf_list.append(0.2)
                        continue

                    # local FFT around global estimate
                    # shift signal by negative global to centre search
                    from scipy.ndimage import shift as nd_shift
                    res_m = res
                    gdx_px = gdx / res_m
                    gdy_px = gdy / res_m
                    shifted_signal = nd_shift(signal, shift=[gdy_px, -gdx_px], order=1)

                    # scale search radius to plot size: small plots → tighter search
                    plot_diam = float(np.sqrt(row.get("map_area_sqm") or 5000))
                    local_radius = float(np.clip(plot_diam * 0.3, 5.0, SEARCH_RADIUS_M / 2))

                    dx_local, dy_local, bs, bl = _fft_offset(
                        shifted_signal, mask, res_m, local_radius
                    )
                    total_dx = gdx + dx_local
                    total_dy = gdy + dy_local
                    conf = _confidence(bs, bl, float(signal.mean()))

                except Exception:
                    total_dx, total_dy, conf = gdx, gdy, 0.2

                pn_list.append(str(pn))
                dx_list.append(total_dx)
                dy_list.append(total_dy)
                conf_list.append(conf)

            # ── 3. Neighbour consistency ──────────────────────────────────
            print("\n[3] Neighbour smoothing...")
            utm_plots = plots.to_crs(utm_crs)
            centroids = np.array([
                [utm_plots.loc[pn, "geometry"].centroid.x,
                 utm_plots.loc[pn, "geometry"].centroid.y]
                for pn in pn_list
            ])
            dx_arr = np.array(dx_list)
            dy_arr = np.array(dy_list)
            conf_arr = np.array(conf_list)

            dx_s, dy_s, conf_s = _smooth_shifts(
                pn_list, centroids, dx_arr, dy_arr, conf_arr,
                k=NEIGHBOR_K, radius_m=NEIGHBOR_RADIUS_M
            )

            # ── 4. Build predictions ──────────────────────────────────────
            print("\n[4] Building predictions...")
            records = []

            for pn, dx, dy, conf in zip(pn_list, dx_s, dy_s, conf_s):
                geom = plots.loc[pn, "geometry"]
                try:
                    new_geom = _apply(geom, float(dx), float(dy), utm_crs)
                except Exception:
                    new_geom = geom
                    conf = 0.1
                records.append({
                    "plot_number": pn,
                    "status": "corrected",
                    "confidence": round(float(np.clip(conf, 0.0, 1.0)), 3),
                    "method_note": f"dx={dx:.1f}m dy={dy:.1f}m",
                    "geometry": new_geom,
                })

            for pn, reason in flag_list.items():
                records.append({
                    "plot_number": pn,
                    "status": "flagged",
                    "confidence": None,
                    "method_note": f"area mismatch: {reason}",
                    "geometry": plots.loc[pn, "geometry"],
                })

        finally:
            if bsrc_obj is not None:
                bsrc_obj.close()

    n_corr = sum(r["status"] == "corrected" for r in records)
    n_flag = sum(r["status"] == "flagged" for r in records)
    print(f"  {n_corr} corrected · {n_flag} flagged")

    preds = gpd.GeoDataFrame(records, crs="EPSG:4326")
    out = Path(village_dir) / "predictions.geojson"
    write_predictions(out, preds)
    print(f"\n[5] Written → {out}")

    if village.example_truths is not None:
        print("\n[6] Self-score:")
        print(score(preds, village))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run solve.py data/<village_folder>")
        sys.exit(1)
    solve(sys.argv[1])