"""
BhuMe Alignment Pipeline — Phase 1
====================================
Track 1: Geometry Sanity Check (area discrepancy ratio).

For every plot in a village:
  1. Compute the polygon's geodesic area in UTM (true metres, not EPSG:4326 degrees).
  2. Compare against the TOTAL recorded area = recorded_area_sqm + pot_kharaba_ha * 10_000.
  3. If the ratio is outside [AREA_RATIO_MIN, AREA_RATIO_MAX], flag the plot immediately
     as "area_error" — do not attempt a spatial correction.
  4. Plots with no recorded area data are flagged as "no_record".
  5. Everything else passes the sanity check and is flagged as "pending_alignment"
     (ready for Track 2 in Phase 2).

Output: predictions.geojson (EPSG:4326, contract-valid via write_predictions).

Usage:
    python pipeline.py data/<village_slug>/
    python pipeline.py data/<village_slug>/ --area-min 0.7 --area-max 1.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.geometry import mapping
from shapely.ops import transform as shp_transform

# ── bring in the starter kit (assumes pipeline.py lives next to the bhume/ package) ──
sys.path.insert(0, str(Path(__file__).parent))
from bhume import load, write_predictions


# ---------------------------------------------------------------------------
# Configuration — thresholds kept as named constants so they're easy to tune
# ---------------------------------------------------------------------------

# Flag if computed_area / total_recorded_area falls outside this window.
# 0.75–1.35 gives ±25-35% tolerance; generous enough for surveying drift
# but tight enough to catch genuinely wrong shapes.
DEFAULT_AREA_RATIO_MIN: float = 0.75
DEFAULT_AREA_RATIO_MAX: float = 1.35

# Confidence assigned to plots that *pass* Track 1 but await Track 2.
# We set this low because Track 1 passing alone tells us nothing about
# spatial alignment quality; Track 2 will overwrite this.
PASS_CONFIDENCE_PLACEHOLDER: float = 0.0


# ---------------------------------------------------------------------------
# Geodesic area helper
# ---------------------------------------------------------------------------

def _utm_epsg_for_geom(geom) -> str:
    """Return the best UTM EPSG string for a geometry in EPSG:4326."""
    lon = geom.centroid.x
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + zone}"


def compute_geodesic_area_sqm(geom_4326) -> float:
    """
    Reproject a Shapely geometry from EPSG:4326 to its local UTM zone and
    return the area in square metres.

    Why UTM and not shapely.geometry.area on EPSG:4326?
    Because EPSG:4326 area is in *degrees squared* — meaningless for comparison
    against recorded areas in m².  A single UTM zone gives sub-percent accuracy
    for field-sized polygons anywhere in Maharashtra.
    """
    epsg = _utm_epsg_for_geom(geom_4326)
    transformer = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    geom_utm = shp_transform(
        lambda xs, ys, zs=None: transformer.transform(xs, ys),
        geom_4326,
    )
    return float(geom_utm.area)


# ---------------------------------------------------------------------------
# Track 1: Geometry Sanity Check
# ---------------------------------------------------------------------------

def total_recorded_area_sqm(row) -> float | None:
    """
    Compute the TOTAL recorded area for a plot row:
        total = cultivable (recorded_area_sqm) + uncultivable (pot_kharaba_ha × 10_000)

    Returns None if both fields are null/missing — the plot has no reference area.

    Important nuance from the CONTRACT:
        "The parcel's full recorded extent ≈ recorded_area + pot_kharaba,
         so compare your geometry against that TOTAL, not the cultivable figure alone."
    We follow this exactly.
    """
    cultivable = row.get("recorded_area_sqm")
    pot_kharaba_ha = row.get("pot_kharaba_ha")

    cultivable_sqm = float(cultivable) if cultivable is not None and not _is_nan(cultivable) else 0.0
    pot_kharaba_sqm = float(pot_kharaba_ha) * 10_000 if pot_kharaba_ha is not None and not _is_nan(pot_kharaba_ha) else 0.0

    total = cultivable_sqm + pot_kharaba_sqm
    return total if total > 0 else None


def _is_nan(v) -> bool:
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return False


def run_track1(
    plots: gpd.GeoDataFrame,
    area_ratio_min: float = DEFAULT_AREA_RATIO_MIN,
    area_ratio_max: float = DEFAULT_AREA_RATIO_MAX,
) -> gpd.GeoDataFrame:
    """
    Run Track 1 (Geometry Sanity Check) over every plot in the GeoDataFrame.

    Returns a new GeoDataFrame in the predictions contract format:
        plot_number | status | confidence | method_note | geometry

    Status values assigned here:
        "flagged"  — definitively skipped (area_error or no_record)
        "flagged"  — passed Track 1 but needs Track 2  (method_note="pending_alignment")

    Track 2 (Phase 2) will iterate over rows where method_note == "pending_alignment"
    and overwrite status/confidence/geometry/method_note in place.
    """
    records = []

    for plot_number, row in plots.iterrows():
        geom = row.geometry

        # ── Guard: degenerate geometry ──────────────────────────────────────
        if geom is None or geom.is_empty or not geom.is_valid:
            records.append(_flag(
                plot_number=plot_number,
                geom=geom,
                note="flagged: null or invalid geometry",
            ))
            continue

        # ── Step 1: Compute geodesic area of the drawn polygon ──────────────
        try:
            computed_area = compute_geodesic_area_sqm(geom)
        except Exception as exc:
            records.append(_flag(
                plot_number=plot_number,
                geom=geom,
                note=f"flagged: area projection failed ({exc})",
            ))
            continue

        # ── Step 2: Retrieve total recorded area ────────────────────────────
        recorded_total = total_recorded_area_sqm(row)

        if recorded_total is None:
            # No reference area at all — we cannot validate anything.
            records.append(_flag(
                plot_number=plot_number,
                geom=geom,
                note="flagged: no_record — recorded_area_sqm and pot_kharaba_ha both null/zero",
            ))
            continue

        # ── Step 3: Compute ratio and apply threshold ───────────────────────
        ratio = computed_area / recorded_total

        if ratio < area_ratio_min or ratio > area_ratio_max:
            # Shape is fundamentally inconsistent with the written record.
            # Moving it would be meaningless — flag immediately.
            records.append(_flag(
                plot_number=plot_number,
                geom=geom,
                note=(
                    f"flagged: area_error — "
                    f"computed={computed_area:.1f} m², "
                    f"recorded={recorded_total:.1f} m², "
                    f"ratio={ratio:.3f} (outside [{area_ratio_min}, {area_ratio_max}])"
                ),
            ))
            continue

        # ── Step 4: Passed sanity check — hand off to Track 2 ───────────────
        records.append({
            "plot_number": str(plot_number),
            "status": "flagged",           # conservative default; Track 2 upgrades to "corrected"
            "confidence": PASS_CONFIDENCE_PLACEHOLDER,
            "method_note": (
                f"pending_alignment — "
                f"area check passed: computed={computed_area:.1f} m², "
                f"recorded={recorded_total:.1f} m², "
                f"ratio={ratio:.3f}"
            ),
            "geometry": geom,
        })

    result = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return result


def _flag(plot_number: str, geom, note: str) -> dict:
    """Helper: build a flagged record. Geometry is the original (contract rule)."""
    return {
        "plot_number": str(plot_number),
        "status": "flagged",
        "confidence": 0.0,
        "method_note": note,
        "geometry": geom,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _print_summary(preds: gpd.GeoDataFrame, village_slug: str) -> None:
    total = len(preds)
    flagged = preds[preds["status"] == "flagged"]
    pending = flagged[flagged["method_note"].str.startswith("pending_alignment")]
    area_errors = flagged[flagged["method_note"].str.contains("area_error")]
    no_record = flagged[flagged["method_note"].str.contains("no_record")]
    geom_errors = flagged[
        flagged["method_note"].str.contains("null or invalid|projection failed")
    ]

    print(f"\n{'='*60}")
    print(f"  Village : {village_slug}")
    print(f"  Total plots evaluated : {total}")
    print(f"{'='*60}")
    print(f"  ✓ Passed Track 1 (→ Track 2)  : {len(pending):>5}  ({100*len(pending)/total:.1f}%)")
    print(f"  ✗ Flagged — area_error         : {len(area_errors):>5}  ({100*len(area_errors)/total:.1f}%)")
    print(f"  ✗ Flagged — no_record          : {len(no_record):>5}  ({100*len(no_record)/total:.1f}%)")
    print(f"  ✗ Flagged — geometry error     : {len(geom_errors):>5}  ({100*len(geom_errors)/total:.1f}%)")
    print(f"{'='*60}\n")

    if len(area_errors) > 0:
        # Show the 5 most extreme ratio violations to aid threshold tuning
        area_err_rows = []
        for _, r in area_errors.iterrows():
            # Extract ratio from the note string
            note = r["method_note"]
            try:
                ratio_str = [p for p in note.split() if p.startswith("ratio=")][0]
                ratio_val = float(ratio_str.split("=")[1].rstrip(")"))
            except (IndexError, ValueError):
                ratio_val = float("nan")
            area_err_rows.append((r["plot_number"], ratio_val))

        area_err_rows.sort(key=lambda x: abs(x[1] - 1.0), reverse=True)
        print("  Top area-error plots (by deviation from ratio=1.0):")
        for pn, rv in area_err_rows[:5]:
            print(f"    plot {pn:>10}  ratio={rv:.3f}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(village_dir: str, area_ratio_min: float, area_ratio_max: float) -> None:
    village_path = Path(village_dir)

    print(f"Loading village from: {village_path}")
    village = load(village_path)

    print(
        f"Loaded {village.slug}: "
        f"{len(village.plots)} plots  |  "
        f"boundaries={'yes' if village.boundaries_path else 'none'}  |  "
        f"example_truths={0 if village.example_truths is None else len(village.example_truths)}"
    )

    print(f"\nRunning Track 1 — Geometry Sanity Check "
          f"(ratio window: [{area_ratio_min}, {area_ratio_max}]) ...")

    predictions = run_track1(
        village.plots,
        area_ratio_min=area_ratio_min,
        area_ratio_max=area_ratio_max,
    )

    _print_summary(predictions, village.slug)

    out_path = village_path / "predictions.geojson"
    write_predictions(out_path, predictions)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BhuMe pipeline Phase 1 — Track 1 geometry sanity check."
    )
    parser.add_argument(
        "village_dir",
        help="Path to the village bundle directory (must contain input.geojson + imagery.tif)",
    )
    parser.add_argument(
        "--area-min",
        type=float,
        default=DEFAULT_AREA_RATIO_MIN,
        help=f"Lower bound for area ratio (default: {DEFAULT_AREA_RATIO_MIN})",
    )
    parser.add_argument(
        "--area-max",
        type=float,
        default=DEFAULT_AREA_RATIO_MAX,
        help=f"Upper bound for area ratio (default: {DEFAULT_AREA_RATIO_MAX})",
    )
    args = parser.parse_args()

    main(args.village_dir, args.area_min, args.area_max)
