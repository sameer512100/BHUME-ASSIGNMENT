# BhuMe — Land Boundary Correction

## The Problem

India's cadastral maps were drawn by hand decades ago and later scanned and digitised. When georeferenced onto modern satellite imagery, the fit introduces errors: plot boundaries end up shifted off the real fields they describe.

The task: for each plot, decide if the official boundary can be nudged onto the real field, and if so, where it should go.

---

## Method

### Step 1 — Global village offset

Most of the error is a single coherent shift — the whole map slid a few metres in one direction during georeferencing.

To find it:
- Sample 80 plots near the **median plot size** (avoids bias toward large or small plots)
- For each, blend boundary hints + imagery Sobel edges into a signal map
- Rasterise the official polygon boundary as a thin ring mask
- FFT cross-correlate the ring mask against the signal — the peak location is where the polygon best fits the real field edges
- Take the **median** of all 80 estimates → single `(dx, dy)` in metres for the whole village

### Step 2 — Per-plot local refinement

After applying the global shift, each plot gets its own local FFT search:
- Shift the polygon by `(gdx, gdy)` first
- Extract a fresh patch centred on the shifted polygon
- Run FFT cross-correlation with a search radius scaled to plot size (small plots → tighter search)
- Local peak gives a residual correction on top of the global shift

**Total shift = global + local**

### Step 3 — Neighbour consistency

Adjacent plots share bunds and should shift by similar amounts:
- Find each plot's 8 nearest neighbours within 300m
- If a plot's shift diverges >20m from its neighbours → outlier, pull 75% toward neighbourhood median, cut confidence
- If consistent → small confidence boost

### Step 4 — Confidence

Two signals:
- **Peak sharpness** = `peak_value / mean_of_search_region` — sharp isolated peak = polygon snapped cleanly to an edge
- **Area ratio proximity** = how close `drawn_area / recorded_area` is to 1.0 — ratio near 1.0 means the error is likely pure translation (trustworthy)

```
confidence = 0.7 × sharpness + 0.3 × area_ratio_proximity
```

### Step 5 — Flagging

If `drawn_area / (recorded_area + pot_kharaba)` is outside `[0.5, 2.0]`, the plot is flagged. Moving a geometrically wrong boundary won't help — the shape itself disagrees with the record.

---

## Signal blending

`boundaries.tif` is rough — strong on open land, unreliable under trees. Rather than switching to imagery-only when hints fail, we always blend:

```
signal = 0.6 × boundary_hints + 0.4 × imagery_sobel_edges
```

---

## Results (self-scored on example truths)

| Village | Official IoU | Predicted IoU | Centroid Error |
|---|---|---|---|
| Vadnerbhairav (2,457 plots) | 0.612 | 0.820 | 3.9m |
| Malatavadi (2,508 plots) | 0.510 | 0.596 | 8.8m |

---

## Running

```bash
uv sync
uv run solve.py data/34855_vadnerbhairav_chandavad_nashik
uv run solve.py data/malatavadi_folder
# writes predictions.geojson into each village folder
```

---

## AI Usage

Built with Claude (Anthropic). Transcripts in `/transcripts`.
