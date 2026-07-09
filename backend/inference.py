"""Viewport scoring -> region overlay (Phase 6, current-view scope).

Two ways to turn the visible region into an overlay the frontend draws on
OpenSeadragon, both scoring only what's on screen so they stay responsive on
gigapixel slides:

  - predict()    : SUPERVISED. Runs the trained classifier head over each tissue
                   tile; outlines the regions confidently the chosen class
                   (P(class) >= PREDICT_MIN_CONFIDENCE). Needs a trained model.
  - similarity() : UNSUPERVISED. The user selects one annotation; outlines the
                   visible regions most similar to the annotation's mean Virchow 2
                   embedding. No training, no labels — a "find more like this
                   region" tool you can use before training.

Both return the confident regions as tile CELLS (rects, for a tinted fill) plus
their BOUNDARY segments (for the outline); the frontend draws one vector overlay.
Both reuse the same tiling + tissue mask + embedding (an in-memory, process-lived
tile cache SEPARATE from the Phase 5 training cache).
"""

from __future__ import annotations

import math
import time

import numpy as np
from PIL import Image

from backend import classifier, embeddings, patches, slides

# Upper bound on tiles per call. If the visible region grids to more than this
# (zoomed too far out), we ask the user to zoom in rather than hang.
MAX_TILES = 4000

# Predict outlines regions whose tiles score at least this probability for the
# chosen class; similarity outlines tiles at least this fraction of the way up the
# (min-max stretched) per-view similarity range. Fixed for now.
PREDICT_MIN_CONFIDENCE = 0.9
SIMILARITY_MIN = 0.9

# Session-lived embedding cache, keyed by (slide_id, model_id, col, row).
_tile_cache: dict[tuple, np.ndarray] = {}


# --- Shared geometry + tiling ------------------------------------------------

def _geometry(slide, region: dict):
    """Grid geometry for a DZI-image-pixel region snapped to the global tile grid.

    Returns (size0, level, off_x, off_y, col0, row0, cols, rows).
    """
    mpp0 = patches._slide_mpp(slide)
    size0 = max(1, round(patches.PATCH_SIZE * patches.TARGET_MPP / mpp0))
    level = slide.get_best_level_for_downsample(max(1.0, size0 / patches.PATCH_SIZE))
    off_x, off_y = patches._bounds_offset(slide)
    rx, ry = float(region.get("x", 0)), float(region.get("y", 0))
    rw, rh = float(region.get("w", 0)), float(region.get("h", 0))
    col0, row0 = math.floor(rx / size0), math.floor(ry / size0)
    col1, row1 = math.ceil((rx + rw) / size0), math.ceil((ry + rh) / size0)
    return size0, level, off_x, off_y, col0, row0, max(0, col1 - col0), max(0, row1 - row0)


def _collect_embeddings(slide, slide_id, model_id, geom):
    """Embed the tissue tiles of the region (cache-first). Returns {(r,c): vec}.

    Non-tissue / off-slide cells are simply absent. Misses are embedded in
    batches and added to the session cache. May raise EmbedderError.
    """
    size0, level, off_x, off_y, col0, row0, cols, rows = geom
    dim_x, dim_y = slide.dimensions
    embedder = embeddings.get_embedder()

    cell_vec: dict[tuple[int, int], np.ndarray] = {}
    to_imgs: list[Image.Image] = []
    to_cells: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            col, row = col0 + c, row0 + r
            x0, y0 = col * size0 + off_x, row * size0 + off_y
            if x0 < off_x or y0 < off_y or x0 + size0 > off_x + dim_x or y0 + size0 > off_y + dim_y:
                continue
            key = (slide_id, model_id, col, row)
            if key in _tile_cache:
                cell_vec[(r, c)] = _tile_cache[key]
                continue
            patch = patches.read_patch(slide, x0, y0, size0, level)
            if not patches._is_tissue(patch):
                continue
            to_imgs.append(patch)
            to_cells.append((r, c))

    for start in range(0, len(to_imgs), embedder.BATCH):
        chunk_imgs = to_imgs[start:start + embedder.BATCH]
        chunk_cells = to_cells[start:start + embedder.BATCH]
        for (r, c), vec in zip(chunk_cells, embeddings.embed_images(embedder, chunk_imgs)):
            cell_vec[(r, c)] = vec
            _tile_cache[(slide_id, model_id, col0 + c, row0 + r)] = vec
    return cell_vec


def _region_echo(size0, col0, row0, cols, rows) -> dict:
    """The snapped grid region in DZI-image pixels (where the frontend overlays)."""
    return {"x": col0 * size0, "y": row0 * size0, "w": cols * size0, "h": rows * size0}


# --- Supervised prediction ---------------------------------------------------

def predict(slide_id: str, region: dict, target_class: str | None = None) -> dict:
    """Score the region's tissue tiles with the head; outline the confident ones."""
    slide = slides.get_slide(slide_id)
    if slide is None:
        return {"status": "no_slide"}
    head = classifier.load_head()
    if head is None:
        return {"status": "no_model"}

    model_id = embeddings.active_model_id()
    geom = _geometry(slide, region)
    size0, _, _, _, col0, row0, cols, rows = geom
    if cols * rows == 0:
        return {"status": "empty_region"}
    if cols * rows > MAX_TILES:
        return {"status": "too_many_tiles", "requested": cols * rows, "limit": MAX_TILES}

    classes = list(head.classes_)
    target_class = target_class if target_class in classes else classes[0]
    class_col = classes.index(target_class)

    t0 = time.perf_counter()
    cell_vec = _collect_embeddings(slide, slide_id, model_id, geom)
    score_by_cell = {}
    if cell_vec:
        cells = list(cell_vec.keys())
        proba = head.predict_proba(np.vstack([cell_vec[c] for c in cells]))[:, class_col]
        score_by_cell = {cell: float(p) for cell, p in zip(cells, proba)}

    # Outline + tint the confident regions (frontend draws one vector overlay).
    shapes = _region_shapes(score_by_cell, size0, PREDICT_MIN_CONFIDENCE)

    return {
        "status": "ok", "mode": "predict", "model_id": model_id,
        "class": target_class, "classes": classes,
        "region": _region_echo(size0, col0, row0, cols, rows),
        "grid": {"cols": cols, "rows": rows},
        "n_tiles": cols * rows, "n_tissue": len(cell_vec),
        "n_above": shapes["n_above"], "min_confidence": PREDICT_MIN_CONFIDENCE,
        "cells": shapes["cells"], "boundaries": shapes["boundaries"],
        "seconds": round(time.perf_counter() - t0, 3),
    }


def _region_shapes(score_by_cell, size0, thr):
    """Confident-region shapes in region-local pixels: filled cells + outline.

    A tile is "inside" when its score >= thr. `cells` are the inside tiles as
    [x, y, w, h] rects (for a tinted fill); `boundaries` are axis-aligned edge
    segments [x1, y1, x2, y2] wherever an inside tile borders a non-inside neighbor
    (the outline of every region, holes included). Cell (r, c) spans
    [c*size0, (c+1)*size0] x [r*size0, (r+1)*size0] — region-local, since the region
    origin (col0*size0, row0*size0) cancels out.

    Returns {"cells": [...], "boundaries": [...], "n_above": N}.
    """
    inside = {rc for rc, p in score_by_cell.items() if p >= thr}
    cells, segs = [], []
    for (r, c) in inside:
        x0, y0 = c * size0, r * size0
        x1, y1 = (c + 1) * size0, (r + 1) * size0
        cells.append([x0, y0, size0, size0])
        if (r - 1, c) not in inside:
            segs.append([x0, y0, x1, y0])  # top
        if (r + 1, c) not in inside:
            segs.append([x0, y1, x1, y1])  # bottom
        if (r, c - 1) not in inside:
            segs.append([x0, y0, x0, y1])  # left
        if (r, c + 1) not in inside:
            segs.append([x1, y0, x1, y1])  # right
    return {"cells": cells, "boundaries": segs, "n_above": len(inside)}


# --- Unsupervised similarity-to-an-annotation (no model needed) --------------

def _annotation_reference_vector(slide, slide_id, model_id, annotation):
    """Mean embedding of the tissue tiles inside an annotation, L2-normalized.

    Reuses the Phase 3 patch machinery (parse geometry, grid it, drop background)
    so the reference is computed exactly like the training patches were. Returns
    the normalized reference vector, or None if the annotation covers no tissue.
    """
    parsed = patches._parse_annotation(annotation)
    if parsed is None:
        return None
    _label, kind, geom = parsed

    mpp0 = patches._slide_mpp(slide)
    size0 = max(1, round(patches.PATCH_SIZE * patches.TARGET_MPP / mpp0))
    level = slide.get_best_level_for_downsample(max(1.0, size0 / patches.PATCH_SIZE))
    off_x, off_y = patches._bounds_offset(slide)
    dim_x, dim_y = slide.dimensions
    embedder = embeddings.get_embedder()

    imgs = []
    for x0, y0 in patches._grid_origins(patches._bbox(kind, geom), size0):
        if kind == "polygon" and not patches._point_in_polygon(x0 + size0 / 2, y0 + size0 / 2, geom):
            continue
        px, py = x0 + off_x, y0 + off_y
        if px < off_x or py < off_y or px + size0 > off_x + dim_x or py + size0 > off_y + dim_y:
            continue
        patch = patches.read_patch(slide, px, py, size0, level)
        if patches._is_tissue(patch):
            imgs.append(patch)
    if not imgs:
        return None

    vecs = []
    for start in range(0, len(imgs), embedder.BATCH):
        vecs.extend(embeddings.embed_images(embedder, imgs[start:start + embedder.BATCH]))
    ref = np.mean(np.vstack(vecs), axis=0)
    return ref / (np.linalg.norm(ref) + 1e-8)


def _stretch_scores(cell_vec, ref_norm):
    """Cosine-similarity of each cell to ref_norm, min-max stretched to [0,1]."""
    cells = list(cell_vec.keys())
    mat = np.vstack([cell_vec[c] for c in cells])
    norms = np.linalg.norm(mat, axis=1) + 1e-8
    sims = (mat @ ref_norm) / norms
    lo, hi = float(sims.min()), float(sims.max())
    rng = max(1e-6, hi - lo)
    return {cell: (float(s) - lo) / rng for cell, s in zip(cells, sims)}


def similarity(slide_id: str, region: dict, annotation: dict) -> dict:
    """Colour each visible tile by embedding similarity to a selected annotation.

    Works with no trained model — it only needs the (Virchow 2) embedder.
    """
    slide = slides.get_slide(slide_id)
    if slide is None:
        return {"status": "no_slide"}

    model_id = embeddings.active_model_id()
    geom = _geometry(slide, region)
    size0, _, _, _, col0, row0, cols, rows = geom
    if cols * rows == 0:
        return {"status": "empty_region"}
    if cols * rows > MAX_TILES:
        return {"status": "too_many_tiles", "requested": cols * rows, "limit": MAX_TILES}

    t0 = time.perf_counter()
    ref_norm = _annotation_reference_vector(slide, slide_id, model_id, annotation)
    if ref_norm is None:
        return {"status": "empty_annotation"}

    cell_vec = _collect_embeddings(slide, slide_id, model_id, geom)
    score_by_cell = _stretch_scores(cell_vec, ref_norm) if cell_vec else {}

    # Same tinted-region overlay as predict: outline the tiles most similar to the
    # reference (top of the per-view similarity range).
    shapes = _region_shapes(score_by_cell, size0, SIMILARITY_MIN)

    return {
        "status": "ok", "mode": "similarity", "model_id": model_id,
        "ref_label": patches._label_of(annotation),
        "region": _region_echo(size0, col0, row0, cols, rows),
        "grid": {"cols": cols, "rows": rows},
        "n_tiles": cols * rows, "n_tissue": len(cell_vec),
        "n_above": shapes["n_above"], "min_confidence": SIMILARITY_MIN,
        "cells": shapes["cells"], "boundaries": shapes["boundaries"],
        "seconds": round(time.perf_counter() - t0, 3),
    }
