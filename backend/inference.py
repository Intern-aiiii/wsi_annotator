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

Neither reads pixels or runs the model. Both are pure lookups into the slide's
feature bank (features.py), which is why they are fast and why they are GATED: a
slide with no completed sweep returns "no_features" rather than quietly embedding
tiles on the interactive path, which is what the old version did.
"""

from __future__ import annotations

import time

import numpy as np

from backend import classifier, embeddings, features, patches, slides

# Upper bound on tiles per call. If the visible region grids to more than this
# (zoomed too far out), we ask the user to zoom in rather than hang. Now that the
# vectors are all in the bank this is a memory bound, not a compute one, so it is
# cheaply raisable.
MAX_TILES = 4000

# Predict outlines regions whose tiles score at least this probability for the
# chosen class; similarity outlines tiles at least this fraction of the way up the
# (min-max stretched) per-view similarity range. Fixed for now.
PREDICT_MIN_CONFIDENCE = 0.95
SIMILARITY_MIN = 0.95


# --- Shared setup: the gate + the region's cells ------------------------------

def _prepare(slide_id: str, region: dict):
    """Resolve the slide, check the feature gate, snap the region to the grid.

    Returns (context, None) when good, or (None, error_dict) when the caller should
    return that error straight to the client. The check ORDER matters: "no_features"
    has to precede "too_many_tiles", or a first-time user zoomed out over a whole
    slide is told to zoom in when the real problem is that nothing is extracted yet.
    """
    slide = slides.get_slide(slide_id)
    if slide is None:
        return None, {"status": "no_slide"}

    model_id = embeddings.active_model_id()
    feat = features.state(slide_id, model_id)
    if feat["state"] != "complete":
        return None, {"status": "no_features", "features": feat}

    grid = feat["grid"]
    col0, row0, cols, rows = patches.cells_in_region(grid, region)
    if cols * rows == 0:
        return None, {"status": "empty_region"}
    if cols * rows > MAX_TILES:
        return None, {"status": "too_many_tiles", "requested": cols * rows, "limit": MAX_TILES}

    return {
        "slide_id": slide_id, "model_id": model_id, "grid": grid,
        "col0": col0, "row0": row0, "cols": cols, "rows": rows,
    }, None


def _cell_vectors(ctx) -> dict:
    """The region's tissue vectors from the bank, keyed by REGION-LOCAL (r, c).

    Background cells simply have no vector, so they're absent — same contract the
    tissue mask gave us before, just resolved at sweep time instead of query time.
    """
    col0, row0 = ctx["col0"], ctx["row0"]
    want = [
        (col0 + c, row0 + r)
        for r in range(ctx["rows"])
        for c in range(ctx["cols"])
    ]
    got = features.vectors(ctx["slide_id"], ctx["model_id"], ctx["grid"], want)
    return {(row - row0, col - col0): vec for (col, row), vec in got.items()}


def _region_echo(ctx) -> dict:
    """The snapped grid region in DZI-image pixels (where the frontend overlays)."""
    size0 = ctx["grid"]["size0"]
    return {"x": ctx["col0"] * size0, "y": ctx["row0"] * size0,
            "w": ctx["cols"] * size0, "h": ctx["rows"] * size0}


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


# --- Supervised prediction ---------------------------------------------------

def predict(project_id: str, slide_id: str, region: dict,
            target_class: str | None = None) -> dict:
    """Score the region's tissue tiles with THIS PROJECT's head; outline the confident ones.

    The project decides which head is loaded — and only that. The vectors come from the
    slide's shared feature bank, so two projects score the very same tiles through two
    different classifiers, which is exactly what "two experiments coexist" means.
    """
    ctx, err = _prepare(slide_id, region)
    if err is not None:
        return err

    head = classifier.load_head(project_id)
    if head is None:
        return {"status": "no_model"}

    classes = list(head.classes_)
    target_class = target_class if target_class in classes else classes[0]
    class_col = classes.index(target_class)

    t0 = time.perf_counter()
    cell_vec = _cell_vectors(ctx)
    score_by_cell = {}
    if cell_vec:
        cells = list(cell_vec.keys())
        proba = head.predict_proba(np.vstack([cell_vec[c] for c in cells]))[:, class_col]
        score_by_cell = {cell: float(p) for cell, p in zip(cells, proba)}

    # Outline + tint the confident regions (frontend draws one vector overlay).
    shapes = _region_shapes(score_by_cell, ctx["grid"]["size0"], PREDICT_MIN_CONFIDENCE)

    return {
        "status": "ok", "mode": "predict", "model_id": ctx["model_id"],
        "class": target_class, "classes": classes,
        "region": _region_echo(ctx),
        "grid": {"cols": ctx["cols"], "rows": ctx["rows"]},
        "n_tiles": ctx["cols"] * ctx["rows"], "n_tissue": len(cell_vec),
        "n_above": shapes["n_above"], "min_confidence": PREDICT_MIN_CONFIDENCE,
        "cells": shapes["cells"], "boundaries": shapes["boundaries"],
        "seconds": round(time.perf_counter() - t0, 3),
    }


# --- Unsupervised similarity-to-an-annotation (no model needed) --------------

def _annotation_reference_vector(ctx, annotation):
    """Mean bank embedding of the annotation's tissue cells, L2-normalized.

    Pure geometry + lookups: the annotation's cells come off the same grid the sweep
    used, so its vectors are already in the bank. Returns None if the annotation
    covers no tissue (all its cells were masked out as background).
    """
    cells = patches.cells_in_annotation(ctx["grid"], annotation)
    if not cells:
        return None
    vecs = features.vectors(ctx["slide_id"], ctx["model_id"], ctx["grid"], cells)
    if not vecs:
        return None
    ref = np.mean(np.vstack(list(vecs.values())), axis=0)
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
    """Outline the visible tiles most similar to a selected annotation.

    Works with no trained model — it only needs the slide's feature bank.
    """
    ctx, err = _prepare(slide_id, region)
    if err is not None:
        return err

    # "I can't READ this shape" and "this shape covers no tissue" both surface as zero
    # cells, but they need completely different fixes — so check the selector FIRST.
    # Reporting "covers no tissue" for a shape we simply failed to parse sends the user
    # hunting for a tissue problem that does not exist.
    if patches.parse_annotation(annotation) is None:
        return {"status": "bad_selector"}

    t0 = time.perf_counter()
    ref_norm = _annotation_reference_vector(ctx, annotation)
    if ref_norm is None:
        return {"status": "empty_annotation"}

    cell_vec = _cell_vectors(ctx)
    score_by_cell = _stretch_scores(cell_vec, ref_norm) if cell_vec else {}

    # Same tinted-region overlay as predict: outline the tiles most similar to the
    # reference (top of the per-view similarity range).
    shapes = _region_shapes(score_by_cell, ctx["grid"]["size0"], SIMILARITY_MIN)

    return {
        "status": "ok", "mode": "similarity", "model_id": ctx["model_id"],
        "ref_label": patches.label_of(annotation),
        "region": _region_echo(ctx),
        "grid": {"cols": ctx["cols"], "rows": ctx["rows"]},
        "n_tiles": ctx["cols"] * ctx["rows"], "n_tissue": len(cell_vec),
        "n_above": shapes["n_above"], "min_confidence": SIMILARITY_MIN,
        "cells": shapes["cells"], "boundaries": shapes["boundaries"],
        "seconds": round(time.perf_counter() - t0, 3),
    }
