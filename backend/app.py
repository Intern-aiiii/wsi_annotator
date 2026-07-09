"""FastAPI entrypoint (Phase 1: one slide on screen).

Wires the backend together:
  - API routes the frontend calls (currently just: list slides).
  - DeepZoom routes OpenSeadragon calls to fetch the .dzi descriptor and tiles.
  - Serving the static frontend (index.html + js/) so the whole app lives at
    http://127.0.0.1:8000.

The DeepZoom URL scheme is fixed by the DZI spec and by what OpenSeadragon
expects for a given tile source:
    tile source :  /slides/<id>.dzi
    tiles       :  /slides/<id>_files/<level>/<col>_<row>.jpeg

Run locally with:
    uvicorn backend.app:app --reload --port 8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import annotations, classifier, embeddings, inference, jobs, patches, slides
from backend.models import AnnotationCollection, PredictRegion, SimilarityRegion

app = FastAPI(title="SlideProbe")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# --- API -------------------------------------------------------------------

@app.get("/api/slides")
def api_list_slides():
    """List the slides available under data/slides/."""
    return {"slides": slides.list_slides()}


@app.get("/api/slides/{slide_id}/annotations")
def api_get_annotations(slide_id: str):
    """Return a slide's saved annotations (W3C JSON), or 404 if no such slide."""
    if annotations.annotations_path(slide_id) is None:
        return JSONResponse({"error": f"slide '{slide_id}' not found"}, status_code=404)
    return {"annotations": annotations.load(slide_id)}


@app.put("/api/slides/{slide_id}/annotations")
def api_put_annotations(slide_id: str, body: AnnotationCollection):
    """Overwrite a slide's annotations with the posted list.

    The frontend sends the whole collection on every change, so this is a plain
    replace. Returns how many annotations were saved.
    """
    if annotations.annotations_path(slide_id) is None:
        return JSONResponse({"error": f"slide '{slide_id}' not found"}, status_code=404)
    saved = annotations.save(slide_id, body.annotations)
    # Kick off the background learning pipeline (extract -> embed -> train). It's
    # debounced, so rapid successive edits collapse into a single training pass.
    jobs.schedule(slide_id)
    return {"saved": saved}


@app.get("/api/learn/status")
def api_learn_status():
    """Current state of the background learning worker (for the topbar chip)."""
    return jobs.status()


# --- Patch extraction (Phase 3) --------------------------------------------

@app.post("/api/slides/{slide_id}/patches")
def api_extract_patches(slide_id: str):
    """Extract labelled 224x224 tissue patches from the slide's annotations.

    Runs OpenSlide reads + tissue masking, writes a manifest + preview montage
    under data/cache/patches/, and returns a summary (counts, per-class).
    """
    manifest = patches.extract_patches(slide_id)
    if manifest is None:
        return JSONResponse({"error": f"slide '{slide_id}' not found"}, status_code=404)
    return manifest


@app.get("/api/slides/{slide_id}/patches")
def api_get_patches(slide_id: str):
    """Return the last-extracted patch manifest, or 404 if none exists yet."""
    if slides.get_slide(slide_id) is None:
        return JSONResponse({"error": f"slide '{slide_id}' not found"}, status_code=404)
    manifest = patches.load_manifest(slide_id)
    if manifest is None:
        return JSONResponse(
            {"error": "no patches extracted yet; POST this endpoint first"},
            status_code=404,
        )
    return manifest


@app.get("/api/slides/{slide_id}/patches/preview.jpg")
def api_patches_preview(slide_id: str):
    """Serve the preview montage image from the last extraction."""
    path = patches.preview_path(slide_id)
    if not path.exists():
        return JSONResponse({"error": "no preview available"}, status_code=404)
    return Response(content=path.read_bytes(), media_type="image/jpeg")


# --- Embeddings (Phase 4) --------------------------------------------------

@app.post("/api/slides/{slide_id}/embeddings")
def api_compute_embeddings(slide_id: str):
    """Embed the slide's extracted patches with the active model (cached).

    Runs the selected embedder (default: the dev stand-in; Virchow 2 is opt-in via
    the SLIDEPROBE_EMBEDDER env var) and caches vectors under data/cache/embeddings/.
    """
    try:
        result = embeddings.embed_slide(slide_id)
    except embeddings.EmbedderError as e:
        # Selected model unavailable (e.g. Virchow 2 deps/access not set up).
        return JSONResponse({"error": str(e)}, status_code=503)

    status = result.get("status")
    if status == "no_slide":
        return JSONResponse({"error": f"slide '{slide_id}' not found"}, status_code=404)
    if status == "no_manifest":
        return JSONResponse(
            {"error": "no patches to embed — extract patches first"}, status_code=400
        )
    return result


@app.get("/api/slides/{slide_id}/embeddings")
def api_get_embeddings(slide_id: str):
    """Return the saved embedding index/summary for the active model, or 404."""
    if slides.get_slide(slide_id) is None:
        return JSONResponse({"error": f"slide '{slide_id}' not found"}, status_code=404)
    index = embeddings.load_index(slide_id)
    if index is None:
        return JSONResponse(
            {"error": "no embeddings yet; POST this endpoint first"}, status_code=404
        )
    # Don't ship the full row list back by default — just the summary.
    rows = index.get("rows", [])
    per_class: dict[str, int] = {}
    for r in rows:
        per_class[r.get("label")] = per_class.get(r.get("label"), 0) + 1
    return {
        "model_id": index.get("model_id"),
        "dim": index.get("dim"),
        "n_total": len(rows),
        "per_class": per_class,
    }


# --- Classifier head (Phase 5) ---------------------------------------------

@app.post("/api/train")
def api_train():
    """Train the classifier head on all cached embeddings for the active model.

    Pools every embedded slide, splits by region for honest metrics, and persists
    the head + metadata under data/models/. Returns the training summary.
    """
    result = classifier.train()
    status = result.get("status")
    if status == "no_data":
        return JSONResponse(
            {"error": "no embeddings to train on — extract patches and compute embeddings first"},
            status_code=400,
        )
    if status == "need_2_classes":
        return JSONResponse(
            {"error": "need at least two labelled classes to train", "classes": result.get("classes", [])},
            status_code=400,
        )
    return result


@app.get("/api/model")
def api_get_model():
    """Return the trained head's metadata + metrics for the active model, or 404."""
    meta = classifier.head_metadata()
    if meta is None:
        return JSONResponse(
            {"error": "no trained model yet; POST /api/train first"}, status_code=404
        )
    return meta


# --- Predict + region overlay (Phase 6) ------------------------------------

@app.post("/api/slides/{slide_id}/predict")
def api_predict(slide_id: str, body: PredictRegion):
    """Score the visible tissue tiles and return the confident regions for a class."""
    region = {"x": body.x, "y": body.y, "w": body.w, "h": body.h}
    try:
        result = inference.predict(slide_id, region, body.target_class)
    except embeddings.EmbedderError as e:
        return JSONResponse({"error": str(e)}, status_code=503)

    status = result.get("status")
    if status == "no_slide":
        return JSONResponse({"error": f"slide '{slide_id}' not found"}, status_code=404)
    if status == "no_model":
        return JSONResponse({"error": "train a classifier first"}, status_code=400)
    if status == "too_many_tiles":
        return JSONResponse(
            {"error": "zoom in to predict a smaller area",
             "requested": result.get("requested"), "limit": result.get("limit")},
            status_code=400,
        )
    if status == "empty_region":
        return JSONResponse({"error": "empty region"}, status_code=400)
    return result


@app.post("/api/slides/{slide_id}/similarity")
def api_similarity(slide_id: str, body: SimilarityRegion):
    """Unsupervised: outline the visible regions most similar to a selected annotation.

    Needs no trained model — only the embedder. Great for a first look: draw a
    region, then find everything that looks like it.
    """
    region = {"x": body.x, "y": body.y, "w": body.w, "h": body.h}
    try:
        result = inference.similarity(slide_id, region, body.annotation)
    except embeddings.EmbedderError as e:
        return JSONResponse({"error": str(e)}, status_code=503)

    status = result.get("status")
    if status == "no_slide":
        return JSONResponse({"error": f"slide '{slide_id}' not found"}, status_code=404)
    if status == "too_many_tiles":
        return JSONResponse(
            {"error": "zoom in to predict a smaller area",
             "requested": result.get("requested"), "limit": result.get("limit")},
            status_code=400,
        )
    if status == "empty_region":
        return JSONResponse({"error": "empty region"}, status_code=400)
    if status == "empty_annotation":
        return JSONResponse(
            {"error": "the selected annotation covers no tissue"}, status_code=400
        )
    return result


# --- DeepZoom (consumed by OpenSeadragon) ----------------------------------

@app.get("/slides/{slide_id}.dzi")
def slide_dzi(slide_id: str):
    """Return the DeepZoom XML descriptor for a slide."""
    dzi = slides.get_dzi(slide_id)
    if dzi is None:
        return JSONResponse({"error": f"slide '{slide_id}' not found"}, status_code=404)
    # DZI descriptors are XML.
    return Response(content=dzi, media_type="application/xml")


@app.get("/slides/{slide_id}_files/{level}/{tile}")
def slide_tile(slide_id: str, level: int, tile: str):
    """Return one DeepZoom tile.

    `tile` arrives as "<col>_<row>.<format>" (e.g. "3_5.jpeg"); we parse it
    here rather than as separate path params because a single URL segment
    can't be split on "_" by the router cleanly.
    """
    try:
        name, _, _fmt = tile.partition(".")
        col_str, _, row_str = name.partition("_")
        col, row = int(col_str), int(row_str)
    except (ValueError, AttributeError):
        return JSONResponse({"error": "malformed tile request"}, status_code=400)

    data = slides.get_tile(slide_id, level, col, row)
    if data is None:
        return JSONResponse({"error": "tile not found"}, status_code=404)
    return Response(content=data, media_type="image/jpeg")


# --- Static frontend -------------------------------------------------------
# Mounted LAST so the API/DeepZoom routes above take precedence. `html=True`
# serves index.html at "/" and exposes js/ files (e.g. /js/viewer.js).
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
