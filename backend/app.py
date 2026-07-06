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

from backend import slides

app = FastAPI(title="SlideProbe")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# --- API -------------------------------------------------------------------

@app.get("/api/slides")
def api_list_slides():
    """List the slides available under data/slides/."""
    return {"slides": slides.list_slides()}


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
