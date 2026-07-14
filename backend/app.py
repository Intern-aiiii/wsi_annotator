"""FastAPI entrypoint.

Wires the backend together:
  - API routes the frontend calls.
  - DeepZoom routes OpenSeadragon calls to fetch the .dzi descriptor and tiles.
  - Serving the static frontend (index.html + js/) so the whole app lives at
    http://127.0.0.1:8000.

THE ROUTE RULE (Phase 7). Every route is one of two kinds, and which one it is follows
from what the thing IS, not from what the handler happens to read:

    GLOBAL           it is about a FILE on disk, or a cache derived only from that file
                     -> /api/slides, /api/slides/{id}/features*, the DeepZoom routes

    PROJECT-SCOPED   it is about the user's EXPERIMENT
                     -> /api/projects/{pid}/...  annotations, classes, train, model,
                        learn/status, predict, similarity

The feature routes stay global because the feature bank is deliberately SHARED across
projects: it is keyed by (slide, embedding model, tissue mask), so a slide swept once
with Virchow 2 is reused by every project that contains it. That is the single most
expensive step in the pipeline, and project-scoping it would re-run it per project for
no gain.

Note `similarity` is project-scoped even though it does not need the project (it takes
the reference annotation in the request body and only reads the shared bank). Predict
and Find-similar are twin buttons in the UI; giving them different URL shapes to save
one existence check would leave the frontend with two URL conventions forever.

The DeepZoom URL scheme is fixed by the DZI spec and by what OpenSeadragon expects:
    tile source :  /slides/<id>.dzi
    tiles       :  /slides/<id>_files/<level>/<col>_<row>.jpeg

Run locally with:
    uvicorn backend.app:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import annotations, classifier, features, inference, jobs, projects, slides
from backend.models import (
    AnnotationCollection,
    ClassList,
    PredictRegion,
    ProjectCreate,
    ProjectRename,
    SimilarityRegion,
    SlideRef,
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Import any pre-Phase-7 data into a "default" project, once, on first boot.

    In the lifespan hook rather than at module import: an import-time side-effect that
    writes to disk is hard to reason about and impossible to skip in a test.
    """
    report = projects.migrate_legacy()
    if report["status"] == "imported":
        print(f"[slideprobe] imported legacy data into project "
              f"'{report['project_id']}': {report['annotations']} annotation file(s), "
              f"{report['heads']} head file(s). data/annotations/ and data/models/ were "
              f"copied, not moved — they remain as a backup.")
    yield


app = FastAPI(title="SlideProbe", lifespan=lifespan)


# --- Shared guards -----------------------------------------------------------

def _project_or_404(project_id: str):
    """(project, None) when it exists, else (None, a 404 response)."""
    project = projects.load(project_id)
    if project is None:
        return None, JSONResponse(
            {"error": f"project '{project_id}' not found"}, status_code=404
        )
    return project, None


def _member_or_404(project: dict, slide_id: str):
    """A 404 unless the slide is in this project's picker, else None.

    The UI can never trigger this (its picker only ever lists members), so if it fires,
    something is wired wrong — and a loud 404 beats silently writing annotations into a
    project that doesn't claim the slide.
    """
    if slide_id not in project.get("slides", []):
        return JSONResponse(
            {"error": f"slide '{slide_id}' is not in project '{project['id']}'"},
            status_code=404,
        )
    return None


# --- Projects (Phase 7) ------------------------------------------------------

@app.get("/api/projects")
def api_list_projects():
    """Every workspace on disk. The frontend picks one and scopes everything to it."""
    return {"projects": projects.list_projects()}


@app.post("/api/projects")
def api_create_project(body: ProjectCreate):
    """Create a workspace. Omit `slides` to include every slide currently on disk."""
    name = body.name.strip()
    if not name:
        return JSONResponse({"error": "project name cannot be empty"}, status_code=400)
    return projects.create(name, body.slides)


@app.get("/api/projects/{project_id}")
def api_get_project(project_id: str):
    """One project in full: name, slide membership, and class list (with colours).

    Deliberately ONE call: the frontend needs classes AND slides before it can open
    anything, and two calls would be two chances to race a project switch.
    """
    project, err = _project_or_404(project_id)
    return err or project


@app.patch("/api/projects/{project_id}")
def api_rename_project(project_id: str, body: ProjectRename):
    """Rename a project. The id and its directory never move."""
    _, err = _project_or_404(project_id)
    if err:
        return err
    try:
        return projects.rename(project_id, body.name)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.delete("/api/projects/{project_id}")
def api_delete_project(project_id: str):
    """Delete a project and everything it owns — its annotations, its classes, its head.

    Never touches data/slides/ (the WSIs) or data/cache/features/ (the shared bank), so
    another project over the same slides is completely unaffected and does not have to
    re-sweep anything.
    """
    _, err = _project_or_404(project_id)
    if err:
        return err
    result = projects.delete(project_id)
    jobs.forget(project_id)     # drop any queued retrain + stale status for it
    return result


@app.get("/api/projects/{project_id}/slides")
def api_project_slides(project_id: str):
    """This project's slides, JOINED against what is actually on disk.

    A slide can be listed in the project but missing from data/slides/ (someone deleted
    the file), so we report `missing` rather than trusting project.json — otherwise the
    viewer would try to open it and die. We never PRUNE here: a GET must not write.
    """
    project, err = _project_or_404(project_id)
    if err:
        return err
    on_disk = {s["id"]: s["name"] for s in slides.list_slides()}
    return {
        "slides": [
            {"id": sid, "name": on_disk.get(sid, sid), "missing": sid not in on_disk}
            for sid in project.get("slides", [])
        ]
    }


@app.post("/api/projects/{project_id}/slides")
def api_add_project_slide(project_id: str, body: SlideRef):
    """Add a slide on disk to this project's picker."""
    _, err = _project_or_404(project_id)
    if err:
        return err
    if slides.get_slide(body.slide_id) is None:
        return JSONResponse(
            {"error": f"slide '{body.slide_id}' not found"}, status_code=404
        )
    return projects.add_slide(project_id, body.slide_id)


@app.delete("/api/projects/{project_id}/slides/{slide_id}")
def api_remove_project_slide(project_id: str, slide_id: str, force: bool = False):
    """Take a slide out of this project's picker.

    409 if the project still holds annotations for it, unless ?force=true — a mis-click
    must not be able to silently destroy drawn work. With force, we delete only THIS
    project's annotation file for the slide: never the slide, never the shared feature
    bank, never another project's annotations.
    """
    _, err = _project_or_404(project_id)
    if err:
        return err
    result = projects.remove_slide(project_id, slide_id, force=force)
    if result["status"] == "has_annotations":
        return JSONResponse(
            {"error": f"'{slide_id}' still has annotations in this project; "
                      "removing it will delete them",
             "status": "has_annotations"},
            status_code=409,
        )
    return result


@app.get("/api/projects/{project_id}/classes")
def api_get_classes(project_id: str):
    """The project's class vocabulary + the colour each is drawn in."""
    project, err = _project_or_404(project_id)
    return err or {"classes": project.get("classes", [])}


@app.put("/api/projects/{project_id}/classes")
def api_put_classes(project_id: str, body: ClassList):
    """Replace the whole class list (same convention as annotations).

    Removing a class here is purely COSMETIC: annotation tags are the source of truth
    for training, so regions labelled with a removed class keep their tag and keep
    training. The trained head reports them as `classes_not_in_project`.
    """
    _, err = _project_or_404(project_id)
    if err:
        return err
    try:
        project = projects.set_classes(
            project_id, [c.model_dump() for c in body.classes]
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"classes": project["classes"]}


# --- Slides on disk (GLOBAL — a file is not owned by any project) -------------

@app.get("/api/slides")
def api_list_slides():
    """Every slide under data/slides/, regardless of project. This is what the
    "add slides to this project" picker offers."""
    return {"slides": slides.list_slides()}


# --- Annotations (project-scoped) --------------------------------------------

@app.get("/api/projects/{project_id}/slides/{slide_id}/annotations")
def api_get_annotations(project_id: str, slide_id: str):
    """Return this project's annotations for a slide (W3C JSON)."""
    project, err = _project_or_404(project_id)
    if err:
        return err
    err = _member_or_404(project, slide_id)
    if err:
        return err
    return {"annotations": annotations.load(project_id, slide_id)}


@app.put("/api/projects/{project_id}/slides/{slide_id}/annotations")
def api_put_annotations(project_id: str, slide_id: str, body: AnnotationCollection):
    """Overwrite this project's annotations for a slide with the posted list.

    The frontend sends the whole collection on every change, so this is a plain
    replace. Returns how many annotations were saved.
    """
    project, err = _project_or_404(project_id)
    if err:
        return err
    err = _member_or_404(project, slide_id)
    if err:
        return err
    saved = annotations.save(project_id, slide_id, body.annotations)
    # Retrain THIS project in the background. Cheap — geometry + feature-bank lookups,
    # no pixels and no model — and debounced, so a flurry of edits collapses into one
    # training pass. Embedding is NOT on this path; it's the explicit feature sweep.
    jobs.schedule_project(project_id)
    return {"saved": saved}


@app.get("/api/projects/{project_id}/learn/status")
def api_learn_status(project_id: str):
    """State of the background learning worker FOR THIS PROJECT (the topbar chip)."""
    return jobs.status(project_id)


# --- Whole-slide feature extraction (GLOBAL — the bank is shared) -------------
# The one expensive step, and the only one the user triggers by hand. Everything
# downstream (training, predict, find-similar) is a lookup into what this produces.
#
# NOT project-scoped, on purpose: the bank is keyed by (slide, embedding model, tissue
# mask), so it is a property of the FILE. Sweep a slide once and every project that
# contains it can train and predict immediately.

@app.post("/api/slides/{slide_id}/features")
def api_start_features(slide_id: str):
    """Start the whole-slide feature sweep in the background. Returns immediately.

    Slow (minutes on a real slide with Virchow 2), so it does NOT block the request.
    Poll GET .../features for progress, POST .../features/cancel to stop it. Safe to
    re-run: it resumes from whatever is already in the bank.
    """
    if slides.get_slide(slide_id) is None:
        return JSONResponse({"error": f"slide '{slide_id}' not found"}, status_code=404)
    return features.start(slide_id)


@app.get("/api/slides/{slide_id}/features")
def api_feature_state(slide_id: str):
    """This slide's feature state + live sweep progress. The UI gates its buttons on it.

    state: "none" | "partial" | "complete". Predict and Find similar require
    "complete" — a partial bank would silently score only part of the view.
    """
    if slides.get_slide(slide_id) is None:
        return JSONResponse({"error": f"slide '{slide_id}' not found"}, status_code=404)

    result = features.state(slide_id)
    sweep = features.status()
    running = sweep.get("slide_id") if sweep.get("state") in ("running", "cancelling") else None
    # `running_slide_id` on every response: one sweep runs at a time, so a user who
    # switches slides mid-sweep needs to see whose sweep is holding the worker.
    result["running_slide_id"] = running
    result["sweep"] = sweep if sweep.get("slide_id") == slide_id else {"state": "idle"}
    return result


@app.post("/api/slides/{slide_id}/features/cancel")
def api_cancel_features(slide_id: str):
    """Stop the running sweep. Whatever it already embedded is kept and resumable."""
    return features.cancel()


@app.delete("/api/slides/{slide_id}/features")
def api_clear_features(slide_id: str):
    """Drop this slide's feature bank, so the next sweep starts from scratch.

    Affects EVERY project containing the slide — the bank is shared. That's the price of
    not re-embedding the same tiles once per project, and it's the right trade.
    """
    if slides.get_slide(slide_id) is None:
        return JSONResponse({"error": f"slide '{slide_id}' not found"}, status_code=404)
    sweep = features.status()
    if sweep.get("state") in ("running", "cancelling") and sweep.get("slide_id") == slide_id:
        return JSONResponse(
            {"error": "a sweep is running for this slide; cancel it first"}, status_code=409
        )
    features.clear_bank(slide_id)
    return {"status": "cleared"}


# --- Classifier head (project-scoped) ----------------------------------------

@app.post("/api/projects/{project_id}/train")
def api_train(project_id: str):
    """Train this project's head on the annotated cells of its slides' feature banks.

    Normally you don't call this: it runs automatically (debounced) whenever an
    annotation changes, and once at the end of a feature sweep. Kept as an explicit
    trigger for testing.
    """
    _, err = _project_or_404(project_id)
    if err:
        return err

    result = classifier.train(project_id)
    status = result.get("status")
    if status == "no_data":
        hints = {
            "no_annotations": "no annotations to train on — draw some regions first",
            "no_features": "the annotated slides have no features — extract features first",
            "no_labels": "the regions have no class — tag them first",
            "no_shapes": "could not read the shape of any region (unsupported selector)",
            "no_cells": "the annotations are too small to cover a single tile",
            "no_tissue": "the annotated regions contain no tissue",
        }
        return JSONResponse(
            {"error": hints.get(result.get("reason"), "nothing to train on"),
             "reason": result.get("reason")},
            status_code=400,
        )
    if status == "need_2_classes":
        return JSONResponse(
            {"error": "need at least two labelled classes to train",
             "classes": result.get("classes", [])},
            status_code=400,
        )
    return result


@app.get("/api/projects/{project_id}/model")
def api_get_model(project_id: str):
    """This project's trained head: metadata + honest (region-grouped CV) metrics."""
    _, err = _project_or_404(project_id)
    if err:
        return err
    meta = classifier.head_metadata(project_id)
    if meta is None:
        return JSONResponse(
            {"error": "no trained model yet in this project"}, status_code=404
        )
    return meta


# --- Predict + region overlay (project-scoped) -------------------------------

def _no_features_response(result: dict) -> JSONResponse:
    """409 for a slide whose features aren't extracted (or only partly).

    Deliberately NOT a 400: the frontend renders this one as "Extract features
    first" rather than a generic failure, and 409 (conflict — the resource isn't in
    a state that allows this) is the honest code for it.
    """
    feat = result.get("features", {})
    if feat.get("state") == "partial":
        msg = ("feature extraction is incomplete for this slide "
               f"({feat.get('n_covered', 0):,} / {feat.get('n_cells', 0):,} tiles) — resume it first")
    else:
        msg = "extract features for this slide first"
    return JSONResponse({"error": msg, "status": "no_features", "features": feat}, status_code=409)


def _scoring_error(result: dict, slide_id: str) -> JSONResponse | None:
    """The failure modes predict and similarity share, in the ORDER THAT MATTERS.

    "no_features" has to precede "too_many_tiles", or a first-time user zoomed out over
    a whole slide is told to zoom in when the real problem is that nothing is extracted.
    """
    status = result.get("status")
    if status == "no_slide":
        return JSONResponse({"error": f"slide '{slide_id}' not found"}, status_code=404)
    if status == "no_features":
        return _no_features_response(result)
    if status == "too_many_tiles":
        return JSONResponse(
            {"error": "zoom in to predict a smaller area",
             "requested": result.get("requested"), "limit": result.get("limit")},
            status_code=400,
        )
    if status == "empty_region":
        return JSONResponse({"error": "empty region"}, status_code=400)
    return None


@app.post("/api/projects/{project_id}/slides/{slide_id}/predict")
def api_predict(project_id: str, slide_id: str, body: PredictRegion):
    """Score the visible tissue tiles with THIS PROJECT's head; outline a class."""
    project, err = _project_or_404(project_id)
    if err:
        return err
    err = _member_or_404(project, slide_id)
    if err:
        return err

    region = {"x": body.x, "y": body.y, "w": body.w, "h": body.h}
    result = inference.predict(project_id, slide_id, region, body.target_class)

    err = _scoring_error(result, slide_id)
    if err:
        return err
    if result.get("status") == "no_model":
        return JSONResponse({"error": "train a classifier first"}, status_code=400)
    return result


@app.post("/api/projects/{project_id}/slides/{slide_id}/similarity")
def api_similarity(project_id: str, slide_id: str, body: SimilarityRegion):
    """Unsupervised: outline the visible regions most similar to a selected annotation.

    Needs no trained model — just the slide's (shared) feature bank. Great for a first
    look: draw a region, then find everything that looks like it.

    The project id is validated but NOT passed down: similarity genuinely doesn't need
    it (the reference annotation arrives in the request body). It is in the path so that
    every route the workspace uses has the same shape — Predict and Find-similar are
    twin buttons, and two URL conventions for them would be a permanent trip hazard.
    """
    project, err = _project_or_404(project_id)
    if err:
        return err
    err = _member_or_404(project, slide_id)
    if err:
        return err

    region = {"x": body.x, "y": body.y, "w": body.w, "h": body.h}
    result = inference.similarity(slide_id, region, body.annotation)

    err = _scoring_error(result, slide_id)
    if err:
        return err
    if result.get("status") == "bad_selector":
        return JSONResponse(
            {"error": "could not read this annotation's shape (unsupported selector)",
             "status": "bad_selector"},
            status_code=400,
        )
    if result.get("status") == "empty_annotation":
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
