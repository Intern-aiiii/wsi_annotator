"""Train / persist / load the classifier head (Phase 5).

The "head" is a lightweight classifier trained on the frozen embeddings — it
never sees pixels, only vectors + labels. This is the crux of the approach: the
foundation model already did the hard visual representation learning, so a simple
linear probe on its embeddings separates the annotated structures from a handful
of examples.

Design (per CLAUDE.md's Phase 5 detail):
  - Assemble (X = embeddings, y = labels) by asking each annotation which GRID CELLS
    it covers (pure geometry) and looking those cells up in the slide's feature bank.
    No pixels are read and the model never runs, so a retrain after an annotation
    edit costs a fraction of a second.
  - Split by region (NOT random cell) to avoid leakage between adjacent tiles:
    grouped out-of-fold cross-validation keyed by each cell's `group` (= the drawn
    region it came from).
  - class_weight="balanced" for imbalance; StandardScaler + LogisticRegression.
  - Report per-class precision/recall/F1 + confusion matrix (accuracy misleads
    under imbalance).
  - Persist the head with joblib alongside the embedding config that produced it
    (embedding model_id, dim, magnification) so inference uses matching features.

Training and inference now read the same vectors, for the same cells, off the same
grid. They used to disagree: training tiled each annotation from its bounding-box
corner while inference tiled the viewport on a global grid, so the head was applied
to tile alignments it had never seen.

PROJECT-SCOPED (Phase 7). A head belongs to ONE project and is trained only on that
project's annotations, so two experiments over the same slides cannot contaminate each
other. It is keyed by (project, EMBEDDING MODEL):

    data/projects/<project_id>/models/head__<embedding_model_id>.{joblib,json}

The embedding model has to stay in the filename. The dev embedder is 62-dim and
Virchow 2 is 2560-dim, so a head trained on one is not merely worse on the other — it
is dimensionally invalid. And because the feature BANK is keyed by embedder too, one
project can legitimately hold two valid heads at once, one per embedder, with
SLIDEPROBE_EMBEDDER choosing which is live.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend import annotations, embeddings, features, patches, projects, slides

# patches._label_of() falls back to this for an annotation with no class tag. It must
# never become a training class: it would show up in the predict dropdown, have no
# colour in the project palette, and — worst — could satisfy the "need 2 classes"
# check on its own, training a head that distinguishes "gland" from "you forgot to
# label this".
UNLABELLED = "unlabeled"


# `project_id` is a REQUIRED positional arg on everything below — deliberately no
# `project_id=None -> "default"` fallback. A default is exactly how you get a route
# that silently trains or loads the wrong project's head and nobody notices for a
# week; without one, Python raises TypeError at any call site you forgot to update.

def _head_stem(project_id: str, model_id: str) -> Path | None:
    models = projects.models_dir(project_id)
    return None if models is None else models / f"head__{model_id}"


def head_path(project_id: str, model_id: str) -> Path | None:
    stem = _head_stem(project_id, model_id)
    return None if stem is None else stem.with_suffix(".joblib")


def meta_path(project_id: str, model_id: str) -> Path | None:
    stem = _head_stem(project_id, model_id)
    return None if stem is None else stem.with_suffix(".json")


def delete_head(project_id: str, model_id: str) -> None:
    """Remove a persisted head + metadata (if present) so the app reports no model."""
    for path in (head_path(project_id, model_id), meta_path(project_id, model_id)):
        if path is not None:
            path.unlink(missing_ok=True)


# --- Dataset assembly --------------------------------------------------------

def _gather_dataset(project_id: str, model_id: str):
    """Turn ONE PROJECT's annotations into (X, y, groups) using the shared feature bank.

    For each annotation: which grid cells does it cover (geometry only), and what
    vector does the bank hold for each? No OpenSlide reads, no embedder — that is
    what makes this cheap enough to re-run on every annotation save.

    We iterate the project's ANNOTATION FILES, not slides.list_slides() (which is what
    the pre-Phase-7 version did, pooling every slide on disk into one head) and not
    project["slides"] (which is only a display list — see projects.py, invariant 1).
    The training set is therefore defined by the annotations that actually exist, so
    adding or removing a slide from the picker can never silently change what the head
    learnt.

    Everything downstream of this loop is UNCHANGED from the global version, because
    the feature bank is shared across projects: same cells, same vectors, same lookup.

    Returns (X (N,DIM) float32, y (N,) str, groups (N,) str, per_slide_counts,
    slides_used, stats).
    """
    X_parts, y, groups = [], [], []
    per_slide: dict[str, int] = {}
    n_annotated = n_missing = n_overlap = n_unlabelled = n_unparseable = 0
    slides_with_annotations = 0
    without_features: list[str] = []

    for slide_id in projects.annotated_slides(project_id):
        annos = annotations.load(project_id, slide_id)
        if not annos:
            continue
        slide = slides.get_slide(slide_id)
        if slide is None:
            continue  # annotated once, but the slide has since left data/slides/
        slides_with_annotations += 1
        grid = patches.grid_config(slide)

        # A slide with no feature bank can't contribute anything, so skip it whole.
        # (Not just an optimisation: without this, an annotation covering a whole
        # un-swept slide would report tens of thousands of "missing" cells, which
        # reads like something is broken when the real answer is "extract features".)
        if features.state(slide_id, model_id)["state"] == "none":
            without_features.append(slide_id)
            continue

        # One cell -> one row. The global grid means two overlapping annotations can
        # claim the same cell; if their labels differ, emitting both would hand the
        # head contradictory targets. First annotation in file order wins.
        claimed: dict[tuple[int, int], tuple[str, str]] = {}
        for i, anno in enumerate(annos):
            label = patches.label_of(anno)
            if label == UNLABELLED:
                # An untagged region is a region the user hasn't finished. Training on
                # it would invent a class called "unlabeled". Skip it, but COUNT it, so
                # the UI can say "3 regions have no class yet" instead of silently
                # training on less than was drawn.
                n_unlabelled += 1
                continue
            if patches.parse_annotation(anno) is None:
                # We cannot READ this region's shape (an unsupported selector). Without
                # this branch it would contribute zero cells and vanish from training
                # with no counter and no error — silent data loss, and the user's only
                # clue would be a model that ignores regions they can plainly see.
                n_unparseable += 1
                continue
            group = f"{slide_id}::{anno.get('id') or f'anno{i}'}"
            for cell in patches.cells_in_annotation(grid, anno):
                if cell in claimed:
                    n_overlap += 1
                    continue
                claimed[cell] = (label, group)

        n_annotated += len(claimed)
        vecs = features.vectors(slide_id, model_id, grid, claimed.keys())
        kept = 0
        for cell, (label, group) in claimed.items():
            vec = vecs.get(cell)
            if vec is None:
                # No vector: the cell is background (the mask rejected it), or the
                # slide hasn't been swept here yet.
                n_missing += 1
                continue
            X_parts.append(vec)
            y.append(label)
            groups.append(group)
            kept += 1
        if kept:
            per_slide[slide_id] = kept

    stats = {
        "n_cells_annotated": n_annotated,
        "n_cells_missing": n_missing,          # annotated, but background (or unswept)
        "n_overlap_dropped": n_overlap,
        "n_unlabelled_skipped": n_unlabelled,  # regions drawn but never given a class
        "n_unparseable": n_unparseable,        # regions whose selector we cannot read
        "slides_with_annotations": slides_with_annotations,
        "slides_without_features": without_features,
    }
    if not X_parts:
        return np.zeros((0, 0), np.float32), np.array([]), np.array([]), {}, [], stats

    X = np.vstack(X_parts).astype(np.float32)
    return X, np.array(y), np.array(groups), per_slide, sorted(per_slide), stats


# --- Training ----------------------------------------------------------------

def _make_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def _grouped_oof_predictions(X, y, groups, classes):
    """Out-of-fold predictions from region-grouped CV, or None if not possible.

    Returns (y_pred, validation_info) or (None, validation_info) when there
    aren't enough regions to hold any out.
    """
    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        return None, {"mode": "none", "reason": "need at least 2 regions/slides to validate"}

    n_splits = min(5, n_groups)
    # Prefer stratified grouping (keeps class balance across folds); fall back to
    # plain grouped folds if stratification can't be satisfied on this data.
    for splitter in (StratifiedGroupKFold(n_splits=n_splits), GroupKFold(n_splits=n_splits)):
        try:
            y_pred = cross_val_predict(_make_pipeline(), X, y, groups=groups, cv=splitter)
            return y_pred, {"mode": f"grouped {n_splits}-fold CV", "n_splits": n_splits}
        except Exception:
            continue
    return None, {"mode": "none", "reason": "grouped cross-validation failed on this data"}


def train(project_id: str, model_id: str | None = None) -> dict:
    """Train + persist ONE PROJECT's head for an embedding model. Returns a status dict."""
    if model_id is None:
        model_id = embeddings.active_model_id()

    # Refuse to train into a project that isn't there. This is not just an input check:
    # the persistence step below would otherwise mkdir the project's models/ directory
    # back into existence, so a project deleted while a debounced retrain was in flight
    # would partially come back from the dead. (jobs.py also holds projects.LOCK across
    # this call; this is the belt to that's braces.)
    if projects.project_dir(project_id) is None:
        return {"status": "no_project", "project_id": project_id}

    X, y, groups, per_slide, slides_used, stats = _gather_dataset(project_id, model_id)
    if X.shape[0] == 0:
        # Nothing to train on -> drop any stale head so the app reports "no model"
        # instead of serving classes it can no longer predict. `reason` tells the
        # user which of the four quite different dead-ends they're in.
        delete_head(project_id, model_id)
        if stats["slides_with_annotations"] == 0:
            reason = "no_annotations"
        elif stats["n_cells_annotated"] == 0:
            # Four quite different dead-ends, four different fixes. Distinguish them, or
            # the user is told "annotations too small" when the truth is something else.
            if stats["slides_without_features"]:
                reason = "no_features"
            elif stats["n_unparseable"]:
                reason = "no_shapes"       # we cannot read the regions' selectors
            elif stats["n_unlabelled_skipped"]:
                reason = "no_labels"       # regions drawn, but not one carries a class
            else:
                reason = "no_cells"        # too small to cover a single tile
        else:
            reason = "no_tissue"         # annotated cells exist, but all background
        return {"status": "no_data", "reason": reason, "project_id": project_id, **stats}
    classes = sorted(set(y.tolist()))
    if len(classes) < 2:
        delete_head(project_id, model_id)
        return {"status": "need_2_classes", "classes": classes,
                "project_id": project_id, **stats}

    # Honest metrics from region-grouped out-of-fold CV.
    y_pred, validation = _grouped_oof_predictions(X, y, groups, classes)
    if y_pred is not None:
        report = classification_report(y, y_pred, labels=classes, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, y_pred, labels=classes).tolist()
    else:
        report, cm = None, None

    # The persisted head is trained on ALL annotations (standard practice).
    pipeline = _make_pipeline()
    pipeline.fit(X, y)

    # A class the user annotated but has since removed from the project palette. We
    # still TRAIN on it — the annotations are real data, and silently dropping them
    # would destroy work invisibly — but we report it so the UI can offer to re-add
    # the class rather than render those regions in a fallback colour forever.
    project = projects.load(project_id) or {}
    palette = {c["name"].lower() for c in project.get("classes", [])}
    classes_not_in_project = [c for c in classes if c.lower() not in palette]

    metadata = {
        "project_id": project_id,
        "embedding_model_id": model_id,
        "dim": int(X.shape[1]),
        "classes": classes,
        "classes_not_in_project": classes_not_in_project,
        "n_samples": int(X.shape[0]),
        "n_groups": len(set(groups.tolist())),
        "slides_used": slides_used,
        "per_slide_counts": per_slide,
        "target_mpp": patches.TARGET_MPP,
        "patch_size": patches.PATCH_SIZE,
        "tissue_mask": patches.tissue_mask_name(),
        "validation": validation,
        "metrics": {"report": report, "confusion_matrix": cm, "labels": classes},
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **stats,
    }

    # Persist BOTH files atomically (tmp + os.replace), the same way annotations.py
    # does. Retraining is debounced and runs in a background thread, so a predict can
    # land in the middle of it — dumping straight onto the live path let joblib.load
    # read a half-written file. Note we do NOT mkdir here: the project's models/ dir is
    # created with the project, and re-creating it would resurrect a deleted project.
    _atomic_dump(pipeline, head_path(project_id, model_id))
    _atomic_json(metadata, meta_path(project_id, model_id))

    return {"status": "ok", **metadata}


def _atomic_dump(pipeline, path: Path) -> None:
    tmp = path.with_suffix(".joblib.tmp")
    joblib.dump(pipeline, tmp)
    os.replace(tmp, path)


def _atomic_json(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


# --- Load / inspect (used by GET route + Phase 6) ----------------------------

def head_metadata(project_id: str, model_id: str | None = None) -> dict | None:
    if model_id is None:
        model_id = embeddings.active_model_id()
    path = meta_path(project_id, model_id)
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_head(project_id: str, model_id: str | None = None):
    """Load a project's trained pipeline for inference, or None if not trained yet."""
    if model_id is None:
        model_id = embeddings.active_model_id()
    path = head_path(project_id, model_id)
    if path is None or not path.exists():
        return None
    return joblib.load(path)
