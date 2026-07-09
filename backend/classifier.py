"""Train / persist / load the classifier head (Phase 5).

The "head" is a lightweight classifier trained on the frozen embeddings — it
never sees pixels, only vectors + labels. This is the crux of the approach: the
foundation model already did the hard visual representation learning, so a simple
linear probe on its embeddings separates the annotated structures from a handful
of examples.

Design (per CLAUDE.md's Phase 5 detail):
  - Assemble (X = embeddings, y = labels) by pooling EVERY slide's cache for the
    active embedding model, so the head generalizes across slides.
  - Split by region (NOT random patch) to avoid leakage between adjacent tiles:
    grouped out-of-fold cross-validation keyed by each patch's `group`.
  - class_weight="balanced" for imbalance; StandardScaler + LogisticRegression.
  - Report per-class precision/recall/F1 + confusion matrix (accuracy misleads
    under imbalance).
  - Persist the head with joblib alongside the embedding config that produced it
    (embedding model_id, dim, magnification) so inference uses matching features.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend import embeddings, slides

MODELS_DIR = slides.REPO_ROOT / "data" / "models"


def _head_stem(model_id: str) -> Path:
    return MODELS_DIR / f"head__{model_id}"


def head_path(model_id: str) -> Path:
    return _head_stem(model_id).with_suffix(".joblib")


def meta_path(model_id: str) -> Path:
    return _head_stem(model_id).with_suffix(".json")


def delete_head(model_id: str) -> None:
    """Remove a persisted head + metadata (if present) so the app reports no model."""
    head_path(model_id).unlink(missing_ok=True)
    meta_path(model_id).unlink(missing_ok=True)


# --- Dataset assembly --------------------------------------------------------

def _gather_dataset(model_id: str):
    """Pool every slide's cached embeddings for `model_id`.

    Returns (X (N,DIM) float32, y (N,) str, groups (N,) str, per_slide_counts,
    slides_used). Skips any cache whose matrix and index disagree in length.
    """
    X_parts, y, groups = [], [], []
    per_slide: dict[str, int] = {}
    suffix = f"__{model_id}.json"

    for index_file in sorted(embeddings.EMBEDDINGS_DIR.glob(f"*__{model_id}.json")):
        slide_id = index_file.name[: -len(suffix)]
        matrix_file = embeddings._matrix_path(slide_id, model_id)
        if not matrix_file.exists():
            continue
        try:
            matrix = np.load(matrix_file)
            with index_file.open("r", encoding="utf-8") as fh:
                index = json.load(fh)
        except Exception:
            continue
        rows = index.get("rows", [])
        if matrix.shape[0] != len(rows) or not rows:
            continue

        X_parts.append(matrix.astype(np.float32))
        for r in rows:
            y.append(r["label"])
            groups.append(r.get("group", slide_id))
        per_slide[slide_id] = len(rows)

    if not X_parts:
        return np.zeros((0, 0), np.float32), np.array([]), np.array([]), {}, []

    X = np.vstack(X_parts)
    return X, np.array(y), np.array(groups), per_slide, sorted(per_slide)


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


def train(model_id: str | None = None) -> dict:
    """Train + persist the head for an embedding model. Returns a status dict."""
    if model_id is None:
        model_id = embeddings.active_model_id()

    X, y, groups, per_slide, slides_used = _gather_dataset(model_id)
    if X.shape[0] == 0:
        # Nothing left to train on -> drop any stale head so the app reports "no
        # model" instead of serving old classes.
        delete_head(model_id)
        return {"status": "no_data"}
    classes = sorted(set(y.tolist()))
    if len(classes) < 2:
        delete_head(model_id)
        return {"status": "need_2_classes", "classes": classes}

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

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, head_path(model_id))

    # Pull embedding config (magnification etc.) from any one slide's index.
    dim = int(X.shape[1])
    target_mpp = patch_size = None
    if slides_used:
        idx = embeddings.load_index(slides_used[0]) or {}
        target_mpp, patch_size = idx.get("target_mpp"), idx.get("patch_size")

    metadata = {
        "embedding_model_id": model_id,
        "dim": dim,
        "classes": classes,
        "n_samples": int(X.shape[0]),
        "n_groups": len(set(groups.tolist())),
        "slides_used": slides_used,
        "per_slide_counts": per_slide,
        "target_mpp": target_mpp,
        "patch_size": patch_size,
        "validation": validation,
        "metrics": {"report": report, "confusion_matrix": cm, "labels": classes},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with meta_path(model_id).open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return {"status": "ok", **metadata}


# --- Load / inspect (used by GET route + Phase 6) ----------------------------

def head_metadata(model_id: str | None = None) -> dict | None:
    if model_id is None:
        model_id = embeddings.active_model_id()
    path = meta_path(model_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_head(model_id: str | None = None):
    """Load the trained pipeline for inference, or None if not trained yet."""
    if model_id is None:
        model_id = embeddings.active_model_id()
    path = head_path(model_id)
    if not path.exists():
        return None
    return joblib.load(path)
