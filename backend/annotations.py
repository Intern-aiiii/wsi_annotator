"""Annotation persistence (Phase 2).

Stores the regions a user draws on a slide. Each slide gets ONE JSON file under
``data/annotations/<slide_id>.json`` holding the full list of annotations for
that slide, in W3C WebAnnotation format (the shape Annotorious emits in the
browser). The frontend saves the whole collection on every change, so this
module only has to load and overwrite that list — no per-annotation bookkeeping.

Why W3C JSON is stored as-is: it already carries both the region geometry (in
image-pixel coordinates) and the class label (a ``tagging`` body), which is
exactly what Phase 3 (patch extraction) and Phase 5 (training) will read. We
avoid remodeling every W3C field so we stay robust to Annotorious's exact shape.

Security: annotations are only ever read/written for a slide that actually
exists. We reuse ``slides._resolve_slide_path`` — which already guards against
path traversal — so an attacker can't steer the filename outside the
annotations directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from backend import slides

# Annotations live next to the other data, alongside slides/ and cache/.
ANNOTATIONS_DIR = slides.REPO_ROOT / "data" / "annotations"


def annotations_path(slide_id: str) -> Path | None:
    """Return the JSON path for a slide's annotations, or None for a bad id.

    Returns a path only when ``slide_id`` names a real slide on disk. Because
    that check goes through ``slides._resolve_slide_path`` (which rejects path
    traversal and anything outside the slides directory), the id is a safe,
    plain filename by the time we build the annotations path from it.
    """
    if slides._resolve_slide_path(slide_id) is None:
        return None
    return ANNOTATIONS_DIR / f"{slide_id}.json"


def load(slide_id: str) -> list[dict]:
    """Return the saved annotations for a slide, or [] if none exist yet."""
    path = annotations_path(slide_id)
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save(slide_id: str, annotations: list[dict]) -> int:
    """Overwrite a slide's annotations with ``annotations``; return the count.

    Writes atomically: we serialize to a temp file in the same directory and
    then ``os.replace`` it into place, so an interrupted write can never leave a
    half-written (corrupt) JSON file behind.
    """
    path = annotations_path(slide_id)
    if path is None:
        raise ValueError(f"unknown slide '{slide_id}'")
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(annotations, fh, indent=2)
    os.replace(tmp, path)
    return len(annotations)
