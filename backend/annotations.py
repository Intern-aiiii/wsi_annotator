"""Annotation persistence (Phase 2; project-scoped in Phase 7).

Stores the regions a user draws on a slide. Each (project, slide) pair gets ONE JSON
file at ``data/projects/<project_id>/annotations/<slide_id>.json`` holding the full
list of annotations, in W3C WebAnnotation format (the shape Annotorious emits in the
browser). The frontend saves the whole collection on every change, so this module
only has to load and overwrite that list — no per-annotation bookkeeping.

The path is keyed by PROJECT as well as slide (Phase 7): the same slide annotated in
two projects has two independent files, which is what lets two experiments coexist
over the same tissue. Before Phase 7 these lived in a flat data/annotations/ and every
project shared them, so there was only ever one experiment.

Why W3C JSON is stored as-is: it already carries both the region geometry (in
image-pixel coordinates) and the class label (a ``tagging`` body), which is exactly
what patch geometry (patches.py) and training (classifier.py) read. We avoid
remodeling every W3C field so we stay robust to Annotorious's exact shape.

Security: a path is built only when BOTH the project and the slide really exist. Those
checks go through projects.project_dir and slides._resolve_slide_path, which each
reject path traversal, so neither id can steer the filename out of the project.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from backend import projects, slides


def annotations_path(project_id: str, slide_id: str) -> Path | None:
    """The JSON path for one project's annotations on one slide, or None for a bad id.

    Two guards, and BOTH must pass:
      - projects.project_dir  — rejects traversal, and proves the project exists
      - slides._resolve_slide_path — rejects traversal, and proves the slide exists
    By the time we join them, each id is a safe plain filename.
    """
    annos = projects.annotations_dir(project_id)
    if annos is None:
        return None
    if slides._resolve_slide_path(slide_id) is None:
        return None
    return annos / f"{slide_id}.json"


def load(project_id: str, slide_id: str) -> list[dict]:
    """Return this project's annotations for a slide, or [] if there are none yet."""
    path = annotations_path(project_id, slide_id)
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save(project_id: str, slide_id: str, annotations: list[dict]) -> int:
    """Overwrite this project's annotations for a slide; return the count.

    Writes atomically: we serialize to a temp file in the same directory and then
    ``os.replace`` it into place, so an interrupted write can never leave a
    half-written (corrupt) JSON file behind.
    """
    path = annotations_path(project_id, slide_id)
    if path is None:
        raise ValueError(f"unknown project '{project_id}' or slide '{slide_id}'")
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(annotations, fh, indent=2)
    os.replace(tmp, path)
    return len(annotations)
