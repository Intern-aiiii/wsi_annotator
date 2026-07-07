"""Data models: projects, slides, annotations, classes.

Plain data structures (e.g. Pydantic models) shared across the backend and sent
to/from the frontend as JSON. Keeping them in one place avoids mismatched shapes
between modules.

Sketch of the core entities:
  - Project   : a named workspace grouping slides, classes, and a trained head.
  - Slide     : a WSI on disk + its metadata (dimensions, levels, MPP).
  - Class     : a label the user is training for (e.g. "gland", "not-gland").
  - Annotation: a drawn region (W3C annotation JSON) tied to a slide + class.

This grows as phases 2, 5, and 7 need more structure.
"""

from __future__ import annotations

from pydantic import BaseModel


class AnnotationCollection(BaseModel):
    """The full set of annotations for one slide (Phase 2).

    We keep each annotation as a permissive ``dict`` rather than modeling every
    W3C WebAnnotation field: the frontend (Annotorious) is the source of truth
    for that shape, and storing it as-is keeps the region geometry and the class
    label (a ``tagging`` body) intact for Phases 3 and 5 to read later.
    """

    annotations: list[dict] = []
