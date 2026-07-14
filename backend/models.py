"""Data models: projects, slides, annotations, classes.

Plain data structures (e.g. Pydantic models) shared across the backend and sent
to/from the frontend as JSON. Keeping them in one place avoids mismatched shapes
between modules.

The core entities:
  - Project   : a named workspace grouping slides, classes, and a trained head.
  - Slide     : a WSI on disk + its metadata. NOT modelled here — a slide is just a
                file, discovered by scanning data/slides/ (see slides.list_slides).
  - Class     : a label the user is training for (e.g. "gland", "not-gland"), with a
                colour. Stored inside project.json; see projects.py.
  - Annotation: a drawn region (W3C annotation JSON) tied to a project + slide + class.

Only the shapes the frontend POSTs/PUTs need to be modelled — a response is built as a
plain dict by the module that owns the data.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictRegion(BaseModel):
    """A viewport region to score (Phase 6), in DZI-image pixels.

    `class` is the (optional) label whose confident regions to outline; it's
    aliased because `class` is a Python keyword.
    """

    model_config = ConfigDict(populate_by_name=True)

    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0
    target_class: str | None = Field(default=None, alias="class")


class SimilarityRegion(BaseModel):
    """A viewport region + a reference annotation (unsupervised similarity map).

    {x,y,w,h} is the visible region to score (DZI-image pixels). `annotation` is
    the full W3C annotation object the user selected — the backend reads its
    geometry and uses the mean embedding of the tissue tiles inside it as the
    reference, then colours the visible region by similarity to that reference.

    We send the whole annotation (not just an id) so a freshly-drawn region works
    even before it has been persisted server-side.
    """

    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0
    annotation: dict


class AnnotationCollection(BaseModel):
    """The full set of annotations for one slide (Phase 2).

    We keep each annotation as a permissive ``dict`` rather than modeling every
    W3C WebAnnotation field: the frontend (Annotorious) is the source of truth
    for that shape, and storing it as-is keeps the region geometry and the class
    label (a ``tagging`` body) intact for Phases 3 and 5 to read later.
    """

    annotations: list[dict] = []


# --- Projects (Phase 7) ------------------------------------------------------

class ProjectCreate(BaseModel):
    """A new workspace. `slides` defaults to None = "every slide on disk"."""

    name: str
    slides: list[str] | None = None


class ProjectRename(BaseModel):
    """Changes the DISPLAY NAME only — the id and its directory are frozen at creation."""

    name: str


class ClassDef(BaseModel):
    """One label the user trains for, plus the colour its regions are drawn in.

    The colour is stored EXPLICITLY (rather than derived from the class's position in
    the list, as the browser used to do) so that deleting a class cannot silently
    repaint every class after it.
    """

    name: str
    color: str = ""          # #rrggbb; blank => the server assigns the next unused


class ClassList(BaseModel):
    """The whole class list, replaced in one shot — same "send the whole collection"
    convention as annotations, so there is one convention to learn, not two."""

    classes: list[ClassDef] = []


class SlideRef(BaseModel):
    """Names a slide to add to a project's picker."""

    slide_id: str
