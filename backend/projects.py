"""Projects: named workspaces that own annotations, classes, and a trained head (Phase 7).

A PROJECT is one experiment. Two projects over the same slides train independently
and predict differently, so "glands vs stroma" and "tumor vs normal" can coexist
instead of contaminating a single pooled classifier (which is what happened before
this module existed: one global head, trained on every slide on disk).

WHAT A PROJECT OWNS — and, just as importantly, what it does NOT:

    data/projects/<project_id>/
        project.json                 # name, slides, classes
        annotations/<slide_id>.json  # the regions drawn IN THIS PROJECT
        models/head__<embed_id>.*    # the head trained on THIS PROJECT's annotations

    NOT owned (deliberately global, shared by every project):
        data/slides/           the WSI files themselves
        data/cache/features/   THE FEATURE BANK
        data/cache/            DeepZoom tiles

The feature bank is the one that matters. It is keyed by (slide, embedding model,
tissue mask) — nothing about a user's experiment enters it — so it is a derived
property of a FILE, not of a project. Sharing it means a slide swept once with
Virchow 2 (minutes of GPU) is reused by every project that contains it. Scoping it
per project would re-run the single most expensive step in the whole pipeline, once
per project, for zero gain.

TWO INVARIANTS. Both are load-bearing; violating either reintroduces a bug class we
specifically designed away.

  1. `project.json["slides"]` is a DISPLAY LIST. It decides what the slide picker
     shows. It NEVER decides what the model trains on — training iterates this
     project's annotations/ directory (see `annotated_slides`). Without this rule you
     get "I removed the slide from the project but the head still knows about it".

  2. Annotation tags are the SOURCE OF TRUTH for training; `project.json["classes"]`
     is only a vocabulary + colour registry. So deleting a class is purely cosmetic:
     its annotations keep their tag and keep training, and the head reports them as
     `classes_not_in_project` so the UI can offer to re-add the class.

     Corollary: there is deliberately NO class rename. A label is a free-text string
     inside every annotation that uses it, so renaming would mean rewriting every
     annotation file in the project — a migration inside a migration. Add / remove /
     recolour only.

The project id is a slug of the name ("Glands vs stroma" -> "glands-vs-stroma") so
that `ls data/projects/` is readable. THE ID IS FROZEN AT CREATION: renaming changes
only the display name, never the id or the directory.

This module imports ONLY `slides` (for REPO_ROOT and the traversal-guard pattern). It
must never import classifier / jobs / features / annotations — that is what keeps it
out of the existing features -> jobs -> classifier -> features import cycle.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path

from backend import slides

PROJECTS_DIR = slides.REPO_ROOT / "data" / "projects"

# Serialises "delete a project" against "train a project". Without it, a train that
# is already in flight can re-create the directory tree it is writing into (joblib
# needs the dir to exist), and a project you just deleted partially comes back from
# the dead. jobs.py holds this across a training pass; delete() takes it too.
# Re-entrant because delete() calls other functions in this module that also take it.
LOCK = threading.RLock()

# The class vocabulary a NEW project starts with. Lifted from the frontend, which
# used to own this list (annotate.js PRESET_CLASSES + CLASS_PALETTE). It lives here
# now because a class carries an explicit stored COLOUR, and colour has to survive a
# browser cache clear.
DEFAULT_CLASSES = [
    "gland",
    "epithelium",
    "stroma",
    "tumor",
    "necrosis",
    "lymphocytes",
    "blood vessel",
    "adipose",
    "background",
]

# Distinct and reasonably colourblind-friendly. A new class takes the first colour
# NOT already in use (see `_next_color`) rather than the one at its index — indexing
# by position meant deleting a class silently repainted every class after it.
DEFAULT_PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#9A6324", "#800000", "#808000", "#000075", "#a9a9a9",
]

_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


# --- Ids and paths -----------------------------------------------------------

def _slug(name: str) -> str:
    """Turn a display name into a filesystem- and URL-safe id."""
    s = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
    return s or "project"


def _safe_dir(project_id: str) -> Path | None:
    """The directory a project id MAPS TO, whether or not it exists yet.

    THE TRAVERSAL GUARD. `delete()` runs shutil.rmtree on the path this returns, and
    the id arrives from a URL path segment — so this function is the only thing
    standing between a malformed request and an arbitrary directory on disk. It
    mirrors slides._resolve_slide_path: reject anything that isn't a plain name, then
    prove the resolved path sits DIRECTLY inside PROJECTS_DIR.
    """
    if not project_id or "/" in project_id or "\\" in project_id or project_id.startswith("."):
        return None
    path = PROJECTS_DIR / project_id
    # `resolve()` collapses any "..", so this catches traversal even if the checks
    # above were somehow bypassed. A symlinked project dir also fails here.
    if path.resolve().parent != PROJECTS_DIR.resolve():
        return None
    return path


def project_dir(project_id: str) -> Path | None:
    """The directory of an EXISTING project, or None (bad id, or no such project)."""
    path = _safe_dir(project_id)
    if path is None or not (path / "project.json").exists():
        return None
    return path


def annotations_dir(project_id: str) -> Path | None:
    path = project_dir(project_id)
    return None if path is None else path / "annotations"


def models_dir(project_id: str) -> Path | None:
    path = project_dir(project_id)
    return None if path is None else path / "models"


# --- Read --------------------------------------------------------------------

def load(project_id: str) -> dict | None:
    """Parse a project's project.json, or None if there is no such project.

    Read straight from disk every time. These files are ~1 KB and the app is local;
    an mtime-checked cache would be a bug factory for no measurable gain.
    """
    path = project_dir(project_id)
    if path is None:
        return None
    try:
        with (path / "project.json").open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def list_projects() -> list[dict]:
    """Every project on disk, oldest first. Summary shape, for the picker."""
    if not PROJECTS_DIR.exists():
        return []
    out = []
    for path in sorted(PROJECTS_DIR.iterdir()):
        if not path.is_dir():
            continue
        project = load(path.name)
        if project is None:
            continue
        out.append({
            "id": project["id"],
            "name": project["name"],
            "created_at": project.get("created_at", ""),
            "n_slides": len(project.get("slides", [])),
            "n_classes": len(project.get("classes", [])),
        })
    out.sort(key=lambda p: p["created_at"])
    return out


def projects_with_slide(slide_id: str) -> list[str]:
    """Ids of the projects containing `slide_id`. Used by the end-of-sweep retrain hook:
    fresh vectors for a slide invalidate the head of EVERY project that uses it."""
    return [p["id"] for p in list_projects()
            if slide_id in (load(p["id"]) or {}).get("slides", [])]


def annotated_slides(project_id: str) -> list[str]:
    """The slides this project has an annotation file for — i.e. WHAT TRAINING READS.

    Deliberately NOT project["slides"] (invariant 1): membership is a display list.
    The training set is defined by the annotation files that actually exist, so adding
    or removing a slide from the picker can never silently change what the head learnt.
    """
    path = annotations_dir(project_id)
    if path is None or not path.exists():
        return []
    return sorted(p.stem for p in path.iterdir() if p.suffix == ".json")


def has_annotations(project_id: str, slide_id: str) -> bool:
    """True if this project holds a NON-EMPTY annotation file for the slide."""
    path = annotations_dir(project_id)
    if path is None:
        return False
    file = path / f"{slide_id}.json"
    if not file.exists():
        return False
    try:
        with file.open("r", encoding="utf-8") as fh:
            return len(json.load(fh)) > 0
    except (json.JSONDecodeError, OSError):
        return False


# --- Write -------------------------------------------------------------------

def save(project: dict) -> dict:
    """Overwrite a project's project.json atomically.

    Same tmp + os.replace dance as annotations.py: an interrupted write must never
    leave a half-written project.json, because that would read back as "no such
    project" and hide the user's annotations.
    """
    path = _safe_dir(project["id"])
    if path is None:
        raise ValueError(f"unsafe project id {project['id']!r}")
    path.mkdir(parents=True, exist_ok=True)

    target = path / "project.json"
    tmp = target.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(project, fh, indent=2)
    os.replace(tmp, target)
    return project


def _next_color(classes: list[dict]) -> str:
    """The first palette colour not already taken, so adding a class never re-tints
    an existing one. Wraps once the palette is exhausted."""
    used = {c.get("color", "").lower() for c in classes}
    for color in DEFAULT_PALETTE:
        if color.lower() not in used:
            return color
    return DEFAULT_PALETTE[len(classes) % len(DEFAULT_PALETTE)]


def _clean_classes(raw: list) -> list[dict]:
    """Validate + normalise a class list. Raises ValueError on bad input.

    Names are unique case-insensitively (they're matched against annotation tags,
    which are free text); colours must be #rrggbb. A class with no colour gets the
    next unused palette entry.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for entry in raw:
        if isinstance(entry, str):
            entry = {"name": entry}
        if not isinstance(entry, dict):
            raise ValueError(f"class must be a name or {{name, color}}, got {entry!r}")

        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError("class name cannot be empty")
        if name.lower() in seen:
            raise ValueError(f"duplicate class name {name!r}")
        seen.add(name.lower())

        color = str(entry.get("color", "")).strip()
        if not color:
            color = _next_color(out)
        elif not _COLOR_RE.match(color):
            raise ValueError(f"colour must be #rrggbb, got {color!r}")

        out.append({"name": name, "color": color})
    return out


def create(name: str, slide_ids: list[str] | None = None,
           classes: list | None = None) -> dict:
    """Create a project. Ids collide-suffix (-2, -3) and are frozen thereafter.

    `slide_ids` defaults to every slide on disk — the common case is one experiment
    over everything you have, and it means a fresh project never starts with orphaned
    slides that are invisible in the picker.
    """
    with LOCK:
        base = _slug(name)
        project_id, n = base, 1
        while (PROJECTS_DIR / project_id).exists():
            n += 1
            project_id = f"{base}-{n}"

        if slide_ids is None:
            slide_ids = [s["id"] for s in slides.list_slides()]

        project = {
            "id": project_id,
            "name": str(name).strip() or project_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "slides": list(slide_ids),
            "classes": _clean_classes(classes if classes is not None else DEFAULT_CLASSES),
        }
        save(project)
        # Create both subdirectories up front so every other module can assume they
        # exist and never has to mkdir into a project (which is how a deleted project
        # comes back from the dead — see LOCK).
        (PROJECTS_DIR / project_id / "annotations").mkdir(parents=True, exist_ok=True)
        (PROJECTS_DIR / project_id / "models").mkdir(parents=True, exist_ok=True)
        return project


def rename(project_id: str, name: str) -> dict | None:
    """Change the DISPLAY NAME only. The id and directory never move — every stored
    URL, every localStorage key, and every path on disk stays valid."""
    with LOCK:
        project = load(project_id)
        if project is None:
            return None
        new_name = str(name).strip()
        if not new_name:
            raise ValueError("project name cannot be empty")
        project["name"] = new_name
        return save(project)


def delete(project_id: str) -> dict | None:
    """Remove a project and EVERYTHING IT OWNS — and nothing else.

    Gone:  data/projects/<pid>/  (project.json, its annotations, its heads)
    Kept:  data/slides/**        the WSIs
           data/cache/**         the shared feature bank + DeepZoom tiles

    Returns a receipt of what was removed, so a caller (or a test) can assert on it.
    """
    with LOCK:
        path = project_dir(project_id)
        if path is None:
            return None

        n_annotations = len(annotated_slides(project_id))
        models = models_dir(project_id)
        n_heads = len(list(models.glob("head__*.joblib"))) if models.exists() else 0

        # Belt and braces before an rmtree driven by a URL path segment. project_dir()
        # already proved this, but the cost of being wrong here is somebody's home
        # directory, so we prove it again immediately before the destructive call.
        resolved = path.resolve()
        if resolved.parent != PROJECTS_DIR.resolve():
            raise ValueError(f"refusing to delete {resolved} — outside {PROJECTS_DIR}")
        shutil.rmtree(resolved)

        return {"deleted": project_id,
                "removed": {"annotations": n_annotations, "heads": n_heads}}


def set_classes(project_id: str, classes: list) -> dict | None:
    """Replace the whole class list (same 'send the whole collection' convention as
    annotations — one convention, learned once). Raises ValueError on bad input."""
    with LOCK:
        project = load(project_id)
        if project is None:
            return None
        project["classes"] = _clean_classes(classes)
        return save(project)


def add_slide(project_id: str, slide_id: str) -> dict | None:
    """Put a slide in this project's picker. No-op if it's already there."""
    with LOCK:
        project = load(project_id)
        if project is None:
            return None
        if slide_id not in project["slides"]:
            project["slides"].append(slide_id)
            save(project)
        return project


def migrate_legacy() -> dict:
    """One-time import of the pre-Phase-7 layout into a project called "default".

    Before projects existed, annotations lived in a flat data/annotations/ and the one
    global head in data/models/. This lifts them into data/projects/default/ so an
    existing install keeps its work and behaves exactly as it did (the default project
    contains every slide on disk, which is what the old global head trained on).

    COPY, NEVER MOVE — and guarded by a SENTINEL FILE, not by "does data/projects/
    exist?". That distinction is the whole design:

        The naive trigger resurrects data. Create projects, delete them all, restart:
        data/projects/ is empty again, so the migration re-runs and the annotations you
        deliberately deleted come back from the dead.

    The sentinel is written LAST, so a half-finished migration simply runs again. And
    because we copy, data/annotations/ and data/models/ survive untouched as a frozen
    backup — which also makes `rm -rf data/projects/` a clean, complete "start over".

    Idempotent. Safe to call on every boot. Returns a small report for the log.
    """
    with LOCK:
        sentinel = PROJECTS_DIR / ".legacy_imported"
        if sentinel.exists():
            return {"status": "already_imported"}

        legacy_annotations = slides.REPO_ROOT / "data" / "annotations"
        legacy_models = slides.REPO_ROOT / "data" / "models"

        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        project = load("default") or create("Default")

        copied_annotations = copied_heads = 0
        if legacy_annotations.exists():
            target = PROJECTS_DIR / project["id"] / "annotations"
            target.mkdir(parents=True, exist_ok=True)
            for src in sorted(legacy_annotations.glob("*.json")):
                dst = target / src.name
                if dst.exists():
                    continue          # never overwrite work already in the project
                shutil.copy2(src, dst)
                copied_annotations += 1

        if legacy_models.exists():
            target = PROJECTS_DIR / project["id"] / "models"
            target.mkdir(parents=True, exist_ok=True)
            for src in sorted(legacy_models.glob("head__*")):
                dst = target / src.name
                if dst.exists():
                    continue
                shutil.copy2(src, dst)
                copied_heads += 1

        sentinel.write_text(
            f"legacy data/annotations + data/models imported into "
            f"'{project['id']}' at {datetime.now().isoformat(timespec='seconds')}\n"
            "Delete data/projects/ entirely to re-run this import.\n",
            encoding="utf-8",
        )
        return {"status": "imported", "project_id": project["id"],
                "annotations": copied_annotations, "heads": copied_heads}


def remove_slide(project_id: str, slide_id: str, force: bool = False) -> dict | None:
    """Take a slide out of this project's picker.

    Returns {"status": "has_annotations", ...} instead of acting when the project
    still holds annotations for it and `force` is false — a mis-click must not be
    able to silently destroy drawn work. With force=True we delete THIS PROJECT'S
    annotation file for the slide and nothing else: never the slide, never the bank,
    never another project's annotations.
    """
    with LOCK:
        project = load(project_id)
        if project is None:
            return None

        if has_annotations(project_id, slide_id) and not force:
            return {"status": "has_annotations", "slide_id": slide_id}

        if force:
            annos = annotations_dir(project_id)
            if annos is not None:
                (annos / f"{slide_id}.json").unlink(missing_ok=True)

        project["slides"] = [s for s in project["slides"] if s != slide_id]
        save(project)
        return {"status": "ok", "project": project}
