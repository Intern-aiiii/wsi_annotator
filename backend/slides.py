"""Slide I/O and DeepZoom tile serving (Phase 1).

Reads whole slide images (.svs, .ndpi, .mrxs, ...) with OpenSlide and produces
the tiled DeepZoom (DZI) pyramid that OpenSeadragon consumes in the browser.

The heavy lifting is done by OpenSlide's `DeepZoomGenerator`, which takes a WSI
and hands back:
  - an XML ".dzi" descriptor (image size, tile size, tile format), and
  - individual JPEG tiles addressed by (level, column, row).

We keep one `DeepZoomGenerator` per slide in memory so repeated tile requests
don't reopen the file every time.

NOTE: this module needs both the `openslide-python` package AND the OpenSlide
system library. If either is missing the import below fails with a clear error;
see the README for install instructions (`apt install openslide-tools`).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator

# --- Where slides live -------------------------------------------------------
# Repo root = two levels up from this file (backend/slides.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
SLIDES_DIR = REPO_ROOT / "data" / "slides"

# --- DeepZoom tiling parameters ----------------------------------------------
# 254 + an overlap of 1 pixel on each side gives 256-px tiles, the values the
# OpenSlide DeepZoom example uses and that OpenSeadragon handles smoothly.
TILE_SIZE = 254
OVERLAP = 1
TILE_FORMAT = "jpeg"  # what we encode tiles as; must match the .dzi descriptor

# File extensions OpenSlide can typically open. Used only to list slides.
_SLIDE_EXTENSIONS = {".svs", ".ndpi", ".mrxs", ".tif", ".tiff", ".scn", ".svslide", ".bif"}

# Cache of opened DeepZoom generators, keyed by slide_id.
_generators: dict[str, DeepZoomGenerator] = {}


def list_slides() -> list[dict]:
    """Return the slides available in data/slides/ as [{id, name}, ...].

    `id` is the filename without its extension (used in URLs); `name` is the
    original filename (shown to the user).
    """
    if not SLIDES_DIR.exists():
        return []
    slides = []
    for path in sorted(SLIDES_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in _SLIDE_EXTENSIONS:
            slides.append({"id": path.stem, "name": path.name})
    return slides


def _resolve_slide_path(slide_id: str) -> Path | None:
    """Map a slide_id back to a file on disk, or None if there is no match.

    Guards against path traversal: slide_id must be a plain name, and the
    resolved file must sit directly inside SLIDES_DIR.
    """
    if not slide_id or "/" in slide_id or "\\" in slide_id or slide_id.startswith("."):
        return None
    for path in SLIDES_DIR.iterdir():
        if path.is_file() and path.stem == slide_id and path.suffix.lower() in _SLIDE_EXTENSIONS:
            # Extra safety: ensure it is really inside SLIDES_DIR.
            if path.resolve().parent == SLIDES_DIR.resolve():
                return path
    return None


def _get_generator(slide_id: str) -> DeepZoomGenerator | None:
    """Return (opening and caching if needed) the DeepZoomGenerator for a slide."""
    if slide_id in _generators:
        return _generators[slide_id]
    path = _resolve_slide_path(slide_id)
    if path is None:
        return None
    slide = OpenSlide(str(path))
    generator = DeepZoomGenerator(
        slide, tile_size=TILE_SIZE, overlap=OVERLAP, limit_bounds=True
    )
    _generators[slide_id] = generator
    return generator


def get_dzi(slide_id: str) -> str | None:
    """Return the DeepZoom XML descriptor for a slide, or None if not found."""
    generator = _get_generator(slide_id)
    if generator is None:
        return None
    return generator.get_dzi(TILE_FORMAT)


def get_tile(slide_id: str, level: int, col: int, row: int) -> bytes | None:
    """Return one JPEG tile as bytes, or None if the slide/tile does not exist.

    `level`, `col`, `row` are DeepZoom coordinates as requested by
    OpenSeadragon (they do NOT map 1:1 to OpenSlide pyramid levels — the
    generator handles that translation).
    """
    generator = _get_generator(slide_id)
    if generator is None:
        return None
    try:
        tile = generator.get_tile(level, (col, row))
    except (ValueError, IndexError):
        # Out-of-range level/column/row: treat as "no such tile".
        return None
    buffer = BytesIO()
    tile.save(buffer, format=TILE_FORMAT)
    return buffer.getvalue()
