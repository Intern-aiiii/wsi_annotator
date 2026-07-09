"""Patch extraction from annotations + tissue masking (Phase 3).

Turns the user's labelled regions into the training material the ML pipeline
needs: for each annotation we cut 224x224 tiles (the input size Virchow 2
expects) from the slide with OpenSlide at the model's target magnification, drop
background/white tiles with a simple tissue mask, and record each kept tile's
coordinates + class label.

The output is a per-slide **manifest** (JSON) plus a **preview montage** (one
stitched image) so the extraction can be eyeballed. The manifest is exactly what
Phase 4 (embeddings) and Phase 5 (training) consume — no model is involved here.

Deliberately dependency-light: only PIL + OpenSlide + stdlib. Tissue masking uses
PIL's saturation channel; point-in-polygon is pure Python. numpy/scipy/shapely
are intentionally avoided at this phase.

Coordinate note: annotations are stored in the coordinate space of the DeepZoom
base image. Because the DZI is built with limit_bounds=True (see slides.py), that
space starts at the slide's bounds origin, so we offset annotation coords by
(bounds-x, bounds-y) before calling OpenSlide's read_region (which works in
full level-0 coordinates). For slides with a zero bounds origin (like the sample)
the offset is (0, 0).
"""

from __future__ import annotations

import json
import os
import re
from math import ceil
from pathlib import Path

import openslide
from PIL import Image

from backend import annotations, slides

# --- Extraction parameters ---------------------------------------------------
PATCH_SIZE = 224          # pixels per side; the input size Virchow 2 expects
TARGET_MPP = 0.5          # microns/pixel (~20x). Matches Virchow 2 and is stored
                          # in the manifest so Phase 4 uses matching features.

# Tissue mask: a patch is kept if at least MIN_TISSUE_FRACTION of its pixels have
# saturation >= SAT_THRESHOLD. White/grey background is near-zero saturation;
# H&E-stained tissue is well above it. Both are easy to tune later.
SAT_THRESHOLD = 25        # 0..255 saturation cutoff
MIN_TISSUE_FRACTION = 0.30

# Where manifests + montages are cached (gitignored).
PATCHES_DIR = slides.REPO_ROOT / "data" / "cache" / "patches"

# How many patches to show in the preview montage, and the thumbnail size.
MONTAGE_MAX = 64
MONTAGE_COLS = 8
MONTAGE_THUMB = 96


# --- Slide geometry helpers --------------------------------------------------

def _slide_mpp(slide: openslide.OpenSlide) -> float:
    """Microns-per-pixel at level 0. Falls back to objective power, then 0.5."""
    val = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if val:
        try:
            return float(val)
        except ValueError:
            pass
    # Fall back to objective magnification (40x ~= 0.25, 20x ~= 0.5 µm/px).
    power = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if power:
        try:
            return 10.0 / float(power)
        except ValueError:
            pass
    return 0.5  # last-resort default; documented so results stay interpretable


def _bounds_offset(slide: openslide.OpenSlide) -> tuple[int, int]:
    """(bounds-x, bounds-y) in level-0 pixels; (0, 0) if the slide has no bounds."""
    bx = slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_X)
    by = slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y)
    return (int(bx) if bx else 0, int(by) if by else 0)


# --- Annotation parsing ------------------------------------------------------

def _label_of(anno: dict) -> str:
    """The class label = the annotation's `tagging` body value, or 'unlabeled'."""
    body = anno.get("body")
    items = body if isinstance(body, list) else ([body] if body else [])
    for b in items:
        if isinstance(b, dict) and b.get("purpose") == "tagging" and b.get("value"):
            return str(b["value"])
    return "unlabeled"


def _parse_annotation(anno: dict):
    """Return (label, kind, geometry) or None if the selector isn't understood.

    kind == "rect"    -> geometry is (x, y, w, h)
    kind == "polygon" -> geometry is [(x1, y1), (x2, y2), ...]
    Coordinates are in the DeepZoom base-image space (see module docstring).
    """
    label = _label_of(anno)
    selector = (anno.get("target") or {}).get("selector") or {}
    stype = selector.get("type")
    value = selector.get("value") or ""

    if stype == "FragmentSelector":
        m = re.search(r"xywh=(?:pixel:)?([0-9.]+),([0-9.]+),([0-9.]+),([0-9.]+)", value)
        if m:
            x, y, w, h = (float(g) for g in m.groups())
            return label, "rect", (x, y, w, h)

    if stype == "SvgSelector":
        pm = re.search(r'points="([^"]+)"', value)
        if pm:
            pairs = re.findall(r"(-?[0-9.]+)[ ,]+(-?[0-9.]+)", pm.group(1))
            pts = [(float(a), float(b)) for a, b in pairs]
            if len(pts) >= 3:
                return label, "polygon", pts
        # Be tolerant of an <rect> expressed as SVG too.
        rm = re.search(
            r'<rect[^>]*\bx="([0-9.]+)"[^>]*\by="([0-9.]+)"'
            r'[^>]*\bwidth="([0-9.]+)"[^>]*\bheight="([0-9.]+)"',
            value,
        )
        if rm:
            x, y, w, h = (float(g) for g in rm.groups())
            return label, "rect", (x, y, w, h)

    return None


def _bbox(kind: str, geom) -> tuple[float, float, float, float]:
    """Bounding box (x, y, w, h) of a parsed geometry."""
    if kind == "rect":
        return geom
    xs = [p[0] for p in geom]
    ys = [p[1] for p in geom]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def _point_in_polygon(x: float, y: float, pts) -> bool:
    """Ray-casting point-in-polygon test (no external deps)."""
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


# --- Pixel reading + tissue mask ---------------------------------------------

def _read_patch(slide, x0: int, y0: int, size0: int, level: int) -> Image.Image:
    """Read a size0-square region at level-0 coords (x0, y0) and return 224x224 RGB.

    Reads from the pyramid level best matching the requested downsample, then
    resizes to PATCH_SIZE. read_region always takes level-0 coordinates.
    """
    ds = slide.level_downsamples[level]
    read_size = max(1, round(size0 / ds))
    region = slide.read_region((int(x0), int(y0)), level, (read_size, read_size))
    # region is RGBA; composite onto white to flatten transparency, then RGB.
    rgb = Image.new("RGB", region.size, (255, 255, 255))
    rgb.paste(region, mask=region.split()[-1])
    if rgb.size != (PATCH_SIZE, PATCH_SIZE):
        rgb = rgb.resize((PATCH_SIZE, PATCH_SIZE))
    return rgb


def read_patch(slide, x: int, y: int, size: int, level: int) -> Image.Image:
    """Public wrapper: read one manifest patch as a 224x224 RGB image.

    Phase 4 (embeddings) reuses this to fetch patch pixels from a manifest entry
    ({x, y, size, level}) without duplicating the read/resize logic.
    """
    return _read_patch(slide, x, y, size, level)


def _is_tissue(img: Image.Image) -> bool:
    """True if enough of the patch is saturated (stained) rather than background."""
    saturation = img.convert("HSV").split()[1]  # S channel, 0..255
    hist = saturation.histogram()               # 256 bins
    total = sum(hist)
    if total == 0:
        return False
    tissue_pixels = sum(hist[SAT_THRESHOLD:])
    return (tissue_pixels / total) >= MIN_TISSUE_FRACTION


# --- Grid + orchestration ----------------------------------------------------

def _grid_origins(bbox, size0: int):
    """Non-overlapping grid of patch top-left corners covering a bbox.

    Always yields at least one cell, so regions smaller than a patch still
    produce a single (bbox-anchored) patch.
    """
    x, y, w, h = bbox
    origins = []
    yy = y
    while yy < y + h:
        xx = x
        while xx < x + w:
            origins.append((xx, yy))
            xx += size0
        yy += size0
    if not origins:
        origins.append((x, y))
    return origins


def _save_montage(images, path: Path) -> bool:
    """Stitch up to MONTAGE_MAX patch thumbnails into one JPEG for eyeballing."""
    if not images:
        return False
    imgs = images[:MONTAGE_MAX]
    cols = min(MONTAGE_COLS, len(imgs))
    rows = ceil(len(imgs) / cols)
    canvas = Image.new("RGB", (cols * MONTAGE_THUMB, rows * MONTAGE_THUMB), (30, 34, 40))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas.paste(im.resize((MONTAGE_THUMB, MONTAGE_THUMB)), (c * MONTAGE_THUMB, r * MONTAGE_THUMB))
    canvas.save(path, "JPEG", quality=80)
    return True


def manifest_path(slide_id: str) -> Path:
    return PATCHES_DIR / f"{slide_id}.json"


def preview_path(slide_id: str) -> Path:
    return PATCHES_DIR / f"{slide_id}_preview.jpg"


def load_manifest(slide_id: str) -> dict | None:
    """Return a previously-extracted manifest, or None if not extracted yet."""
    path = manifest_path(slide_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_patches(slide_id: str) -> dict | None:
    """Extract labelled tissue patches for a slide; write manifest + montage.

    Returns the manifest dict, or None if the slide doesn't exist.
    """
    slide = slides.get_slide(slide_id)
    if slide is None:
        return None

    annos = annotations.load(slide_id)
    mpp0 = _slide_mpp(slide)
    # Side length (in level-0 pixels) of a patch that is PATCH_SIZE px at TARGET_MPP.
    size0 = max(1, round(PATCH_SIZE * TARGET_MPP / mpp0))
    level = slide.get_best_level_for_downsample(max(1.0, size0 / PATCH_SIZE))
    off_x, off_y = _bounds_offset(slide)
    dim_x, dim_y = slide.dimensions

    patches: list[dict] = []
    per_class: dict[str, int] = {}
    montage_imgs: list[Image.Image] = []
    candidates = dropped_background = skipped = 0

    for anno_index, anno in enumerate(annos):
        parsed = _parse_annotation(anno)
        if parsed is None:
            skipped += 1
            continue
        label, kind, geom = parsed
        bbox = _bbox(kind, geom)
        # Identity of the drawn region this patch came from. Phase 5 splits
        # train/val by this so spatially-adjacent tiles from one region never
        # straddle the split (leakage). Prefer the annotation's own id.
        group = f"{slide_id}::{anno.get('id') or f'anno{anno_index}'}"

        for (gx, gy) in _grid_origins(bbox, size0):
            # For polygons, keep only cells whose centre falls inside the shape.
            if kind == "polygon":
                if not _point_in_polygon(gx + size0 / 2, gy + size0 / 2, geom):
                    continue
            candidates += 1

            # Convert to level-0 pixel coords and skip anything off the slide.
            x0 = int(round(gx)) + off_x
            y0 = int(round(gy)) + off_y
            if x0 < off_x or y0 < off_y or x0 + size0 > off_x + dim_x or y0 + size0 > off_y + dim_y:
                continue

            patch = _read_patch(slide, x0, y0, size0, level)
            if not _is_tissue(patch):
                dropped_background += 1
                continue

            patches.append({"x": x0, "y": y0, "size": size0, "level": level, "label": label, "group": group})
            per_class[label] = per_class.get(label, 0) + 1
            if len(montage_imgs) < MONTAGE_MAX:
                montage_imgs.append(patch)

    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    _save_montage(montage_imgs, preview_path(slide_id))

    manifest = {
        "slide_id": slide_id,
        "config": {
            "patch_size": PATCH_SIZE,
            "target_mpp": TARGET_MPP,
            "slide_mpp": mpp0,
            "level": level,
        },
        "counts": {
            "candidates": candidates,
            "kept": len(patches),
            "dropped_background": dropped_background,
            "skipped_unparseable": skipped,
        },
        "per_class": per_class,
        "patches": patches,
    }

    # Atomic write (temp file + os.replace), mirroring annotations.save.
    path = manifest_path(slide_id)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp, path)

    return manifest
