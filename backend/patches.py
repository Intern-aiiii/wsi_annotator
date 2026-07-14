"""The tile grid, annotation geometry, and the tissue mask.

Every part of the pipeline — feature extraction, training, and inference — tiles a
slide the SAME way: one fixed, slide-wide grid of `size0`-pixel cells, addressed by
integer `(col, row)`. This module is the only place that grid is defined.

That matters more than it sounds. An earlier version tiled each annotation starting
at *its own bounding-box corner* while inference tiled the viewport on a global grid.
Same tile size, same magnification, but a different phase — so the classifier was
trained on one set of tile alignments and applied to another. Keeping a single
`grid_config()` is what makes that class of bug impossible.

Coordinate note: annotations and viewport regions are in the coordinate space of the
DeepZoom base image. Because the DZI is built with limit_bounds=True (see slides.py),
that space starts at the slide's bounds origin — so a cell's level-0 pixel origin adds
(bounds-x, bounds-y) before OpenSlide's read_region sees it (read_region always works
in full level-0 coordinates). For slides with a zero bounds origin (the CMU samples)
the offset is (0, 0).

Deliberately dependency-light: PIL + OpenSlide + stdlib. No numpy, no shapely.
"""

from __future__ import annotations

import math
import os
import re

import openslide
from PIL import Image

# --- Tiling parameters -------------------------------------------------------
PATCH_SIZE = 224          # pixels per side; the input size Virchow 2 expects
TARGET_MPP = 0.5          # microns/pixel (~20x). Matches Virchow 2's training data.

# --- Tissue mask -------------------------------------------------------------
# A tile is tissue if at least MIN_TISSUE_FRACTION of its pixels have saturation at or
# above a cutoff. White/grey background is near-zero saturation; H&E-stained tissue is
# well above it. Two ways to pick that cutoff (SLIDEPROBE_TISSUE env var):
#
#   "otsu" (default) — derive it from the slide's own saturation histogram, so the mask
#                      adapts to how faintly or darkly this slide happens to be stained.
#   "hsv"            — a fixed cutoff of SAT_THRESHOLD. Predictable; stain-blind.
MIN_TISSUE_FRACTION = 0.30
SAT_THRESHOLD = 25        # 0..255; the fixed cutoff used by the "hsv" mask

OVERVIEW_MAX = 2048       # longest side of the low-res slide overview Otsu reads
OTSU_MIN_SAT = 8          # sanity band for a derived threshold (see _otsu_gate)
OTSU_MAX_SAT = 80


# --- Slide geometry ----------------------------------------------------------

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


def slide_mpp(slide: openslide.OpenSlide) -> float:
    """Public accessor for the slide's level-0 µm/px (recorded in feature banks)."""
    return _slide_mpp(slide)


def _bounds_offset(slide: openslide.OpenSlide) -> tuple[int, int]:
    """(bounds-x, bounds-y) in level-0 pixels; (0, 0) if the slide has no bounds."""
    bx = slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_X)
    by = slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y)
    return (int(bx) if bx else 0, int(by) if by else 0)


def _bounded_size(slide: openslide.OpenSlide) -> tuple[int, int]:
    """Size of the DZI base image: the bounded region, not the whole level-0 image."""
    bw = slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_WIDTH)
    bh = slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT)
    dim_x, dim_y = slide.dimensions
    return (int(bw) if bw else dim_x, int(bh) if bh else dim_y)


def grid_config(slide: openslide.OpenSlide) -> dict:
    """THE slide-wide tile grid. Nothing else derives geometry on its own.

    Returns {"size0", "level", "off_x", "off_y", "cols", "rows"}:
      size0        side of a cell in LEVEL-0 pixels (so it is PATCH_SIZE px at TARGET_MPP)
      level        the pyramid level to read from (best match for the downsample)
      off_x/off_y  the bounds origin, added to get level-0 coords for read_region
      cols/rows    the grid's extent

    cols/rows use floor division, so a partial strip at the right/bottom edge (too
    narrow to make a full tile) is simply outside the grid. Every downstream bounds
    check is therefore just `0 <= col < cols and 0 <= row < rows`.

    All values are ints: a grid dict is stored in each feature bank and compared for
    equality on load, so a bank built under a different tiling is rejected outright.
    """
    mpp0 = _slide_mpp(slide)
    # Side length (in level-0 pixels) of a tile that is PATCH_SIZE px at TARGET_MPP.
    size0 = max(1, round(PATCH_SIZE * TARGET_MPP / mpp0))
    level = int(slide.get_best_level_for_downsample(max(1.0, size0 / PATCH_SIZE)))
    off_x, off_y = _bounds_offset(slide)
    width, height = _bounded_size(slide)
    return {
        "size0": size0,
        "level": level,
        "off_x": off_x,
        "off_y": off_y,
        "cols": width // size0,
        "rows": height // size0,
    }


def in_grid(grid: dict, col: int, row: int) -> bool:
    return 0 <= col < grid["cols"] and 0 <= row < grid["rows"]


def cell_origin(grid: dict, col: int, row: int) -> tuple[int, int]:
    """Level-0 pixel origin of a cell, ready for read_region (bounds offset included)."""
    return col * grid["size0"] + grid["off_x"], row * grid["size0"] + grid["off_y"]


def cell_index(grid: dict, col: int, row: int) -> int:
    """Flatten (col, row) to a single int — how cells are stored in a feature bank."""
    return row * grid["cols"] + col


def index_cell(grid: dict, idx: int) -> tuple[int, int]:
    """Inverse of cell_index."""
    return idx % grid["cols"], idx // grid["cols"]


# --- Annotation parsing ------------------------------------------------------

def _label_of(anno: dict) -> str:
    """The class label = the annotation's `tagging` body value, or 'unlabeled'."""
    body = anno.get("body")
    items = body if isinstance(body, list) else ([body] if body else [])
    for b in items:
        if isinstance(b, dict) and b.get("purpose") == "tagging" and b.get("value"):
            return str(b["value"])
    return "unlabeled"


def label_of(anno: dict) -> str:
    """Public accessor for an annotation's class label."""
    return _label_of(anno)


def parse_annotation(anno: dict):
    """Public accessor: (label, kind, geometry), or None if the selector isn't understood.

    Exposed so callers can tell "I cannot READ this shape" apart from "this shape covers
    no tissue". They look identical from cells_in_annotation() (both give []), and
    conflating them produces a very confusing error: the app used to tell the user their
    region contained no tissue when the truth was that the backend couldn't parse it.
    """
    return _parse_annotation(anno)


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


# --- Which cells does a shape / a viewport cover? ----------------------------

def cells_in_region(grid: dict, region: dict) -> tuple[int, int, int, int]:
    """Snap a DZI-pixel viewport {x,y,w,h} to the grid. -> (col0, row0, cols, rows).

    Clipped to the grid, so the returned block contains only real, full cells.
    """
    size0 = grid["size0"]
    rx, ry = float(region.get("x", 0)), float(region.get("y", 0))
    rw, rh = float(region.get("w", 0)), float(region.get("h", 0))
    col0 = max(0, min(grid["cols"], math.floor(rx / size0)))
    row0 = max(0, min(grid["rows"], math.floor(ry / size0)))
    col1 = max(0, min(grid["cols"], math.ceil((rx + rw) / size0)))
    row1 = max(0, min(grid["rows"], math.ceil((ry + rh) / size0)))
    return col0, row0, max(0, col1 - col0), max(0, row1 - row0)


def cells_in_annotation(grid: dict, annotation: dict) -> list[tuple[int, int]]:
    """The grid cells an annotation covers. PURE GEOMETRY — reads no pixels.

    A cell belongs to the annotation when its CENTRE falls inside the shape. This
    is what lets training assemble its dataset without touching the slide: the
    cells' embeddings are already in the feature bank.

    Returns [] for an unparseable annotation. An annotation smaller than one cell
    falls back to the single cell containing its centroid, so a tiny region still
    contributes one training sample rather than silently none.
    """
    parsed = _parse_annotation(annotation)
    if parsed is None:
        return []
    _label, kind, geom = parsed
    size0 = grid["size0"]
    x, y, w, h = _bbox(kind, geom)

    cells: list[tuple[int, int]] = []
    for row in range(math.floor(y / size0), math.ceil((y + h) / size0)):
        for col in range(math.floor(x / size0), math.ceil((x + w) / size0)):
            if not in_grid(grid, col, row):
                continue
            cx, cy = (col + 0.5) * size0, (row + 0.5) * size0
            if kind == "polygon":
                if not _point_in_polygon(cx, cy, geom):
                    continue
            elif not (x <= cx < x + w and y <= cy < y + h):
                continue
            cells.append((col, row))

    if not cells:
        col, row = math.floor((x + w / 2) / size0), math.floor((y + h / 2) / size0)
        if in_grid(grid, col, row):
            cells.append((col, row))
    return cells


# --- Pixel reading + the tissue mask -----------------------------------------

def read_cell(slide, grid: dict, col: int, row: int) -> Image.Image:
    """Read one grid cell as a PATCH_SIZE-square RGB image.

    Reads from the pyramid level best matching the target downsample, then resizes
    to PATCH_SIZE. read_region always takes level-0 coordinates.
    """
    x0, y0 = cell_origin(grid, col, row)
    size0, level = grid["size0"], grid["level"]
    ds = slide.level_downsamples[level]
    read_size = max(1, round(size0 / ds))
    region = slide.read_region((int(x0), int(y0)), level, (read_size, read_size))
    # region is RGBA; composite onto white to flatten transparency, then RGB.
    rgb = Image.new("RGB", region.size, (255, 255, 255))
    rgb.paste(region, mask=region.split()[-1])
    if rgb.size != (PATCH_SIZE, PATCH_SIZE):
        rgb = rgb.resize((PATCH_SIZE, PATCH_SIZE))
    return rgb


def _saturation_histogram(img: Image.Image) -> list[int]:
    """256-bin histogram of the HSV saturation channel.

    Saturation is the signal that separates stain from slide: H&E tissue is strongly
    coloured, while background glass/scanner white is near-grey (saturation ~0).
    """
    return img.convert("HSV").split()[1].histogram()


def _overview(slide) -> Image.Image:
    """A small RGB image of the whole BOUNDED slide, for slide-level statistics.

    Read from the pyramid level closest to the target downsample, so this costs one
    cheap low-res read rather than a scan of level 0. Note it reads the BOUNDED region
    (not slide.get_thumbnail, which would include the scanner's dead margin and skew
    any statistic computed from it).
    """
    off_x, off_y = _bounds_offset(slide)
    width, height = _bounded_size(slide)
    scale = max(1.0, max(width, height) / OVERVIEW_MAX)
    level = int(slide.get_best_level_for_downsample(scale))
    ds = slide.level_downsamples[level]
    size = (max(1, int(width / ds)), max(1, int(height / ds)))

    region = slide.read_region((off_x, off_y), level, size)
    rgb = Image.new("RGB", region.size, (255, 255, 255))
    rgb.paste(region, mask=region.split()[-1])
    if max(rgb.size) > OVERVIEW_MAX:  # the level may still be larger than we need
        rgb.thumbnail((OVERVIEW_MAX, OVERVIEW_MAX))
    return rgb


def _otsu(hist: list[int]) -> int:
    """Otsu's threshold on a 256-bin histogram.

    Returns the LOWEST bin belonging to the foreground (bright/saturated) class, i.e.
    the value `t` such that the split {0..t-1} vs {t..255} maximizes between-class
    variance. Pure Python over 256 bins — no numpy, no scikit-image needed.
    """
    total = sum(hist)
    if total == 0:
        return 0
    sum_all = sum(i * h for i, h in enumerate(hist))
    sum_bg = 0.0
    w_bg = 0
    best_var, best_t = -1.0, 0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if between > best_var:
            best_var, best_t = between, t
    return best_t + 1  # first foreground bin


class TissueGate:
    """The tissue test for ONE slide: `gate(tile) -> bool`, plus an `id` for the bank.

    A tile counts as tissue when at least `min_fraction` of its pixels have saturation
    >= `threshold`. Where that threshold comes from is what distinguishes the masks.

    `id` is recorded in every feature bank and checked on load, so changing the mask (or
    its parameters) correctly invalidates existing banks rather than silently mixing
    tiles judged by two different rules.
    """

    def __init__(self, name: str, threshold: int, min_fraction: float):
        self.name = name
        self.threshold = int(threshold)
        self.min_fraction = float(min_fraction)
        self.id = f"{name}-sat{self.threshold}-frac{self.min_fraction}"

    def __call__(self, img: Image.Image) -> bool:
        hist = _saturation_histogram(img)
        total = sum(hist)
        if total == 0:
            return False
        return sum(hist[self.threshold:]) / total >= self.min_fraction


def _hsv_gate(slide) -> TissueGate:
    """Fixed saturation cutoff. Simple and predictable; ignores the slide entirely."""
    return TissueGate("hsv", SAT_THRESHOLD, MIN_TISSUE_FRACTION)


def _otsu_gate(slide) -> TissueGate:
    """Otsu's method — the cutoff is LEARNED from this slide's own saturation histogram.

    Why it reads a whole-slide overview rather than thresholding each tile:
    Otsu always finds a split. Run it on a tile of blank glass and it will happily
    divide the sensor noise into "dark" and "light" and hand back a threshold that
    calls half the background tissue. The histogram it sees must actually contain both
    populations — so the threshold is derived ONCE, from an overview of the whole
    slide, and then applied per tile.

    This is the point of Otsu here: it adapts to the stain. A faintly-stained slide gets
    a lower cutoff and a darkly-stained one a higher cutoff, where the fixed hsv=25
    would under- or over-call tissue on both.

    The clamp guards the degenerate case where the overview has only ONE population
    (a slide that is nearly all tissue, or nearly all background): Otsu would then be
    splitting noise, so we refuse to trust a threshold outside a sane band.
    """
    threshold = _otsu(_saturation_histogram(_overview(slide)))
    threshold = max(OTSU_MIN_SAT, min(OTSU_MAX_SAT, threshold))
    return TissueGate("otsu", threshold, MIN_TISSUE_FRACTION)


_TISSUE_GATES = {"hsv": _hsv_gate, "otsu": _otsu_gate}


def tissue_mask_name() -> str:
    """Which tissue mask is selected: "otsu" (default) or "hsv"."""
    name = os.environ.get("SLIDEPROBE_TISSUE", "otsu").lower()
    return name if name in _TISSUE_GATES else "otsu"


def tissue_gate(slide) -> TissueGate:
    """Build the tissue test for a slide. Called ONCE per slide (features.py caches it).

    Otsu reads a low-res overview to derive its threshold, so this is not free — do not
    call it per tile.
    """
    return _TISSUE_GATES[tissue_mask_name()](slide)
