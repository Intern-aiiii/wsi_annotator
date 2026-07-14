// Freehand (lasso) region drawing.
//
// Drag a loop around a structure; it becomes a normal polygon annotation that tags,
// colours, edits, trains and deletes exactly like one drawn with the polygon tool.
//
// WHY IT WORKS THIS WAY (read before changing anything)
// -----------------------------------------------------
// Two earlier attempts used @recogito/annotorious-selector-pack's freehand tool and both
// were reverted: the shape drew, but the editor never opened, so the region could not be
// tagged or deleted (docs/log.txt). The cause was in the pack: it vendored a copy of the
// core's format(shape, annotation, formatters) -- which takes an ARRAY -- but calls it
// with the SINGULAR config.formatter. Since annotate.js configures a formatter (to colour
// regions by class), the pack did fn.reduce(...) and threw "n.reduce is not a function"
// deep inside createEditableShape, so the editor never rendered.
//
// A freehand stroke, though, is just a POLYGON with a lot of points -- and polygons are
// NATIVE to Annotorious, handled by the core's own EditablePolygon, which calls the
// correct plural API. So this file does not add a shape type or touch Annotorious's
// internals at all. It captures the stroke with plain pointer events, simplifies it, and
// hands the points to SlideAnnotate.addPolygon(). The backend needs no change either:
// patches.py already parses an SvgSelector polygon.
//
// The capture layer is a transparent div over the viewer that SWALLOWS every pointer
// event, so neither OpenSeadragon (pan/zoom) nor Annotorious (select) sees the stroke.
// Nothing has to be disabled or coordinated; they simply never find out.
//
// STICKY: the tool stays armed after each region, so you can draw a row of structures
// without re-clicking the button. Escape or a second click switches it off.

(function () {
  // Simplification tolerance, in SCREEN pixels -- so it is independent of zoom.
  const RDP_TOLERANCE = 2.0;

  // Hard ceiling on vertices, and it is NOT arbitrary. Annotorious's EditablePolygon
  // draws ONE DRAG HANDLE PER VERTEX and rescales every one of them on every OSD
  // animation frame while the shape is selected -- a few thousand points locks the tab.
  // And backend/patches.py's `_point_in_polygon` is O(vertices) per candidate grid cell,
  // re-run on EVERY debounced retrain, so a fat polygon quietly breaks the "retraining is
  // so cheap it can run on every save" property the whole app is built on. A real lasso
  // simplifies to 20-60 points, so this is generous.
  const MAX_VERTICES = 100;

  // A stroke smaller than this isn't a region, it's a slip of the mouse. The backend
  // needs >= 3 points or it cannot read the shape at all, so we must never emit fewer.
  const MIN_POINTS = 3;
  const MIN_DIAGONAL = 12;   // screen px, bounding box
  const MIN_AREA = 100;      // screen px^2, |shoelace|

  let armed = false;         // is the tool switched on?
  let captureEl = null;      // the transparent event-swallowing layer (null when idle)
  let previewEl = null;      // the live <polyline> shown while dragging
  let pointerId = null;      // the one pointer we track (ignore a second finger)
  let strokeSlideId = null;  // which slide the stroke began on -- see finishStroke()
  let screenPts = [];        // for the preview + simplification (resolution-independent)
  let imagePts = [];         // the truth, converted as we go (see below). Same indices.

  const btn = () => document.getElementById("freehand-btn");

  // --- Geometry ---------------------------------------------------------------

  // Perpendicular distance from p to the line ab.
  function perpDist(p, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    if (dx === 0 && dy === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
    const t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy);
    const cx = a[0] + t * dx, cy = a[1] + t * dy;
    return Math.hypot(p[0] - cx, p[1] - cy);
  }

  // Ramer-Douglas-Peucker. Returns the INDICES it keeps, so the caller can pull the
  // matching image-space points out of the parallel array.
  function simplifyIndices(pts, tol, lo, hi, keep) {
    let worst = 0, idx = -1;
    for (let i = lo + 1; i < hi; i++) {
      const d = perpDist(pts[i], pts[lo], pts[hi]);
      if (d > worst) { worst = d; idx = i; }
    }
    if (worst > tol && idx > 0) {
      simplifyIndices(pts, tol, lo, idx, keep);
      keep.push(idx);
      simplifyIndices(pts, tol, idx, hi, keep);
    }
  }

  function simplify(pts, tol) {
    if (pts.length < 3) return pts.map((_, i) => i);
    const keep = [];
    simplifyIndices(pts, tol, 0, pts.length - 1, keep);
    return [0, ...keep.sort((a, b) => a - b), pts.length - 1];
  }

  // Simplify, then raise the tolerance until we're under the vertex cap. Raising the
  // tolerance (rather than dropping every Nth point) keeps the shape RDP just preserved;
  // plain decimation would put the error straight back.
  function simplifyToCap(pts) {
    let tol = RDP_TOLERANCE;
    let idx = simplify(pts, tol);
    while (idx.length > MAX_VERTICES && tol < 1e4) {
      tol *= 1.6;
      idx = simplify(pts, tol);
    }
    return idx;
  }

  function bboxDiagonal(pts) {
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    return Math.hypot(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  }

  function shoelaceArea(pts) {
    let a = 0;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      a += pts[j][0] * pts[i][1] - pts[i][0] * pts[j][1];
    }
    return Math.abs(a / 2);
  }

  // --- Coordinates ------------------------------------------------------------

  // Browser pointer -> DZI base-image pixels, the space annotations are stored in.
  //
  // Converted on every pointermove, NOT once at the end: OpenSeadragon animates for
  // ~0.3s, so a stroke begun while a zoom is still settling would otherwise have every
  // screen point mapped through a viewport that has since moved -- a smeared polygon.
  // Doing it as we go makes the geometry correct even if the view shifts mid-stroke.
  function toImage(clientX, clientY) {
    const v = window.osdViewer;
    const item = v && v.world.getItemAt(0);
    if (!item) return null;                       // mid slide-switch: nothing loaded
    const rect = v.element.getBoundingClientRect();
    const p = v.viewport.viewerElementToImageCoordinates(
      new OpenSeadragon.Point(clientX - rect.left, clientY - rect.top)
    );
    // Clamp into the image. Dragging onto the black surround would otherwise give
    // negative coordinates, which blow up the bounding box the backend loops over.
    const size = item.getContentSize();
    return [
      Math.min(Math.max(p.x, 0), size.x),
      Math.min(Math.max(p.y, 0), size.y),
    ];
  }

  // --- The capture layer ------------------------------------------------------

  function addCapture() {
    if (captureEl) return;
    const host = document.getElementById("viewer");
    if (!host) return;

    captureEl = document.createElement("div");
    captureEl.className = "freehand-capture";

    previewEl = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    previewEl.setAttribute("class", "freehand-preview");
    captureEl.appendChild(previewEl);

    captureEl.addEventListener("pointerdown", onDown);
    captureEl.addEventListener("pointermove", onMove);
    captureEl.addEventListener("pointerup", onUp);
    captureEl.addEventListener("pointercancel", abortStroke);
    // Without this the wheel still reaches OpenSeadragon and zooms mid-stroke.
    captureEl.addEventListener("wheel", (e) => { e.preventDefault(); e.stopPropagation(); },
                               { passive: false });
    host.appendChild(captureEl);
  }

  function removeCapture() {
    if (!captureEl) return;
    captureEl.remove();
    captureEl = null;
    previewEl = null;
    pointerId = null;
    screenPts = [];
    imagePts = [];
  }

  function drawPreview() {
    if (!previewEl) return;
    previewEl.innerHTML = "";
    if (screenPts.length < 2) return;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    line.setAttribute("points", screenPts.map((p) => `${p[0]},${p[1]}`).join(" "));
    previewEl.appendChild(line);
  }

  // --- Stroke ------------------------------------------------------------------

  function onDown(e) {
    if (pointerId !== null) return;             // a second finger: ignore it
    pointerId = e.pointerId;
    // Capture the pointer, or dragging up over the topbar loses the pointerup and the
    // stroke never finishes -- the app just looks frozen.
    captureEl.setPointerCapture(e.pointerId);
    strokeSlideId = window.SlideAnnotate.slideId();
    screenPts = [];
    imagePts = [];
    addPoint(e);
    e.preventDefault();
  }

  function onMove(e) {
    if (e.pointerId !== pointerId) return;
    addPoint(e);
    drawPreview();
    e.preventDefault();
  }

  function addPoint(e) {
    const rect = captureEl.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    // Skip near-duplicate points so the raw buffer stays in the hundreds, not thousands.
    const last = screenPts[screenPts.length - 1];
    if (last && Math.hypot(sx - last[0], sy - last[1]) < 2) return;
    const img = toImage(e.clientX, e.clientY);
    if (!img) return;
    screenPts.push([sx, sy]);
    imagePts.push(img);        // parallel arrays: same index, same point
  }

  function onUp(e) {
    if (e.pointerId !== pointerId) return;
    e.preventDefault();
    const screen = screenPts.slice();
    const image = imagePts.slice();
    const slideAtStart = strokeSlideId;
    // Take the capture layer down BEFORE opening the editor. Annotorious renders the
    // editor popup inside the viewer element, so a surviving capture div would paint on
    // top of it and swallow clicks on the tag box -- which looks exactly like the two
    // prior failures, and would send the next person hunting the wrong ghost.
    removeCapture();
    finishStroke(screen, image, slideAtStart);
  }

  function abortStroke() {
    if (!captureEl) return;
    screenPts = [];
    imagePts = [];
    pointerId = null;
    drawPreview();
  }

  function finishStroke(screen, image, slideAtStart) {
    const reArm = () => { if (armed) addCapture(); };

    // The slide changed under us mid-stroke. Committing now would write a region drawn
    // on one slide into another slide's annotation file -- silent and permanent.
    if (!window.SlideAnnotate.isReady() || slideAtStart !== window.SlideAnnotate.slideId()) {
      reArm();
      return;
    }

    const idx = simplifyToCap(screen);
    const kept = idx.map((i) => image[i]).filter(Boolean);
    const keptScreen = idx.map((i) => screen[i]).filter(Boolean);

    // Reject a stroke that isn't a region. Say so out loud: a silent no-op just looks
    // broken. (The backend needs >= 3 points, so we must never be able to emit fewer.)
    if (kept.length < MIN_POINTS ||
        bboxDiagonal(keptScreen) < MIN_DIAGONAL ||
        shoelaceArea(keptScreen) < MIN_AREA) {
      window.SlideAnnotate.status("stroke too small — drag a loop around a region");
      reArm();
      return;
    }

    // The polygon closes itself (SVG closes it, and so does the backend's ray cast), so
    // do NOT repeat the first point at the end — that just adds a degenerate edge.
    const id = window.SlideAnnotate.addPolygon(kept);
    window.__freehand.lastCommittedId = id;   // deterministic hook for the e2e test
    window.SlideAnnotate.status(`region drawn (${kept.length} points) — give it a class`);
    // NOTE: no save here. The editor's OK fires updateAnnotation, which annotate.js
    // already wires to saveCurrent. Saving now would rebuild Annotorious out from under
    // the open editor, and could write the region twice. See SlideAnnotate.addPolygon.
  }

  // --- Arm / disarm -------------------------------------------------------------

  function arm() {
    if (armed || !window.SlideAnnotate.isReady()) return;
    armed = true;
    window.SlideAnnotate.cancelSelected();   // an open editor would sit under our layer
    addCapture();
    const b = btn();
    if (b) b.classList.add("active");
    window.SlideAnnotate.status("freehand: drag a loop around a region");
  }

  function disarm() {
    if (!armed) return;
    armed = false;
    removeCapture();
    const b = btn();
    if (b) b.classList.remove("active");
  }

  function refreshButton() {
    const b = btn();
    if (!b) return;
    const ready = window.SlideAnnotate.isReady();
    b.disabled = !ready;
    if (!ready) disarm();
  }

  // --- Wiring --------------------------------------------------------------------

  window.addEventListener("DOMContentLoaded", () => {
    const b = btn();
    if (b) b.addEventListener("click", () => (armed ? disarm() : arm()));

    // Re-arm once the editor closes (OK / Cancel / Delete): that is what makes the tool
    // STICKY — draw, tag, and the lasso is ready again without touching the button.
    window.SlideAnnotate.onEditorClosed(() => { if (armed) addCapture(); });
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && armed) {
      abortStroke();
      disarm();
    }
  });

  // A stroke in progress must never survive a context change. Each of these also
  // rebuilds or replaces Annotorious, so anything half-finished has to go.
  document.addEventListener("slideprobe:project-opened", disarm);
  document.addEventListener("slideprobe:classes-changed", disarm);
  document.addEventListener("slideprobe:slide-opened", () => {
    disarm();
    refreshButton();          // slideId === null (an empty project) => nothing to draw on
  });

  // Pure functions + a commit hook, exposed for tests (no browser interaction needed).
  window.__freehand = {
    simplify,
    simplifyToCap,
    bboxDiagonal,
    shoelaceArea,
    lastCommittedId: null,
    isArmed: () => armed,
    hasCapture: () => !!captureEl,
  };
})();
