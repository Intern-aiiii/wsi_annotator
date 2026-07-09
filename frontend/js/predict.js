// Predict the current view (Phase 6) and the annotation similarity map (Phase 7),
// each overlaid on OpenSeadragon.
//
//   - "Predict view": supervised. The backend scores each tissue tile with the
//     trained head and returns the regions that are confidently the chosen class
//     (>= a min-confidence threshold) as tile CELLS + BOUNDARY segments.
//   - "Find similar": unsupervised. Triggered from an annotation's editor popup
//     (see annotate.js, which dispatches "slideprobe:find-similar"); the backend
//     returns the regions most similar to that annotation in the same shape.
//
// Both are drawn as one SVG overlay: a faint tinted fill over the qualifying tiles
// plus a crisp outline. A shared "overlay control" (opacity + clear) manages it.

let predictSlideId = null;
let overlayEl = null; // the <svg> (predict) or <img> (similarity) currently overlaid

function pdEl(id) {
  return document.getElementById(id);
}

function pdStatus(msg) {
  const el = pdEl("predict-status");
  if (el) el.textContent = msg;
}

// Status line inside the shared overlay control.
function ocStatus(msg) {
  const el = pdEl("overlay-status");
  if (el) el.textContent = msg;
}

function removeOverlay() {
  if (overlayEl && window.osdViewer) {
    window.osdViewer.removeOverlay(overlayEl);
  }
  overlayEl = null;
}

// Fill the class dropdown from the trained model, or explain if there's none.
async function refreshClasses() {
  const select = pdEl("predict-class");
  try {
    const model = await API.getModel();
    if (!model || !model.classes || !model.classes.length) {
      select.innerHTML = "";
      pdEl("predict-run").disabled = true; // caller sets the status message
      return null;
    }
    const prev = select.value;
    select.innerHTML = "";
    for (const c of model.classes) {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      select.appendChild(opt);
    }
    if (model.classes.includes(prev)) select.value = prev;
    pdEl("predict-run").disabled = false;
    return model;
  } catch (err) {
    pdStatus(`could not load model: ${err.message}`);
    return null;
  }
}

// The image-pixel rectangle currently visible in the viewer, clamped to the image.
function visibleImageRegion() {
  const v = window.osdViewer;
  const item = v && v.world.getItemAt(0);
  if (!item) return null;
  const r = v.viewport.viewportToImageRectangle(v.viewport.getBounds());
  const size = item.getContentSize();
  const x = Math.max(0, r.x);
  const y = Math.max(0, r.y);
  const x2 = Math.min(size.x, r.x + r.width);
  const y2 = Math.min(size.y, r.y + r.height);
  return { x, y, w: Math.max(0, x2 - x), h: Math.max(0, y2 - y) };
}

async function runPredict() {
  if (!predictSlideId) return;
  const region = visibleImageRegion();
  if (!region || region.w < 1 || region.h < 1) {
    pdStatus("nothing visible to predict");
    return;
  }
  const cls = pdEl("predict-class").value || undefined;
  const btn = pdEl("predict-run");
  btn.disabled = true;
  pdStatus("predicting…");
  try {
    const result = await API.predictView(predictSlideId, region, cls);
    placeOverlay(result);
    const pct = Math.round((result.min_confidence || 0) * 100);
    const summary = `${result.class}: ${result.n_above} tiles ≥ ${pct}%`;
    pdStatus(summary);
    ocStatus(summary);
  } catch (err) {
    pdStatus(err.message);
  } finally {
    btn.disabled = false;
  }
}

// --- Unsupervised similarity-to-a-selected-annotation ----------------------

// Triggered by the "Find similar" button in an annotation's editor popup.
async function runFindSimilar(annotation) {
  if (!predictSlideId || !annotation) return;
  const region = visibleImageRegion();
  if (!region || region.w < 1) return;
  ocStatus("finding similar regions…");
  pdEl("overlay-ctrl").hidden = false;
  try {
    const result = await API.similarityByAnnotation(predictSlideId, region, annotation);
    placeOverlay(result);
    const label = result.ref_label ? ` to “${result.ref_label}”` : "";
    const pct = Math.round((result.min_confidence || 0) * 100);
    ocStatus(`similar${label} · ${result.n_above} tiles ≥ ${pct}%`);
  } catch (err) {
    ocStatus(err.message);
  }
}

// Fixed high-contrast color for region overlays (outline + tint). A later
// per-class color would just replace this constant with a lookup.
const REGION_COLOR = "#00e5ff";
const REGION_FILL_OPACITY = "0.2"; // the "slight tint" inside each region
const SVG_NS = "http://www.w3.org/2000/svg";

// Build the SVG overlay for a scored region: a faint fill over the qualifying
// tiles (`cells`) plus a crisp outline (`boundaries`). Both predict and "Find
// similar" return this shape data. The viewBox matches the scored region so OSD
// can stretch it to the right rectangle; non-scaling-stroke keeps the outline a
// constant width on screen at any zoom.
function buildShapesOverlay(result) {
  const r = result.region;
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${r.w} ${r.h}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("class", "shapes-overlay");
  svg.style.width = "100%";
  svg.style.height = "100%";
  svg.style.opacity = pdEl("predict-opacity").value;

  // Tinted interior: one fill path of all qualifying tile rectangles.
  let fillD = "";
  for (const [x, y, w, h] of result.cells || []) {
    fillD += `M${x} ${y}L${x + w} ${y}L${x + w} ${y + h}L${x} ${y + h}Z`;
  }
  const fill = document.createElementNS(SVG_NS, "path");
  fill.setAttribute("d", fillD);
  fill.setAttribute("fill", REGION_COLOR);
  fill.setAttribute("fill-opacity", REGION_FILL_OPACITY);
  fill.setAttribute("stroke", "none");
  svg.appendChild(fill);

  // Outline: the region boundary segments.
  let lineD = "";
  for (const [x1, y1, x2, y2] of result.boundaries || []) {
    lineD += `M${x1} ${y1}L${x2} ${y2}`;
  }
  const outline = document.createElementNS(SVG_NS, "path");
  outline.setAttribute("d", lineD);
  outline.setAttribute("fill", "none");
  outline.setAttribute("stroke", REGION_COLOR);
  outline.setAttribute("stroke-width", "2");
  outline.setAttribute("vector-effect", "non-scaling-stroke");
  svg.appendChild(outline);
  return svg;
}

// Drop the region overlay over the exact rectangle the backend scored, and reveal
// the shared overlay control (opacity + clear). Both predict and similarity return
// the same {cells, boundaries} shape data.
function placeOverlay(result) {
  const v = window.osdViewer;
  const r = result.region;
  const location = v.viewport.imageToViewportRectangle(
    new OpenSeadragon.Rect(r.x, r.y, r.w, r.h)
  );
  removeOverlay();
  overlayEl = buildShapesOverlay(result);
  v.addOverlay({ element: overlayEl, location });
  pdEl("overlay-ctrl").hidden = false;
}

function clearOverlay() {
  removeOverlay();
  pdEl("overlay-ctrl").hidden = true;
  ocStatus("");
}

window.addEventListener("DOMContentLoaded", () => {
  const openBtn = pdEl("predict-btn");
  if (openBtn) {
    openBtn.addEventListener("click", async () => {
      pdEl("predict-panel").hidden = false;
      const model = await refreshClasses();
      if (model) {
        runPredict(); // predict immediately with the default class
      } else {
        pdStatus("No trained model yet — annotate two classes, then predict.");
      }
    });
  }
  const runBtn = pdEl("predict-run");
  if (runBtn) runBtn.addEventListener("click", runPredict);

  const clearBtn = pdEl("overlay-clear");
  if (clearBtn) clearBtn.addEventListener("click", clearOverlay);

  const close = pdEl("predict-close");
  if (close) close.addEventListener("click", () => { pdEl("predict-panel").hidden = true; });

  const cls = pdEl("predict-class");
  if (cls) cls.addEventListener("change", runPredict); // re-score (cached; fast)

  const opacity = pdEl("predict-opacity");
  if (opacity) {
    opacity.addEventListener("input", () => {
      if (overlayEl) overlayEl.style.opacity = opacity.value;
    });
  }
});

// The annotation editor's "Find similar" button dispatches this.
document.addEventListener("slideprobe:find-similar", (event) => {
  runFindSimilar(event.detail && event.detail.annotation);
});

// When the background worker finishes training, refresh the class dropdown so a
// newly-available (or changed) model shows up without reopening the panel.
document.addEventListener("slideprobe:model-ready", () => {
  if (!pdEl("predict-panel").hidden) refreshClasses();
});

document.addEventListener("slideprobe:slide-opened", (event) => {
  predictSlideId = event.detail.slideId;
  clearOverlay();
  const panel = pdEl("predict-panel");
  if (panel) panel.hidden = true;
});
