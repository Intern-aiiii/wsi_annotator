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
let overlayEl = null; // the <svg> region overlay currently placed on the viewer

// The two INDEPENDENT conditions for predicting. Both must hold, and the messaging
// has to say which one is missing — they need completely different fixes.
//   (a) this slide's features are extracted   -> Predict view AND Find similar
//   (b) a classifier head exists (>=2 classes) -> Predict view only
// Find similar uses no classifier, so it never mentions (b).
let pdFeaturesReady = false;
let featStatus = null; // last slideprobe:features detail, for a precise message
let hasModel = false;

function pdEl(id) {
  return document.getElementById(id);
}

const fmtN = (n) => Number(n || 0).toLocaleString();

// Why is predicting blocked? null when it isn't.
function gateReason() {
  if (pdFeaturesReady && hasModel) return null;

  const s = featStatus;
  let featMsg = null;
  if (!pdFeaturesReady) {
    if (!s || s.state === "unknown") featMsg = "Checking feature status…";
    else if (s.state === "running")
      featMsg = `Extracting features… ${s.pct}% — Predict unlocks when it finishes.`;
    else if (s.state === "cancelling") featMsg = "Cancelling feature extraction…";
    else if (s.state === "partial")
      featMsg = `Feature extraction is incomplete (${fmtN(s.done)} / ${fmtN(s.total)} tiles). Resume it to enable Predict.`;
    else if (s.state === "error")
      featMsg = `Feature extraction failed: ${s.detail || "unknown error"}`;
    else if (s.state === "complete" && !s.nTissue)
      featMsg = "No tissue found on this slide.";
    else featMsg = "Extract features for this slide first.";
  }
  const modelMsg = hasModel
    ? null
    : "No trained model yet — annotate two classes, then predict.";

  if (featMsg && modelMsg) return `${featMsg} Also: ${modelMsg[0].toLowerCase()}${modelMsg.slice(1)}`;
  return featMsg || modelMsg;
}

// The single place the two conditions are combined into the button's state.
function updatePredictGate() {
  const reason = gateReason();
  const run = pdEl("predict-run");
  if (run) run.disabled = !!reason;
  // A disabled button doesn't fire hover events, so a title on it would never show.
  // Put it on the wrapper, and put the real explanation in visible text.
  const actions = pdEl("predict-actions");
  if (actions) actions.title = reason || "";
  const open = pdEl("predict-btn");
  if (open) open.title = reason || "predict the visible region";
  const panel = pdEl("predict-panel");
  if (reason && panel && !panel.hidden) pdStatus(reason);
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

// Fill the class dropdown from the trained model. It no longer owns #predict-run's
// disabled state — that depends on the feature gate too, so updatePredictGate() owns
// it and this just reports whether condition (b) holds.
async function refreshClasses() {
  const select = pdEl("predict-class");
  if (!API.projectId()) {
    hasModel = false;
    select.innerHTML = "";
    updatePredictGate();
    return null;
  }
  // Capture the project, and drop the answer if it changed mid-flight — otherwise
  // project A's classes can land in the dropdown while you're looking at project B.
  const pid = API.projectId();
  try {
    const model = await API.getModel();
    if (API.projectId() !== pid) return null;
    hasModel = !!(model && model.classes && model.classes.length);
    if (!hasModel) {
      select.innerHTML = "";
      updatePredictGate();
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
    updatePredictGate();
    return model;
  } catch (err) {
    hasModel = false;
    updatePredictGate();
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
  // Belt and braces: even if a stale button somehow survives, don't fire a request
  // the backend would refuse anyway.
  const blocked = gateReason();
  if (blocked) {
    pdStatus(blocked);
    return;
  }
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
    // Restore via the gate, not `disabled = false` — the gate may have shut while the
    // request was in flight (slide change, model invalidated), and blindly re-enabling
    // would hand back a button that no longer has the right to run.
    updatePredictGate();
  }
}

// --- Unsupervised similarity-to-a-selected-annotation ----------------------

// Triggered by the "Find similar" button in an annotation's editor popup.
async function runFindSimilar(annotation) {
  if (!predictSlideId || !annotation) return;
  // Find similar needs the feature bank, but NOT a trained model — so it must never
  // complain about the model, only about features.
  if (!pdFeaturesReady) {
    pdEl("overlay-ctrl").hidden = false;
    ocStatus("Extract features for this slide first.");
    return;
  }
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

// The overlay is drawn in the CLASS's colour (Phase 7). It used to be a fixed cyan
// because the frontend had no authoritative per-class colour; the project stores one
// now. Tinting a predicted region in the same colour as the annotations of that class
// is the point: you can see at a glance whether the model agrees with what you drew.
const FALLBACK_REGION_COLOR = "#00e5ff"; // Find-similar on an UNLABELLED annotation
const REGION_FILL_OPACITY = "0.2"; // the "slight tint" inside each region
const SVG_NS = "http://www.w3.org/2000/svg";

// predictView() returns the class as `class`; similarityByAnnotation() as `ref_label`.
function regionColor(result) {
  const label = result.class || result.ref_label;
  if (!label || !window.SlideProject) return FALLBACK_REGION_COLOR;
  return window.SlideProject.colorOf(label);
}

// Build the SVG overlay for a scored region: a faint fill over the qualifying
// tiles (`cells`) plus a crisp outline (`boundaries`). Both predict and "Find
// similar" return this shape data. The viewBox matches the scored region so OSD
// can stretch it to the right rectangle; non-scaling-stroke keeps the outline a
// constant width on screen at any zoom.
function buildShapesOverlay(result) {
  const r = result.region;
  const color = regionColor(result);
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
  fill.setAttribute("fill", color);
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
  outline.setAttribute("stroke", color);
  outline.setAttribute("stroke-width", "2");
  outline.setAttribute("vector-effect", "non-scaling-stroke");
  svg.appendChild(outline);
  return svg;
}

// Legend swatch in the overlay control, so the tint explains itself.
function setOverlaySwatch(color) {
  const sw = pdEl("overlay-swatch");
  if (sw) sw.style.background = color || "transparent";
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
  setOverlaySwatch(regionColor(result));
  pdEl("overlay-ctrl").hidden = false;
}

function clearOverlay() {
  removeOverlay();
  pdEl("overlay-ctrl").hidden = true;
  setOverlaySwatch(null);
  ocStatus("");
}

window.addEventListener("DOMContentLoaded", () => {
  const openBtn = pdEl("predict-btn");
  if (openBtn) {
    // The top-bar button only OPENS the panel — it stays enabled even when the gate
    // is shut, because the panel is the only place we can explain *why* it's shut
    // (a disabled button can't show a tooltip). What's gated is the prediction
    // itself: no auto-run below, and #predict-run is disabled.
    openBtn.addEventListener("click", async () => {
      window.SlidePanels.open("predict-panel");
      await refreshClasses();
      const blocked = gateReason();
      if (blocked) {
        pdStatus(blocked);
        return;
      }
      runPredict(); // predict immediately with the default class
    });
  }
  const runBtn = pdEl("predict-run");
  if (runBtn) runBtn.addEventListener("click", runPredict);

  const clearBtn = pdEl("overlay-clear");
  if (clearBtn) clearBtn.addEventListener("click", clearOverlay);

  const close = pdEl("predict-close");
  if (close) close.addEventListener("click", () => window.SlidePanels.close("predict-panel"));

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
// newly-available (or changed) model shows up without reopening the panel. Runs even
// when the panel is closed — the gate needs `hasModel` to stay current either way.
document.addEventListener("slideprobe:model-ready", () => {
  refreshClasses();
});

// Feature state changed (features.js owns it). Condition (a) of the gate.
document.addEventListener("slideprobe:features", (event) => {
  featStatus = event.detail;
  const was = pdFeaturesReady;
  pdFeaturesReady = event.detail.ready;
  // A sweep can finish while the panel sits open. Re-check the model too (it may have
  // trained in the meantime) and re-open the gate without the user reopening the
  // panel — but do NOT auto-predict: never fire a job the user walked away from.
  if (pdFeaturesReady !== was) {
    refreshClasses();
    return;
  }
  updatePredictGate();
});

document.addEventListener("slideprobe:slide-opened", (event) => {
  predictSlideId = event.detail.slideId; // null when the project has no slides
  // Features are per-slide, so BOTH conditions reset on a slide change — and they
  // reset synchronously, so the gate shuts on the same tick rather than after a
  // round-trip during which Predict would be wrongly clickable.
  pdFeaturesReady = false;
  featStatus = null;
  hasModel = false;
  clearOverlay();
  window.SlidePanels.close("predict-panel");
  updatePredictGate();
});

// A different project is active: the HEAD changed (the feature bank did not — it's
// shared). Reset synchronously; predict.js must never score with the old head.
document.addEventListener("slideprobe:project-opened", () => {
  hasModel = false;
  clearOverlay();
  window.SlidePanels.close("predict-panel");
  const select = pdEl("predict-class");
  if (select) select.innerHTML = "";
  updatePredictGate();
});
