// Annotorious integration: draw regions, label them, save/load (Phase 2).
//
// viewer.js creates the OpenSeadragon viewer and, each time a slide finishes
// opening, dispatches a "slideprobe:slide-opened" event. We hook that to:
//   - initialize Annotorious once (on the first slide) and wire up saving, and
//   - (re)load the annotations belonging to whichever slide is now on screen.
//
// Each annotation carries a class label (e.g. "gland") entered via Annotorious's
// TAG widget. That label is stored in the standard W3C annotation body, so the
// saved JSON already has region + label — the shape Phase 3 (patch extraction)
// and Phase 5 (training) need.
//
// Class labels: a built-in PRESET list plus any custom classes you add in the
// "Classes" panel. Custom classes are persisted in the browser (localStorage),
// and each class gets a stable color so regions are visually distinct by class.

let anno = null; // the Annotorious instance (created lazily on first slide)
let currentSlideId = null; // which slide's annotations we're editing/saving

// --- Class labels -----------------------------------------------------------

// Common H&E categories offered out of the box. The TAG widget is free-text, so
// these are only a starting point — add your own in the Classes panel.
const PRESET_CLASSES = [
  "gland",
  "epithelium",
  "stroma",
  "tumor",
  "necrosis",
  "lymphocytes",
  "blood vessel",
  "adipose",
  "background",
];

const CUSTOM_CLASSES_KEY = "slideprobe.customClasses";

// A fixed palette; class N gets color N (wrapping if there are more classes than
// colors). Kept distinct and reasonably colorblind-friendly.
const CLASS_PALETTE = [
  "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
  "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
  "#9A6324", "#800000", "#808000", "#000075", "#a9a9a9",
];

function loadCustomClasses() {
  try {
    const raw = localStorage.getItem(CUSTOM_CLASSES_KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

function saveCustomClasses(list) {
  localStorage.setItem(CUSTOM_CLASSES_KEY, JSON.stringify(list));
}

// Presets first, then custom classes, de-duplicated case-insensitively. This is
// the canonical ordered class list everything else derives from.
function allClasses() {
  const seen = new Set();
  const out = [];
  for (const name of [...PRESET_CLASSES, ...loadCustomClasses()]) {
    const trimmed = String(name).trim();
    const key = trimmed.toLowerCase();
    if (!trimmed || seen.has(key)) continue;
    seen.add(key);
    out.push(trimmed);
  }
  return out;
}

function isPreset(name) {
  return PRESET_CLASSES.some((p) => p.toLowerCase() === name.toLowerCase());
}

function colorForClass(name) {
  const classes = allClasses();
  const idx = classes.findIndex((c) => c.toLowerCase() === name.toLowerCase());
  // Unknown label (typed but not in the list) falls at the end of the palette.
  return CLASS_PALETTE[(idx >= 0 ? idx : classes.length) % CLASS_PALETTE.length];
}

// A CSS-safe class name derived from a label, used both by the formatter (which
// tags each annotation's SVG element) and by the injected color stylesheet.
function slugForClass(name) {
  return (
    "sp-cls-" +
    name.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "")
  );
}

// --- Coloring annotations by class ------------------------------------------

// Pull the class label out of a W3C annotation's tagging body. Annotorious hands
// the formatter its internal object, so we look in a few likely spots.
function tagOf(annotation) {
  if (!annotation) return null;
  // Prefer .bodies, but fall back to .body — note an empty [] is truthy, so we
  // must check length rather than rely on ||.
  let raw = annotation.bodies;
  if (!raw || (Array.isArray(raw) && raw.length === 0)) raw = annotation.body;
  if (!raw && annotation.underlying) raw = annotation.underlying.body;
  const arr = Array.isArray(raw) ? raw : raw ? [raw] : [];
  const tag = arr.find((b) => b && b.purpose === "tagging");
  return tag ? tag.value : null;
}

// Annotorious formatter: return a CSS class string, which it adds to the
// annotation's SVG element. Our injected stylesheet then colors it by class.
function annotationFormatter(annotation) {
  const tag = tagOf(annotation);
  return tag ? slugForClass(tag) : undefined;
}

// (Re)write a <style> block giving each class its stroke/fill color. Broad
// selectors cover the class landing on either the group or the inner shape.
function applyClassColors() {
  let styleEl = document.getElementById("class-colors");
  if (!styleEl) {
    styleEl = document.createElement("style");
    styleEl.id = "class-colors";
    document.head.appendChild(styleEl);
  }
  styleEl.textContent = allClasses()
    .map((name) => {
      const color = colorForClass(name);
      const cls = slugForClass(name);
      return (
        `.${cls} .a9s-inner, .a9s-inner.${cls}{` +
        `stroke:${color} !important;fill:${color} !important;fill-opacity:.25 !important;}`
      );
    })
    .join("\n");
}

// --- Classes panel UI -------------------------------------------------------

function renderClassPanel() {
  const list = document.getElementById("classes-list");
  if (!list) return;
  list.innerHTML = "";

  for (const name of allClasses()) {
    const row = document.createElement("div");
    row.className = "cp-row";

    const swatch = document.createElement("span");
    swatch.className = "cp-swatch";
    swatch.style.background = colorForClass(name);

    const label = document.createElement("span");
    label.className = "cp-name";
    label.textContent = name;

    row.appendChild(swatch);
    row.appendChild(label);

    if (isPreset(name)) {
      const tag = document.createElement("span");
      tag.className = "cp-tag";
      tag.textContent = "preset";
      row.appendChild(tag);
    } else {
      // Custom classes can be removed; presets can't.
      const remove = document.createElement("button");
      remove.className = "cp-remove";
      remove.type = "button";
      remove.title = `remove "${name}"`;
      remove.textContent = "×";
      remove.addEventListener("click", () => removeClass(name));
      row.appendChild(remove);
    }

    list.appendChild(row);
  }
}

// Add a class if it's new (case-insensitive). Returns true if it was added.
function addClass(name) {
  const trimmed = name.trim();
  if (!trimmed) return false;
  if (allClasses().some((c) => c.toLowerCase() === trimmed.toLowerCase())) return false;
  saveCustomClasses([...loadCustomClasses(), trimmed]);
  onClassesChanged();
  return true;
}

function removeClass(name) {
  saveCustomClasses(
    loadCustomClasses().filter((c) => c.toLowerCase() !== name.toLowerCase())
  );
  onClassesChanged();
}

// Called whenever the class list changes: refresh colors + panel, and rebuild
// Annotorious so the TAG widget's autocomplete offers the new vocabulary.
function onClassesChanged() {
  applyClassColors();
  renderClassPanel();
  if (anno) rebuildAnnotorious();
}

// If the user typed a brand-new label directly into the tag box, remember it as
// a custom class so it gets a color and shows up in the panel. We DON'T rebuild
// here (that would disrupt the interaction) — colors/panel update immediately;
// the new label joins the autocomplete list on the next rebuild/reload.
function rememberTypedTags(annotations) {
  const known = new Set(allClasses().map((c) => c.toLowerCase()));
  const fresh = [];
  for (const a of annotations) {
    const tag = tagOf(a);
    if (tag && !known.has(tag.toLowerCase())) {
      known.add(tag.toLowerCase());
      fresh.push(tag);
    }
  }
  if (fresh.length) {
    saveCustomClasses([...loadCustomClasses(), ...fresh]);
    applyClassColors();
    renderClassPanel();
  }
}

// --- Saving / loading -------------------------------------------------------

function setAnnoStatus(message) {
  const el = document.getElementById("anno-status");
  if (el) el.textContent = message;
}

// Push the CURRENT full set of annotations to the backend. We always send the
// whole collection (the server does a plain replace), so create/update/delete
// all funnel through here.
async function saveCurrent() {
  if (!anno || !currentSlideId) return;
  const annotations = anno.getAnnotations();
  rememberTypedTags(annotations);
  setAnnoStatus("saving…");
  try {
    await API.saveAnnotations(currentSlideId, annotations);
    setAnnoStatus(`saved (${annotations.length})`);
  } catch (err) {
    setAnnoStatus(`save failed: ${err.message}`);
  }
}

// Load (and display) the annotations for the slide now on screen.
async function loadForSlide(slideId) {
  currentSlideId = slideId;
  try {
    const annotations = await API.getAnnotations(slideId);
    // setAnnotations replaces the whole set, so this also clears the previous
    // slide's annotations when switching slides.
    anno.setAnnotations(annotations);
    rememberTypedTags(annotations); // color any labels saved before we knew them
    setAnnoStatus(
      annotations.length ? `${annotations.length} loaded` : "no annotations yet"
    );
  } catch (err) {
    setAnnoStatus(`could not load annotations: ${err.message}`);
  }
}

// --- Annotorious lifecycle --------------------------------------------------

// Build the Annotorious instance, its toolbar, and its event wiring.
function buildAnnotorious() {
  anno = OpenSeadragon.Annotorious(window.osdViewer, {
    // The TAG widget is the class label; its vocabulary is our class list.
    widgets: [{ widget: "TAG", vocabulary: allClasses() }],
    // Color each annotation by its class label.
    formatter: annotationFormatter,
  });

  // Rect/polygon draw buttons + a draw-vs-pan mode toggle, rendered into the
  // topbar container. Optional — if the toolbar plugin didn't load, drawing
  // still works, you just won't have the shape buttons.
  const toolbar = window.Annotorious && window.Annotorious.Toolbar;
  if (toolbar) toolbar(anno, document.getElementById("anno-toolbar"));

  // Any change to the set → save the whole collection. Loading annotations via
  // setAnnotations() does NOT fire these, so there's no save loop.
  anno.on("createAnnotation", saveCurrent);
  anno.on("updateAnnotation", saveCurrent);
  anno.on("deleteAnnotation", saveCurrent);
}

// Recreate Annotorious (the only clean way to refresh the TAG vocabulary),
// preserving whatever is currently displayed.
function rebuildAnnotorious() {
  const current = anno ? anno.getAnnotations() : [];
  if (anno) {
    anno.destroy();
    anno = null;
  }
  // destroy() leaves the toolbar's buttons behind; clear them before re-adding.
  const toolbarEl = document.getElementById("anno-toolbar");
  if (toolbarEl) toolbarEl.innerHTML = "";
  buildAnnotorious();
  anno.setAnnotations(current);
}

// Create Annotorious once, the first time a slide opens (the OSD viewer must
// exist before we can attach to it). Returns true on success; on failure the
// reason is shown in the top bar instead of failing silently.
function initAnnotorious() {
  if (typeof OpenSeadragon.Annotorious !== "function") {
    setAnnoStatus("annotation plugin failed to load (offline? check network/console)");
    return false;
  }
  try {
    buildAnnotorious();
    applyClassColors();
    renderClassPanel();
    return true;
  } catch (err) {
    setAnnoStatus(`annotation init failed: ${err.message}`);
    return false;
  }
}

// --- Wiring -----------------------------------------------------------------

// Classes panel: toggle button + add form. Wired once, up front.
window.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("classes-btn");
  const panel = document.getElementById("classes-panel");
  if (btn && panel) {
    btn.addEventListener("click", () => (panel.hidden = !panel.hidden));
  }

  const form = document.getElementById("classes-add");
  const input = document.getElementById("classes-input");
  if (form && input) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      if (addClass(input.value)) input.value = "";
      input.focus();
    });
  }
});

document.addEventListener("slideprobe:slide-opened", (event) => {
  const { slideId } = event.detail;
  if (!anno && !initAnnotorious()) return; // init failed; message already shown
  loadForSlide(slideId);
});
