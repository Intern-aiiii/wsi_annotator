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
// Class labels belong to the PROJECT (Phase 7), not to the browser. projects.js owns
// them — the list, each class's stored colour, and the edits — and this file just
// reads them. They used to live in localStorage with a colour derived from a class's
// INDEX in the list, which meant deleting one class silently repainted every class
// after it. The colour is now stored explicitly, per class, on the server.

let anno = null; // the Annotorious instance (created lazily on first slide)
let currentSlideId = null; // which slide's annotations we're editing/saving

// --- Class labels (delegated to the active project) --------------------------

// The canonical ordered class list. Empty until a project is open.
function allClasses() {
  return window.SlideProject ? window.SlideProject.classNames() : [];
}

function colorForClass(name) {
  return window.SlideProject ? window.SlideProject.colorOf(name) : "#a9a9a9";
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
  const classes = window.SlideProject ? window.SlideProject.classes() : [];
  styleEl.textContent = classes
    .map(({ name, color }) => {
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

  const classes = window.SlideProject ? window.SlideProject.classes() : [];
  if (!classes.length) {
    const empty = document.createElement("div");
    empty.className = "cp-empty";
    empty.textContent = "No classes — add one below.";
    list.appendChild(empty);
    return;
  }

  for (const { name, color } of classes) {
    const row = document.createElement("div");
    row.className = "cp-row";

    // The colour is stored per class on the server, so it can be EDITED — and the
    // change repaints both the annotations and the prediction overlay for that class.
    const swatch = document.createElement("input");
    swatch.type = "color";
    swatch.className = "cp-swatch";
    swatch.value = color;
    swatch.title = `colour for "${name}"`;
    swatch.addEventListener("change", () =>
      window.SlideProject.setClassColor(name, swatch.value)
    );

    const label = document.createElement("span");
    label.className = "cp-name";
    label.textContent = name;

    // Every class can be removed now — there are no immovable "presets", because the
    // preset list is just what a NEW project is seeded with. Removing is safe and
    // purely cosmetic: annotation tags are what training reads, so regions labelled
    // with a removed class keep their tag and keep training.
    const remove = document.createElement("button");
    remove.className = "cp-remove";
    remove.type = "button";
    remove.title = `remove "${name}" from this project's palette`;
    remove.textContent = "×";
    remove.addEventListener("click", () => window.SlideProject.removeClass(name));

    row.appendChild(swatch);
    row.appendChild(label);
    row.appendChild(remove);
    list.appendChild(row);
  }
}

// If the user typed a brand-new label straight into the tag box, register it with the
// project so it gets a colour and appears in the panel. We DON'T rebuild Annotorious
// here (that would yank the popup out from under the interaction) — colours and the
// panel update immediately, and the new label joins the autocomplete on the next
// rebuild, which the classes-changed listener performs.
function rememberTypedTags(annotations) {
  if (!window.SlideProject) return;
  const tags = annotations.map(tagOf).filter(Boolean);
  if (tags.length) window.SlideProject.ensureClasses(tags);
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

// Load (and display) THIS PROJECT's annotations for the slide now on screen.
async function loadForSlide(slideId) {
  currentSlideId = slideId;
  // Capture which project we're loading for. If the user switches project while this
  // request is in flight, the answer that comes back belongs to the OLD project and
  // must be dropped — otherwise project A's regions land on project B's slide.
  const pid = API.projectId();
  try {
    const annotations = await API.getAnnotations(slideId);
    if (API.projectId() !== pid || currentSlideId !== slideId) return; // stale; drop it
    // setAnnotations replaces the whole set, so this also clears the previous
    // slide's annotations when switching slides.
    anno.setAnnotations(annotations);
    rememberTypedTags(annotations); // adopt any label saved before the project knew it
    setAnnoStatus(
      annotations.length ? `${annotations.length} loaded` : "no annotations yet"
    );
  } catch (err) {
    setAnnoStatus(`could not load annotations: ${err.message}`);
  }
}

// --- "Find similar" editor widget -------------------------------------------

// A custom Annotorious editor widget: a "Find similar" button rendered inside
// the annotation popup (alongside the class TAG widget). Clicking it asks the
// backend to highlight regions of the current view whose embedding resembles
// THIS whole annotation — unsupervised, so it needs no trained model. We only
// dispatch an event here; predict.js does the API call + overlay, keeping the
// annotation module decoupled from the viewer/heatmap code.

// Does the CURRENT slide have a complete feature bank? (Prefixed `anno` because
// these scripts share one global scope — a bare `featuresReady` here would collide
// with predict.js's and throw a duplicate-declaration SyntaxError.)
// Does the CURRENT slide have a complete feature bank? Find similar compares this
// annotation's embedding against the visible tiles', so without one there is
// nothing to compare. features.js owns this; we mirror it to gate the button.
// (It needs no classifier, so it is NOT gated on a trained model.)
let annoFeaturesReady = false;
const FS_DISABLED_MSG =
  "Extract features for this slide first (top bar) — Find similar needs its feature bank.";

function findSimilarWidget(args) {
  const container = document.createElement("div");
  container.className = "find-similar-widget";
  // Tooltip on the CONTAINER, not the button: a disabled button fires no hover
  // events, so a title on it would never appear.
  container.title = annoFeaturesReady ? "" : FS_DISABLED_MSG;

  const button = document.createElement("button");
  button.type = "button";
  button.className = "find-similar-btn";
  button.textContent = "🔍 Find similar";
  button.disabled = !annoFeaturesReady;
  button.addEventListener("click", () => {
    // Prefer the plain W3C object (the backend parses its selector geometry);
    // fall back to the wrapper if `.underlying` isn't present.
    const annotation = (args.annotation && args.annotation.underlying) || args.annotation;
    document.dispatchEvent(
      new CustomEvent("slideprobe:find-similar", { detail: { annotation } })
    );
  });
  container.appendChild(button);
  if (!annoFeaturesReady) container.appendChild(findSimilarNote());
  return container;
}

function findSimilarNote() {
  const note = document.createElement("div");
  note.className = "find-similar-note";
  note.textContent = "Extract features first";
  return note;
}

// The button above is minted fresh for every editor popup, so there's no persistent
// handle to it. To update one that's ALREADY open, query the DOM. That's not a
// shortcut: Annotorious removes the popup node itself when it closes, so the query is
// self-cleaning, and it survives rebuildAnnotorious() (which destroys and recreates
// the whole instance) — neither of which a handle registry would do for free.
function updateFindSimilarButtons() {
  document.querySelectorAll(".find-similar-btn").forEach((button) => {
    button.disabled = !annoFeaturesReady;
    const box = button.closest(".find-similar-widget");
    if (!box) return;
    box.title = annoFeaturesReady ? "" : FS_DISABLED_MSG;
    const note = box.querySelector(".find-similar-note");
    if (annoFeaturesReady && note) note.remove();
    else if (!annoFeaturesReady && !note) box.appendChild(findSimilarNote());
  });
}

// --- Annotorious lifecycle --------------------------------------------------

// Build the Annotorious instance, its toolbar, and its event wiring.
function buildAnnotorious() {
  anno = OpenSeadragon.Annotorious(window.osdViewer, {
    // Editor widgets: the TAG widget is the class label (vocabulary = our class
    // list); the custom "Find similar" button drives the similarity heatmap.
    // `force: "plainjs"` is REQUIRED: recogito-client-core's getWidget otherwise
    // auto-detects widget type with a brittle regex over the function source and
    // misclassifies our plain-DOM widget (it calls document.createElement) as a
    // React component, so its returned node is discarded and nothing renders.
    widgets: [
      { widget: "TAG", vocabulary: allClasses() },
      { widget: findSimilarWidget, force: "plainjs" },
    ],
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
  //
  // NOTE this is also what saves a FREEHAND region: addAnnotation() fires nothing, but
  // the editor's OK on it fires updateAnnotation (not createAnnotation — the shape is
  // not a "selection"), so the save below happens for free. freehand.js must therefore
  // NOT call saveCurrent() itself; see SlideAnnotate.addPolygon.
  anno.on("createAnnotation", (a) => { pendingRegionId = null; saveCurrent(); notifyEditorClosed(a); });
  anno.on("updateAnnotation", (a) => { pendingRegionId = null; saveCurrent(); notifyEditorClosed(a); });
  anno.on("deleteAnnotation", (a) => { pendingRegionId = null; saveCurrent(); notifyEditorClosed(a); });
  anno.on("cancelSelected", onCancelSelected);
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
  if (btn) btn.addEventListener("click", () => window.SlidePanels.toggle("classes-panel"));

  const close = document.getElementById("classes-close");
  if (close) close.addEventListener("click", () => window.SlidePanels.close("classes-panel"));

  const form = document.getElementById("classes-add");
  const input = document.getElementById("classes-input");
  if (form && input) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const added = await window.SlideProject.addClass(input.value);
      if (added) input.value = "";
      input.focus();
    });
  }
});

// A different project is now active: forget everything about the old one. This is a
// SYNCHRONOUS RESET ONLY — the annotations themselves reload on the slide-opened event
// that projects.js fires immediately afterwards.
document.addEventListener("slideprobe:project-opened", () => {
  currentSlideId = null;
  annoFeaturesReady = false;
  setAnnoStatus("");
  applyClassColors();
  renderClassPanel();
  // The TAG widget's autocomplete vocabulary is baked in at construction, so the only
  // clean way to swap it for the new project's classes is to rebuild. Guard on `anno`:
  // on first boot it doesn't exist yet (it's created lazily when the first slide opens).
  if (anno) {
    rebuildAnnotorious();
    anno.setAnnotations([]);
  }
});

// The class list or a colour changed WITHIN the current project.
document.addEventListener("slideprobe:classes-changed", () => {
  applyClassColors();
  renderClassPanel();
  if (anno) rebuildAnnotorious(); // refresh the TAG autocomplete vocabulary
});

document.addEventListener("slideprobe:slide-opened", (event) => {
  const { slideId } = event.detail;
  // Features are per-slide: shut the Find-similar gate immediately on a slide change
  // and let features.js re-open it once it has confirmed the new slide's bank.
  annoFeaturesReady = false;
  updateFindSimilarButtons();

  // slideId is null when the project has no slides. Nothing to annotate.
  if (!slideId) {
    currentSlideId = null;
    if (anno) anno.setAnnotations([]);
    setAnnoStatus("");
    return;
  }

  if (!anno && !initAnnotorious()) return; // init failed; message already shown
  loadForSlide(slideId);
});

// features.js is the single source of truth for whether this slide has features.
document.addEventListener("slideprobe:features", (event) => {
  annoFeaturesReady = !!event.detail.ready;
  updateFindSimilarButtons(); // un-grey an already-open popup, no reopen needed
});

// --- Programmatic regions (used by the freehand tool) ------------------------
//
// annotate.js stays the ONLY owner of `anno` and `saveCurrent`. freehand.js talks to
// Annotorious exclusively through the facade below, and hands over nothing but a list
// of image-space points. That matters because rebuildAnnotorious() DESTROYS and
// recreates `anno` whenever the class list changes: a module that cached the instance
// would be holding a corpse. Reading the binding `anno` by name (as we do here) is
// always fresh.

// The freehand region that is currently in the editor but NOT yet saved. If the user
// dismisses the editor without giving it a class, we throw it away — matching how a
// cancelled rect/polygon disappears. Only THIS id is ever auto-removed, so cancelling
// the editor on an existing, saved annotation can never destroy it.
let pendingRegionId = null;
const editorClosedCallbacks = [];

function notifyEditorClosed(annotation) {
  for (const cb of editorClosedCallbacks) {
    try {
      cb(annotation);
    } catch (err) {
      console.error("editor-closed callback failed:", err);
    }
  }
}

function onCancelSelected(annotation) {
  const id = annotation && (annotation.id || (annotation.underlying || {}).id);
  if (id && id === pendingRegionId) {
    // Drawn, then abandoned: it was never saved, so just take it back off the layer.
    // No save needed — the layer now matches what's on disk again.
    anno.removeAnnotation(id);
    setAnnoStatus("region discarded (no class given)");
  }
  pendingRegionId = null;
  notifyEditorClosed(annotation);
}

// A stable, unique annotation id. `crypto.randomUUID` is undefined on a non-secure
// origin (this app is reachable over a plain-HTTP LAN address), so fall back rather
// than mint "#undefined" — the id is also the CROSS-VALIDATION GROUP KEY in
// classifier.py, and duplicate ids would merge two regions into one CV group and
// silently inflate the reported metrics.
function newAnnotationId() {
  const uuid = crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now().toString(16)}-${Math.random().toString(16).slice(2, 10)}`;
  return `#${uuid}`;
}

window.SlideAnnotate = {
  isReady: () => !!anno && !!currentSlideId,
  slideId: () => currentSlideId,
  status: setAnnoStatus,
  cancelSelected: () => { try { anno && anno.cancelSelected(); } catch { /* none open */ } },

  // Called when the editor closes for any reason (OK / Cancel / Delete). freehand.js
  // uses it to re-arm its sticky mode once the popup is out of the way.
  onEditorClosed(cb) {
    editorClosedCallbacks.push(cb);
  },

  // Add a polygon region from `points` ([[x, y], ...] in DZI base-image pixels) and
  // open the editor on it so the user can give it a class. Returns the new id.
  //
  // THREE THINGS HERE ARE LOAD-BEARING. Each one silently breaks the region if changed:
  //
  //  1. The SVG must be ONE LINE. Annotorious picks the editing tool with
  //     /^<svg.*<polygon/ — no `s` flag, so `.` does not cross newlines. Pretty-print
  //     this string and the shape still renders, but with NO vertex handles and no
  //     dragging, and nothing tells you why.
  //  2. fill-rule="evenodd". SVG fills with `nonzero` by default, but BOTH hit-testers
  //     (Annotorious's and backend/patches.py's `_point_in_polygon`) are even-odd ray
  //     casts. A lasso that crosses itself would then RENDER solid but TRAIN with a
  //     hole punched in it — a silent disagreement between what you see and what the
  //     model learns.
  //  3. NO saveCurrent() here. The editor's OK fires updateAnnotation, which is already
  //     wired to saveCurrent (see buildAnnotorious). Saving here instead would (a) run
  //     rememberTypedTags -> ensureClasses -> "classes-changed" -> rebuildAnnotorious,
  //     which destroys the very instance whose editor we just opened, and (b) land
  //     inside a ~1ms window where Annotorious has the shape in the DOM twice, so the
  //     region would be written to disk twice.
  addPolygon(points) {
    if (!this.isReady() || !points || points.length < 3) return null;

    const round = (n) => Math.round(n * 10) / 10;
    // "x,y x,y" — comma WITHIN a pair, space BETWEEN pairs. Annotorious's area helper
    // splits on space then comma, so a space-only separator yields NaN.
    const pts = points.map(([x, y]) => `${round(x)},${round(y)}`).join(" ");

    const item = window.osdViewer.world.getItemAt(0);
    const annotation = {
      "@context": "http://www.w3.org/ns/anno.jsonld",
      type: "Annotation",
      id: newAnnotationId(),
      body: [],
      target: {
        // Annotorious REQUIRES target.source; without it its parser throws on load.
        source: (item && item.source && item.source["@id"]) || document.baseURI,
        selector: {
          type: "SvgSelector",
          value: `<svg><polygon fill-rule="evenodd" points="${pts}"/></svg>`,
        },
      },
    };

    anno.addAnnotation(annotation);
    pendingRegionId = annotation.id;
    anno.selectAnnotation(annotation.id); // opens the editor so it can be tagged
    return annotation.id;
  },
};
