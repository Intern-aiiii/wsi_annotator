// fetch() calls to the backend.
//
// The single place the frontend talks to the Python server, so no other file
// hard-codes URLs. Exposes a small global `API` object.
//
// The pipeline the calls below follow: pick a project -> open one of its slides ->
// annotate (saving also kicks a background retrain) -> extract features once ->
// predict or find-similar.
//
// TWO KINDS OF ROUTE (Phase 7), and the split is the whole architecture:
//
//   PROJECT-SCOPED — about the user's EXPERIMENT. Annotations, classes, the trained
//     head, predict, find-similar. Each project has its own, so two experiments over
//     the same slides never contaminate each other.
//
//   GLOBAL — about a FILE on disk, or a cache derived only from that file. The slide
//     list, the DeepZoom tiles, and THE FEATURE BANK.
//
// The feature bank being global is the load-bearing decision. It is keyed by
// (slide, embedding model, tissue mask) — nothing about an experiment enters it — so
// a slide swept once with Virchow 2 (minutes of GPU) is reused by every project that
// contains it. Scoping it per project would re-run the single most expensive step in
// the pipeline once per project, for no gain.

// The ONE project the whole UI is looking at. Held here rather than passed to every
// call because it is genuinely global UI state (exactly one project is open at a
// time), and because it keeps every existing call site unchanged — getAnnotations
// (slideId) still takes just a slide id, it simply resolves to a different URL now.
// projects.js is the ONLY writer.
let ACTIVE_PROJECT = null;

// Build a project-scoped URL. Throws LOUDLY rather than quietly fetching
// "/api/projects/null/..." and returning a baffling 404, so a boot-ordering mistake
// announces itself the first time it happens instead of looking like a server bug.
function projectPath(rest) {
  if (!ACTIVE_PROJECT) {
    throw new Error("no active project yet (has projects.js finished booting?)");
  }
  return `/api/projects/${encodeURIComponent(ACTIVE_PROJECT)}${rest}`;
}

const API = {
  // --- Which project are we in? (local only, no network) ---------------------

  setProject(projectId) {
    ACTIVE_PROJECT = projectId || null;
  },

  projectId() {
    return ACTIVE_PROJECT;
  },

  // --- Projects (Phase 7) ---------------------------------------------------

  // [{ id, name, created_at, n_slides, n_classes }, ...]
  async listProjects() {
    const res = await fetch("/api/projects");
    if (!res.ok) throw new Error(`listProjects failed: HTTP ${res.status}`);
    const data = await res.json();
    return data.projects || [];
  },

  // One project in full: { id, name, created_at, slides: [...], classes: [{name,color}] }.
  // Deliberately ONE call — the UI needs classes AND slides before it can open
  // anything, and two calls would be two chances to race a project switch.
  async getProject(projectId) {
    const res = await fetch(`/api/projects/${encodeURIComponent(projectId)}`);
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`getProject failed: HTTP ${res.status}`);
    return res.json();
  },

  // Omit `slides` to include every slide currently on disk.
  async createProject(name, slides) {
    const body = { name };
    if (slides) body.slides = slides;
    const res = await fetch("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `createProject failed: HTTP ${res.status}`);
    return data;
  },

  // Rename. The project's id and its directory on disk never move.
  async renameProject(projectId, name) {
    const res = await fetch(`/api/projects/${encodeURIComponent(projectId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `renameProject failed: HTTP ${res.status}`);
    return data;
  },

  // Deletes the project's annotations, classes and head. Never the slides, and never
  // the shared feature bank — another project over the same slides is unaffected and
  // does not have to re-sweep.
  async deleteProject(projectId) {
    const res = await fetch(`/api/projects/${encodeURIComponent(projectId)}`, {
      method: "DELETE",
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `deleteProject failed: HTTP ${res.status}`);
    return data;
  },

  async addProjectSlide(projectId, slideId) {
    const res = await fetch(`/api/projects/${encodeURIComponent(projectId)}/slides`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slide_id: slideId }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `addProjectSlide failed: HTTP ${res.status}`);
    return data;
  },

  // Refuses with HTTP 409 if the project still holds annotations for the slide,
  // unless force=true — a mis-click must not silently destroy drawn work. With force,
  // only THIS project's annotation file for the slide is deleted.
  async removeProjectSlide(projectId, slideId, force) {
    const url =
      `/api/projects/${encodeURIComponent(projectId)}/slides/${encodeURIComponent(slideId)}` +
      (force ? "?force=true" : "");
    const res = await fetch(url, { method: "DELETE" });
    const data = await res.json().catch(() => ({}));
    if (res.status === 409) return { status: "has_annotations" };
    if (!res.ok) throw new Error(data.error || `removeProjectSlide failed: HTTP ${res.status}`);
    return data;
  },

  // Replace the whole class list: [{ name, color }, ...]. Same "send the whole
  // collection" convention as annotations, so there is one convention, not two.
  async setClasses(projectId, classes) {
    const res = await fetch(`/api/projects/${encodeURIComponent(projectId)}/classes`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ classes }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `setClasses failed: HTTP ${res.status}`);
    return data.classes || [];
  },

  // --- GLOBAL: slides on disk ------------------------------------------------

  // Return [{ id, name }, ...] for EVERY slide in data/slides/, regardless of
  // project. This is what the "add slides to this project" picker offers.
  async listSlides() {
    const res = await fetch("/api/slides");
    if (!res.ok) throw new Error(`listSlides failed: HTTP ${res.status}`);
    const data = await res.json();
    return data.slides || [];
  },

  // The DeepZoom tile-source URL OpenSeadragon should open for a slide.
  dziUrl(slideId) {
    return `/slides/${encodeURIComponent(slideId)}.dzi`;
  },

  // --- Annotations (project-scoped) -----------------------------------------

  // Return the active project's annotations (W3C JSON array) for a slide.
  async getAnnotations(slideId) {
    const res = await fetch(projectPath(`/slides/${encodeURIComponent(slideId)}/annotations`));
    if (!res.ok) throw new Error(`getAnnotations failed: HTTP ${res.status}`);
    const data = await res.json();
    return data.annotations || [];
  },

  // Overwrite this project's annotations for a slide. Sends the whole collection
  // each time (the backend does a plain replace) and kicks a debounced retrain.
  async saveAnnotations(slideId, annotations) {
    const res = await fetch(
      projectPath(`/slides/${encodeURIComponent(slideId)}/annotations`),
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ annotations }),
      }
    );
    if (!res.ok) throw new Error(`saveAnnotations failed: HTTP ${res.status}`);
    return res.json();
  },

  // --- GLOBAL: whole-slide feature extraction --------------------------------
  // The one slow step, and the only one the user starts by hand. It embeds every
  // tissue tile of the slide once; training and prediction are then just lookups.
  //
  // NOT project-scoped, on purpose (see the header): the bank belongs to the SLIDE.
  // Sweep a slide once and every project containing it can train and predict
  // immediately — which is also why clearing a bank affects them all.

  // Start the sweep. Returns IMMEDIATELY with { status: "started" | "busy" } — the
  // work happens on the server. Poll featureState() for progress and stop it with
  // cancelFeatures(); there is no long-lived request here, so nothing to abort
  // client-side. Safe to call again: it resumes rather than restarting.
  async startFeatures(slideId) {
    const res = await fetch(
      `/api/slides/${encodeURIComponent(slideId)}/features`,
      { method: "POST" }
    );
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `startFeatures failed: HTTP ${res.status}`);
    return data;
  },

  // This slide's feature state + live sweep progress:
  //   { state: "none"|"partial"|"complete", n_cells, n_covered, n_tissue, progress,
  //     running_slide_id, sweep: { state, done, total, n_embedded, detail } }
  // Predict and Find similar require state === "complete".
  async featureState(slideId) {
    const res = await fetch(`/api/slides/${encodeURIComponent(slideId)}/features`);
    if (!res.ok) throw new Error(`featureState failed: HTTP ${res.status}`);
    return res.json();
  },

  // Ask the running sweep to stop. Whatever it already embedded is kept, and the
  // next startFeatures() carries on from there.
  async cancelFeatures(slideId) {
    const res = await fetch(
      `/api/slides/${encodeURIComponent(slideId)}/features/cancel`,
      { method: "POST" }
    );
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `cancelFeatures failed: HTTP ${res.status}`);
    return data;
  },

  // --- Classifier (project-scoped) ------------------------------------------

  // Metadata + metrics for THIS PROJECT's trained head, or null if none trained yet.
  async getModel() {
    const res = await fetch(projectPath("/model"));
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`getModel failed: HTTP ${res.status}`);
    return res.json();
  },

  // --- Predict / region overlays (project-scoped) ---------------------------

  // Score the given image-pixel region {x,y,w,h} for `cls` with this project's head.
  // Returns the summary (region covered, grid, counts, class) plus the confident-region
  // shapes: `cells` ([x,y,w,h] tile rects, for the tint) and `boundaries`
  // ([x1,y1,x2,y2] segments, for the outline).
  async predictView(slideId, region, cls) {
    const body = { ...region };
    if (cls) body.class = cls;
    const res = await fetch(
      projectPath(`/slides/${encodeURIComponent(slideId)}/predict`),
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    );
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `predict failed: HTTP ${res.status}`);
    return data;
  },

  // Unsupervised: regions of the visible {x,y,w,h} most similar to a selected
  // annotation (same {cells, boundaries} shape as predictView). `annotation` is
  // the full W3C annotation object. Needs no trained model — only the feature bank.
  //
  // Project-scoped even though similarity doesn't actually need the project: it is a
  // twin of Predict in the UI, and two URL conventions for two adjacent buttons is a
  // permanent trip hazard. The backend validates the project and ignores it.
  async similarityByAnnotation(slideId, region, annotation) {
    const res = await fetch(
      projectPath(`/slides/${encodeURIComponent(slideId)}/similarity`),
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...region, annotation }) }
    );
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `similarity failed: HTTP ${res.status}`);
    return data;
  },

  // --- Background learning status (project-scoped) --------------------------

  // State of the auto-learning worker FOR THIS PROJECT:
  //   { state, detail, updated, queued, training }
  // state is one of: idle | learning | ready | waiting | error.
  // `queued` matters: without it the chip shows a stale "model ready" for the whole
  // debounce window after an edit, and the user concludes the save didn't take.
  async learnStatus() {
    const res = await fetch(projectPath("/learn/status"));
    if (!res.ok) throw new Error(`learnStatus failed: HTTP ${res.status}`);
    return res.json();
  },
};
