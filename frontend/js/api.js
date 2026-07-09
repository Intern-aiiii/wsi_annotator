// fetch() calls to the backend.
//
// The single place the frontend talks to the Python server, so no other file
// hard-codes URLs. Exposes a small global `API` object.
//
// Phase 1 needs just one call: list the available slides. More are added as
// later phases land (save/load annotations, train, heatmap).

const API = {
  // Return [{ id, name }, ...] for the slides in data/slides/.
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

  // --- Annotations (Phase 2) ------------------------------------------------

  // Return the saved annotations (W3C JSON array) for a slide.
  async getAnnotations(slideId) {
    const res = await fetch(
      `/api/slides/${encodeURIComponent(slideId)}/annotations`
    );
    if (!res.ok) throw new Error(`getAnnotations failed: HTTP ${res.status}`);
    const data = await res.json();
    return data.annotations || [];
  },

  // Overwrite a slide's annotations with the given list. Sends the whole
  // collection each time (the backend does a plain replace).
  async saveAnnotations(slideId, annotations) {
    const res = await fetch(
      `/api/slides/${encodeURIComponent(slideId)}/annotations`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ annotations }),
      }
    );
    if (!res.ok) throw new Error(`saveAnnotations failed: HTTP ${res.status}`);
    return res.json();
  },

  // --- Patch extraction (Phase 3) -------------------------------------------

  // Run patch extraction for a slide; returns the manifest summary (counts,
  // per_class, ...). This can take a while on big slides (it reads many tiles).
  async extractPatches(slideId) {
    const res = await fetch(
      `/api/slides/${encodeURIComponent(slideId)}/patches`,
      { method: "POST" }
    );
    if (!res.ok) throw new Error(`extractPatches failed: HTTP ${res.status}`);
    return res.json();
  },

  // URL of the preview montage from the last extraction. Cache-busted so a
  // re-run shows the fresh image.
  patchesPreviewUrl(slideId) {
    return `/api/slides/${encodeURIComponent(slideId)}/patches/preview.jpg?t=${Date.now()}`;
  },

  // --- Embeddings (Phase 4) -------------------------------------------------

  // Embed the slide's extracted patches with the active model (cached
  // server-side). Returns a summary (model_id, dim, counts, per_class, timing).
  // On a backend error (e.g. the model isn't available) throws with the message.
  async computeEmbeddings(slideId) {
    const res = await fetch(
      `/api/slides/${encodeURIComponent(slideId)}/embeddings`,
      { method: "POST" }
    );
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || `computeEmbeddings failed: HTTP ${res.status}`);
    }
    return data;
  },

  // --- Classifier (Phase 5) -------------------------------------------------

  // Train the head on all cached embeddings for the active model. Returns the
  // training summary (metrics, counts). Throws with the backend message on 400.
  async trainClassifier() {
    const res = await fetch("/api/train", { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `train failed: HTTP ${res.status}`);
    return data;
  },

  // Metadata + metrics for the trained head, or null if none trained yet.
  async getModel() {
    const res = await fetch("/api/model");
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`getModel failed: HTTP ${res.status}`);
    return res.json();
  },

  // --- Predict / region overlays (Phase 6) ----------------------------------

  // Score the given image-pixel region {x,y,w,h} for `cls`. Returns the summary
  // (region covered, grid, counts, class) plus the confident-region shapes:
  // `cells` ([x,y,w,h] tile rects, for the tint) and `boundaries` ([x1,y1,x2,y2]
  // segments, for the outline) of the tiles predicted `cls` with >= min_confidence.
  async predictView(slideId, region, cls) {
    const body = { ...region };
    if (cls) body.class = cls;
    const res = await fetch(
      `/api/slides/${encodeURIComponent(slideId)}/predict`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    );
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `predict failed: HTTP ${res.status}`);
    return data;
  },

  // Unsupervised: regions of the visible {x,y,w,h} most similar to a selected
  // annotation (same {cells, boundaries} shape as predictView). `annotation` is
  // the full W3C annotation object. Needs no trained model.
  async similarityByAnnotation(slideId, region, annotation) {
    const res = await fetch(
      `/api/slides/${encodeURIComponent(slideId)}/similarity`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...region, annotation }) }
    );
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `similarity failed: HTTP ${res.status}`);
    return data;
  },

  // --- Background learning status (Phase 7) ---------------------------------

  // Current state of the auto-learning worker: { state, detail, updated }.
  // state is one of: idle | learning | ready | waiting | error.
  async learnStatus() {
    const res = await fetch("/api/learn/status");
    if (!res.ok) throw new Error(`learnStatus failed: HTTP ${res.status}`);
    return res.json();
  },
};
