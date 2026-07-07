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
};
