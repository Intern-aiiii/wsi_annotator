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
};
