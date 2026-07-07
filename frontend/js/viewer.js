// OpenSeadragon setup + DZI source (Phase 1).
//
// Fetches the list of available slides, fills the picker, and opens the first
// one in an OpenSeadragon viewer pointed at the backend's DeepZoom source.
// Switching the picker re-opens the viewer on the chosen slide.

let viewer = null;

function setStatus(message) {
  const el = document.getElementById("status");
  if (el) el.textContent = message;
}

// Create the OpenSeadragon viewer once; reuse it for subsequent slides.
function ensureViewer() {
  if (viewer) return viewer;
  viewer = OpenSeadragon({
    id: "viewer",
    prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/images/",
    showNavigator: true,
    animationTime: 0.3
    // The backend serves plain .dzi tile sources; defaults are fine otherwise.
  });
  // Expose the instance so annotate.js (Phase 2) can attach Annotorious to it.
  window.osdViewer = viewer;
  return viewer;
}

function openSlide(slideId) {
  setStatus(`opening ${slideId}…`);
  const osd = ensureViewer();
  osd.open(API.dziUrl(slideId));
  osd.addOnceHandler("open", () => {
    setStatus(`showing ${slideId}`);
    // Tell annotate.js (Phase 2) which slide is now on screen so it can load
    // that slide's annotations and route saves to the right file.
    document.dispatchEvent(
      new CustomEvent("slideprobe:slide-opened", { detail: { slideId } })
    );
  });
  osd.addOnceHandler("open-failed", () =>
    setStatus(`failed to open ${slideId} (is the slide readable?)`)
  );
}

async function init() {
  let slides = [];
  try {
    slides = await API.listSlides();
  } catch (err) {
    setStatus(`could not reach backend: ${err.message}`);
    return;
  }

  if (slides.length === 0) {
    setStatus("no slides found — drop a WSI into data/slides/ and reload");
    return;
  }

  // Populate the picker.
  const picker = document.getElementById("slide-picker");
  picker.innerHTML = "";
  for (const s of slides) {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = s.name;
    picker.appendChild(opt);
  }
  picker.addEventListener("change", () => openSlide(picker.value));

  // Open the first slide automatically.
  openSlide(slides[0].id);
}

window.addEventListener("DOMContentLoaded", init);
