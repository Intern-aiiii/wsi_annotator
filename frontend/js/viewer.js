// OpenSeadragon setup + DZI source (Phase 1; project-aware in Phase 7).
//
// Shows the slides of whichever PROJECT is open, and opens one of them in an
// OpenSeadragon viewer pointed at the backend's DeepZoom source.
//
// This file no longer boots itself. It used to auto-open slides[0] on DOMContentLoaded,
// which raced projects.js — and script order could NOT have fixed that, because
// projects.js's boot handler is async: it yields at its first `await` and this one
// would run immediately after, on the same tick. So projects.js is now the single boot
// driver and CALLS showProjectSlides() once every module has reset. See the event
// contract at the top of projects.js.

let viewer = null;

// Which slide is on screen. Named `openSlideId`, NOT `currentSlideId`: these scripts
// are plain <script>s sharing ONE global scope, and annotate.js already declares a
// top-level `currentSlideId`. Two top-level `let`s of the same name is a
// duplicate-declaration SyntaxError that silently kills the whole second file — you
// get a working viewer and an annotation module that never loaded. (Same reason
// annotate.js/predict.js use annoFeaturesReady/pdFeaturesReady rather than one name.)
let openSlideId = null;

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

// "This slide is now on screen" — the signal annotate.js / features.js / predict.js
// key all their per-slide state off. `slideId` is null when there is nothing open.
function announceSlide(slideId) {
  openSlideId = slideId;
  document.dispatchEvent(
    new CustomEvent("slideprobe:slide-opened", { detail: { slideId } })
  );
}

function openSlide(slideId) {
  setStatus(`opening ${slideId}…`);
  const osd = ensureViewer();
  osd.open(API.dziUrl(slideId));
  osd.addOnceHandler("open", () => {
    setStatus(`showing ${slideId}`);
    announceSlide(slideId);
  });
  osd.addOnceHandler("open-failed", () =>
    setStatus(`failed to open ${slideId} (is the slide readable?)`)
  );
}

// The project's slides, INTERSECTED with what is actually on disk, in the project's
// order. A project can name a slide whose file has since been deleted; without this
// filter the viewer would try to open it and the app would look dead.
function usableSlideIds(project, allSlides) {
  const onDisk = new Set(allSlides.map((s) => s.id));
  return (project.slides || []).filter((id) => onDisk.has(id));
}

// Called by projects.js AFTER every module has handled slideprobe:project-opened.
function showProjectSlides(project, allSlides) {
  const nameById = new Map(allSlides.map((s) => [s.id, s.name]));
  const usable = usableSlideIds(project, allSlides);

  const picker = document.getElementById("slide-picker");
  picker.innerHTML = "";
  picker.disabled = usable.length === 0;

  if (usable.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "— no slides —";
    picker.appendChild(opt);
    setStatus(`"${project.name}" has no slides — click Projects and add some.`);
    showNothing();
    return;
  }

  for (const id of usable) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = nameById.get(id) || id;
    picker.appendChild(opt);
  }

  // If the slide already on screen is also in the new project, KEEP it — don't re-open.
  // This is what makes switching projects feel like switching experiments rather than
  // reloading the app: the same tissue stays put, at the same zoom, and simply
  // re-annotates itself with the other project's regions.
  if (openSlideId && usable.includes(openSlideId)) {
    picker.value = openSlideId;
    setStatus(`showing ${openSlideId}`);
    announceSlide(openSlideId); // re-announce so the modules reload for THIS project
    return;
  }

  picker.value = usable[0];
  openSlide(usable[0]);
}

// No slide on screen (an empty project, or no project at all). Blank the viewer and
// SAY SO — otherwise features.js keeps polling the last slide and cheerfully reports
// "✓ 4,281 tissue tiles" for a slide that isn't open in this project.
function showNothing() {
  if (viewer) viewer.close();
  announceSlide(null);
}

window.addEventListener("DOMContentLoaded", () => {
  const picker = document.getElementById("slide-picker");
  if (picker) picker.addEventListener("change", () => openSlide(picker.value));
});

window.SlideViewer = { showProjectSlides, showNothing };
