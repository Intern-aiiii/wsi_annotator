// Projects: named workspaces, and the app's BOOT DRIVER (Phase 7).
//
// A project owns its annotations, its class list, and its own trained head, so two
// experiments over the same slides can coexist instead of contaminating one pooled
// classifier. It does NOT own the slides (files on disk) or the feature bank — the
// bank is keyed by (slide, embedding model, tissue mask) and is deliberately SHARED,
// so a slide swept once with Virchow 2 is reused by every project containing it.
//
// THE EVENT CONTRACT (three lines; the rest of the frontend depends on them):
//
//   slideprobe:project-opened   "forget everything about the old project."
//                               Handlers must be SYNCHRONOUS STATE RESETS ONLY — they
//                               must not open a slide or start long work.
//                               detail = { project: {id, name, slides, classes} }
//
//   slideprobe:slide-opened     "this slide is now on screen; load its stuff."
//                               Unchanged, EXCEPT slideId is now `null` when the
//                               active project has no slides. Every consumer must
//                               treat null as "nothing on screen".
//
//   slideprobe:classes-changed  the class list/colours changed WITHIN this project.
//                               detail = { classes }. Deliberately not a re-dispatch
//                               of project-opened, which would needlessly re-open the
//                               slide and flicker the viewer.
//
// WHY THIS FILE OWNS BOOT, and why script order alone could not fix it:
// viewer.js used to auto-open slides[0] on DOMContentLoaded. Registering our listener
// first would NOT have helped — ours is async, so it yields at its first `await` and
// viewer's handler runs immediately after, on the same tick. So viewer.js no longer
// boots itself: it exposes SlideViewer.showProjectSlides() and we CALL it, after
// dispatchEvent has returned. dispatchEvent is synchronous, so by then every module
// has finished resetting. Two strict phases, no timers, no setTimeout(…, 0).

(function () {
  const ACTIVE_KEY = "slideprobe.activeProject";

  let ALL_SLIDES = [];   // every slide on disk (global; the "add slides" source)
  let PROJECTS = [];     // the summary list, for the picker
  let ACTIVE = null;     // the full active project: {id, name, slides, classes}

  const el = (id) => document.getElementById(id);

  function setProjStatus(msg, isError) {
    const s = el("projects-status");
    if (!s) return;
    s.textContent = msg || "";
    s.className = isError ? "err" : "";
  }

  // --- Colour -----------------------------------------------------------------

  // A label that isn't (yet) a project class still has to be drawn in SOMETHING. Hash
  // its NAME to a colour, never its position in a list: position is exactly what the
  // old colorForClass() used, which is why deleting one class silently repainted every
  // class after it.
  const FALLBACK_PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
  ];

  function colorOf(name) {
    const key = String(name || "").trim().toLowerCase();
    const hit = (ACTIVE ? ACTIVE.classes : []).find(
      (c) => c.name.toLowerCase() === key
    );
    if (hit) return hit.color;
    let h = 0;
    for (const ch of key) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    return FALLBACK_PALETTE[h % FALLBACK_PALETTE.length];
  }

  const classes = () => (ACTIVE ? ACTIVE.classes.slice() : []);
  const classNames = () => classes().map((c) => c.name);

  // --- Class edits (the project is the source of truth; the server assigns colours) --

  async function putClasses(next) {
    if (!ACTIVE) return;
    try {
      ACTIVE.classes = await API.setClasses(ACTIVE.id, next);
      renderPanel();
      document.dispatchEvent(
        new CustomEvent("slideprobe:classes-changed", {
          detail: { classes: classes() },
        })
      );
    } catch (err) {
      setProjStatus(err.message, true);
    }
  }

  // Add a class if it's new (case-insensitive). A blank `color` tells the server to
  // pick the first UNUSED palette entry.
  async function addClass(name) {
    const trimmed = String(name || "").trim();
    if (!trimmed || !ACTIVE) return false;
    if (classNames().some((c) => c.toLowerCase() === trimmed.toLowerCase())) return false;
    await putClasses([...classes(), { name: trimmed, color: "" }]);
    return true;
  }

  // Removing a class is purely COSMETIC. Annotation tags are the source of truth for
  // training, so regions labelled with it keep their tag and keep training — they just
  // fall back to a hashed colour until the class is added back.
  async function removeClass(name) {
    await putClasses(classes().filter((c) => c.name.toLowerCase() !== name.toLowerCase()));
  }

  async function setClassColor(name, color) {
    await putClasses(
      classes().map((c) => (c.name === name ? { ...c, color } : c))
    );
  }

  // Any label found in a loaded annotation that isn't in the project's class list gets
  // added, so the vocabulary reconstructs itself from the annotations. This is also the
  // safety net for the retired browser-local class list: nothing can be orphaned.
  async function ensureClasses(names) {
    if (!ACTIVE) return;
    const known = new Set(classNames().map((c) => c.toLowerCase()));
    const fresh = [];
    for (const n of names) {
      const t = String(n || "").trim();
      if (t && !known.has(t.toLowerCase())) {
        known.add(t.toLowerCase());
        fresh.push({ name: t, color: "" });
      }
    }
    if (fresh.length) await putClasses([...classes(), ...fresh]);
  }

  // --- Opening a project ------------------------------------------------------

  // The ONE code path used by boot, by switching, and by recovering from a delete.
  async function openProject(projectId) {
    let project = null;
    try {
      project = await API.getProject(projectId);
    } catch (err) {
      setProjStatus(err.message, true);
    }
    if (!project) {
      // Vanished (another tab deleted it, or a stale localStorage id). Re-enter boot.
      localStorage.removeItem(ACTIVE_KEY);
      await boot();
      setProjStatus(`project "${projectId}" no longer exists — opened another one`);
      return;
    }

    ACTIVE = project;
    API.setProject(project.id);
    localStorage.setItem(ACTIVE_KEY, project.id); // only AFTER a successful load
    renderPicker();
    renderPanel();

    // ---- PHASE 1: RESET. Every listener is a synchronous state reset. ----------
    document.dispatchEvent(
      new CustomEvent("slideprobe:project-opened", { detail: { project } })
    );
    // dispatchEvent has RETURNED, so every module has finished resetting.

    // ---- PHASE 2: WORK. Now, and only now, put a slide on screen. --------------
    window.SlideViewer.showProjectSlides(project, ALL_SLIDES);
  }

  function setStatus(msg) {
    const s = el("status");
    if (s) s.textContent = msg;
  }

  // --- Panel + picker ---------------------------------------------------------

  function renderPicker() {
    const picker = el("project-picker");
    if (!picker) return;
    picker.innerHTML = "";
    for (const p of PROJECTS) {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.name;
      picker.appendChild(opt);
    }
    if (ACTIVE) picker.value = ACTIVE.id;
  }

  function renderPanel() {
    renderProjectList();
    renderSlideChooser();
    const name = el("projects-active-name");
    if (name) name.textContent = ACTIVE ? ACTIVE.name : "—";
  }

  function renderProjectList() {
    const list = el("projects-list");
    if (!list) return;
    list.innerHTML = "";

    if (!PROJECTS.length) {
      const empty = document.createElement("div");
      empty.className = "prj-empty";
      empty.textContent = "No projects yet — create one below.";
      list.appendChild(empty);
      return;
    }

    for (const p of PROJECTS) {
      const row = document.createElement("div");
      row.className = "prj-row" + (ACTIVE && p.id === ACTIVE.id ? " active" : "");
      row.title = p.id; // two projects may share a name; the id disambiguates

      const name = document.createElement("span");
      name.className = "prj-name";
      name.textContent = p.name;
      name.addEventListener("click", () => {
        if (!ACTIVE || p.id !== ACTIVE.id) openProject(p.id);
      });

      const rename = document.createElement("button");
      rename.className = "prj-btn";
      rename.type = "button";
      rename.title = `rename "${p.name}"`;
      rename.textContent = "✎";
      rename.addEventListener("click", async () => {
        const next = prompt("Rename project", p.name);
        if (!next || next.trim() === p.name) return;
        try {
          await API.renameProject(p.id, next.trim());
          await refreshProjects();
          if (ACTIVE && ACTIVE.id === p.id) {
            ACTIVE.name = next.trim();
            renderPicker();
          }
          renderPanel();
        } catch (err) {
          setProjStatus(err.message, true);
        }
      });

      const del = document.createElement("button");
      del.className = "prj-btn prj-del";
      del.type = "button";
      del.title = `delete "${p.name}"`;
      del.textContent = "×";
      del.addEventListener("click", () => deleteProject(p));

      row.appendChild(name);
      row.appendChild(rename);
      row.appendChild(del);
      list.appendChild(row);
    }
  }

  async function deleteProject(p) {
    const ok = confirm(
      `Delete project "${p.name}"?\n\n` +
      "This removes its annotations, its classes and its trained model.\n" +
      "Your slides and their extracted features are NOT touched — any other " +
      "project over the same slides keeps working, and nothing has to be re-swept."
    );
    if (!ok) return;
    try {
      await API.deleteProject(p.id);
    } catch (err) {
      setProjStatus(err.message, true);
      return;
    }
    if (ACTIVE && ACTIVE.id === p.id) {
      // Drop the dead id BEFORE anything else can try to fetch with it.
      API.setProject(null);
      ACTIVE = null;
      localStorage.removeItem(ACTIVE_KEY);
      await boot();          // the same path as first load: pick another, or empty state
    } else {
      await refreshProjects();
      renderPanel();
    }
  }

  // Every slide on disk, checked if it's in this project. A slide that belongs to NO
  // project is still listed here — this chooser is the only way it can be adopted back
  // into one, which is what stops it from silently vanishing from the app.
  function renderSlideChooser() {
    const box = el("projects-slides");
    if (!box) return;
    box.innerHTML = "";
    if (!ACTIVE) return;

    const inProject = new Set(ACTIVE.slides);
    for (const s of ALL_SLIDES) {
      const row = document.createElement("label");
      row.className = "sl-row";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = inProject.has(s.id);
      cb.addEventListener("change", () => toggleSlide(s.id, cb));

      const name = document.createElement("span");
      name.className = "sl-name";
      name.textContent = s.name;

      row.appendChild(cb);
      row.appendChild(name);
      box.appendChild(row);
    }

    // A slide listed in the project but no longer on disk: report it rather than
    // letting the viewer try to open a file that isn't there.
    const onDisk = new Set(ALL_SLIDES.map((s) => s.id));
    const missing = ACTIVE.slides.filter((id) => !onDisk.has(id));
    const outside = ALL_SLIDES.filter((s) => !inProject.has(s.id)).length;

    const notes = [];
    if (missing.length) notes.push(`${missing.length} missing from data/slides/`);
    if (outside) notes.push(`${outside} slide(s) on disk not in this project`);
    setProjStatus(notes.join(" · "));
  }

  async function toggleSlide(slideId, checkbox) {
    if (!ACTIVE) return;
    try {
      if (checkbox.checked) {
        await API.addProjectSlide(ACTIVE.id, slideId);
      } else {
        const res = await API.removeProjectSlide(ACTIVE.id, slideId, false);
        if (res.status === "has_annotations") {
          const ok = confirm(
            `"${slideId}" has annotations in this project.\n\n` +
            "Removing it will DELETE those annotations (only in this project — the " +
            "slide file and its extracted features stay).\n\nRemove it anyway?"
          );
          if (!ok) {
            checkbox.checked = true; // put the tick back; nothing happened
            return;
          }
          await API.removeProjectSlide(ACTIVE.id, slideId, true);
        }
      }
    } catch (err) {
      setProjStatus(err.message, true);
      checkbox.checked = !checkbox.checked; // the UI must not lie about the server
      return;
    }
    // Re-open the project so the slide picker and every module see the new membership.
    await refreshProjects();
    await openProject(ACTIVE.id);
  }

  async function refreshProjects() {
    PROJECTS = await API.listProjects();
    renderPicker();
  }

  // --- Empty state --------------------------------------------------------------

  function showEmptyState() {
    ACTIVE = null;
    API.setProject(null);
    localStorage.removeItem(ACTIVE_KEY);
    renderPicker();
    renderPanel();
    setStatus("no project — create one to start");
    window.SlidePanels.open("projects-panel");
    const input = el("projects-input");
    if (input) input.focus();
    // Nothing on screen: tell every module, so none of them keeps showing the last
    // project's slide state (features.js in particular would otherwise happily go on
    // reporting "✓ 4,281 tissue tiles" for a slide that is no longer open).
    window.SlideViewer.showNothing();
  }

  // --- Boot ---------------------------------------------------------------------

  async function boot() {
    try {
      [ALL_SLIDES, PROJECTS] = await Promise.all([
        API.listSlides(),
        API.listProjects(),
      ]);
    } catch (err) {
      setStatus(`could not reach backend: ${err.message}`);
      return;
    }

    if (!PROJECTS.length) {
      showEmptyState();
      return;
    }

    const stored = localStorage.getItem(ACTIVE_KEY);
    const known = PROJECTS.some((p) => p.id === stored);
    await openProject(known ? stored : PROJECTS[0].id);

    // Report the fallback AFTER opening, and in the Projects panel rather than in
    // #status: #status belongs to the viewer, which overwrites it with "showing <slide>"
    // as soon as the slide loads — so a notice written there is gone before it's read.
    if (stored && !known) {
      setProjStatus(`project "${stored}" no longer exists — opened "${PROJECTS[0].name}"`);
    }
  }

  // --- Wiring --------------------------------------------------------------------

  window.addEventListener("DOMContentLoaded", () => {
    // Phase 7: class labels moved server-side into project.json, where they carry an
    // explicit stored colour. Nothing on disk ever used the old browser-local list, so
    // there is nothing to migrate — drop the stale key rather than leave a ghost class
    // list in devtools. (And if any label DID exist, ensureClasses() above rebuilds the
    // vocabulary from the annotations themselves, so nothing can be lost.)
    localStorage.removeItem("slideprobe.customClasses");

    const btn = el("projects-btn");
    if (btn) btn.addEventListener("click", () => window.SlidePanels.toggle("projects-panel"));

    const close = el("projects-close");
    if (close) close.addEventListener("click", () => window.SlidePanels.close("projects-panel"));

    const picker = el("project-picker");
    if (picker) {
      picker.addEventListener("change", () => {
        if (picker.value && (!ACTIVE || picker.value !== ACTIVE.id)) openProject(picker.value);
      });
    }

    const form = el("projects-add");
    const input = el("projects-input");
    if (form && input) {
      form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const name = input.value.trim();
        if (!name) return;
        try {
          const created = await API.createProject(name); // defaults to every slide on disk
          input.value = "";
          await refreshProjects();
          await openProject(created.id);
        } catch (err) {
          setProjStatus(err.message, true);
        }
      });
    }

    boot();
  });

  // What the rest of the frontend is allowed to know about the active project.
  window.SlideProject = {
    current: () => ACTIVE,
    id: () => (ACTIVE ? ACTIVE.id : null),
    classes,
    classNames,
    colorOf,
    addClass,
    removeClass,
    setClassColor,
    ensureClasses,
  };
})();
