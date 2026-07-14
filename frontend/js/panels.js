// The floating right-hand panels, as a radio group: opening one closes the others.
//
// Projects, Classes and Predict all sit at the SAME coordinates (top-right). Before
// this file they simply drew on top of each other — open Classes while Predict was
// up and you got two overlapping cards with no way to tell which was which. Rather
// than find three different corners for them, we make them mutually exclusive: they
// are three different phases of work (set up the experiment / define the vocabulary /
// run it) and you never need two at once.
//
// #overlay-ctrl is deliberately NOT in the group. It lives bottom-right and you must
// be able to drag its opacity slider while a panel is open.
//
// If you ever DO want two visible at once: wrap them in a
// `#panel-stack { position: fixed; top: 48px; right: 14px; display: flex;
// flex-direction: column; gap: 10px; }`, drop position/top/right from each panel, and
// delete this file. Nothing here blocks that.

(function () {
  const PANELS = ["projects-panel", "classes-panel", "predict-panel"];

  function open(id) {
    for (const p of PANELS) {
      const el = document.getElementById(p);
      if (el) el.hidden = p !== id;
    }
  }

  function close(id) {
    const el = document.getElementById(id);
    if (el) el.hidden = true;
  }

  function closeAll() {
    for (const p of PANELS) close(p);
  }

  function toggle(id) {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.hidden) open(id);
    else close(id);
  }

  function isOpen(id) {
    const el = document.getElementById(id);
    return !!(el && !el.hidden);
  }

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeAll();
  });

  window.SlidePanels = { open, close, closeAll, toggle, isOpen };
})();
