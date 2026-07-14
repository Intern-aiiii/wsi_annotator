// Background-learning status chip (Phase 7).
//
// The annotate -> train pipeline runs automatically on the server whenever an
// annotation changes (see backend/jobs.py). There's no streaming, so we poll
// GET /api/projects/{id}/learn/status and reflect it in a small topbar chip. When a
// training run finishes we announce "slideprobe:model-ready" so predict.js can refresh
// its class list without the user reopening the panel.
//
// PER PROJECT: each project has its own head, so one project's "ready" says nothing
// about another's. The poll starts on slideprobe:project-opened (not DOMContentLoaded —
// there is no project to ask about yet at that point) and restarts on every switch.

(function () {
  const chipEl = () => document.getElementById("learn-status");
  const FAST_MS = 1500; // poll rate while actively learning
  const SLOW_MS = 4000; // poll rate when idle / settled

  let seq = 0;          // bumped on every project switch; a stale in-flight poll bails
  let prevState = null;
  let timer = null;

  function render(status) {
    const chip = chipEl();
    if (!chip) return;
    const state = status.state || "idle";
    const detail = status.detail || "";
    chip.classList.toggle("err", state === "error");

    // `queued` is why the chip doesn't lie: an edit sits in the debounce window for a
    // second and a half before training starts, and without this the chip would go on
    // showing "✓ model ready" the whole time — so the user concludes the save failed.
    if (status.queued && state !== "learning") {
      chip.textContent = "⏳ queued…";
      return;
    }

    if (state === "learning") {
      chip.textContent = detail ? `⏳ ${detail}` : "⏳ learning…";
    } else if (state === "ready") {
      chip.textContent = detail ? `✓ model ready · ${detail}` : "✓ model ready";
    } else if (state === "waiting") {
      chip.textContent = detail ? `• ${detail}` : "";
    } else if (state === "error") {
      chip.textContent = `⚠ ${detail || "learning error"}`;
    } else {
      chip.textContent = "";
    }
  }

  async function poll(mySeq) {
    if (mySeq !== seq || !API.projectId()) return;
    let state = "idle";
    try {
      const status = await API.learnStatus();
      if (mySeq !== seq) return; // the project changed mid-flight; drop this answer
      state = status.state || "idle";
      render(status);
      // Announce whenever a training run settles — "ready" (model refreshed) or
      // "waiting" (model invalidated: <2 classes). predict.js re-reads the class
      // list so the dropdown reflects the current annotations either way.
      const settled = state === "ready" || state === "waiting";
      if (settled && prevState && prevState !== state) {
        document.dispatchEvent(new CustomEvent("slideprobe:model-ready"));
      }
      prevState = state;
    } catch {
      // Server unreachable / transient — leave the chip as-is and retry.
    }
    if (mySeq !== seq) return;
    clearTimeout(timer);
    timer = setTimeout(() => poll(mySeq), state === "learning" ? FAST_MS : SLOW_MS);
  }

  document.addEventListener("slideprobe:project-opened", () => {
    seq++;              // invalidate any in-flight poll for the previous project
    // prevState MUST reset. Without it, project A's settled "learning -> ready"
    // transition is compared against project B's first reading and fires a spurious
    // model-ready — telling predict.js to load a model for a project it isn't in.
    prevState = null;
    render({ state: "idle" });
    clearTimeout(timer);
    poll(seq);
  });
})();
