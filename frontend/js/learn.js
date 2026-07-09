// Background-learning status chip (Phase 7).
//
// The extract -> embed -> train pipeline runs automatically on the server
// whenever an annotation changes (see backend/jobs.py). There's no streaming, so
// we poll GET /api/learn/status and reflect it in a small topbar chip. When a
// training run finishes we announce "slideprobe:model-ready" so predict.js can
// refresh its class list without the user reopening the panel.

(function () {
  const chipEl = () => document.getElementById("learn-status");
  const FAST_MS = 1500; // poll rate while actively learning
  const SLOW_MS = 4000; // poll rate when idle / settled
  let prevState = null;

  function render(status) {
    const chip = chipEl();
    if (!chip) return;
    const state = status.state || "idle";
    const detail = status.detail || "";
    chip.classList.toggle("err", state === "error");
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

  async function poll() {
    let state = "idle";
    try {
      const status = await API.learnStatus();
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
    setTimeout(poll, state === "learning" ? FAST_MS : SLOW_MS);
  }

  window.addEventListener("DOMContentLoaded", poll);
})();
