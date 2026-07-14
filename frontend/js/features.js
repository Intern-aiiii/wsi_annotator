// Whole-slide feature extraction: the "Extract features" button, its progress
// readout, and its cancel control.
//
// This is the pipeline's one slow step (minutes on a real slide with Virchow 2), so
// it runs on the server and this module polls it. It also owns the single source of
// truth for "does this slide have features?" — which predict.js and annotate.js gate
// their buttons on, via the `slideprobe:features` event below. Predict view and Find
// similar are refused until a sweep has COMPLETED: a partial bank would silently
// score only the part of the view it happens to cover.
//
// The poll runs unconditionally, not just after the user clicks Start. That one
// choice is what makes reloading mid-sweep, switching slides and coming back, and a
// second browser tab all work without any extra code — the server is the only state.

(function () {
  const FAST_MS = 1000; // while a sweep is running
  const SLOW_MS = 4000; // otherwise (matches learn.js)

  let slideId = null;
  let seq = 0; // bumped on every slide change; a stale in-flight poll compares and bails
  let timer = null;
  let cur = null; // last status for `slideId`

  const el = (id) => document.getElementById(id);
  const fmt = (n) => Number(n || 0).toLocaleString();

  // --- Status shape ---------------------------------------------------------

  function normalize(raw, id) {
    const state = (raw && raw.state) || "unknown";
    const sweep = (raw && raw.sweep) || {};
    const sweeping = sweep.state === "running" || sweep.state === "cancelling";
    const total = Number((raw && raw.n_cells) || 0);
    const done = Number((raw && raw.n_covered) || 0);
    return {
      slideId: id,
      // none | partial | complete, or "running"/"cancelling" while the sweep is live
      state: sweeping ? sweep.state : state,
      bankState: state,
      // THE GATE. A complete sweep that found no tissue is still nothing to predict
      // on, so require at least one tissue tile.
      ready: state === "complete" && Number((raw && raw.n_tissue) || 0) > 0,
      done,
      total,
      pct: total ? Math.round((done / total) * 100) : 0,
      nTissue: Number((raw && raw.n_tissue) || 0),
      detail: sweep.detail || "",
      sweepState: sweep.state || "idle",
      // Non-null when the single sweep worker is busy with a DIFFERENT slide.
      busyWith:
        raw && raw.running_slide_id && raw.running_slide_id !== id
          ? raw.running_slide_id
          : null,
    };
  }

  // --- Rendering ------------------------------------------------------------

  function setStatus(text, cls) {
    const s = el("features-status");
    if (!s) return;
    s.textContent = text;
    s.className = cls || "";
  }

  function render(st) {
    const btn = el("features-btn");
    const cancelBtn = el("features-cancel");
    const bar = el("features-bar");
    const fill = el("features-bar-fill");
    if (!btn) return;

    // No slide on screen (an empty project, or no project). Say nothing rather than
    // keep showing the LAST slide's "✓ 4,281 tissue tiles", which would be a claim
    // about a slide the user can't even see.
    if (!st.slideId) {
      cancelBtn.hidden = true;
      bar.hidden = true;
      btn.disabled = true;
      btn.textContent = "Extract features";
      btn.title = "open a slide first";
      setStatus("");
      return;
    }

    const running = st.state === "running" || st.state === "cancelling";
    cancelBtn.hidden = !running;
    cancelBtn.disabled = st.state === "cancelling";
    bar.hidden = !running;
    if (running && fill) fill.style.width = `${st.pct}%`;

    if (st.busyWith) {
      // One sweep at a time. Say whose, rather than offering a button that bounces.
      btn.disabled = true;
      btn.textContent = "Extract features";
      btn.title = `busy extracting ${st.busyWith} — wait for it to finish`;
      setStatus(`busy: extracting ${st.busyWith}`);
      return;
    }

    btn.disabled = running || st.state === "unknown";
    btn.title = "";

    switch (st.state) {
      case "unknown":
        btn.textContent = "Extract features";
        setStatus("checking…");
        break;
      case "running":
        btn.textContent = "Extracting…";
        setStatus(`${fmt(st.done)} / ${fmt(st.total)} tiles · ${st.pct}%`);
        break;
      case "cancelling":
        btn.textContent = "Extracting…";
        setStatus("cancelling…");
        break;
      case "complete":
        btn.textContent = "Re-extract";
        setStatus(`✓ ${fmt(st.nTissue)} tissue tiles`, "ok");
        break;
      case "partial":
        btn.textContent = "Resume extraction";
        setStatus(`partial · ${fmt(st.done)} / ${fmt(st.total)} — resume to enable Predict`);
        break;
      case "error":
        btn.textContent = "Retry extraction";
        setStatus(`⚠ ${st.detail || "extraction failed"}`, "err");
        break;
      default: // "none"
        btn.textContent = "Extract features";
        setStatus("");
    }
  }

  // --- Poll loop ------------------------------------------------------------

  function apply(st) {
    const prev = cur;
    cur = st;
    render(st);
    // Consumers only care about the gate, so don't wake them once a second with
    // progress ticks — emit only when something gate-relevant actually changed.
    if (!prev || prev.state !== st.state || prev.ready !== st.ready) {
      document.dispatchEvent(
        new CustomEvent("slideprobe:features", { detail: { ...st } })
      );
    }
  }

  function schedule(ms, mySeq) {
    clearTimeout(timer);
    timer = setTimeout(() => poll(mySeq), ms);
  }

  async function poll(mySeq) {
    if (mySeq !== seq || !slideId) return;
    const id = slideId;
    let running = false;
    try {
      const st = normalize(await API.featureState(id), id);
      if (mySeq !== seq) return; // the slide changed mid-flight; drop this answer
      apply(st);
      running = st.state === "running" || st.state === "cancelling" || !!st.busyWith;
    } catch (err) {
      // Server hiccup: keep the last render AND the last gate value. A blip must not
      // flap Predict between enabled and disabled.
    }
    if (mySeq !== seq) return;
    schedule(running ? FAST_MS : SLOW_MS, mySeq);
  }

  // --- Actions --------------------------------------------------------------

  async function start() {
    if (!slideId) return;
    el("features-btn").disabled = true;
    setStatus("starting…");
    try {
      const res = await API.startFeatures(slideId);
      if (res.status === "busy") {
        apply({ ...cur, busyWith: res.slide_id });
        return;
      }
    } catch (err) {
      apply(normalize({ state: "error", sweep: { detail: err.message } }, slideId));
      return;
    }
    schedule(0, seq); // let the next poll pick up the real state
  }

  async function cancel() {
    if (!slideId) return;
    el("features-cancel").disabled = true;
    setStatus("cancelling…");
    try {
      await API.cancelFeatures(slideId);
    } catch (err) {
      setStatus(`cancel failed: ${err.message}`, "err");
    }
    schedule(0, seq); // poll for the real terminal state
  }

  // --- Wiring ---------------------------------------------------------------

  window.addEventListener("DOMContentLoaded", () => {
    const btn = el("features-btn");
    if (btn) btn.addEventListener("click", start);
    const c = el("features-cancel");
    if (c) c.addEventListener("click", cancel);
  });

  // NOTE there is deliberately no slideprobe:project-opened listener here. The feature
  // bank is keyed by (slide, embedding model, tissue mask) and is SHARED across
  // projects, so switching project changes nothing about it. That is the whole point:
  // sweep a slide once, and every project containing it can predict immediately. The
  // slide-opened event that follows a project switch is all this module needs.
  document.addEventListener("slideprobe:slide-opened", (event) => {
    slideId = event.detail.slideId; // null when the project has no slides
    seq++; // invalidate any in-flight poll for the previous slide
    cur = null;
    // Emit "unknown"/not-ready synchronously, so the gates SHUT on the same tick as
    // the slide change rather than one network round-trip later.
    apply(normalize({ state: "unknown" }, slideId));
    if (!slideId) return;   // nothing open: rendered as disabled above, don't poll
    schedule(0, seq); // and immediately ask the server — this is what picks up a
                      // sweep already running after a page reload
  });

  // Small read-only surface for defensive checks in the consumers.
  window.SlideFeatures = {
    status: () => (cur ? { ...cur } : null),
    ready: () => !!(cur && cur.ready),
  };
})();
