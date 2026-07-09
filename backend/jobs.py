"""Background learning worker (Phase 7: close the loop).

The active-learning cycle used to be three manual buttons — Extract patches,
Compute embeddings, Train classifier. This module makes that pipeline run
automatically whenever the user changes an annotation: `schedule(slide_id)` is
called from the PUT-annotations route, and a single background thread debounces
the request and runs extract -> embed -> train.

Design notes for a novice reader:
  - ONE worker thread does all the heavy work, so trainings never overlap
    (training pools every slide's embeddings, so running two at once would be
    wasteful and could corrupt the shared on-disk caches). Overlapping edits just
    re-queue and get picked up on the next pass.
  - We DEBOUNCE: after each annotation change we wait a short quiet period before
    running, so dragging out five boxes in a row triggers one training, not five.
  - Progress is exposed via `status()` and surfaced in the UI as a small topbar
    chip (there's no streaming; the frontend polls GET /api/learn/status).
"""

from __future__ import annotations

import threading
import time

from backend import classifier, embeddings, patches

# How long to wait (seconds) after the last annotation change before running, so
# a flurry of edits collapses into a single pipeline pass.
DEBOUNCE_SECONDS = 1.5

_cv = threading.Condition()
_pending: set[str] = set()      # slide_ids whose patches need re-extracting
_last_change = 0.0              # time.monotonic() of the most recent schedule()
_worker: threading.Thread | None = None

# What the UI shows. `state` is one of: idle | learning | ready | waiting | error.
_status: dict = {"state": "idle", "detail": "", "updated": 0.0}


def _set_status(state: str, detail: str = "") -> None:
    with _cv:
        _status.update(state=state, detail=detail, updated=time.time())


def status() -> dict:
    """Snapshot of the current learning state (for GET /api/learn/status)."""
    with _cv:
        return dict(_status)


def _ensure_worker() -> None:
    """Start the single worker thread on first use (idempotent)."""
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_run, name="slideprobe-learn", daemon=True)
        _worker.start()


def schedule(slide_id: str) -> None:
    """Queue a learn pass for `slide_id`; called from the annotations route.

    Cheap and non-blocking: it just records the slide and pokes the worker.
    """
    global _last_change
    _ensure_worker()
    with _cv:
        _pending.add(slide_id)
        _last_change = time.monotonic()
        _cv.notify()


def _wait_for_work() -> set[str]:
    """Block until there's pending work, then wait out the debounce window.

    Returns the set of slide_ids to process (draining `_pending`).
    """
    with _cv:
        while not _pending:
            _cv.wait()
        # Debounce: keep waiting until the slide has been quiet for DEBOUNCE_SECONDS.
        while True:
            remaining = DEBOUNCE_SECONDS - (time.monotonic() - _last_change)
            if remaining <= 0:
                break
            _cv.wait(timeout=remaining)
        drained = set(_pending)
        _pending.clear()
        return drained


def _run() -> None:
    """Worker loop: debounce, then extract -> embed -> train, forever."""
    while True:
        slide_ids = _wait_for_work()
        _set_status("learning", "updating patches & embeddings…")
        try:
            for slide_id in slide_ids:
                patches.extract_patches(slide_id)
                embeddings.embed_slide(slide_id)

            _set_status("learning", "training classifier…")
            result = classifier.train()
            _finish(result)
        except embeddings.EmbedderError as e:
            _set_status("error", str(e))
        except Exception as e:  # keep the worker alive across unexpected failures
            _set_status("error", f"{type(e).__name__}: {e}")


def _finish(result: dict) -> None:
    """Translate a classifier.train() result into a user-facing status."""
    st = result.get("status")
    if st == "ok":
        n_patches = result.get("n_samples", 0)
        n_classes = len(result.get("classes", []))
        _set_status("ready", f"{n_patches} patches · {n_classes} classes")
    elif st == "need_2_classes":
        _set_status("waiting", "draw a second class to train")
    elif st == "no_data":
        _set_status("waiting", "annotate some regions to train")
    else:
        _set_status("error", f"train: {st}")
