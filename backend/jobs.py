"""Background learning worker (Phase 7: close the loop).

Retrains a PROJECT's classifier head whenever that project's annotations change:
`schedule_project(pid)` is called from the PUT-annotations route, and a single
background thread debounces the request and runs classifier.train(pid).

Retraining is CHEAP — it reads no pixels and never touches the foundation model. It
asks each annotation which grid cells it covers and looks those cells up in the
slide's feature bank (see features.py), so a pass is a fraction of a second. It used
to re-extract patches and re-embed them on every save, which put the GPU squarely on
the interactive path; the explicit "Extract features" sweep now owns that work.

Design notes for a novice reader:
  - ONE worker thread, so trainings never overlap. Several projects can be queued at
    once; they are trained one after another.
  - We DEBOUNCE: after each annotation change we wait a short quiet period before
    running, so dragging out five boxes in a row triggers one training, not five.
  - Progress is exposed PER PROJECT via `status(pid)` and surfaced in the UI as a small
    topbar chip (there's no streaming; the frontend polls
    GET /api/projects/{pid}/learn/status).

TWO WAYS IN, and the difference matters:
  - schedule_project(pid) — this project's annotations changed. Retrain just it.
  - schedule_slide(sid)   — this slide's feature bank changed (a sweep finished). Fresh
                            vectors invalidate the head of EVERY project containing
                            that slide, so we fan out over all of them.
The fan-out lives here, not in features.py: features.py must never learn what a project
is (it is deliberately project-agnostic, because the bank is shared).
"""

from __future__ import annotations

import threading
import time

from backend import classifier, projects

# How long to wait (seconds) after the last annotation change before running, so
# a flurry of edits collapses into a single training pass.
DEBOUNCE_SECONDS = 1.5

_cv = threading.Condition()
_pending: set[str] = set()      # project_ids that changed since the last pass
_last_change = 0.0              # time.monotonic() of the most recent schedule()
_worker: threading.Thread | None = None
_current: str | None = None     # the project_id being trained RIGHT NOW, if any

# What the UI shows, per project. state: idle | learning | ready | waiting | error.
# Keyed by project_id, because two projects have two independent heads and one's
# "ready" says nothing about the other.
_status: dict[str, dict] = {}

_IDLE = {"state": "idle", "detail": "", "updated": 0.0}


def _set_status(project_id: str, state: str, detail: str = "") -> None:
    with _cv:
        _status[project_id] = {"state": state, "detail": detail, "updated": time.time()}


def status(project_id: str) -> dict:
    """Snapshot of one project's learning state (for GET /api/projects/{id}/learn/status).

    `queued` and `training` are what stop the chip from lying: without them, a project
    you just edited keeps showing a stale "✓ model ready" for the whole debounce window
    and the user concludes the save didn't take.
    """
    with _cv:
        snapshot = dict(_status.get(project_id, _IDLE))
        snapshot["queued"] = project_id in _pending
        snapshot["training"] = project_id == _current
        return snapshot


def forget(project_id: str) -> None:
    """Drop a deleted project's queue entry + status, so nothing lingers under an id
    that no longer exists. Called from the DELETE route (projects.py can't call it —
    it must not import this module, or we'd close an import cycle)."""
    with _cv:
        _pending.discard(project_id)
        _status.pop(project_id, None)


def _ensure_worker() -> None:
    """Start the single worker thread on first use (idempotent)."""
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_run, name="slideprobe-learn", daemon=True)
        _worker.start()


def schedule_project(project_id: str) -> None:
    """Queue a learn pass for one project; called from the annotations route.

    Cheap and non-blocking: it just records the project and pokes the worker.
    """
    global _last_change
    _ensure_worker()
    with _cv:
        _pending.add(project_id)
        _last_change = time.monotonic()
        _cv.notify()


def schedule_slide(slide_id: str) -> None:
    """Queue a learn pass for every project containing `slide_id`.

    Called when a feature sweep finishes: the slide now has vectors it didn't have
    before, so every project that uses that slide can train on more of what its user
    already drew.
    """
    for project_id in projects.projects_with_slide(slide_id):
        schedule_project(project_id)


def _wait_for_work() -> set[str]:
    """Block until there's pending work, then wait out the debounce window.

    Returns the set of project_ids to process (draining `_pending`).
    """
    with _cv:
        while not _pending:
            _cv.wait()
        # Debounce: keep waiting until things have been quiet for DEBOUNCE_SECONDS.
        # ONE clock for all projects, not one each: only one project is being edited at
        # a time, and schedule_slide() queues several at once on purpose — a single
        # quiet window then trains them all.
        while True:
            remaining = DEBOUNCE_SECONDS - (time.monotonic() - _last_change)
            if remaining <= 0:
                break
            _cv.wait(timeout=remaining)
        drained = set(_pending)
        _pending.clear()
        return drained


def _run() -> None:
    """Worker loop: debounce, then retrain each queued project, forever."""
    global _current
    while True:
        for project_id in sorted(_wait_for_work()):
            # The project may have been deleted while it sat in the queue.
            if projects.load(project_id) is None:
                forget(project_id)
                continue
            with _cv:
                _current = project_id
            _set_status(project_id, "learning", "training classifier…")
            try:
                # Hold projects.LOCK across the whole training pass so a DELETE landing
                # mid-train blocks until we're done, rather than racing us: otherwise we
                # could write a head into a directory that is being removed, and the
                # deleted project would partially come back from the dead. Training is
                # ~60ms, so the delete never notices the wait.
                with projects.LOCK:
                    result = classifier.train(project_id)
                _finish(project_id, result)
            except Exception as e:  # keep the worker alive across unexpected failures
                _set_status(project_id, "error", f"{type(e).__name__}: {e}")
            finally:
                with _cv:
                    _current = None


def _finish(project_id: str, result: dict) -> None:
    """Translate a classifier.train() result into a user-facing status."""
    st = result.get("status")
    if st == "ok":
        n_tiles = result.get("n_samples", 0)
        n_classes = len(result.get("classes", []))
        detail = f"{n_tiles} tiles · {n_classes} classes"
        # An annotation over cells the sweep hasn't reached contributes nothing.
        # Say so, rather than silently training on less than the user drew.
        missing = result.get("n_cells_missing", 0)
        if missing:
            detail += f" ({missing} without features)"
        unlabelled = result.get("n_unlabelled_skipped", 0)
        if unlabelled:
            detail += f" ({unlabelled} unlabelled)"
        unreadable = result.get("n_unparseable", 0)
        if unreadable:
            # Loud on purpose: a region we can't parse trains on nothing, and before this
            # counter existed it did so completely silently.
            detail += f" ⚠ {unreadable} unreadable"
        _set_status(project_id, "ready", detail)
    elif st == "need_2_classes":
        _set_status(project_id, "waiting", "draw a second class to train")
    elif st == "no_project":
        # Deleted between the queue check and the train. Nothing to report to anyone.
        forget(project_id)
    elif st == "no_data":
        reasons = {
            "no_annotations": "annotate some regions to train",
            "no_features": "extract features to train",
            "no_labels": "give your regions a class to train",
            "no_cells": "annotations are too small to cover a tile",
            "no_tissue": "the annotated regions contain no tissue",
        }
        _set_status(project_id, "waiting",
                    reasons.get(result.get("reason"), "nothing to train on"))
    else:
        _set_status(project_id, "error", f"train: {st}")
