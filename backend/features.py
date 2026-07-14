"""The feature bank: every tissue tile of a slide, embedded once, stored.

This is the pipeline's one expensive step, and the user triggers it explicitly with
the "Extract features" button. It sweeps the whole slide on the grid defined by
patches.grid_config(), tissue-masks each cell, embeds the tissue ones with the
frozen foundation model, and persists the vectors.

Everything downstream is then a lookup:

    Extract features (minutes, once)  ->  feature bank
                                            |
                    annotate --------------> train (geometry + lookups, <1s, no GPU)
                    Predict / Find similar -> score  (lookups, no GPU)

That ordering is the whole point. Embedding used to run on the interactive path —
every annotation save re-read pixels and hit the GPU — which made annotating slow
and predicting slower. Now the model runs exactly once per tile, ever.

The sweep is:
  - CANCELLABLE — a threading.Event checked between blocks (a block is <=256 cells,
    so cancel latency is a few seconds even on Virchow 2).
  - RESUMABLE — cells already in the bank are skipped, and the bank is flushed on
    exit whether the sweep finished or was cancelled. Re-running continues where it
    stopped. This is why cancelling is safe: no work is ever thrown away.

Bank layout (data/cache/features/, gitignored), per (slide, model_id):
  <slide_id>__<model_id>.npy   -> (n_tissue, DIM) float16, row-aligned to index["tissue"]
  <slide_id>__<model_id>.json  -> grid + the tissue/nontissue cell ids + counts

Cells are stored as flat ints (row * cols + col; see patches.cell_index). "nontissue"
records cells the mask rejected, so a resumed sweep never re-reads known background.

`complete` is DERIVED, never trusted: covered = len(tissue) + len(nontissue), and the
bank is complete only when covered == cols*rows. A truncated or hand-edited file
cannot claim to be a finished sweep — which matters, because Predict and Find similar
are gated on exactly that.

float16 on disk is free for the production path: Virchow2Embedder runs model.half()
on CUDA, so the values already carry fp16 precision before they are ever widened.
It halves the bank (~38 MB rather than ~77 MB for a slide like CMU-3).
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from backend import embeddings, patches, slides

FEATURES_DIR = slides.REPO_ROOT / "data" / "cache" / "features"
BANK_VERSION = 2

SWEEP_BLOCK = 16          # cells per side of a sweep block (16x16 = 256 cells)
READ_WORKERS = 4          # OpenSlide reads + tissue mask, overlapped with the model
FLUSH_EVERY_BLOCKS = 32   # ~8k cells between disk flushes; bounds work lost to a crash
BANK_MEMORY_SLIDES = 2    # resident banks (LRU)


# --- Resident banks ----------------------------------------------------------
# A bank in memory: {"grid", "dim", "tissue": {(col,row): vec}, "nontissue": {(col,row)}}
#
# LOCK ORDER: _BANK_LOCK, then release, then _IO_LOCK. Never the reverse, and NEVER
# call embeddings.embed_images() while holding _BANK_LOCK — the model can take
# seconds per batch, and that is the one way this design could deadlock or stall
# every reader.
_BANK_LOCK = threading.Lock()
_IO_LOCK = threading.Lock()
_banks: dict[tuple[str, str], dict] = {}
_bank_order: list[tuple[str, str]] = []   # LRU, oldest first


def bank_paths(slide_id: str, model_id: str) -> tuple[Path, Path]:
    stem = f"{slide_id}__{model_id}"
    return FEATURES_DIR / f"{stem}.npy", FEATURES_DIR / f"{stem}.json"


# The tissue gate is PER SLIDE (the Otsu mask derives its threshold from the slide's own
# saturation histogram, which costs a low-res read), so build it once and keep it.
_gates: dict[tuple[str, str], patches.TissueGate] = {}
_GATE_LOCK = threading.Lock()


def tissue_gate(slide_id: str) -> patches.TissueGate | None:
    """This slide's tissue test, built once and cached. None if the slide is unknown.

    Keyed by (slide, mask name) so flipping SLIDEPROBE_TISSUE gives a genuinely different
    gate rather than a stale cached one.
    """
    key = (slide_id, patches.tissue_mask_name())
    with _GATE_LOCK:
        gate = _gates.get(key)
    if gate is not None:
        return gate

    slide = slides.get_slide(slide_id)
    if slide is None:
        return None
    gate = patches.tissue_gate(slide)   # may read a low-res overview; outside the lock
    with _GATE_LOCK:
        _gates[key] = gate
    return gate


def _empty_bank(grid: dict, dim: int | None = None) -> dict:
    return {"grid": grid, "dim": dim, "tissue": {}, "nontissue": set()}


def _read_index(slide_id: str, model_id: str, grid: dict) -> dict | None:
    """The bank's JSON header, or None if absent/unreadable/incompatible.

    Rejects a bank built under a different grid, model, or tissue mask outright —
    mixing tiles judged by two different rules is worse than having no bank at all.
    """
    _, ipath = bank_paths(slide_id, model_id)
    if not ipath.exists():
        return None
    try:
        with ipath.open("r", encoding="utf-8") as fh:
            index = json.load(fh)
    except Exception:
        return None
    if index.get("version") != BANK_VERSION:
        return None
    if index.get("model_id") != model_id:
        return None
    if index.get("grid") != grid:
        return None
    # Checked LAST: building the gate can cost a low-res slide read (Otsu), and there's
    # no point paying for it to validate a bank the cheap checks already rejected.
    gate = tissue_gate(slide_id)
    if gate is None or index.get("tissue_mask") != gate.id:
        return None
    return index


def load_bank(slide_id: str, model_id: str, grid: dict) -> dict | None:
    """Read a bank from disk. None if missing or not valid for this grid/model/mask."""
    index = _read_index(slide_id, model_id, grid)
    if index is None:
        return None
    mpath, _ = bank_paths(slide_id, model_id)
    try:
        matrix = np.load(mpath)
    except Exception:
        return None

    tissue_ids = index.get("tissue", [])
    if matrix.shape[0] != len(tissue_ids):
        return None  # matrix and index disagree -> the pair is untrustworthy

    tissue = {patches.index_cell(grid, i): matrix[k] for k, i in enumerate(tissue_ids)}
    nontissue = {patches.index_cell(grid, i) for i in index.get("nontissue", [])}
    dim = int(matrix.shape[1]) if matrix.shape[0] else index.get("dim")
    return {"grid": grid, "dim": dim, "tissue": tissue, "nontissue": nontissue}


def save_bank(slide_id: str, model_id: str, bank: dict, slide_mpp: float | None = None) -> None:
    """Persist a bank atomically. Safe to call while the sweep is still appending."""
    grid = bank["grid"]
    # Snapshot under the lock, then do the (slow) disk work without it.
    with _BANK_LOCK:
        cells = sorted(bank["tissue"], key=lambda cr: patches.cell_index(grid, *cr))
        vecs = [bank["tissue"][c] for c in cells]
        nontissue = sorted(patches.cell_index(grid, *c) for c in bank["nontissue"])
        dim = bank["dim"]

    matrix = (
        np.vstack(vecs).astype(np.float16) if vecs
        else np.zeros((0, dim or 0), dtype=np.float16)
    )
    gate = tissue_gate(slide_id)
    n_cells = grid["cols"] * grid["rows"]
    n_covered = len(cells) + len(nontissue)
    index = {
        "version": BANK_VERSION,
        "slide_id": slide_id,
        "model_id": model_id,
        "dim": dim,
        "grid": grid,
        "patch_size": patches.PATCH_SIZE,
        "target_mpp": patches.TARGET_MPP,
        "slide_mpp": slide_mpp,
        # Which tissue mask judged these cells, and the cutoff it used. Validated on
        # load, so re-masking a slide can't silently mix two rules' verdicts.
        "tissue_mask": gate.id if gate else None,
        "tissue_threshold": gate.threshold if gate else None,
        "n_cells": n_cells,
        "n_tissue": len(cells),
        "n_covered": n_covered,
        "complete": n_covered >= n_cells,   # a convenience; readers re-derive it
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "tissue": [patches.cell_index(grid, *c) for c in cells],
        "nontissue": nontissue,
    }

    mpath, ipath = bank_paths(slide_id, model_id)
    with _IO_LOCK:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        # Unique tmp names: two flushes (or two processes) must never share one.
        uniq = f"{os.getpid()}.{threading.get_ident()}"
        tmp_m = mpath.with_suffix(f".npy.{uniq}.tmp")
        with tmp_m.open("wb") as fh:
            np.save(fh, matrix)
        os.replace(tmp_m, mpath)

        tmp_i = ipath.with_suffix(f".json.{uniq}.tmp")
        with tmp_i.open("w", encoding="utf-8") as fh:
            json.dump(index, fh)     # compact: the cell lists get large
        os.replace(tmp_i, ipath)


def clear_bank(slide_id: str, model_id: str | None = None) -> None:
    """Drop a slide's bank from disk and memory (forces a fresh sweep)."""
    model_id = model_id or embeddings.active_model_id()
    mpath, ipath = bank_paths(slide_id, model_id)
    mpath.unlink(missing_ok=True)
    ipath.unlink(missing_ok=True)
    with _BANK_LOCK:
        key = (slide_id, model_id)
        _banks.pop(key, None)
        if key in _bank_order:
            _bank_order.remove(key)


def _resident(slide_id: str, model_id: str, grid: dict, create: bool = False) -> dict | None:
    """The in-memory bank for (slide, model), loading it from disk on first use."""
    key = (slide_id, model_id)
    with _BANK_LOCK:
        bank = _banks.get(key)
        if bank is not None and bank["grid"] == grid:
            _touch(key)
            return bank

    bank = load_bank(slide_id, model_id, grid)   # disk I/O, outside the lock
    if bank is None:
        if not create:
            return None
        bank = _empty_bank(grid)

    with _BANK_LOCK:
        _banks[key] = bank
        _touch(key)
        # Never evict the slide currently being swept.
        keep = _sweeping
        while len(_bank_order) > BANK_MEMORY_SLIDES:
            for victim in list(_bank_order):
                if victim != keep and victim != key:
                    _bank_order.remove(victim)
                    _banks.pop(victim, None)
                    break
            else:
                break
    return bank


def _touch(key) -> None:
    """LRU bookkeeping. Caller holds _BANK_LOCK."""
    if key in _bank_order:
        _bank_order.remove(key)
    _bank_order.append(key)


# --- The two accessors everything else uses ----------------------------------

def vectors(slide_id: str, model_id: str, grid: dict, cells) -> dict:
    """Bank vectors for the given (col, row) cells, as float32. No pixels, no model.

    Cells with no vector (background, or not swept yet) are simply absent from the
    result. Returns a fresh dict, so callers never hold a reference into the live
    bank while the sweep is appending to it.

    This is the ONLY way training and inference get embeddings.
    """
    bank = _resident(slide_id, model_id, grid)
    if bank is None:
        return {}
    with _BANK_LOCK:
        tissue = bank["tissue"]
        return {c: tissue[c].astype(np.float32) for c in cells if c in tissue}


def state(slide_id: str, model_id: str | None = None) -> dict:
    """Feature state for a slide — what the UI gates its buttons on.

    {"state": "none" | "partial" | "complete", n_cells, n_covered, n_tissue,
     progress, grid, dim, model_id, updated_at}

    Prefers the resident bank, so progress ticks between disk flushes.
    """
    model_id = model_id or embeddings.active_model_id()
    slide = slides.get_slide(slide_id)
    if slide is None:
        return {"state": "none", "model_id": model_id, "n_cells": 0,
                "n_covered": 0, "n_tissue": 0, "progress": 0.0}

    grid = patches.grid_config(slide)
    n_cells = grid["cols"] * grid["rows"]
    updated_at, dim = None, None

    with _BANK_LOCK:
        bank = _banks.get((slide_id, model_id))
        if bank is not None and bank["grid"] == grid:
            n_tissue = len(bank["tissue"])
            n_covered = n_tissue + len(bank["nontissue"])
            dim = bank["dim"]
        else:
            bank = None

    if bank is None:
        index = _read_index(slide_id, model_id, grid)
        if index is None:
            n_tissue = n_covered = 0
        else:
            n_tissue = len(index.get("tissue", []))
            n_covered = n_tissue + len(index.get("nontissue", []))
            updated_at, dim = index.get("updated_at"), index.get("dim")

    if n_covered <= 0:
        st = "none"
    elif n_covered >= n_cells:
        st = "complete"
    else:
        st = "partial"

    return {
        "state": st,
        "model_id": model_id,
        "grid": grid,
        "dim": dim,
        "n_cells": n_cells,
        "n_covered": n_covered,
        "n_tissue": n_tissue,
        "progress": round(n_covered / n_cells, 4) if n_cells else 0.0,
        "updated_at": updated_at,
    }


def is_ready(slide_id: str, model_id: str | None = None) -> bool:
    """True only for a COMPLETE sweep. Predict + Find similar are gated on this."""
    return state(slide_id, model_id)["state"] == "complete"


# --- The sweep ---------------------------------------------------------------

def _sweep(slide_id: str, model_id: str, cancelled, progress) -> dict:
    """Embed every not-yet-known cell of the slide, block by block. Resumable."""
    slide = slides.get_slide(slide_id)
    if slide is None:
        return {"status": "no_slide"}

    grid = patches.grid_config(slide)
    n_cells = grid["cols"] * grid["rows"]
    if n_cells == 0:
        return {"status": "empty_slide"}

    embedder = embeddings.get_embedder()      # may raise EmbedderError
    bank = _resident(slide_id, model_id, grid, create=True)
    bank["dim"] = embedder.DIM
    mpp = patches.slide_mpp(slide)
    # THE tissue gate — built once for this slide (Otsu derives its cutoff from the
    # slide's own histogram), then applied to every tile. This is its only call site.
    is_tissue = tissue_gate(slide_id)

    def covered() -> int:
        with _BANK_LOCK:
            return len(bank["tissue"]) + len(bank["nontissue"])

    def read_and_mask(cell):
        img = patches.read_cell(slide, grid, cell[0], cell[1])
        return cell, (img if is_tissue(img) else None)

    n_embedded = 0
    since_flush = 0
    progress(covered(), n_cells, len(bank["tissue"]), n_embedded)

    with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
        for brow in range(0, grid["rows"], SWEEP_BLOCK):
            for bcol in range(0, grid["cols"], SWEEP_BLOCK):
                if cancelled():
                    break

                # Skip cells we already have a verdict for — this is what makes a
                # re-run resume rather than redo.
                with _BANK_LOCK:
                    todo = [
                        (col, row)
                        for row in range(brow, min(brow + SWEEP_BLOCK, grid["rows"]))
                        for col in range(bcol, min(bcol + SWEEP_BLOCK, grid["cols"]))
                        if (col, row) not in bank["tissue"] and (col, row) not in bank["nontissue"]
                    ]
                if not todo:
                    continue

                # Read + tissue-mask on the pool, overlapping disk I/O with the model.
                results = list(pool.map(read_and_mask, todo))
                keep = [(cell, img) for cell, img in results if img is not None]
                background = [cell for cell, img in results if img is None]

                for i in range(0, len(keep), embedder.BATCH):
                    chunk = keep[i:i + embedder.BATCH]
                    # NOT holding _BANK_LOCK: this is the slow call.
                    vecs = embeddings.embed_images(embedder, [img for _, img in chunk])
                    with _BANK_LOCK:
                        for (cell, _), vec in zip(chunk, vecs):
                            bank["tissue"][cell] = np.asarray(vec, dtype=np.float16)
                    n_embedded += len(chunk)

                with _BANK_LOCK:
                    bank["nontissue"].update(background)
                    n_tissue = len(bank["tissue"])
                    done = n_tissue + len(bank["nontissue"])
                progress(done, n_cells, n_tissue, n_embedded)

                since_flush += 1
                if since_flush >= FLUSH_EVERY_BLOCKS:
                    save_bank(slide_id, model_id, bank, mpp)
                    since_flush = 0
            else:
                continue    # inner loop wasn't broken -> next block row
            break           # cancelled

    # Always flush — a cancelled sweep must leave a valid bank to resume from.
    save_bank(slide_id, model_id, bank, mpp)
    done = covered()
    with _BANK_LOCK:
        n_tissue = len(bank["tissue"])
    return {"status": "ok", "done": done, "total": n_cells,
            "n_tissue": n_tissue, "n_embedded": n_embedded}


# --- The background worker (start / status / cancel) --------------------------

_worker_lock = threading.Lock()
# state: idle | running | cancelling | done | cancelled | error
_status: dict = {"state": "idle", "slide_id": None, "done": 0, "total": 0,
                 "n_tissue": 0, "n_embedded": 0, "detail": "", "updated": 0.0}
_thread: threading.Thread | None = None
_cancel = threading.Event()
_sweeping: tuple[str, str] | None = None    # bank key being swept; never evicted


def _set(**kw) -> None:
    with _worker_lock:
        _status.update(updated=time.time(), **kw)


def status() -> dict:
    """Snapshot of the sweep (the frontend polls this for progress)."""
    with _worker_lock:
        return dict(_status)


def start(slide_id: str) -> dict:
    """Begin sweeping `slide_id` in the background. Cheap + non-blocking.

    One slide at a time: a second call while a sweep runs returns "busy". Resumes
    from whatever is already in the bank.
    """
    global _thread
    with _worker_lock:
        if _status["state"] in ("running", "cancelling"):
            return {"status": "busy", "slide_id": _status["slide_id"]}
        # Flip to running under the same lock as the check, so two rapid clicks
        # can't both start a sweep.
        _status.update(state="running", slide_id=slide_id, done=0, total=0,
                       n_tissue=0, n_embedded=0, detail="starting…", updated=time.time())
    _cancel.clear()
    _thread = threading.Thread(target=_run, args=(slide_id,),
                               name="slideprobe-features", daemon=True)
    _thread.start()
    return {"status": "started", "slide_id": slide_id}


def cancel() -> dict:
    """Ask a running sweep to stop after the current block. Keeps what it embedded."""
    with _worker_lock:
        if _status["state"] != "running":
            return {"status": "idle"}
        _status.update(state="cancelling", detail="cancelling…", updated=time.time())
        slide_id = _status["slide_id"]
    _cancel.set()
    return {"status": "cancelling", "slide_id": slide_id}


def _run(slide_id: str) -> None:
    global _sweeping
    model_id = embeddings.active_model_id()
    _sweeping = (slide_id, model_id)

    def progress(done, total, n_tissue, n_embedded):
        # Deliberately does not touch "state" — a cancel() in flight must not be
        # overwritten back to "running" by the next block's progress report.
        _set(done=done, total=total, n_tissue=n_tissue, n_embedded=n_embedded,
             detail=f"{done:,} / {total:,} tiles")

    try:
        res = _sweep(slide_id, model_id, _cancel.is_set, progress)
        if res["status"] == "no_slide":
            _set(state="error", detail="slide not found")
        elif res["status"] == "empty_slide":
            _set(state="error", detail="slide is smaller than one tile at this magnification")
        elif _cancel.is_set():
            _set(state="cancelled",
                 detail=f"stopped at {res['done']:,} / {res['total']:,} tiles")
        else:
            _set(state="done", done=res["total"], total=res["total"],
                 detail=f"{res['n_tissue']:,} tissue tiles ({res['n_embedded']:,} new)")
    except embeddings.EmbedderError as e:
        _set(state="error", detail=str(e))
    except Exception as e:  # surface failures instead of dying silently
        _set(state="error", detail=f"{type(e).__name__}: {e}")
    finally:
        _sweeping = None
        # Retrain against the vectors we just produced. Without this the FIRST sweep
        # leaves the head untrained: the annotations didn't change, so nothing else
        # would wake the learn worker. Imported here, not at module scope, to break
        # the cycle features -> jobs -> classifier -> features.
        from backend import jobs
        # Fans out to every PROJECT containing this slide (jobs.py owns that knowledge;
        # this module stays deliberately project-agnostic, because the bank is shared).
        jobs.schedule_slide(slide_id)
