"""Patch embeddings (Phase 4) — swappable extractor + aggressive cache.

Turns each 224x224 tissue patch (from the Phase 3 manifest) into a numeric
fingerprint (embedding) and caches it, because recomputation is the pipeline's
main bottleneck. These cached vectors are what Phase 5 (the classifier) trains on.

The extractor lives behind a single swappable interface (CLAUDE.md: keep the
foundation model a one-file change). Two backends:

  - "dev" (default): a lightweight, dependency-light color/texture descriptor
    (numpy + PIL). It is a DEVELOPMENT STAND-IN, *not* Virchow 2 — it exists so
    the whole pipeline (and Phase 5) runs today without the gated model, and it
    still gives a weak-but-real signal to test training on.

  - "virchow2" (opt-in): the real foundation model — frozen ViT-H/14, 2560-dim
    (class token concatenated with the mean of the patch tokens). Gated + heavy;
    imported lazily so the default path never needs torch/timm/HF.

Select the backend with the SLIDEPROBE_EMBEDDER env var ("dev" | "virchow2").
Every embedding records its `model_id`, so cached vectors from different backends
never mix and Phase 5 knows which features it is using.

Cache layout (data/cache/embeddings/, gitignored), per (slide, model_id):
  <slide_id>__<model_id>.npy   -> (N, DIM) float32 matrix, row-aligned to:
  <slide_id>__<model_id>.json  -> {model_id, dim, ...config, rows:[{x,y,level,label}]}
The per-patch cache key is (slide_id, x, y, level, model_id).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from backend import patches, slides

EMBEDDINGS_DIR = slides.REPO_ROOT / "data" / "cache" / "embeddings"


class EmbedderError(RuntimeError):
    """Raised when the selected embedder can't be used (e.g. Virchow 2 not set up)."""


# --- Backends ---------------------------------------------------------------

class DevEmbedder:
    """Deterministic color/texture descriptor. A stand-in, NOT Virchow 2.

    Features per patch (all in [0, 1], fixed length 62):
      - RGB histograms, 16 bins/channel (48)
      - RGB mean + std (6)
      - HSV mean + std (6)
      - mean |horizontal| and |vertical| grayscale gradient (2)
    """

    MODEL_ID = "dev-colorstats-v1"
    DIM = 62
    BATCH = 64

    def embed(self, images) -> np.ndarray:
        out = np.zeros((len(images), self.DIM), dtype=np.float32)
        for i, img in enumerate(images):
            out[i] = self._descriptor(img)
        return out

    def _descriptor(self, img) -> np.ndarray:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32)  # H x W x 3, 0..255
        flat = rgb.reshape(-1, 3)
        feats = []
        for c in range(3):
            hist, _ = np.histogram(rgb[:, :, c], bins=16, range=(0, 255))
            feats.append(hist / max(1, hist.sum()))
        feats.append(flat.mean(0) / 255.0)
        feats.append(flat.std(0) / 255.0)
        hsv = np.asarray(img.convert("HSV"), dtype=np.float32).reshape(-1, 3) / 255.0
        feats.append(hsv.mean(0))
        feats.append(hsv.std(0))
        gray = rgb.mean(2)
        dx = np.abs(np.diff(gray, axis=1)).mean() / 255.0
        dy = np.abs(np.diff(gray, axis=0)).mean() / 255.0
        feats.append(np.array([dx, dy], dtype=np.float32))
        return np.concatenate([np.ravel(f) for f in feats]).astype(np.float32)


class Virchow2Embedder:
    """Frozen Virchow 2 (ViT-H/14) -> 2560-dim (CLS token + mean of patch tokens).

    Heavy + gated: imported lazily. First use downloads ~2.5 GB and needs HF
    access (`huggingface-cli login`). Runs on GPU (fp16) when available.
    """

    MODEL_ID = "virchow2-clsmean-2560"
    DIM = 2560
    BATCH = 8  # keep small: a 6 GB GPU is tight for a 632M ViT-H

    def __init__(self):
        try:
            import timm
            import torch
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            from timm.layers import SwiGLUPacked
        except ImportError as e:
            raise EmbedderError(
                "Virchow 2 needs torch + timm + huggingface_hub. Install them with:\n"
                "  pip install torch timm huggingface_hub"
            ) from e

        self._torch = torch
        try:
            model = timm.create_model(
                "hf-hub:paige-ai/Virchow2",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        except Exception as e:  # network / gated-access / auth failures
            raise EmbedderError(
                "Could not load paige-ai/Virchow2. Request access on Hugging Face and run "
                "`huggingface-cli login` first."
            ) from e

        model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = self.device == "cuda"
        if self.use_fp16:
            model = model.half()
        self.model = model.to(self.device)
        cfg = resolve_data_config(model.pretrained_cfg, model=model)
        self.transform = create_transform(**cfg)

    def embed(self, images) -> np.ndarray:
        torch = self._torch
        batch = torch.stack([self.transform(im.convert("RGB")) for im in images]).to(self.device)
        if self.use_fp16:
            batch = batch.half()
        with torch.inference_mode():
            out = self.model(batch)                       # (N, 261, 1280)
            cls = out[:, 0]                                # (N, 1280)
            patch_tokens = out[:, 5:].mean(1)             # skip 4 register tokens
            emb = torch.cat([cls, patch_tokens], dim=-1)  # (N, 2560)
        return emb.float().cpu().numpy().astype(np.float32)


_BACKENDS = {"dev": DevEmbedder, "virchow2": Virchow2Embedder}
_embedder = None  # cached active instance


def _selected_name() -> str:
    return os.environ.get("SLIDEPROBE_EMBEDDER", "dev").lower()


def _selected_class():
    return _BACKENDS.get(_selected_name(), DevEmbedder)


def active_model_id() -> str:
    """The model_id of the selected backend, WITHOUT triggering a heavy load."""
    return _selected_class().MODEL_ID


def get_embedder():
    """Return (constructing + caching once) the active embedder instance.

    May raise EmbedderError if the selected backend can't be initialized (e.g.
    Virchow 2 without deps/access).
    """
    global _embedder
    want = _selected_class()
    if _embedder is None or not isinstance(_embedder, want):
        _embedder = want()
    return _embedder


# Serialize embedder inference: the background learning worker (jobs.py) and a
# live similarity request can both reach the embedder at once, and torch/GPU
# calls aren't safe to run concurrently. This coarse lock keeps them in order.
_EMBED_LOCK = threading.Lock()


def embed_images(embedder, images) -> np.ndarray:
    """Run one batch through the embedder under the shared lock."""
    with _EMBED_LOCK:
        return embedder.embed(images)


# --- Cache -------------------------------------------------------------------

def _matrix_path(slide_id: str, model_id: str) -> Path:
    return EMBEDDINGS_DIR / f"{slide_id}__{model_id}.npy"


def _index_path(slide_id: str, model_id: str) -> Path:
    return EMBEDDINGS_DIR / f"{slide_id}__{model_id}.json"


def clear_cache(slide_id: str, model_id: str | None = None) -> None:
    """Delete a slide's cached embedding matrix + index for a model (if present).

    Called when a slide no longer has any patches to embed, so its now-stale
    embeddings stop being pooled into training (which globs every slide's index).
    """
    if model_id is None:
        model_id = active_model_id()
    _matrix_path(slide_id, model_id).unlink(missing_ok=True)
    _index_path(slide_id, model_id).unlink(missing_ok=True)


def _load_cache(slide_id: str, model_id: str) -> dict:
    """Return {(x, y, level): vector} from a previous run, or {} if none/invalid."""
    mpath, ipath = _matrix_path(slide_id, model_id), _index_path(slide_id, model_id)
    if not mpath.exists() or not ipath.exists():
        return {}
    try:
        matrix = np.load(mpath)
        with ipath.open("r", encoding="utf-8") as fh:
            index = json.load(fh)
    except Exception:
        return {}
    rows = index.get("rows", [])
    if matrix.shape[0] != len(rows):
        return {}
    return {(r["x"], r["y"], r["level"]): matrix[i] for i, r in enumerate(rows)}


def _save_cache(slide_id, model_id, matrix, rows, manifest, dim) -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    mpath, ipath = _matrix_path(slide_id, model_id), _index_path(slide_id, model_id)

    # Atomic .npy write (write to a file object so numpy doesn't rename the suffix).
    tmp_m = mpath.with_suffix(".npy.tmp")
    with tmp_m.open("wb") as fh:
        np.save(fh, matrix)
    os.replace(tmp_m, mpath)

    cfg = manifest.get("config", {})
    index = {
        "model_id": model_id,
        "dim": dim,
        "patch_size": cfg.get("patch_size"),
        "target_mpp": cfg.get("target_mpp"),
        "slide_mpp": cfg.get("slide_mpp"),
        "rows": rows,
    }
    tmp_i = ipath.with_suffix(".json.tmp")
    with tmp_i.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    os.replace(tmp_i, ipath)


def load_index(slide_id: str) -> dict | None:
    """Return the saved embedding index (summary) for the active model, or None."""
    ipath = _index_path(slide_id, active_model_id())
    if not ipath.exists():
        return None
    with ipath.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --- Orchestration -----------------------------------------------------------

def embed_slide(slide_id: str) -> dict:
    """Embed a slide's patches (with caching); write matrix + index.

    Returns a status dict:
      {"status": "no_slide"}                       -> unknown slide
      {"status": "no_manifest"}                    -> patches not extracted yet
      {"status": "ok", model_id, dim, n_total, n_embedded, n_from_cache,
       per_class, seconds}
    May raise EmbedderError if the selected backend is unavailable.
    """
    slide = slides.get_slide(slide_id)
    if slide is None:
        return {"status": "no_slide"}
    manifest = patches.load_manifest(slide_id)
    if not manifest or not manifest.get("patches"):
        # No patches (e.g. all annotations deleted): drop this slide's stale cache
        # so training doesn't keep pooling its old labels.
        clear_cache(slide_id, active_model_id())
        return {"status": "no_manifest"}

    embedder = get_embedder()  # may raise EmbedderError
    model_id, dim = embedder.MODEL_ID, embedder.DIM
    pts = manifest["patches"]

    cache = _load_cache(slide_id, model_id)
    vectors: list = [None] * len(pts)
    rows = []
    to_compute = []  # (row_index, patch)
    n_from_cache = 0
    for i, p in enumerate(pts):
        # Carry the region `group` (Phase 3) through so Phase 5 can split by
        # region; fall back to the slide id for older manifests without it.
        rows.append({
            "x": p["x"], "y": p["y"], "level": p["level"], "label": p["label"],
            "group": p.get("group", slide_id),
        })
        key = (p["x"], p["y"], p["level"])
        if key in cache:
            vectors[i] = cache[key]
            n_from_cache += 1
        else:
            to_compute.append((i, p))

    t0 = time.perf_counter()
    for start in range(0, len(to_compute), embedder.BATCH):
        chunk = to_compute[start:start + embedder.BATCH]
        imgs = [patches.read_patch(slide, p["x"], p["y"], p["size"], p["level"]) for _, p in chunk]
        embs = embed_images(embedder, imgs)
        for (idx, _), vec in zip(chunk, embs):
            vectors[idx] = vec
    seconds = time.perf_counter() - t0

    matrix = (
        np.vstack(vectors).astype(np.float32)
        if vectors else np.zeros((0, dim), dtype=np.float32)
    )
    _save_cache(slide_id, model_id, matrix, rows, manifest, dim)

    per_class: dict[str, int] = {}
    for r in rows:
        per_class[r["label"]] = per_class.get(r["label"], 0) + 1

    return {
        "status": "ok",
        "model_id": model_id,
        "dim": dim,
        "n_total": len(pts),
        "n_embedded": len(to_compute),
        "n_from_cache": n_from_cache,
        "per_class": per_class,
        "seconds": round(seconds, 3),
    }
