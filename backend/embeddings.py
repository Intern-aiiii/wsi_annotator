"""The foundation model — a swappable, frozen feature extractor. Nothing else.

Turns a 224x224 tissue tile into a numeric fingerprint (an embedding). That is this
module's whole job: it holds no cache and knows nothing about slides, grids, or
labels. Storing the vectors is features.py's job; learning from them is
classifier.py's. The model is used FROZEN — we never train or fine-tune it.

The extractor lives behind a single swappable interface (CLAUDE.md: keep the
foundation model a one-file change). Two backends:

  - "dev" (default): a lightweight, dependency-light color/texture descriptor
    (numpy + PIL). It is a DEVELOPMENT STAND-IN, *not* Virchow 2 — it exists so
    the whole pipeline runs today without the gated model, and it still gives a
    weak-but-real signal to test training on.

  - "virchow2" (opt-in): the real foundation model — frozen ViT-H/14, 2560-dim
    (class token concatenated with the mean of the patch tokens). Gated + heavy;
    imported lazily so the default path never needs torch/timm/HF.

Select the backend with the SLIDEPROBE_EMBEDDER env var ("dev" | "virchow2").
Each backend has a distinct `model_id`, and feature banks are filed under it, so
vectors from different backends can never mix.
"""

from __future__ import annotations

import os
import threading

import numpy as np


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


# Serialize model inference. Today the feature sweep (features.py) is the only
# caller, but torch/GPU calls aren't safe to run concurrently, so this coarse lock
# is the seam that guarantees a future second caller can't race it. Route ALL
# inference through embed_images().
_EMBED_LOCK = threading.Lock()


def embed_images(embedder, images) -> np.ndarray:
    """Run one batch through the embedder under the shared lock."""
    with _EMBED_LOCK:
        return embedder.embed(images)
