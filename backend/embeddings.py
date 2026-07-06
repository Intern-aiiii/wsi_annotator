"""Virchow 2 embedding extraction (frozen foundation model) + embedding cache.

Turns a 224x224 tissue patch into a numeric fingerprint (embedding). Virchow 2
is used FROZEN — we never train or fine-tune it. Each patch becomes a 2560-dim
vector (class token 1280-dim concatenated with the mean of the patch tokens).

Keep this module behind a single, swappable interface so the foundation model
can be replaced later (the Virchow 2 license is non-commercial; swapping models
should be a one-file change).

Responsibilities:
  - Load Virchow 2 via timm + huggingface_hub (gated model; requires login).
  - Embed a batch of patches on GPU when available.
  - CACHE embeddings keyed by (slide_id, x, y, level, model_id) under data/cache/.
    Recomputation is the main performance bottleneck, so caching is essential.

This is the Phase 4 module.
"""

# A single interface keeps the model swappable:
#
# def embed(patches) -> "np.ndarray":  # shape (N, 2560)
#     ...
#
# TODO Phase 4: load Virchow 2, run inference, and add the on-disk cache.
