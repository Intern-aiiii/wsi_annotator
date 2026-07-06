"""Whole-slide scoring -> heatmap generation.

Tiles the whole slide, embeds each tile (reusing the cache), runs the trained
classifier head over the embeddings, and turns the per-tile scores into a
heatmap overlay the frontend draws on top of the slide in OpenSeadragon.

Responsibilities:
  - Iterate tissue tiles across the slide at the training magnification.
  - Get embeddings (cached) and classifier scores for each tile.
  - Assemble scores into an overlay/heatmap and stream progress to the frontend.

This is the Phase 6 module.
"""

# TODO Phase 6: score tiles, build the overlay, make the pass cancellable.
