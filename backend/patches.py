"""Patch extraction from annotations + tissue masking.

Given a user's annotated region and a target magnification, cut 224x224 tiles
(the input size Virchow 2 expects) from the slide with OpenSlide, and discard
background/white tiles using a simple tissue mask.

Responsibilities:
  - Convert annotation coordinates into pixel reads at the correct level/MPP.
  - Extract 224x224 patches; keep magnification consistent with the model.
  - Filter out non-tissue (mostly-white) patches before they reach the model.

This is the Phase 3 module.
"""

# TODO Phase 3: read regions with OpenSlide; add a tissue mask (e.g. Otsu on
# the saturation channel) to drop background patches.
