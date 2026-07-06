"""Train / persist / load the classifier head.

The "head" is a lightweight classifier trained on the frozen embeddings — it
never sees pixels, only 2560-dim vectors and their labels. Start with
scikit-learn logistic regression; move to a small PyTorch MLP only if that
underfits.

Responsibilities:
  - Assemble (X = embeddings, y = labels) from annotated patches.
  - Split by slide/region (NOT random patch) to avoid leakage between adjacent
    tiles; handle class imbalance with balanced class weights.
  - Train, report precision/recall + confusion matrix per class.
  - Persist the head (joblib for sklearn / state_dict for PyTorch) alongside the
    embedding config that produced it (model_id, CLS vs CLS+mean, magnification)
    so inference uses matching features.

This is the Phase 5 module.
"""

# TODO Phase 5: LogisticRegression(class_weight="balanced") baseline first.
