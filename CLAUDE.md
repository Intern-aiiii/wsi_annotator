# CLAUDE.md

Guidance for Claude Code when working in this repository.

> **Working name:** SlideProbe (placeholder — rename freely). The "probe" refers to
> the linear-probe pattern at the heart of the design: a lightweight classifier trained
> on frozen foundation-model embeddings.

---

## Change log (do this every time)

Keep a running, human-readable log of the work done on this project in `docs/log.txt`.
After you finish fulfilling each user request, append a short entry describing what changed
(create the file and the `docs/` directory if they don't exist yet). Keep entries brief —
a dated line or two summarizing the change and why — so the file stays a readable history
of how the project evolved rather than a full diff.

---

## What we're building

Local, browser-based software for analyzing **whole slide images (WSI)** in pathology.
A user annotates structures or regions of interest on a slide; the app uses a pathology
**foundation model (Virchow 2)** as a *frozen feature extractor* and trains a lightweight
classifier on top of the resulting embeddings to detect those structures across the slide
and across slides. The reference product is **Aiforia**, but the goal here is deliberately
**lean and local**.

The developer is a **novice software engineer**. Favor clarity, small verifiable milestones,
and well-commented code over cleverness. Explain non-obvious choices inline. When a step is
risky or slow (model download, GPU inference, large file I/O), say so before doing it.

---

## Core architecture principle (read this first)

**This is NOT a pure HTML/JS browser app. It is a local client–server web application.**

- **Frontend** — HTML/CSS/JS in the browser. Responsible for *display and annotation only*.
- **Backend** — a Python server on `localhost`. Responsible for slide I/O, the foundation
  model, training, and inference.

Why it must be split this way:

1. **WSIs cannot be read in a browser.** Formats like `.svs`, `.ndpi`, and `.mrxs` are
   gigapixel, multi-resolution "pyramid" files that require **OpenSlide** (or Bio-Formats) —
   C libraries with Python bindings. There is no browser-native reader.
2. **Virchow 2 runs in PyTorch, not JavaScript.** It is a ~632M-parameter vision transformer
   that realistically needs a GPU.

To the user it *feels* local: they open a browser tab pointed at `127.0.0.1:<port>`. Under
the hood the browser talks to a Python process on the same machine. Do not attempt to move
slide reading or model inference into the browser.

---

## The ML pipeline (the active-learning loop)

```
Browser UI  ──▶  Tile & patch    ──▶  Virchow 2      ──▶  Train classifier ──▶  Predict &
(view +          extraction           embeddings          head                  heatmap
 annotate)       (OpenSlide +         (frozen FM,         (logistic reg /       (score tiles,
                  tissue mask)         2560-dim vecs)      small MLP)            overlay)
     ▲                                                                              │
     └──────────────────────────  iterate: refine annotations, retrain  ◀──────────┘
```

**The single most important design insight:** Virchow 2 is used **frozen**. We never train
or fine-tune it. It was pretrained on 3.1M slides and converts any 224×224 tissue tile into
a numeric fingerprint (embedding). All learning happens in the small classifier on top of
those fingerprints. This is why the app works with only a handful of user annotations — the
hard visual representation learning is already done.

For "point at structures" region tasks, a **per-tile classifier on embeddings** is enough to
start. (Slide-level tasks — one label per whole slide — would instead use multiple-instance
learning / MIL; not needed for the initial region-classification workflow.)

---

## Foundation model: Virchow 2

- **Source:** `paige-ai/Virchow2` on Hugging Face. **Gated** — request access, get approved,
  then authenticate (`huggingface-cli login`) in the environment before download.
- **Architecture:** ViT-H/14, ~632M parameters.
- **Input:** 224×224 tissue tiles. Trained on tiles at 0.25–2.0 microns-per-pixel
  (5×/10×/20×/40×). **Match the extraction magnification to the model** — mismatched
  resolution noticeably degrades embedding quality.
- **Embedding:** class token (1280-dim) concatenated with the mean of the patch tokens
  (1280-dim) → **2560-dim** vector per tile. Class-token-only (1280-dim) is a valid
  lower-storage option; validate before relying on it.
- **Requirements:** `torch>=2.0`, `timm>=0.9.11`, `huggingface_hub`. GPU strongly preferred
  (runs on CPU, but slowly).

### License — a hard constraint, decide intent early

Virchow 2 is released under **CC-BY-NC-ND 4.0**: **non-commercial** and **no-derivatives**.

- Fine for personal / research / internal-only tools.
- **Blocks commercial use.** If commercialization is ever a goal, architect around a
  differently-licensed model instead — the original **Virchow** is Apache 2.0; **UNI2** and
  **H-optimus** have their own terms to check.
- Keep the model backend **swappable** (a single `embeddings.py` interface) so switching
  foundation models later is a one-file change.

*(This is a licensing flag, not legal advice — read the actual model terms.)*

---

## Tech stack

| Layer      | Choice                                    | Notes                                      |
|------------|-------------------------------------------|--------------------------------------------|
| Frontend   | HTML/CSS/JS + **OpenSeadragon**           | Standard gigapixel deep-zoom viewer        |
| Annotation | **Annotorious** (OpenSeadragon plugin)    | Draws polygons/rects; W3C annotation JSON  |
| Tiles      | **DeepZoom (DZI)** generated server-side  | What OpenSeadragon consumes                |
| Backend    | **Python + FastAPI**                      | Beginner-friendly, async, good docs        |
| Slide I/O  | **OpenSlide** (`openslide-python`)        | Needs system lib: `openslide-tools`        |
| Model      | **PyTorch + timm + huggingface_hub**      | Virchow 2 as frozen extractor              |
| Classifier | **scikit-learn** (start) → PyTorch MLP    | Logistic regression first; MLP if needed   |

Before writing slide-handling or feature-extraction code from scratch, evaluate reusing
**TIAToolbox** (Python WSI toolkit with patch extraction + FM feature extraction) or
**Slideflow** (end-to-end DL pathology pipeline). Reusing these for the backend heavy-lifting
can save months. The project's differentiated value is the lean annotation → model UX, not
re-implementing slide I/O.

---

## Proposed repository layout

```
slideprobe/
├── backend/
│   ├── app.py          # FastAPI entrypoint, routes, static frontend serving
│   ├── slides.py       # OpenSlide reading + DeepZoom tile serving
│   ├── patches.py      # patch extraction from annotations + tissue masking
│   ├── embeddings.py   # Virchow 2 load + inference; embedding CACHE (swappable model)
│   ├── classifier.py   # train / persist / load the classifier head
│   ├── inference.py    # whole-slide scoring → heatmap generation
│   └── models.py       # data models: projects, slides, annotations, classes
├── frontend/
│   ├── index.html
│   ├── js/viewer.js    # OpenSeadragon setup + DZI source
│   ├── js/annotate.js  # Annotorious integration, save/load annotations
│   └── js/api.js       # fetch() calls to the backend
├── data/               # gitignored
│   ├── slides/         # input WSIs
│   ├── cache/          # DeepZoom tiles + cached embeddings
│   ├── annotations/    # saved annotation JSON
│   └── models/         # trained classifier heads
├── requirements.txt
├── README.md
└── CLAUDE.md
```

---

## Development phases (build the thinnest end-to-end slice first)

Each phase must produce something that runs. Do not jump ahead; a working Phase N beats a
half-built Phase N+2.

- [x] **Phase 1 — One slide on screen.** FastAPI server generates DeepZoom tiles from a
      sample `.svs` via OpenSlide; OpenSeadragon displays it in the browser. No ML yet. This
      proves the whole frontend↔backend loop. *(Code complete; run once a slide + OpenSlide
      are installed — see README.)*
- [ ] **Phase 2 — Annotation.** Add Annotorious; user draws regions; save coordinates as JSON
      to `data/annotations/`.
- [ ] **Phase 3 — Patch extraction.** For each annotation, use OpenSlide to cut 224×224 tiles
      at the target magnification; discard background/white tiles with a tissue mask.
- [ ] **Phase 4 — Embeddings.** Run patches through Virchow 2; store vectors. **Cache
      aggressively** — recomputation is the main performance bottleneck.
- [ ] **Phase 5 — Train the head.** Logistic regression (scikit-learn) on embeddings + labels;
      train/val split; report precision/recall. Trains in seconds because embeddings are
      precomputed. (Details below.)
- [ ] **Phase 6 — Predict + overlay.** Tile the whole slide, embed, classify, return a heatmap
      overlay to OpenSeadragon.
- [ ] **Phase 7 — Close the loop + polish.** Let the user correct predictions, add
      annotations, and retrain (the active-learning cycle). Then: multiple classes, project/
      slide management, export.

---

## Phase 5 detail — training the classifier head

This is the crux of the whole approach, so get it right:

1. **Assemble the dataset.** For every annotated patch you have an embedding (1280- or
   2560-dim vector `X`) and a label `y` (e.g. `gland` vs `not-gland`, or a class index).
   Because embeddings are cached, this is just loading arrays.
2. **Start simple — logistic regression.** `sklearn.linear_model.LogisticRegression` on the
   embedding vectors. It's fast, interpretable, and a strong baseline on FM embeddings. Only
   move to a small PyTorch MLP (1–2 linear layers + ReLU + dropout) if logistic regression
   underfits.
3. **Split and guard against overfitting.** Hold out a validation set. With few annotations,
   overfitting is the main risk — prefer strong regularization and more annotations over a
   bigger model. **Split by slide/region, not by random patch**, to avoid leakage from
   adjacent tiles.
4. **Handle class imbalance.** Tissue is mostly "background/other"; use class weights
   (`class_weight="balanced"`) or balanced sampling.
5. **Evaluate per class.** Report precision/recall and a confusion matrix, not just accuracy —
   accuracy is misleading under imbalance.
6. **Persist the head.** `joblib.dump` for scikit-learn, `state_dict` for PyTorch. Record
   which foundation model + embedding config (CLS vs CLS+mean, magnification) produced the
   embeddings, so inference uses matching features.

The classifier operates purely on vectors — it never sees pixels. Keep it decoupled from both
the viewer and the embedding extractor.

---

## Conventions & gotchas

- **Cache embeddings and DZI tiles** under `data/cache/`; key embeddings by
  `(slide_id, x, y, level, model_id)`. This is the difference between a responsive app and an
  unusable one.
- **Never commit slides, tiles, embeddings, or models** — they're large and often sensitive.
  `data/` is gitignored.
- **Magnification consistency** end-to-end: extraction, training, and inference must all use
  the same microns-per-pixel. Store it alongside embeddings.
- **Keep `embeddings.py` behind a single interface** so the foundation model is swappable
  (license flexibility).
- **Long/slow operations** (model download, full-slide inference) should stream progress to
  the frontend and be cancellable where feasible.
- **Coordinate systems:** OpenSeadragon uses normalized viewport coords; annotations, OpenSlide
  reads, and DZI tiles use pixel coords at specific levels. Centralize conversions and test
  them — this is a common source of subtle bugs.
- This is medical-adjacent software but **for research/development only** — it is not a
  diagnostic device. Do not add language implying clinical validity.

---

## Commands (fill in as the project is scaffolded)

```bash
# System dependency (once): OpenSlide
sudo apt install openslide-tools           # Debian/Ubuntu

# Python env
pip install -r requirements.txt

# Run the local app (intended)
uvicorn backend.app:app --reload --port 8000
# then open http://127.0.0.1:8000
```

---

## Glossary (for pathology-unfamiliar sessions)

- **WSI** — whole slide image; a digitized microscope slide, often 10k–100k+ px per side.
- **Pyramid / level** — WSIs store the image at multiple downsampled resolutions; level 0 is
  full resolution.
- **MPP** — microns per pixel; the physical scale. Lower MPP = higher magnification.
- **H&E** — hematoxylin & eosin, the standard tissue stain (purple nuclei, pink cytoplasm).
- **Tile / patch** — a small square region (here 224×224) fed to the model.
- **Embedding** — the model's numeric fingerprint of a patch.
- **Frozen feature extractor** — using a pretrained model for embeddings without training it.
- **Linear probe** — a simple classifier trained on frozen embeddings.
- **MIL** — multiple-instance learning; used for slide-level labels (not the initial workflow).
- **DZI / DeepZoom** — the tiled image format OpenSeadragon consumes.
