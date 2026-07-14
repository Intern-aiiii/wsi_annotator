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
Extract features  ──▶  Virchow 2   ──▶  FEATURE BANK        (ONCE per slide, explicit,
(whole slide, on        embeddings       (every tissue tile   abortable, resumable. Minutes.)
 the global grid,       (frozen FM)       on the grid)
 tissue-masked)                                │
                                               ▼
Browser UI  ──▶  which cells does  ──▶  Train classifier ──▶  Predict / Find similar
(view +          each annotation        head                  (score the viewport,
 annotate)       cover? (geometry)      (logistic reg)         tinted region overlay)
     ▲            └─ bank lookups: no pixels, no GPU, <1s ─┘                │
     └──────────────────  iterate: refine annotations, retrain  ◀──────────┘

The bank is the pivot: the model runs ONCE per tile, ever. Annotating and predicting are
lookups. Predict / Find similar are REFUSED until a slide's sweep has completed.
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
│   ├── patches.py      # THE tile grid + annotation geometry + tissue mask
│   ├── embeddings.py   # Virchow 2 load + inference (swappable model). No cache.
│   ├── features.py     # the FEATURE BANK: whole-slide sweep + storage (cancellable)
│   ├── projects.py     # PROJECTS: workspaces owning annotations + classes + a head
│   ├── annotations.py  # per-(project, slide) W3C annotation JSON
│   ├── classifier.py   # train / persist / load a project's classifier head
│   ├── jobs.py         # debounced background retrain worker (per project)
│   ├── inference.py    # viewport scoring → region overlay
│   └── models.py       # data models: projects, classes, annotations, regions
├── frontend/
│   ├── index.html
│   ├── js/api.js       # fetch() calls to the backend; owns the active-project URL
│   ├── js/projects.js  # projects panel + THE BOOT DRIVER (see below)
│   ├── js/panels.js    # the right-hand panels as a radio group (one open at a time)
│   ├── js/viewer.js    # OpenSeadragon setup + DZI source
│   ├── js/annotate.js  # Annotorious integration, save/load annotations
│   ├── js/features.js  # "Extract features": start / progress / cancel; owns the gate
│   ├── js/predict.js   # Predict view + Find similar, region overlay
│   └── js/learn.js     # background-retrain status chip
├── data/               # gitignored
│   ├── slides/         # input WSIs                        ─┐ GLOBAL: a file, or a
│   ├── cache/          # DeepZoom tiles + features/ (banks) ─┘ cache of a file
│   └── projects/       # ONE DIRECTORY PER EXPERIMENT
│       └── <project>/
│           ├── project.json          # name, slide membership, classes (+ colours)
│           ├── annotations/<slide>.json
│           └── models/head__<embedding_model>.{joblib,json}
├── requirements.txt
├── README.md
└── CLAUDE.md
```

**Projects (Phase 7).** A project is one experiment: it owns its annotations, its class
list, and its own trained head, so "glands vs stroma" and "tumor vs normal" can coexist
over the same slides instead of contaminating a single pooled classifier.

The rule that keeps this simple: **a route is GLOBAL if it is about a file on disk (or a
cache derived only from that file); it is PROJECT-SCOPED if it is about the user's
experiment.** So `/api/slides` and every `/api/slides/{id}/features*` route stay global,
while annotations, classes, train, model, predict and similarity live under
`/api/projects/{pid}/…`.

**The feature bank is shared across projects, and that is load-bearing.** It is keyed by
`(slide, embedding model, tissue mask)` — nothing experiment-specific enters it — so a
slide swept once with Virchow 2 is reused by every project containing it. Project-scoping
the bank would re-run the single most expensive step in the pipeline once per project for
no gain. Deleting a project therefore never touches a slide or its bank.

**The frontend boot sequence.** `projects.js` is the ONLY boot driver. It fetches the
active project, dispatches `slideprobe:project-opened` (a synchronous reset — every
module forgets the old project), and only then calls `SlideViewer.showProjectSlides()`.
viewer.js deliberately does NOT boot itself: it used to auto-open a slide on
`DOMContentLoaded`, which raced project loading, and script order could not fix that (an
async handler yields at its first `await`, so the next listener runs on the same tick).

---

## Development phases (build the thinnest end-to-end slice first)

Each phase must produce something that runs. Do not jump ahead; a working Phase N beats a
half-built Phase N+2.

- [x] **Phase 1 — One slide on screen.** FastAPI server generates DeepZoom tiles from a
      sample `.svs` via OpenSlide; OpenSeadragon displays it in the browser. No ML yet. This
      proves the whole frontend↔backend loop. *(Code complete; run once a slide + OpenSlide
      are installed — see README.)*
- [x] **Phase 2 — Annotation.** Annotorious (v2) on the OpenSeadragon viewer; user draws
      rects/polygons/**freehand lassos** and tags each with a class label; the full W3C
      annotation collection is saved per-(project, slide) via
      `GET`/`PUT /api/projects/{pid}/slides/{sid}/annotations`.
      **Freehand** (`frontend/js/freehand.js`) captures the stroke itself and emits a *native
      polygon*, so no new shape type exists anywhere — see the gotchas below and docs/log.txt
      before touching it (two earlier attempts via the Annotorious "selector pack" failed).
- [x] **Phase 3 — The tile grid + tissue mask.** `backend/patches.py` defines **the** grid
      (`grid_config()`): non-overlapping 224×224 cells at `TARGET_MPP=0.5` (~20×), addressed by
      integer `(col, row)`, plus the pure-geometry `cells_in_annotation()` / `cells_in_region()`
      and the tissue mask. The mask is a per-slide `TissueGate` (`SLIDEPROBE_TISSUE` env:
      `otsu` default | `hsv`): **Otsu** derives its saturation cutoff from an overview of the
      whole slide, so it adapts to how faintly or darkly that slide is stained; `hsv` is a
      fixed cutoff of 25. Both then keep a tile whose saturated fraction ≥ 0.30.
      *(Originally cut per-annotation patches to a manifest + montage; that was replaced by
      the whole-slide feature bank in Phase 4b, which removed a train/serve tile-alignment
      skew — see docs/log.txt 2026-07-13.)*
- [x] **Phase 4 — Embeddings.** `backend/embeddings.py` is the swappable frozen extractor
      and *nothing else* — no cache, no knowledge of slides (`SLIDEPROBE_EMBEDDER` env:
      `dev` default | `virchow2`). The default is a lightweight numpy/PIL stand-in
      (`dev-colorstats-v1`, 62-dim) so the pipeline runs without the gated model;
      **Virchow 2** (frozen ViT-H → 2560-dim, CLS ⊕ mean patch tokens) is the opt-in
      production backend. *To enable: `pip install torch timm huggingface_hub`, get HF
      access to paige-ai/Virchow2, `huggingface-cli login`, run with
      `SLIDEPROBE_EMBEDDER=virchow2`.*
- [x] **Phase 4b — The feature bank (user-triggered).** `backend/features.py` sweeps the
      WHOLE slide on the global grid, tissue-masks each cell, embeds the tissue ones, and
      stores them: `data/cache/features/<slide>__<model>.{npy,json}` (float16 + the set of
      cells known to be background). **Cancellable and resumable** — a cancelled sweep
      leaves a valid partial bank the next run continues from. Started by the "Extract
      features" button; `POST/GET/DELETE /api/slides/{id}/features`, `POST .../cancel`.
      `complete` is derived (`covered == cols*rows`), never trusted from the header.
- [x] **Phase 5 — Train the head.** `backend/classifier.py` asks each annotation which grid
      cells it covers (pure geometry) and looks them up in the bank — **no pixels, no GPU**,
      so a retrain is ~60ms and runs automatically (debounced) on every annotation save.
      `StandardScaler` + `LogisticRegression` (`class_weight="balanced"`), honest metrics via
      **region-grouped out-of-fold CV** (group = the drawn region, so adjacent tiles never
      leak across the split). Persists `data/models/head__<model_id>.{joblib,json}`.
      `POST /api/train`, `GET /api/model`.
- [x] **Phase 6 — Predict + overlay.** `backend/inference.py` scores the **current viewport**
      (chosen over a whole-slide sweep to stay responsive): tiles the visible region, pulls
      each tile's vector from the bank, runs `head.predict_proba`, and returns the confident
      regions as tile `cells` + `boundaries`, which the frontend draws as one tinted SVG
      overlay. "Find similar" does the same by cosine similarity to a selected annotation's
      mean embedding (no classifier needed). **Both are gated: `no_features` → HTTP 409
      unless the slide's sweep has COMPLETED.** `POST /api/slides/{id}/{predict,similarity}`.
- [~] **Phase 7 — Close the loop + polish.** *(Partly done.)*
      - [x] **The retrain cycle.** Every annotation save schedules a debounced background
            retrain (`backend/jobs.py`, ~60ms — geometry + bank lookups, no pixels, no GPU),
            surfaced as the topbar chip (`frontend/js/learn.js`).
      - [x] **Multiple classes.** The head is multi-class (`sorted(set(y))` + multinomial
            `LogisticRegression`); the class list lives in the project with an explicit
            per-class colour, and the predict overlay is drawn in that class's colour.
      - [x] **Project management.** Named workspaces owning annotations + classes + a head,
            over a SHARED feature bank. See "Projects" above. `backend/projects.py`,
            `frontend/js/projects.js`.
      - [ ] **Correct predictions.** The active-learning loop is not closed yet: the overlay
            is `pointer-events: none` and keeps no per-cell identity, so a predicted region
            cannot be clicked to confirm/relabel it. The intended design is *one polygon
            annotation per clicked region* — NOT one per tile, because the classifier's CV
            groups by drawn region, so one-annotation-per-tile would split adjacent tiles
            across folds and leak, inflating the metrics.
      - [ ] **Export.** Nothing exports yet. GeoJSON annotations (QuPath-importable) and a
            whole-slide per-class quantification (tile counts / area) are the obvious first
            two — and whole-slide scoring is now nearly free (one `predict_proba` over the
            cached bank), so it no longer needs a viewport.

---

## Phase 5 detail — training the classifier head

This is the crux of the whole approach, so get it right:

1. **Assemble the dataset.** Ask each annotation which grid cells it covers (pure geometry)
   and look those cells up in the slide's feature bank. Every annotated tissue cell gives an
   embedding `X` and a label `y`. No pixels are read and the model never runs — that is what
   makes retraining on every annotation edit affordable.
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

- **One tile grid, everywhere.** `patches.grid_config()` is the single definition; every
  module addresses tiles as integer `(col, row)` on it. Extraction, training, and inference
  MUST agree — an earlier version tiled annotations from their own bbox corner while
  inference tiled on a global grid, which silently trained the head on one set of tile
  alignments and applied it to another.
- **Embedding is never on the interactive path.** It happens once, in the explicit
  user-triggered feature sweep. Annotating and predicting are lookups into the bank.
- **Cache embeddings and DZI tiles** under `data/cache/`; the feature bank is keyed by
  `(slide_id, model_id)` on disk and `(col, row)` within. This is the difference between a
  responsive app and an unusable one.
- **Never commit slides, tiles, embeddings, or models** — they're large and often sensitive.
  `data/` is gitignored.
- **Magnification consistency** end-to-end: extraction, training, and inference must all use
  the same microns-per-pixel. Store it alongside embeddings.
- **Otsu must see the whole slide, never one tile.** Otsu always finds a split — run it on a
  tile of blank glass and it will divide the sensor noise into "dark" and "light" and call
  half the background tissue. The threshold is therefore derived ONCE per slide from a low-res
  overview (`patches._overview`), then applied per tile, and clamped to a sane band in case a
  slide really does have only one population. Each feature bank records the mask id + cutoff
  that judged its cells and rejects itself if either changes, so two masks' verdicts can never
  be mixed in one bank.
- **Keep `embeddings.py` behind a single interface** so the foundation model is swappable
  (license flexibility).
- **`project.json["slides"]` is a DISPLAY list, never a training input.** It decides what
  the slide picker shows. What the head trains on is the set of annotation files in the
  project's `annotations/` directory. Keeping these separate is what stops "I removed the
  slide from the project but the model still knows about it".
- **Annotation tags are the source of truth for classes; `project.json["classes"]` is only
  a vocabulary + colour registry.** So deleting a class is cosmetic — its regions keep
  their tag and keep training, and the head reports them as `classes_not_in_project`.
  Corollary: there is deliberately **no class rename**, because a label is a plain string
  copied into every annotation that uses it; renaming would mean rewriting every file.
- **A class's colour is stored, never derived from its index.** Deriving it from position
  in the list (the pre-Phase-7 behaviour) meant deleting one class silently repainted every
  class after it.
- **`patches.label_of()` returns `"unlabeled"` for an untagged region — never train on it.**
  `classifier._gather_dataset` skips it and counts it. Left in, it becomes a real class that
  appears in the predict dropdown, has no colour, and can even satisfy the "need 2 classes"
  check on its own.
- **The frontend is plain `<script>`s sharing ONE global scope.** Two top-level `let`s of
  the same name in different files is a duplicate-declaration SyntaxError that silently
  kills the *whole* second file — you get a working viewer and an annotation module that
  never loaded. Hence `annoFeaturesReady`/`pdFeaturesReady` and `currentSlideId`
  (annotate.js) vs `openSlideId` (viewer.js). Prefer an IIFE for new modules.
- **Do NOT reach for `@recogito/annotorious-selector-pack`** (freehand/circle/point). It is
  broken *by our own config*: it vendored the core's `format(shape, annotation, formatters)`
  — which takes an **array** — but calls it with the singular `config.formatter`, so the
  per-class formatter in `annotate.js` makes it throw `n.reduce is not a function` and the
  editor never opens. Two attempts died on this. The freehand tool instead captures the
  stroke itself and emits a **native polygon** (`frontend/js/freehand.js`) — see
  `docs/log.txt`. A stroke is just a polygon; no new shape type is needed anywhere.
- **A programmatically-added annotation must be one-line SVG + `fill-rule="evenodd"`.**
  Annotorious picks its editing tool with `/^<svg.*<polygon/` (no `s` flag), so a
  pretty-printed selector silently yields a shape with no vertex handles. And SVG fills
  `nonzero` while *both* hit-testers (Annotorious's and `patches._point_in_polygon`) are
  even-odd — so a self-crossing lasso would render solid but train with a hole in it.
- **Never `saveCurrent()` between `addAnnotation()` and `selectAnnotation()`.** It would run
  `rememberTypedTags → ensureClasses → classes-changed → rebuildAnnotorious()`, destroying
  the instance whose editor you just opened; and Annotorious briefly has the shape in the DOM
  twice during select, so the save would write it to disk twice. The editor's OK fires
  `updateAnnotation`, which is already wired to save.
- **The annotation editor's footer differs by annotation kind.** A freshly *drawn* shape gets
  `[Cancel][Ok]`; an *existing* one gets `[DELETE][Cancel][Ok]`. So `.r6o-btn:not(.outline)`
  clicks **Ok** on one and **Delete** on the other — which looks exactly like the library
  destroying your data. Always select these explicitly (see `tests/`).
- **Deleting a project must never touch a slide or a feature bank**, and a train that is
  already in flight must not re-create the directory it is writing into. `projects.LOCK` is
  held across both `delete()` and a training pass, and `classifier.train()` refuses to
  `mkdir` a project — otherwise a deleted project partially comes back from the dead.
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
