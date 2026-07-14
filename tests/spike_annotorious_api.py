"""SPIKE: does Annotorious accept a polygon we mint ourselves, and open its editor?

WHY THIS EXISTS
---------------
A freehand tool was attempted TWICE via @recogito/annotorious-selector-pack@0.6.1 and
reverted both times (docs/log.txt): the shape drew, but the editor never opened, so the
region could not be tagged or deleted. The page threw:

    "n.reduce is not a function"
    "Cannot read properties of null (reading 'element')"

ROOT CAUSE (found by reading the shipped bundles, not guessed):
The core's format(shape, annotation, formatters) takes an ARRAY. The selector pack
vendored a copy of it but calls it with config.formatter -- the SINGULAR function.
annotate.js configures `formatter: annotationFormatter`, so the pack does fn.reduce(...)
-> throws inside createEditableShape -> selectedShape is never assigned -> selectAnnotation
then reads `.element` of null. And format() early-returns when NO formatter is configured,
which is exactly why the old isolated harness "proved" circle worked: it omitted the one
config key that triggers the bug.

THE BET THIS SPIKE SETTLES
--------------------------
A freehand stroke is just a POLYGON with many points, and polygons are NATIVE to
Annotorious -- handled by the core's EditablePolygon, which calls the correct plural API.
So if we capture the stroke ourselves and hand Annotorious a native polygon, we never
touch the pack's broken code. This script proves (or kills) that, in the REAL app with the
formatter LIVE, using NO MOUSE SIMULATION at all -- the prior post-mortem also complained
that "headless mouse-drag simulation proved flaky", so we simply remove the drag from the
experiment. The stroke capture is ordinary DOM code that cannot fail the way the pack did.

Run:  uvicorn backend.app:app --port 8000     (in another terminal)
      .venv/bin/python tests/spike_annotorious_api.py
"""

import sys
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8000"

# The two strings that ARE the prior post-mortem. If either appears, the approach is dead.
FATAL = ["reduce is not a function", "reading 'element'"]

PASS, FAIL = [], []

# --- Editor selectors, discovered by dumping the real DOM (do NOT guess these) -------
# The footer differs by annotation kind, and getting it wrong is a TRAP that cost an hour:
#
#   freshly DRAWN shape (isSelection=true):   [Cancel(.outline)] [Ok]
#   EXISTING annotation (isSelection=false):  [DELETE(.delete-annotation)] [Cancel(.outline)] [Ok]
#
# So `.r6o-btn:not(.outline)` picks the DELETE button on an existing annotation (it is
# first in the DOM) while picking Ok on a drawn one. That silently "deletes" every shape
# you meant to confirm — and looks exactly like a library bug. Be explicit:
OK_BTN = ".r6o-editor .r6o-footer .r6o-btn:not(.outline):not(.delete-annotation)"
CANCEL_BTN = ".r6o-editor .r6o-footer .r6o-btn.outline"
DELETE_BTN = ".r6o-editor .r6o-footer .r6o-btn.delete-annotation"
TAG_INPUT = ".r6o-editor .r6o-autocomplete input"
# Committed tag chips. NOT `.r6o-tag li` — that also matches the autocomplete's
# suggestion dropdown, so it over-counts.
CHIPS = ".r6o-editor .r6o-taglist li"


def check(name, ok, got=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({got})" if got else ""))


# Mint a polygon exactly the way the real feature will.
#
#   * ONE LINE of SVG. RubberbandPolygonTool.supports matches /^<svg.*<polygon/g -- no `s`
#     flag, so a newline between <svg> and <polygon> fails the match and the shape is added
#     NON-EDITABLE (no vertex handles). That degrades silently, so we must not pretty-print.
#   * fill-rule="evenodd". SVG defaults to nonzero, but BOTH hit-testers (Annotorious's
#     Geom2D.pointInPolygon and our backend patches._point_in_polygon) are even-odd ray
#     casts. A self-crossing lasso would otherwise RENDER solid but TRAIN with a hole.
#   * points as "x,y x,y" -- comma within a pair, space between pairs (svgPolygonArea
#     splits on space then comma).
#   * an id we mint ourselves: WebAnnotation does NOT generate one, and the id is also the
#     cross-validation group key in classifier.py.
MINT_JS = """(n) => {
  const v = window.osdViewer, r = v.element.getBoundingClientRect();
  const cx = r.width / 2, cy = r.height / 2;
  const rad = Math.min(r.width, r.height) * 0.22;
  const round = (x) => Math.round(x * 10) / 10;
  const pts = Array.from({length: n}, (_, i) => {
    const t = 2 * Math.PI * i / n;
    const p = v.viewport.viewerElementToImageCoordinates(
      new OpenSeadragon.Point(cx + rad * Math.cos(t), cy + rad * Math.sin(t)));
    return `${round(p.x)},${round(p.y)}`;
  }).join(' ');
  const id = '#' + (crypto.randomUUID ? crypto.randomUUID()
                    : 'fh-' + Date.now() + '-' + Math.random().toString(16).slice(2));
  const item = v.world.getItemAt(0);
  const src = (item.source && item.source['@id']) || document.baseURI;
  const a = {
    '@context': 'http://www.w3.org/ns/anno.jsonld',
    type: 'Annotation',
    id,
    body: [],
    target: { source: src, selector: {
      type: 'SvgSelector',
      value: `<svg><polygon fill-rule="evenodd" points="${pts}"/></svg>` } },
  };
  const before = anno.getAnnotations().length;
  anno.addAnnotation(a);
  const after = anno.getAnnotations().length;
  anno.selectAnnotation(id);          // NO saveCurrent() -- that is the point (see plan)
  return { id, before, after };
}"""


with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 900})

    errors = []
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.on("console", lambda m: errors.append(f"console.error: {m.text}")
            if m.type == "error" and "404" not in m.text else None)

    print("\n=== boot the REAL app (formatter + toolbar + TAG widget all live) ===")
    page.goto(BASE, wait_until="networkidle")
    page.wait_for_function("window.osdViewer && window.osdViewer.world.getItemCount() > 0",
                           timeout=30000)
    page.wait_for_timeout(2500)          # let OSD's 0.3s animation settle
    print("  project:", page.eval_on_selector("#project-picker", "e => e.value"))
    print("  slide  :", page.inner_text("#status"))
    check("formatter IS live (this is what the old harness omitted)",
          page.evaluate("typeof annotationFormatter === 'function'"))

    print("\n=== mint a 40-gon, addAnnotation(), selectAnnotation() ===")
    res = page.evaluate(MINT_JS, 40)
    page.wait_for_timeout(600)

    # 1. THE assertion the whole approach rests on.
    fatal = [e for e in errors if any(f in e for f in FATAL)]
    check("no 'n.reduce' / 'reading element of null' (the prior post-mortem's two errors)",
          not fatal, "; ".join(fatal)[:110])
    check("zero page errors of any kind", not errors, "; ".join(dict.fromkeys(errors))[:110])

    # 2. The critical one: one drag handle per vertex proves EditablePolygon ran to
    #    completion -- the exact step that threw in BOTH prior attempts.
    handles = page.eval_on_selector_all(".a9s-handle", "es => es.length")
    check("EditablePolygon ran: 40 vertex handles", handles == 40, f"{handles} handles")

    shapes = page.eval_on_selector_all(".a9s-annotation", "es => es.length")
    polys = page.eval_on_selector_all(".a9s-annotation polygon", "es => es.length")
    check("shape rendered as a <polygon>", polys >= 1, f"{shapes} shapes, {polys} polygons")

    # 3. The editor, with BOTH widgets (the custom force:"plainjs" one must survive).
    ed = page.query_selector(".r6o-editor")
    check("editor popup opened", bool(ed) and ed.is_visible())
    check("editor has the TAG widget", bool(page.query_selector(".r6o-editor .r6o-widget")))
    check("editor has the custom Find-similar widget",
          bool(page.query_selector(".r6o-editor .find-similar-btn")))

    # 4. The 1ms duplicate window in selectShape (both the plain shape and the editable
    #    group carry .a9s-annotation for ~1ms; getAnnotations() queries that class).
    n = page.evaluate("anno.getAnnotations().length")
    uniq = page.evaluate("new Set(anno.getAnnotations().map(a => a.id)).size")
    check("no duplicate annotation after select", res["after"] == n and n == uniq,
          f"add {res['before']}->{res['after']}, now {n}, unique {uniq}")

    # 5. Tag it via the real editor, and prove the save happens WITHOUT any manual call.
    print("\n=== tag it through the real editor (no manual saveCurrent anywhere) ===")
    page.click(TAG_INPUT)
    page.type(TAG_INPUT, "gland", delay=30)
    page.keyboard.press("Enter")
    page.wait_for_timeout(400)
    check("tag chip committed in the editor",
          page.eval_on_selector_all(CHIPS, "es => es.length") == 1)
    page.click(OK_BTN)
    page.wait_for_timeout(1500)

    status = page.inner_text("#anno-status")
    check("saved WITHOUT a manual saveCurrent (updateAnnotation wiring fired)",
          "saved (1)" in status, status)
    tagged = page.evaluate("""() => {
      const a = anno.getAnnotations()[0];
      const b = (a && a.body) || [];
      return b.some(x => x.purpose === 'tagging' && x.value === 'gland');
    }""")
    check("tagging body applied", tagged)
    colored = page.evaluate("""() => {
      const el = document.querySelector('.a9s-annotation');
      return el ? el.getAttribute('class') : '';
    }""")
    check("formatter ran (class-colour slug on the shape)", "sp-cls-gland" in (colored or ""),
          colored)

    # 6. Full lifecycle: reload -> it comes back -> click -> editor -> delete.
    print("\n=== reload, re-select, delete ===")
    page.reload(wait_until="networkidle")
    page.wait_for_function("window.osdViewer && window.osdViewer.world.getItemCount() > 0",
                           timeout=30000)
    page.wait_for_timeout(2500)
    back = page.eval_on_selector_all(".a9s-annotation", "es => es.length")
    check("polygon persisted + reloaded from disk", back == 1, f"{back} shapes")

    box = page.evaluate("""() => {
      const el = document.querySelector('.a9s-annotation polygon');
      const r = el.getBoundingClientRect();
      return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
    }""")
    page.mouse.click(box["x"], box["y"])
    page.wait_for_timeout(800)
    check("clicking the shape reopens the editor",
          bool(page.query_selector(".r6o-editor")))
    check("re-selected shape is editable (handles present)",
          page.eval_on_selector_all(".a9s-handle", "es => es.length") == 40)
    check("its tag survived the round-trip",
          page.eval_on_selector_all(CHIPS, "es => es.length") == 1)

    page.click(DELETE_BTN)
    page.wait_for_timeout(1500)
    left = page.eval_on_selector_all(".a9s-annotation", "es => es.length")
    check("deleted from the layer", left == 0, f"{left} shapes left")
    check("delete persisted", "saved (0)" in page.inner_text("#anno-status"),
          page.inner_text("#anno-status"))

    browser.close()

print("\n" + "=" * 66)
print(f"PASS {len(PASS)}   FAIL {len(FAIL)}")
if FAIL:
    print("FAILED: " + "; ".join(FAIL))
print("\nall page errors seen:", list(dict.fromkeys(errors)) or "NONE")
sys.exit(1 if FAIL else 0)
