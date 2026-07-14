"""End-to-end verification of the freehand (lasso) tool, in the REAL app.

The prior post-mortem (docs/log.txt) warns about two things that made the last attempt's
testing lie, and both are addressed here:

  1. "the isolated harness omitted the formatter/toolbar" -- so this drives the real page:
     real project, real slide, real toolbar, real per-class formatter, real TAG widget.
  2. "headless mouse-drag simulation proved flaky" -- that flakiness came from driving
     Annotorious's own OSD MouseTracker, which has a click-vs-drag time threshold and
     internal state. Our capture layer is a plain div with plain pointer listeners and no
     state machine at all, so a synthetic drag is deterministic.

It asserts the WHOLE pipeline, not just "a shape appeared": drawn -> editor -> tagged ->
SAVED TO DISK as a polygon -> geometry resolves to grid cells -> the classifier trains on
it -> re-selectable -> deletable. Plus: any uncaught page error fails the run.

Run:  uvicorn backend.app:app --port 8000    (in another terminal)
      .venv/bin/python tests/e2e_freehand.py
"""

import json
import math
import pathlib
import sys
import time

import httpx as requests
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from backend import patches, slides  # noqa: E402

BASE = "http://127.0.0.1:8000"
SLIDE = "CMU-1-Small-Region"
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"

# See tests/spike_annotorious_api.py: the editor footer has a DELETE button for an
# existing annotation but not for a freshly drawn one, so `.r6o-btn:not(.outline)` would
# pick DELETE. Be explicit.
OK_BTN = ".r6o-editor .r6o-footer .r6o-btn:not(.outline):not(.delete-annotation)"
DELETE_BTN = ".r6o-editor .r6o-footer .r6o-btn.delete-annotation"
TAG_INPUT = ".r6o-editor .r6o-autocomplete input"

PASS, FAIL = [], []


def check(name, ok, got=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({got})" if got else ""))


def saved_annotations(pid):
    f = DATA / "projects" / pid / "annotations" / f"{SLIDE}.json"
    return json.loads(f.read_text()) if f.exists() else []


def points_of(anno):
    val = anno["target"]["selector"]["value"]
    inner = val.split('points="')[1].split('"')[0]
    return [tuple(float(v) for v in p.split(",")) for p in inner.split()]


# Image coords -> page/client coords, so the drag lands where we mean it to.
TO_CLIENT = """(pt) => {
  const v = window.osdViewer, r = v.element.getBoundingClientRect();
  const p = v.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(pt[0], pt[1]));
  return { x: r.left + p.x, y: r.top + p.y };
}"""


def lasso(page, centre_img, radius_img, n=36):
    """Drag a closed loop (in image space) and return the client points used."""
    pts = []
    for i in range(n + 1):
        t = 2 * math.pi * i / n
        pts.append([centre_img[0] + radius_img * math.cos(t),
                    centre_img[1] + radius_img * math.sin(t)])
    client = [page.evaluate(TO_CLIENT, p) for p in pts]
    page.mouse.move(client[0]["x"], client[0]["y"])
    page.mouse.down()
    for c in client[1:]:
        page.mouse.move(c["x"], c["y"])
    page.mouse.up()
    page.wait_for_timeout(700)
    return client


def tag_and_ok(page, label):
    page.click(TAG_INPUT)
    page.type(TAG_INPUT, label, delay=25)
    page.keyboard.press("Enter")
    page.wait_for_timeout(300)
    page.click(OK_BTN)
    page.wait_for_timeout(1500)


def main():
    # A throwaway project. Deleting it never touches the slide or the shared feature bank.
    pid = requests.post(f"{BASE}/api/projects",
                        json={"name": "Freehand Verify", "slides": [SLIDE]}).json()["id"]
    feat = requests.get(f"{BASE}/api/slides/{SLIDE}/features").json()
    if feat.get("state") != "complete":
        print(f"  slide {SLIDE} has no feature bank ({feat.get('state')}) — extracting…")
        requests.post(f"{BASE}/api/slides/{SLIDE}/features")
        while requests.get(f"{BASE}/api/slides/{SLIDE}/features").json()["state"] != "complete":
            time.sleep(1)

    # Two well-separated TISSUE cells to draw around.
    slide = slides.get_slide(SLIDE)
    grid = patches.grid_config(slide)
    idx = json.loads((DATA / "cache" / "features" /
                      f"{SLIDE}__dev-colorstats-v1.json").read_text())
    tissue = [(i % grid["cols"], i // grid["cols"]) for i in idx["tissue"]]
    s0 = grid["size0"]
    centre_a = ((tissue[4][0] + 0.5) * s0, (tissue[4][1] + 0.5) * s0)
    centre_b = ((tissue[-5][0] + 0.5) * s0, (tissue[-5][1] + 0.5) * s0)

    errors = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console.error: {m.text}")
                if m.type == "error" and "404" not in m.text else None)
        page.add_init_script(
            f"localStorage.setItem('slideprobe.activeProject', '{pid}');")

        page.goto(BASE, wait_until="networkidle")
        page.wait_for_function(
            "window.osdViewer && window.osdViewer.world.getItemCount() > 0", timeout=30000)
        page.wait_for_timeout(2500)

        print("\n=== 1. arm the tool ===")
        check("button enabled once a slide is open",
              not page.eval_on_selector("#freehand-btn", "e => e.disabled"))
        page.click("#freehand-btn")
        page.wait_for_timeout(300)
        check("armed: button lit + capture layer present",
              page.evaluate("__freehand.isArmed()") and page.evaluate("__freehand.hasCapture()"))

        print("\n=== 2. draw a lasso -> a native polygon ===")
        lasso(page, centre_a, s0 * 1.4)
        shapes = page.eval_on_selector_all(".a9s-annotation", "es => es.length")
        polys = page.eval_on_selector_all(".a9s-annotation polygon", "es => es.length")
        check("one annotation, drawn as a <polygon>", shapes == 1 and polys >= 1,
              f"{shapes} shapes / {polys} polygons")
        n_verts = page.evaluate("""() => {
          const p = document.querySelector('.a9s-annotation polygon');
          return p.getAttribute('points').trim().split(/\\s+/).length;
        }""")
        handles = page.eval_on_selector_all(".a9s-handle", "es => es.length")
        check("EditablePolygon ran: one drag handle per vertex",
              handles == n_verts, f"{handles} handles / {n_verts} vertices")
        check(f"simplified under the {100}-vertex cap", n_verts <= 100, f"{n_verts} vertices")
        check("capture layer torn down so it can't cover the editor",
              not page.evaluate("__freehand.hasCapture()"))
        check("editor opened with the TAG widget",
              bool(page.query_selector(".r6o-editor .r6o-widget")))
        check("editor has the custom Find-similar widget",
              bool(page.query_selector(".r6o-editor .find-similar-btn")))

        print("\n=== 3. tag it (the save must happen with no manual save call) ===")
        tag_and_ok(page, "gland")
        check("saved", "saved (1)" in page.inner_text("#anno-status"),
              page.inner_text("#anno-status"))
        cls = page.eval_on_selector(".a9s-annotation", "e => e.getAttribute('class')")
        check("formatter coloured it by class", "sp-cls-gland" in (cls or ""), cls)

        on_disk = saved_annotations(pid)
        ok_disk = (len(on_disk) == 1
                   and on_disk[0]["target"]["selector"]["type"] == "SvgSelector"
                   and 'fill-rule="evenodd"' in on_disk[0]["target"]["selector"]["value"]
                   and len(points_of(on_disk[0])) >= 3
                   and any(b.get("purpose") == "tagging" for b in on_disk[0]["body"]))
        check("saved to disk ONCE as an evenodd polygon with a tag (not duplicated)",
              ok_disk, f"{len(on_disk)} on disk")

        print("\n=== 4. STICKY: the tool re-armed itself ===")
        check("capture layer is back after the editor closed",
              page.evaluate("__freehand.isArmed()") and page.evaluate("__freehand.hasCapture()"))
        lasso(page, centre_b, s0 * 1.4)
        tag_and_ok(page, "stroma")
        on_disk = saved_annotations(pid)
        check("second region drawn without touching the button", len(on_disk) == 2,
              f"{len(on_disk)} on disk")
        check("the two regions have distinct ids (the CV group key)",
              len({a["id"] for a in on_disk}) == 2)
        page.click("#freehand-btn")   # disarm
        page.wait_for_timeout(300)
        check("disarms on a second click", not page.evaluate("__freehand.isArmed()"))

        print("\n=== 5. coordinate proof: freehand lands where Annotorious itself would ===")
        # Drive the TOOLBAR RECT over four screen corners, then compare a lasso around the
        # same corners. If both land on the same image rectangle, they share a coord space.
        corners = [page.evaluate(TO_CLIENT, [centre_a[0] - s0, centre_a[1] - s0]),
                   page.evaluate(TO_CLIENT, [centre_a[0] + s0, centre_a[1] + s0])]
        page.click("#anno-toolbar button:first-child")
        page.wait_for_timeout(300)
        page.mouse.move(corners[0]["x"], corners[0]["y"])
        page.mouse.down()
        page.mouse.move(corners[1]["x"], corners[1]["y"], steps=8)
        page.mouse.up()
        page.wait_for_timeout(700)
        rect_xywh = page.evaluate("""() => {
          const a = anno.getAnnotations().find(a =>
            a.target.selector.type === 'FragmentSelector');
          const m = a.target.selector.value.match(
            /xywh=(?:pixel:)?([\\d.]+),([\\d.]+),([\\d.]+),([\\d.]+)/);
          return m.slice(1).map(Number);
        }""")
        page.click(".r6o-editor .r6o-footer .r6o-btn.outline")  # cancel the rect
        page.wait_for_timeout(500)

        want = (centre_a[0] - s0, centre_a[1] - s0, 2 * s0, 2 * s0)
        drift = max(abs(rect_xywh[i] - want[i]) for i in range(4))
        check("toolbar rect lands where we asked (our screen->image maths is right)",
              drift <= 3, f"drift {drift:.1f} px")
        pts = points_of(on_disk[0])
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        off = math.hypot(cx - centre_a[0], cy - centre_a[1])
        check("freehand polygon is centred where it was drawn (same space)",
              off <= 6, f"centre off by {off:.1f} image px")

        print("\n=== 6. the backend can read it, and TRAINS on it ===")
        cells = patches.cells_in_annotation(grid, on_disk[0])
        check("geometry resolves to grid cells", len(cells) > 0, f"{len(cells)} cells")
        r = requests.post(f"{BASE}/api/projects/{pid}/train").json()
        check("trained", r.get("status") == "ok", r.get("error") or r.get("reason") or "")
        check("both freehand classes are in the head",
              set(r.get("classes", [])) >= {"gland", "stroma"}, str(r.get("classes")))
        check("n_unparseable == 0 (nothing silently dropped)",
              r.get("n_unparseable") == 0, str(r.get("n_unparseable")))
        check("training used the freehand cells", r.get("n_samples", 0) > 0,
              f"{r.get('n_samples')} tiles")

        print("\n=== 7. select + delete ===")
        box = page.evaluate("""() => {
          const r = document.querySelector('.a9s-annotation polygon').getBoundingClientRect();
          return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
        }""")
        page.mouse.click(box["x"], box["y"])
        page.wait_for_timeout(700)
        check("clicking a freehand region reopens the editor",
              bool(page.query_selector(".r6o-editor")))
        page.click(DELETE_BTN)
        page.wait_for_timeout(1500)
        check("deleted, and the deletion persisted", len(saved_annotations(pid)) == 1,
              f"{len(saved_annotations(pid))} left on disk")

        print("\n=== 8. page errors ===")
        check("zero uncaught page errors", not errors,
              "; ".join(dict.fromkeys(errors))[:120])
        browser.close()

    requests.delete(f"{BASE}/api/projects/{pid}")
    print("\n" + "=" * 66)
    print(f"PASS {len(PASS)}   FAIL {len(FAIL)}")
    if FAIL:
        print("FAILED: " + "; ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
