#!/usr/bin/env python3
"""Real-browser acceptance test of the offline Bernstein Proof Lab (Playwright).

Run with a Python that has playwright installed and browsers downloaded:
  python tests/browser/proof_lab_browser.py --html reports/bernstein_proof_lab.html \
      --out runs/browser_qa/proof_lab --analytic /tmp/analytic_residuals.json
Checks every feature in both Chromium and Firefox, desktop and mobile viewports,
records console/page errors and any non-file network request, saves screenshots.
Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def run_engine(pw, engine: str, html: Path, out: Path, analytic: Path | None) -> dict:
    browser = getattr(pw, engine).launch(headless=True)
    results, errors, requests = [], [], []

    def check(name, ok, detail=""):
        results.append({"check": name, "pass": bool(ok), "detail": str(detail)[:300]})

    for label, viewport in (("desktop", {"width": 1440, "height": 1000}), ("mobile", {"width": 390, "height": 844})):
        ctx = browser.new_context(viewport=viewport, accept_downloads=True)
        page = ctx.new_page()
        page.on("console", lambda m: errors.append(f"{label}:{m.type}:{m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(f"{label}:pageerror:{e}"))
        page.on("request", lambda r: requests.append(r.url) if not r.url.startswith(("file:", "blob:", "data:")) else None)
        page.goto(html.resolve().as_uri())
        page.wait_for_selector("#verdict")
        verdict = lambda: page.inner_text("#verdict").strip().upper()  # noqa: E731

        def setc(text, strict=None, depth=None, nodes=None, witnesses=None, strict_left=None):
            page.fill("#coeffs", text)
            for sel, val in (("#strict", strict), ("#witnesses", witnesses), ("#strictLeft", strict_left)):
                if val is not None and page.is_checked(sel) != val:
                    page.click(sel) if label == "desktop" else page.locator(sel).check() if val else page.locator(sel).uncheck()
            if depth is not None:
                page.eval_on_selector("#depth", f"e=>{{e.value='{depth}';e.dispatchEvent(new Event('input'))}}")
            if nodes is not None:
                page.eval_on_selector("#nodes", f"e=>{{e.value='{nodes}';e.dispatchEvent(new Event('input'))}}")
            page.click("#run")

        p = f"{engine}/{label}"
        check(f"{p} initial example certified", verdict() == "CERTIFIED", verdict())
        check(f"{p} no horizontal overflow",
              page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"),
              page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]"))
        page.screenshot(path=str(out / f"{engine}_{label}_initial.png"), full_page=True)
        # 1+3 editing and exact bounds
        setc("-1, 2/5, -1", strict=False, depth=8, nodes=2048, witnesses=True)
        check(f"{p} edit coefficients -> CERTIFIED", verdict() == "CERTIFIED", verdict())
        check(f"{p} initial hull exact 2/5", page.inner_text("#initialHull").strip() == "2/5", page.inner_text("#initialHull"))
        page.click("#tree [data-index='0']")
        insp = page.inner_text("#inspector")
        check(f"{p} root exact coefficient bounds", "[-1, 2/5]" in insp, insp[:120])
        # 2 plot
        n_curve = page.eval_on_selector_all("#plot polyline.curve", "e=>e.length")
        n_rect = page.eval_on_selector_all("#plot rect", "e=>e.length")
        check(f"{p} residual plot drawn with hull rectangles", n_curve == 1 and n_rect >= 2, (n_curve, n_rect))
        # 4 global peak bracket
        pk = page.inner_text("#peakBracket")
        check(f"{p} global peak bracket lower -3/10", pk.startswith("[-3/10,"), pk)
        # 5 animation
        page.click("#reset")
        check(f"{p} reset to zero nodes", page.inner_text("#stepCounter").startswith("0 /"), page.inner_text("#stepCounter"))
        page.click("#next")
        check(f"{p} step forward", page.inner_text("#stepCounter").startswith("1 /"), page.inner_text("#stepCounter"))
        page.click("#play")
        page.wait_for_function("document.getElementById('stepCounter').textContent.startsWith('3 / 3')", timeout=5000)
        check(f"{p} play animation completes", True)
        page.screenshot(path=str(out / f"{engine}_{label}_proof_tree.png"), full_page=True)
        # 10 replay
        check(f"{p} structural replay pass", "PASS" in page.inner_text("#replay"), page.inner_text("#replay"))
        # 6 witness
        setc("-1, 2, -1")
        check(f"{p} edited polynomial not labelled as preset", "Custom" in page.inner_text("#exampleTitle"), page.inner_text("#exampleTitle"))
        check(f"{p} violation witness", verdict() == "VIOLATED" and "s = 1/2" in page.inner_text("#inspector"),
              verdict() + " " + page.inner_text("#inspector")[:160])
        n_w = page.eval_on_selector_all("#plot circle.witness-dot", "e=>e.length")
        check(f"{p} witness drawn", n_w == 1, n_w)
        page.screenshot(path=str(out / f"{engine}_{label}_witness.png"), full_page=True)
        setc("-1, 2, -1", witnesses=False, depth=4)
        check(f"{p} no-witness ablation is UNKNOWN", verdict() == "UNKNOWN DEPTH", verdict())
        # 7 strict vs nonstrict
        setc("-1/4, 1/4, -1/4", strict=True, witnesses=True, depth=8)
        check(f"{p} strict tangency violated", verdict() == "VIOLATED STRICT", verdict())
        setc("-1/4, 1/4, -1/4", strict=False)
        check(f"{p} nonstrict tangency certified", verdict() == "CERTIFIED", verdict())
        setc("0, -1/2, -1", strict=True, strict_left=True)
        check(f"{p} strict interior-knot endpoint zero violated", verdict() == "VIOLATED STRICT", verdict())
        setc("0, -1/2, -1", strict=True, strict_left=False)
        check(f"{p} strict global endpoint zero allowed", verdict() == "CERTIFIED", verdict())
        # 8 budgets
        setc("-1, 2/5, -1", strict=False, depth=0)
        check(f"{p} depth budget exhaustion", verdict() == "UNKNOWN DEPTH", verdict())
        setc("-1, 2/5, -1", depth=8, nodes=1)
        check(f"{p} node budget exhaustion", verdict() == "UNKNOWN NODES", verdict())
        setc("1/0, 1")
        check(f"{p} invalid input fails closed", verdict() == "INVALID INPUT", verdict())
        check(f"{p} step buttons disabled after invalid input", page.is_disabled("#next") and page.is_disabled("#prev"))
        # Dispatch directly to exercise the null-proof guard (must not throw a page error).
        page.eval_on_selector("#next", "e=>e.dispatchEvent(new MouseEvent('click'))")
        page.eval_on_selector("#reset", "e=>e.dispatchEvent(new MouseEvent('click'))")
        # 9 export / import
        setc("-1, 2/5, -1", depth=8, nodes=2048)
        with page.expect_download() as dl:
            page.click("#export")
        exported = json.loads(Path(dl.value.path()).read_text())
        check(f"{p} export JSON has proof leaves", len(exported["proof"]["leaves"]) == 2, exported["proof"]["status"])
        good, bad, junk, badstrict = (out / f"{engine}_{label}_{n}.json" for n in ("good", "tampered", "junk", "badstrict"))
        good.write_text(json.dumps(exported))
        exported["proof"]["leaves"][0]["coefficients"][0] = "-2"
        bad.write_text(json.dumps(exported))
        junk.write_text("{not json")
        badstrict.write_text(json.dumps({"coefficients": ["-1", "-1"], "strict": "false"}))
        page.set_input_files("#importFile", str(good))
        page.wait_for_function("document.getElementById('importedReplay').textContent.startsWith('PASS')", timeout=5000)
        check(f"{p} import + replay of exported proof", True)
        page.set_input_files("#importFile", str(bad))
        page.wait_for_function("document.getElementById('importedReplay').textContent.startsWith('REJECTED')", timeout=5000)
        check(f"{p} tampered proof rejected on import", "claimed CERTIFIED" in page.inner_text("#importedReplay"),
              page.inner_text("#importedReplay"))
        page.set_input_files("#importFile", str(junk))
        page.wait_for_function("document.getElementById('validation').textContent.includes('Import rejected')", timeout=5000)
        check(f"{p} malformed JSON rejected", True)
        page.set_input_files("#importFile", str(badstrict))
        page.wait_for_function("document.getElementById('validation').textContent.includes('must be true or false')", timeout=5000)
        check(f"{p} non-boolean strict rejected", True)
        if analytic and analytic.exists():
            page.set_input_files("#importFile", str(analytic))
            page.wait_for_function("document.getElementById('preset').options.length > 6", timeout=5000)
            check(f"{p} analytic residual bundle import", page.eval_on_selector("#preset", "e=>e.options.length") > 6,
                  page.eval_on_selector("#preset", "e=>e.options.length"))
        ctx.close()
    browser.close()
    return {"engine": engine, "version": None, "checks": results, "console_errors": errors,
            "external_requests": requests}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--analytic", type=Path)
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"html": str(args.html), "engines": []}
    import hashlib
    report["html_sha256"] = hashlib.sha256(args.html.read_bytes()).hexdigest()
    with sync_playwright() as pw:
        for engine in ("chromium", "firefox"):
            r = run_engine(pw, engine, args.html, args.out, args.analytic)
            b = getattr(pw, engine).launch(headless=True)
            r["version"] = b.version
            b.close()
            report["engines"].append(r)
    failed = [c for e in report["engines"] for c in e["checks"] if not c["pass"]]
    report["total_checks"] = sum(len(e["checks"]) for e in report["engines"])
    report["failed_checks"] = failed
    (args.out / "browser_acceptance.json").write_text(json.dumps(report, indent=1))
    for e in report["engines"]:
        print(e["engine"], e["version"], "checks", sum(c["pass"] for c in e["checks"]), "/", len(e["checks"]),
              "console", len(e["console_errors"]), "external", len(e["external_requests"]))
    for c in failed:
        print("FAIL", c)
    for e in report["engines"]:
        for msg in e["console_errors"]:
            print("CONSOLE", e["engine"], msg)
    sys.exit(1 if failed or any(e["console_errors"] or e["external_requests"] for e in report["engines"]) else 0)


if __name__ == "__main__":
    main()
