#!/usr/bin/env python3
"""Playwright QA of the offline K4 torque-certification report.

Checks every tab renders without console errors or external requests (Chromium/Firefox, desktop/mobile), the
overview coverage table matches the embedded analysis, the explorer draws certified leaf bands for every
output it is switched to, the payload planning witness is shown, the theme toggle works, and no horizontal
page overflow occurs at phone width.
Usage: <playwright-python> tests/browser/k4_report_browser.py --report <html> --analysis <json> --ext <json> --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

TABS = ["overview", "model", "population", "explorer", "methods", "ext", "retime", "latency", "ledger"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--ext", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    a.out.mkdir(parents=True, exist_ok=True)
    an = json.loads(a.analysis.read_text())
    ex = json.loads(a.ext.read_text())
    n_paths_expected = sum(an["arms"][k]["eligible"] for k in ("primary", "secondary", "stress")) + sum(
        v["arms"]["primary"]["eligible"] for n, v in ex["variants"].items() if n.startswith("cu_"))
    results = []
    with sync_playwright() as pw:
        for eng in ("chromium", "firefox"):
            b = getattr(pw, eng).launch(headless=True)
            for label, vp in (("desktop", {"width": 1400, "height": 1000}), ("mobile", {"width": 390, "height": 844})):
                ctx = b.new_context(viewport=vp)
                pg = ctx.new_page()
                errs, ext = [], []
                pg.on("pageerror", lambda e: errs.append(str(e)))
                pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
                pg.on("request", lambda r: ext.append(r.url) if not r.url.startswith(("file:", "blob:", "data:")) else None)

                def chk(n, ok, d=""):
                    results.append({"check": f"{eng}/{label} {n}", "pass": bool(ok), "detail": str(d)[:300]})
                pg.goto(a.report.resolve().as_uri())
                pg.evaluate("localStorage.clear()")
                pg.goto(a.report.resolve().as_uri())
                pg.wait_for_timeout(500)
                for t in TABS:
                    pg.click(f"#tab-{t}")
                    pg.wait_for_timeout(250)
                    txt = pg.inner_text(f"#sec-{t}")
                    chk(f"tab {t} renders", len(txt) > 50 and not pg.is_hidden(f"#sec-{t}"), len(txt))
                    if label == "mobile":
                        chk(f"tab {t} no horizontal page overflow",
                            pg.evaluate("document.documentElement.scrollWidth<=window.innerWidth+1"),
                            pg.evaluate("document.documentElement.scrollWidth"))
                pg.click("#tab-overview")
                ov = pg.inner_text("#sec-overview")
                p = an["arms"]["primary"]["K3_PLUS_K4"]["coverage"]
                chk("overview shows primary K3+K4 coverage", f'{p["k"]}/{p["n"]}' in ov, ov[:200])
                chk("overview shows BLOCKED_NATIVE_TORQUE", "BLOCKED_NATIVE_TORQUE" in ov)
                chk("overview shows BLOCKED_MODEL_PHYSICAL_VALIDITY", "BLOCKED_MODEL_PHYSICAL_VALIDITY" in ov)
                pg.click("#tab-explorer")
                nopt = pg.eval_on_selector("#sec-explorer select", "e=>e.options.length")
                chk("explorer lists every output", nopt == n_paths_expected, f"{nopt} vs {n_paths_expected}")
                for idx in (0, nopt // 2, nopt - 1):
                    pg.select_option("#sec-explorer select", index=idx)
                    pg.wait_for_timeout(200)
                    rects = pg.eval_on_selector_all("#sec-explorer svg rect", "e=>e.length")
                    chk(f"explorer option {idx} draws certified bands", rects > 0, rects)
                vio = pg.eval_on_selector("#sec-explorer select",
                                          "e=>[...e.options].findIndex(o=>o.text.includes('PLANNING VIOLATED_TORQUE'))")
                chk("payload planning violation is selectable", vio >= 0, vio)
                if vio >= 0:
                    pg.select_option("#sec-explorer select", index=vio)
                    pg.wait_for_timeout(200)
                    chk("planning witness shown", "Verified witness" in pg.inner_text("#sec-explorer"))
                pg.click("#tab-ext")
                et = pg.inner_text("#sec-ext")
                chk("ext tab lists all variants", all(n in et for n in ex["variants"]), [n for n in ex["variants"] if n not in et])
                pg.click("#tab-latency")
                lt = pg.inner_text("#sec-latency")
                chk("latency shows compiled added p50", f'{an["fresh_latency"]["added_ms_K3_plus_K4"]["p50"]:.2f}' in lt)
                before = pg.evaluate("getComputedStyle(document.body).backgroundColor")
                pg.click("#themeBtn")
                pg.wait_for_timeout(100)
                chk("theme toggle changes background", pg.evaluate("getComputedStyle(document.body).backgroundColor") != before)
                pg.screenshot(path=str(a.out / f"{eng}_{label}.png"), full_page=False)
                chk("no console errors", not errs, errs[:3])
                chk("no external requests", not ext, ext[:3])
                ctx.close()
            b.close()
    (a.out / "k4_report_acceptance.json").write_text(json.dumps(results, indent=1))
    failed = [r for r in results if not r["pass"]]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    for r in failed:
        print("FAIL", r)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
