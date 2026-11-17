#!/usr/bin/env python3
"""Playwright QA of the offline formal-qualification report.
Usage: <playwright-python> tests/browser/formal_report_browser.py --html <report.html> --out <dir>
Checks, in Chromium and Firefox at desktop 1400 px and mobile 390 px: every tab renders, no page or
console errors, no external requests, no horizontal page overflow, ledger filter/CSV export, path
explorer opens from a T1 point, dark theme toggle; saves screenshots of each tab.
"""
import argparse
import json
from pathlib import Path

TABS = ["overview", "ledger", "coverage", "durations", "t1", "latency", "explorer", "failures", "review", "prov"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    a.out.mkdir(parents=True, exist_ok=True)
    res = []
    with sync_playwright() as pw:
        for eng in ("chromium", "firefox"):
            b = getattr(pw, eng).launch(headless=True)
            for label, vp in (("desktop", {"width": 1400, "height": 1000}), ("mobile", {"width": 390, "height": 844})):
                ctx = b.new_context(viewport=vp, accept_downloads=True)
                pg = ctx.new_page()
                errs, ext = [], []
                pg.on("pageerror", lambda e: errs.append(str(e)))
                pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
                pg.on("request", lambda r: ext.append(r.url) if not r.url.startswith(("file:", "blob:", "data:")) else None)
                chk = lambda n, ok, d="": res.append({"check": f"{eng}/{label} {n}", "pass": bool(ok), "detail": str(d)[:200]})  # noqa: E731
                pg.goto(a.html.resolve().as_uri())
                pg.evaluate("localStorage.clear()")
                pg.reload()
                pg.wait_for_timeout(500)
                for t in TABS:
                    pg.click(f"#tab-{t}")
                    pg.wait_for_timeout(250)
                    txt = pg.inner_text(f"#sec-{t}")
                    chk(f"tab {t} renders", len(txt) > 40, len(txt))
                    ov = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
                    chk(f"tab {t} no horizontal overflow", ov <= 1, ov)
                    pg.screenshot(path=str(a.out / f"{eng}_{label}_{t}.png"), full_page=False)
                pg.click("#tab-ledger")
                n0 = pg.inner_text("#sec-ledger .filters .note")
                pg.select_option("#sec-ledger select >> nth=2", "native failure")
                pg.wait_for_timeout(200)
                n1 = pg.inner_text("#sec-ledger .filters .note")
                chk("ledger filter narrows rows", n0 != n1, f"{n0} -> {n1}")
                with pg.expect_download() as dl:
                    pg.click("text=Export CSV")
                chk("CSV export", dl.value.suggested_filename == "formal_ledger.csv")
                pg.click("#tab-t1")
                pg.click("#sec-t1 svg circle >> nth=0")
                pg.wait_for_timeout(300)
                chk("T1 point opens explorer", not pg.is_hidden("#sec-explorer") and "NPZ SHA-256" in pg.inner_text("#sec-explorer"))
                pg.click("#themeBtn")
                pg.wait_for_timeout(100)
                chk("theme toggle", pg.evaluate("document.documentElement.dataset.theme") in ("dark", "light"))
                pg.screenshot(path=str(a.out / f"{eng}_{label}_explorer_toggled.png"))
                chk("no console/page errors", not errs, errs[:3])
                chk("no external requests", not ext, ext[:3])
                ctx.close()
            b.close()
    ok = sum(r["pass"] for r in res)
    (a.out / "results.json").write_text(json.dumps({"passed": ok, "total": len(res), "results": res}, indent=1))
    print(f"{ok}/{len(res)} checks passed")
    for r in res:
        if not r["pass"]:
            print("FAIL", r)


if __name__ == "__main__":
    main()
