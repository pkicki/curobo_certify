#!/usr/bin/env python3
"""Playwright QA of the offline margin dashboard (empty state, import, filter, CSV, rejection).
Usage: <playwright-python> tests/browser/margin_dashboard_browser.py --empty <html> --embedded <html> --analysis <json> --out <dir>
"""
import argparse, json, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--empty", type=Path, required=True)
    ap.add_argument("--embedded", type=Path, required=True)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    a.out.mkdir(parents=True, exist_ok=True)
    data = json.loads(a.analysis.read_text())
    n_rows, results = len(data["rows"]), []
    bad = a.out / "bad.json"; bad.write_text(json.dumps({"schema_version": 1, "scope": "OTHER"}))
    with sync_playwright() as pw:
        for eng in ("chromium", "firefox"):
            b = getattr(pw, eng).launch(headless=True)
            for label, vp in (("desktop", {"width": 1400, "height": 1000}), ("mobile", {"width": 390, "height": 844})):
                ctx = b.new_context(viewport=vp, accept_downloads=True); pg = ctx.new_page(); errs = []; ext = []
                pg.on("pageerror", lambda e: errs.append(str(e)))
                pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
                pg.on("request", lambda r: ext.append(r.url) if not r.url.startswith(("file:", "blob:", "data:")) else None)
                chk = lambda n, ok, d="": results.append({"check": f"{eng}/{label} {n}", "pass": bool(ok), "detail": str(d)[:200]})
                pg.goto(a.empty.resolve().as_uri())
                chk("empty state says NOT RUN", "NOT RUN" in pg.inner_text("#status"), pg.inner_text("#status"))
                pg.set_input_files("#upload", str(bad)); pg.wait_for_timeout(400)
                chk("incompatible import rejected", "REJECTED" in pg.inner_text("#status"), pg.inner_text("#status"))
                pg.set_input_files("#upload", str(a.analysis)); pg.wait_for_timeout(800)
                chk("import shows all source queries", pg.inner_text("#queries").strip() == str(n_rows), pg.inner_text("#queries"))
                pg.goto(a.embedded.resolve().as_uri()); pg.wait_for_timeout(800)
                chk("embedded data loads", pg.inner_text("#queries").strip() == str(n_rows), pg.inner_text("#queries"))
                nopt = pg.eval_on_selector("#scene", "e=>e.options.length")
                chk("scene filter populated from embedded data", nopt == 1 + len({r["scene"] for r in data["rows"]}), nopt)
                pg.select_option("#scene", index=1); pg.wait_for_timeout(300)
                chk("scene filter narrows rows", int(pg.inner_text("#queries")) < n_rows, pg.inner_text("#queries"))
                pg.select_option("#scene", "all")
                dots = pg.eval_on_selector_all("#scatter circle", "e=>e.length")
                chk("scatter dots = native successes", dots == sum(r["native_duration_s"] is not None for r in data["rows"]), dots)
                with pg.expect_download() as dl:
                    pg.click("#csv")
                lines = Path(dl.value.path()).read_text().strip().splitlines()
                chk("CSV has all filtered rows", len(lines) == n_rows + 1, len(lines))
                chk("no horizontal overflow", pg.evaluate("document.documentElement.scrollWidth<=window.innerWidth+1"))
                pg.screenshot(path=str(a.out / f"{eng}_{label}.png"), full_page=True)
                chk("no console errors", not errs, errs[:3]); chk("no external requests", not ext, ext[:3])
                ctx.close()
            b.close()
    (a.out / "margin_dashboard_acceptance.json").write_text(json.dumps(results, indent=1))
    failed = [r for r in results if not r["pass"]]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed"); [print("FAIL", r) for r in failed]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
