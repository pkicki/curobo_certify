import sys
from playwright.sync_api import sync_playwright
html, out = sys.argv[1], sys.argv[2]
with sync_playwright() as pw:
    for eng in ("chromium", "firefox"):
        b = getattr(pw, eng).launch(headless=True)
        pg = b.new_page(viewport={"width": 1400, "height": 1000})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto("file://" + html)
        for tab in ["overview", "margins", "timelaw", "explorer", "population", "adaptive", "latency", "native", "correctness", "provenance"]:
            pg.click(f"nav button[data-tab={tab}]")
            pg.wait_for_timeout(400 if tab != "explorer" else 2500)
            pg.screenshot(path=f"{out}/{eng}_{tab}.png", full_page=True)
        pg.click("nav button[data-tab=population]"); pg.click("tr.click >> nth=60"); pg.wait_for_timeout(4000)
        pg.screenshot(path=f"{out}/{eng}_explorer_jump.png", full_page=True)
        print(eng, b.version, "errors:", errs[:5], "status:", pg.inner_text("#xbr")[:160].replace("\n", " "))
        b.close()
