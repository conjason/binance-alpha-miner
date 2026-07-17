from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path(__file__).resolve().parents[1] / "docs" / "images"
out.mkdir(parents=True, exist_ok=True)
url = "http://127.0.0.1:8765/"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.25)
    page.goto(url, wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(2000)
    page.screenshot(path=str(out / "00_hero.png"), full_page=False)
    page.screenshot(path=str(out / "01_train.png"), full_page=True)

    page.locator('button[data-page="backtest"]').click()
    page.wait_for_timeout(1000)
    page.screenshot(path=str(out / "02_backtest.png"), full_page=True)

    page.locator('button[data-page="realtime"]').click()
    page.wait_for_timeout(1500)
    page.screenshot(path=str(out / "03_realtime.png"), full_page=True)
    browser.close()

print("wrote", sorted(p.name for p in out.glob("*.png")))
