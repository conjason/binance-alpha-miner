"""Capture BinanceAlphaMiner UI screenshots for the user tutorial."""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "tutorial_images"
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://127.0.0.1:8765/"


def shot(page, name: str, *, full: bool = False) -> None:
    path = OUT / name
    page.screenshot(path=str(path), full_page=full)
    print("wrote", path.name)


def clip(page, selector: str, name: str, padding: int = 8) -> None:
    loc = page.locator(selector).first
    loc.wait_for(state="visible", timeout=15000)
    box = loc.bounding_box()
    if not box:
        shot(page, name)
        return
    page.screenshot(
        path=str(OUT / name),
        clip={
            "x": max(0, box["x"] - padding),
            "y": max(0, box["y"] - padding),
            "width": box["width"] + 2 * padding,
            "height": box["height"] + 2 * padding,
        },
    )
    print("wrote", name)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1.25,
        )
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(4000)

        # Overview
        shot(page, "01_home_top.png", full=False)
        clip(page, "nav.stepper", "02_stepper.png")
        clip(page, "#page-train .launch-panel", "03_train_launch.png")
        clip(page, "#page-train .chart-panel", "04_train_chart.png")
        clip(page, "#page-train .log-panel", "05_train_log.png")
        shot(page, "06_train_full.png", full=True)

        # Scroll to strategies / AI if present
        for sel, name in [
            ("#page-train .strategies-panel", "07_strategies.png"),
            ("#aiPanelSection", "08_ai_panel.png"),
        ]:
            try:
                loc = page.locator(sel).first
                if loc.count():
                    loc.scroll_into_view_if_needed()
                    page.wait_for_timeout(500)
                    clip(page, sel, name)
            except Exception as exc:
                print("skip", name, exc)

        # Backtest
        page.locator('button[data-page="backtest"]').click()
        page.wait_for_timeout(1500)
        shot(page, "09_backtest_top.png", full=False)
        clip(page, "#page-backtest .launch-panel", "10_backtest_launch.png")
        clip(page, "#page-backtest .bt-summary-panel", "11_backtest_summary.png")
        shot(page, "12_backtest_full.png", full=True)

        # Realtime — first panel of realtime page
        page.locator('button[data-page="realtime"]').click()
        page.wait_for_timeout(1800)
        shot(page, "13_realtime_top.png", full=False)
        try:
            clip(page, "#page-realtime > section.panel", "14_realtime_form.png")
        except Exception as exc:
            print("skip realtime form", exc)
        shot(page, "15_realtime_full.png", full=True)

        browser.close()

    print("done ->", OUT)


if __name__ == "__main__":
    main()
