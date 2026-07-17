"""E2E: import a given training zip from Downloads via real frontend UI."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

URL = "http://127.0.0.1:8765"
API = "http://127.0.0.1:8765"


def overview_step() -> int | None:
    ov = json.loads(urllib.request.urlopen(API + "/api/overview").read())
    prog = ov.get("progress") or {}
    try:
        return int(prog.get("current_step")) if prog.get("current_step") is not None else None
    except Exception:
        return None


def main() -> None:
    downloads = Path(r"C:\Users\Administrator\Downloads")
    zip60 = downloads / "training_ADAUSD_step0060.zip"
    if not zip60.exists():
        raise SystemExit(f"Missing file: {zip60}")

    before = overview_step()
    print("BEFORE overview current_step:", before)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(2000)

        # Ensure stopped
        stop_btn = page.locator("#stopBtn")
        if not stop_btn.is_disabled():
            stop_btn.click()
            page.wait_for_timeout(1500)

        import_btn = page.locator("#importTrainingBtn")
        expect(import_btn).to_be_enabled(timeout=30000)
        import_btn.click()

        page.locator("#importTrainingFile").set_input_files(str(zip60))

        # Allow backend to process + polling refresh
        page.wait_for_timeout(4000)

        ctx.close()
        browser.close()

    # Poll API for step change
    after = None
    for _ in range(10):
        time.sleep(1)
        after = overview_step()
        if after is not None:
            break
    print("AFTER overview current_step:", after)

    if after != 60:
        raise SystemExit(f"Import expected step=60, got {after}")

    print("PASS: frontend import updated progress to 60.")


if __name__ == "__main__":
    main()

