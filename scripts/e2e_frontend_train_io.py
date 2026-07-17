"""Real frontend E2E: click Export/Import training in browser.

This drives the actual web UI (buttons + file chooser) using Playwright.
It verifies:
- Export training triggers a .zip download
- Import training accepts that zip and returns to enabled state

Run:
  python scripts/e2e_frontend_train_io.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8765"


def main() -> None:
    # Ensure there is a valid selected file (so buttons can enable)
    # Use web_settings.json (already set to MT5_K线数据 ADAUSD_H1).
    settings_path = ROOT / "web_settings.json"
    if settings_path.exists():
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        print("web_settings.json last_data_file:", data.get("last_data_file"))

    downloads_dir = ROOT / "tmp_downloads"
    downloads_dir.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        page.on("console", lambda m: print(f"[console:{m.type}] {m.text}"))
        page.on("pageerror", lambda e: print(f"[pageerror] {e}"))

        page.goto(URL, wait_until="networkidle")

        # Wait for app.js to initialize state (loadConfig + refreshOverview)
        for _ in range(20):
            page.wait_for_timeout(500)
            try:
                # Force a refresh tick in case timers are throttled.
                page.evaluate("window.refreshOverview && window.refreshOverview()")
            except Exception:
                pass
            sym = page.evaluate("window.selectedSymbol")
            df = page.evaluate("window.selectedDataFile")
            if sym and df:
                break

        export_btn = page.locator("#exportTrainingBtn")
        import_btn = page.locator("#importTrainingBtn")
        file_input = page.locator("#importTrainingFile")

        print("frontend selectedSymbol:", page.evaluate("window.selectedSymbol"))
        print("frontend selectedDataFile:", page.evaluate("window.selectedDataFile"))
        # Also check backend says checkpoint exists (helps debug why disabled).
        backend_state = page.evaluate(
            """() => fetch('/api/overview').then(r=>r.json()).then(j=>({
                symbol: j?.progress?.symbol,
                has_checkpoint: j?.progress?.has_checkpoint,
                active: j?.training?.active,
                valid: j?.data_file?.valid
            })).catch(e=>({error:String(e)}))"""
        )
        print("backend overview:", backend_state)

        # Make sure we're not in training
        stop_btn = page.locator("#stopBtn")
        if not stop_btn.is_disabled():
            stop_btn.click()
            page.wait_for_timeout(1200)

        # Export training: should be enabled when checkpoint exists and not active
        expect(export_btn).to_be_enabled(timeout=30000)

        with page.expect_download(timeout=30000) as dl_info:
            export_btn.click()
        download = dl_info.value
        suggested = download.suggested_filename
        out_path = downloads_dir / suggested
        download.save_as(str(out_path))

        if not out_path.exists() or out_path.stat().st_size < 1024:
            raise SystemExit(f"export download failed: {out_path}")
        print("exported:", out_path.name, out_path.stat().st_size, "bytes")

        # Import training: enabled when symbol selected and not active.
        expect(import_btn).to_be_enabled(timeout=15000)
        import_btn.click()
        file_input.set_input_files(str(out_path))

        # Import triggers refreshOverview; wait for a client log info line or buttons state stable
        page.wait_for_timeout(2500)

        # After import, export should still be enabled.
        expect(export_btn).to_be_enabled(timeout=15000)

        # Sanity: chart canvas exists
        expect(page.locator("#mainChart")).to_be_visible()

        context.close()
        browser.close()

    print("E2E PASS: export/import via frontend works.")


if __name__ == "__main__":
    main()

