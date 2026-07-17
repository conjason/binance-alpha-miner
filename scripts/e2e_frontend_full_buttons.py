"""Full frontend E2E: click all buttons and validate flows.

This drives the actual web UI using Playwright and validates:
- Debug mode toggle (UI + API)
- Start / Stop training (state changes)
- Export strategy download
- Export training download
- Import training (upload exported zip)

Notes:
- We do NOT click the server-side native file picker ("选择数据文件") because it opens
  a tkinter dialog on the server and can hang in automation/headless runs.
  Instead we rely on the persisted `web_settings.json` and `/api/config` data_file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8765"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        page.on("console", lambda m: print(f"[console:{m.type}] {m.text}"))
        page.on("pageerror", lambda e: print(f"[pageerror] {e}"))

        page.goto(URL, wait_until="networkidle")

        # Basic elements exist
        expect(page.locator("#browseBtn")).to_be_visible()
        expect(page.locator("#startBtn")).to_be_visible()
        expect(page.locator("#stopBtn")).to_be_visible()
        expect(page.locator("#exportBtn")).to_be_visible()
        expect(page.locator("#exportTrainingBtn")).to_be_visible()
        expect(page.locator("#importTrainingBtn")).to_be_visible()
        expect(page.locator("#importTrainingFile")).to_have_attribute("type", "file")
        expect(page.locator("#debugModeCheck")).to_be_visible()

        # Wait for app init (selectedSymbol may take a few polls)
        sym = None
        for _ in range(30):
            page.wait_for_timeout(500)
            sym = page.evaluate("window.selectedSymbol")
            if sym:
                break
        if not sym:
            # Fallback: derive from overview without relying on local JS state.
            sym = page.evaluate(
                "() => fetch('/api/overview').then(r=>r.json()).then(j=>j?.progress?.symbol || j?.data_file?.symbol || null)"
            )
        if not sym:
            raise SystemExit("Could not determine selected symbol from UI/overview")
        print("Selected symbol:", sym)

        # Ensure not training before starting checks
        if not page.locator("#stopBtn").is_disabled():
            page.locator("#stopBtn").click()
            page.wait_for_timeout(1500)

        # Debug mode toggle: on -> /api/config debug_mode true
        dbg = page.locator("#debugModeCheck")
        if not dbg.is_checked():
            dbg.check()
        page.wait_for_timeout(1200)
        cfg_dbg = page.evaluate("() => fetch('/api/config').then(r=>r.json()).then(j=>j?.debug_mode)")
        if not cfg_dbg:
            raise SystemExit("Debug mode toggle did not persist to /api/config")
        print("Debug mode: ON")

        # Debug mode toggle: off
        dbg.uncheck()
        page.wait_for_timeout(1200)
        cfg_dbg = page.evaluate("() => fetch('/api/config').then(r=>r.json()).then(j=>j?.debug_mode)")
        if cfg_dbg:
            raise SystemExit("Debug mode OFF did not persist to /api/config")
        print("Debug mode: OFF")

        # Export strategy
        export_strategy = page.locator("#exportBtn")
        expect(export_strategy).to_be_enabled(timeout=30000)
        with page.expect_download(timeout=30000) as dl_info:
            export_strategy.click()
        dl = dl_info.value
        strat_path = ROOT / "tmp_downloads" / dl.suggested_filename
        strat_path.parent.mkdir(exist_ok=True)
        dl.save_as(str(strat_path))
        if strat_path.stat().st_size < 100:
            raise SystemExit("Strategy export download too small")
        print("Strategy exported:", strat_path.name)

        # Export training
        export_training = page.locator("#exportTrainingBtn")
        expect(export_training).to_be_enabled(timeout=30000)
        with page.expect_download(timeout=30000) as dl_info2:
            export_training.click()
        dl2 = dl_info2.value
        zip_path = ROOT / "tmp_downloads" / dl2.suggested_filename
        zip_path.parent.mkdir(exist_ok=True)
        dl2.save_as(str(zip_path))
        if zip_path.stat().st_size < 1024:
            raise SystemExit("Training export zip too small")
        print("Training exported:", zip_path.name, zip_path.stat().st_size, "bytes")

        # Import training (upload the exported zip)
        import_btn = page.locator("#importTrainingBtn")
        expect(import_btn).to_be_enabled(timeout=30000)
        import_btn.click()
        page.locator("#importTrainingFile").set_input_files(str(zip_path))
        page.wait_for_timeout(2500)
        # Export button should remain enabled after import
        expect(export_training).to_be_enabled(timeout=30000)
        print("Training import: OK")

        # Start training (may take a bit to update UI)
        start_btn = page.locator("#startBtn")
        expect(start_btn).to_be_enabled(timeout=30000)
        start_btn.click()
        page.wait_for_timeout(2000)
        # Stop should become enabled when active
        expect(page.locator("#stopBtn")).to_be_enabled(timeout=30000)
        print("Training start: OK")

        # Stop training
        page.locator("#stopBtn").click()
        page.wait_for_timeout(2000)
        # Stop should be disabled again when inactive
        expect(page.locator("#stopBtn")).to_be_disabled(timeout=30000)
        print("Training stop: OK")

        # Strategies table renders something
        body = page.locator("#strategiesBody")
        expect(body).to_be_visible()
        # not strict about row count; just ensure it has some content
        txt = body.inner_text().strip()
        if not txt:
            raise SystemExit("Strategies table is empty in UI")
        print("Strategies table: OK")

        context.close()
        browser.close()

    print("E2E PASS: all buttons (except native file picker) tested OK.")


if __name__ == "__main__":
    main()

