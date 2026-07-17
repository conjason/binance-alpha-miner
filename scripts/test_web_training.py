"""Quick E2E check: training log + chart stay in sync."""
from __future__ import annotations

import json
import time
import urllib.request

BASE = "http://127.0.0.1:8765"
DATA_FILE = r"D:\K线数据\ADAUSD_H1.parquet"


def post(path: str, data: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        json.dumps(data).encode(),
        {"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=10) as resp:
        return json.loads(resp.read())


def main() -> None:
    post("/api/training/start", {"data_file": DATA_FILE})
    for i in range(12):
        time.sleep(4)
        st = get("/api/training/status")
        ov = get("/api/overview")
        sym = get("/api/symbols/ADAUSD")
        log_tail = st.get("log_tail") or []
        prog = ov.get("progress") or {}
        steps = (sym.get("history") or {}).get("step") or []
        cur = int(prog.get("current_step") or 0)
        print(
            f"t={i * 4:02d}s active={st.get('active')} "
            f"log_lines={len(log_tail)} prog={cur} chart={len(steps)} "
            f"last={steps[-1] if steps else None}"
        )
        if log_tail:
            print("  log:", log_tail[-1][:100])
        if cur >= 22 and len(steps) >= 22 and abs(cur - steps[-1]) <= 1 and st.get("active"):
            print("SYNC OK")
            return
    raise SystemExit("sync check failed")


if __name__ == "__main__":
    main()
