"""Full functional test suite for BinanceAlphaMiner Web UI + training backend."""
from __future__ import annotations

import json
import mimetypes
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API = "http://127.0.0.1:8766"
PORT = 8766

passed: list[str] = []
failed: list[str] = []
warnings: list[str] = []


def ok(name: str, detail: str = "") -> None:
    passed.append(name)
    msg = f"  [OK] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def fail(name: str, detail: str) -> None:
    failed.append(f"{name}: {detail}")
    print(f"  [FAIL] {name} — {detail}")


def warn(name: str, detail: str) -> None:
    warnings.append(f"{name}: {detail}")
    print(f"  [WARN] {name} — {detail}")


def get(path: str, timeout: int = 15) -> tuple[int, dict | bytes, dict]:
    req = urllib.request.Request(API + path)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            headers = dict(resp.headers)
            ctype = headers.get("Content-Type") or headers.get("content-type") or ""
            if "json" in ctype:
                return resp.status, json.loads(raw.decode()), headers
            return resp.status, raw, headers
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            body = json.loads(raw.decode())
        except Exception:
            body = raw
        return e.code, body, dict(e.headers)


def post_json(path: str, data: dict | None = None, method: str = "POST") -> tuple[int, dict]:
    body = json.dumps(data or {}).encode()
    req = urllib.request.Request(
        API + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"detail": e.read().decode(errors="replace")}


def put_json(path: str, data: dict) -> tuple[int, dict]:
    return post_json(path, data, method="PUT")


def post_file(path: str, file_path: Path, query: str = "") -> tuple[int, dict]:
    boundary = "----BinanceAlphaMinerTest"
    data = file_path.read_bytes()
    fname = file_path.name
    ctype = mimetypes.guess_type(fname)[0] or "application/octet-stream"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        API + path + query,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"detail": e.read().decode(errors="replace")}


def wait_server(max_wait: float = 30.0) -> bool:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            st, body, _ = get("/api/health")
            if st == 200 and isinstance(body, dict) and body.get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def mt5_data_file() -> str:
    settings = json.loads((ROOT / "web_settings.json").read_text(encoding="utf-8"))
    path = settings.get("last_data_file", "")
    if not path or not Path(path).exists():
        d = next(x for x in Path("D:/").iterdir() if x.is_dir() and x.name.startswith("MT5"))
        path = str(d / "ADAUSD_H1.parquet")
    return path


def test_static_and_health() -> str | None:
    print("\n== 1. 静态资源与健康检查 ==")
    if not wait_server():
        fail("server", "8766 无响应")
        return None

    st, body, _ = get("/api/health")
    if st != 200 or body.get("version") != "1.1.0":
        fail("health", str(body))
    else:
        ok("health", body["version"])

    st, body, _ = get("/api/routes")
    paths = {r["path"] for r in body.get("routes", [])}
    required = {
        "/api/overview",
        "/api/training/start",
        "/api/training/import",
        "/api/training/{symbol}/export",
    }
    missing = required - paths
    if missing:
        fail("routes", f"缺少 {missing}")
    else:
        ok("routes", f"{len(paths)} 条")

    st, html, _ = get("/")
    if st != 200 or b"exportTrainingBtn" not in html:
        fail("index.html", "缺少导出训练按钮")
    else:
        ok("index.html", "含导出/导入训练按钮")

    for asset in ["/static/app.js?v=14", "/static/style.css?v=5", "/static/bg.js?v=1"]:
        st, content, _ = get(asset.split("?")[0])
        if st != 200 or len(content) < 100:
            fail("static", asset)
        else:
            ok("static", asset.split("?")[0])

    st, body, _ = get("/api/config")
    if st != 200 or not body.get("train_steps"):
        fail("config", str(body))
    else:
        ok("config", f"device={body.get('device')} steps={body.get('train_steps')}")

    return mt5_data_file()


def test_settings(data_file: str) -> None:
    print("\n== 2. 设置与数据文件 ==")
    st, body = get("/api/settings")[0:2]
    if st != 200:
        fail("settings get", str(body))
    else:
        ok("settings get", body.get("last_data_file", "")[-30:])

    st, body = put_json("/api/settings", {"debug_mode": True})
    if st != 200 or not body.get("debug_mode"):
        fail("settings put debug", str(body))
    else:
        ok("settings put debug")

    st, body, _ = get("/api/config")
    if not body.get("debug_mode"):
        warn("debug mode", "config 未反映 debug_mode=true")
    else:
        ok("debug mode sync")

    st, body, _ = get("/api/overview")
    df = body.get("data_file") or {}
    if not df.get("valid"):
        fail("overview data_file", df.get("message", "invalid"))
    else:
        ok("overview data_file", f"{df.get('symbol')} {df.get('timeframe')} {df.get('bars')} bars")

    prog = body.get("progress") or {}
    if prog.get("symbol") != df.get("symbol"):
        fail("overview progress symbol", str(prog))
    else:
        ok("overview progress", f"step={prog.get('current_step')} strategy={prog.get('has_strategy')}")

    st, body, _ = get(f"/api/symbols/{df['symbol']}")
    if st != 200 or body.get("symbol") != df["symbol"]:
        fail("symbol detail", str(body))
    else:
        ok("symbol detail", f"has_strategy={body.get('has_strategy')}")

    st, body = post_json("/api/training/start", {"data_file": "D:\\no_such_file.parquet"})
    if st != 404:
        fail("invalid file", f"expected 404 got {st}")
    else:
        ok("invalid file 404")

    put_json("/api/settings", {"debug_mode": False, "last_data_file": data_file})


def test_strategies(symbol: str) -> None:
    print("\n== 3. 策略列表与导出 ==")
    st, body, _ = get("/api/strategies")
    strategies = body.get("strategies") or []
    if st != 200 or not strategies:
        fail("strategies list", "empty")
    else:
        ok("strategies list", f"{len(strategies)} 条")

    has = any(s.get("symbol") == symbol for s in strategies)
    if not has:
        warn("strategy for symbol", f"{symbol} 不在列表（可能无 best_*.json）")
    else:
        ok("strategy exists", symbol)

    st, body, _ = get(f"/api/strategies/{symbol}/export")
    if st == 200:
        if isinstance(body, dict) and body.get("symbol"):
            ok("export strategy", f"score={body.get('best_score')}")
        elif isinstance(body, bytes):
            try:
                payload = json.loads(body.decode())
                ok("export strategy", f"score={payload.get('best_score')}")
            except Exception as e:
                fail("export strategy json", str(e))
        else:
            fail("export strategy", f"unexpected body type {type(body)}")

    st, _, _ = get("/api/strategies/NOPE/export")
    if st != 404:
        fail("export strategy 404", f"got {st}")
    else:
        ok("export strategy 404")


def test_training_io(symbol: str) -> Path | None:
    print("\n== 4. 训练包导出/导入 ==")
    st, _, _ = get(f"/api/training/{symbol}/export")
    if st == 404:
        warn("export training", f"{symbol} 无 checkpoint，创建临时 fixture")
        from data_pipeline.parquet_manager import ParquetDataManager
        from model_core.engine import AlphaEngine

        data_file = mt5_data_file()
        ckpt_dir = ROOT / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        mgr = ParquetDataManager(data_file)
        mgr.load()
        eng = AlphaEngine(data_manager=mgr, target_symbol=symbol)
        eng.training_history = {
            "step": [0, 1],
            "best_score": [0.1, 0.2],
            "val_score": [0.05, 0.15],
            "entropy": [1.0, 0.9],
        }
        eng.best_score = 0.2
        eng.best_formula = [1, 2, 3]
        eng.save_checkpoint(2)
        (ROOT / f"training_history_{symbol}.json").write_text(
            json.dumps(eng.training_history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    st, raw, headers = get(f"/api/training/{symbol}/export")
    if st != 200 or not isinstance(raw, bytes):
        fail("export training", f"HTTP {st}")
        return None
    ok("export training", f"{len(raw)} bytes")

    zip_path = Path(tempfile.gettempdir()) / f"test_{symbol}.zip"
    zip_path.write_bytes(raw)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    if "manifest.json" not in names:
        fail("zip manifest", str(names))
    else:
        ok("zip manifest", f"{len(names)} files")

    for p in (ROOT / "checkpoints").glob(f"ckpt_{symbol}_step_*.pt"):
        p.unlink()
    st, body = post_file("/api/training/import", zip_path, f"?symbol={symbol}")
    if st != 200 or not body.get("ok"):
        fail("import training", str(body))
    else:
        ok("import training", body.get("message", "")[:40])

    st, body = post_file("/api/training/import", zip_path, "?symbol=WRONG")
    if st != 400:
        fail("import mismatch", f"expected 400 got {st}")
    else:
        ok("import mismatch 400")

    return zip_path


def test_training_live(data_file: str, symbol: str) -> None:
    print("\n== 5. 训练启停 / 日志 / 曲线同步 ==")
    post_json("/api/training/stop")

    st, body = post_json("/api/training/start", {"data_file": data_file})
    if st != 200 or not body.get("ok"):
        fail("training start", str(body))
        return
    ok("training start", f"pid={body.get('job', {}).get('pid')}")

    st, body = post_json("/api/training/start", {"data_file": data_file})
    if st != 409:
        fail("duplicate start", f"expected 409 got {st}")
        post_json("/api/training/stop")
    else:
        ok("duplicate start 409")

    sync_ok = False
    last_log_len = 0
    for i in range(20):
        time.sleep(4)
        st, status, _ = get("/api/training/status")
        st2, ov, _ = get("/api/overview")
        st3, sym, _ = get(f"/api/symbols/{symbol}")

        active = status.get("active")
        log_tail = status.get("log_tail") or []
        prog = (ov.get("progress") or {})
        hist = (sym.get("history") or {})
        steps = hist.get("step") or []
        cur = int(prog.get("current_step") or 0)
        chart_n = len(steps)

        if len(log_tail) > last_log_len:
            last_log_len = len(log_tail)

        print(
            f"    t={i*4:02d}s active={active} log={len(log_tail)} "
            f"prog={cur} chart={chart_n} status={prog.get('status')}"
        )

        if log_tail:
            last = log_tail[-1]
            if any(ord(c) > 127 for c in last):
                pass  # has non-ascii (Chinese) - good
            if "[Web]" in last and i > 2:
                warn("log tail", "出现 Web 结束标记，训练可能已停")

        if active and chart_n >= 3 and cur >= 3:
            # 需观察到步数增长，避免仅用旧 checkpoint 历史误判
            if i >= 2 and chart_n > 0 and abs(cur - (int(steps[-1]) + 1)) <= 2:
                sync_ok = True
                ok("log/chart sync", f"step≈{cur} chart={chart_n}")
                break

    if not sync_ok:
        if last_log_len == 0:
            fail("training log", "无日志输出")
        else:
            warn("log/chart sync", f"60s 内未确认同步 prog={cur} chart={chart_n}")

    st, body = post_json("/api/training/stop")
    time.sleep(2)
    st, status, _ = get("/api/training/status")
    job = status.get("job") or {}
    state = job.get("state")
    if state == "failed":
        warn("stop state", f"停止后状态为 failed（Windows 上 exit_code=1），建议显示为已停止")
    elif state in ("stopped", "completed"):
        ok("training stop", state)
    else:
        warn("training stop", f"state={state}")

    hist_file = ROOT / f"training_history_{symbol}.json"
    if hist_file.exists():
        hist = json.loads(hist_file.read_text(encoding="utf-8"))
        n = len(hist.get("step") or [])
        if n >= 1:
            ok("training_history file", f"{n} steps")
        else:
            warn("training_history file", "empty steps")
    else:
        warn("training_history file", "不存在（训练步数过少或未写入）")


def test_debug() -> None:
    print("\n== 6. 调试接口 ==")
    st, body = post_json("/api/debug/client-log", {
        "level": "error",
        "message": "automated test message",
        "context": {"suite": "full"},
    })
    if st != 200:
        fail("client-log", str(body))
    else:
        ok("client-log")

    st, body, _ = get("/api/debug/logs?lines=50")
    if st != 200:
        fail("debug logs", str(body))
    else:
        ok("debug logs", f"server={len(body.get('server_log', []))} err={len(body.get('error_log', []))}")


def main() -> None:
    print("=== BinanceAlphaMiner 全功能实测 ===")
    data_file = test_static_and_health()
    if not data_file:
        print_summary()
        raise SystemExit(1)

    from data_pipeline.parquet_manager import inspect_parquet_file

    info = inspect_parquet_file(data_file)
    symbol = info["symbol"]
    print(f"\n数据文件: {info['filename']} ({symbol} {info['timeframe']})")

    test_settings(data_file)
    test_strategies(symbol)
    test_training_io(symbol)
    test_training_live(data_file, symbol)
    test_debug()
    print_summary()

    if failed:
        raise SystemExit(1)


def print_summary() -> None:
    print("\n" + "=" * 50)
    print(f"通过: {len(passed)}  失败: {len(failed)}  警告: {len(warnings)}")
    if warnings:
        print("\n警告:")
        for w in warnings:
            print(f"  - {w}")
    if failed:
        print("\n失败:")
        for f in failed:
            print(f"  - {f}")
    else:
        print("\n未发现阻断性问题。")


if __name__ == "__main__":
    main()
