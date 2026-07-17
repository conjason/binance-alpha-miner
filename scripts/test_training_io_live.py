"""Live test: training export/import + strategy export APIs."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API = "http://127.0.0.1:8766"
SYMBOL = "TESTIO"


def header_get(headers: dict, name: str) -> str:
    for k, v in headers.items():
        if k.lower() == name.lower():
            return v or ""
    return ""


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    raise SystemExit(1)


def http_get(path: str, expect_status: int = 200) -> tuple[int, bytes, dict]:
    req = urllib.request.Request(API + path)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
            headers = dict(resp.headers)
            return resp.status, body, headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def http_post_multipart(path: str, file_path: Path, query: str = "") -> tuple[int, dict]:
    import mimetypes

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    data = file_path.read_bytes()
    fname = file_path.name
    ctype = mimetypes.guess_type(fname)[0] or "application/octet-stream"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()

    url = API + path + query
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode())
        except Exception:
            payload = {"detail": e.read().decode(errors="replace")}
        return e.code, payload


def find_parquet() -> Path:
    for base in Path("D:/").iterdir():
        if not base.is_dir():
            continue
        for p in base.glob("*.parquet"):
            if p.stat().st_size > 1000:
                return p
        for p in base.glob("*/*.parquet"):
            if p.stat().st_size > 1000:
                return p
    raise FileNotFoundError("no parquet found")


def setup_fixture() -> Path:
    from data_pipeline.parquet_manager import ParquetDataManager, inspect_parquet_file
    from model_core.engine import AlphaEngine

    src = find_parquet()
    info = inspect_parquet_file(str(src))
    symbol = SYMBOL

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    for old in ckpt_dir.glob(f"ckpt_{symbol}_step_*.pt"):
        old.unlink()

    hist = ROOT / f"training_history_{symbol}.json"
    if hist.exists():
        hist.unlink()

    strat = ROOT / "strategies" / f"best_{symbol}.json"
    if strat.exists():
        strat.unlink()

    mgr = ParquetDataManager(str(src))
    mgr.load()
    engine = AlphaEngine(data_manager=mgr, target_symbol=symbol)
    engine.training_history = {
        "step": [0, 1],
        "best_score": [0.1, 0.2],
        "val_score": [0.05, 0.15],
        "entropy": [1.0, 0.9],
    }
    engine.best_score = 0.2
    engine.best_formula = [1, 2, 3]
    ckpt_path = engine.save_checkpoint(2)
    ok(f"created checkpoint {Path(ckpt_path).name} from {src.name}")

    hist.write_text(
        json.dumps(engine.training_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    strat.parent.mkdir(exist_ok=True)
    strat.write_text(
        json.dumps(
            {
                "vocab_version": "v9217a2c0d91a",
                "symbol": symbol,
                "formula": [1, 2, 3],
                "best_score": 0.2,
                "timeframe": info["timeframe"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    ok(f"fixture ready for symbol {symbol}")
    return src


def test_python_package(symbol: str) -> Path:
    from web.training_package import build_training_export_zip, import_training_package

    print("\n== Python package: export ==")
    data, name = build_training_export_zip(symbol)
    if not data or not name.endswith(".zip"):
        fail(f"bad zip output: {name!r} len={len(data)}")
    ok(f"export zip {name} ({len(data)} bytes)")

    zip_path = Path(tempfile.gettempdir()) / name
    zip_path.write_bytes(data)

    print("\n== Python package: import roundtrip ==")
    ckpt_dir = ROOT / "checkpoints"
    for old in ckpt_dir.glob(f"ckpt_{symbol}_step_*.pt"):
        old.unlink()
    (ROOT / f"training_history_{symbol}.json").unlink(missing_ok=True)
    (ROOT / "strategies" / f"best_{symbol}.json").unlink(missing_ok=True)

    result = import_training_package(data, name, expected_symbol=symbol)
    if not result.get("ok"):
        fail(f"import failed: {result}")
    ok(result["message"])

    if not list(ckpt_dir.glob(f"ckpt_{symbol}_step_*.pt")):
        fail("checkpoint not restored after import")
    ok("checkpoint file on disk")

    if not (ROOT / f"training_history_{symbol}.json").exists():
        fail("training history not restored")
    ok("training history on disk")

    if not (ROOT / "strategies" / f"best_{symbol}.json").exists():
        fail("strategy not restored")
    ok("strategy file on disk")

    print("\n== Python package: symbol mismatch ==")
    try:
        import_training_package(data, name, expected_symbol="WRONG")
        fail("expected ValueError on symbol mismatch")
    except ValueError:
        ok("rejects wrong expected_symbol")

    return zip_path


def test_http(symbol: str, zip_path: Path) -> None:
    print("\n== HTTP: health ==")
    for _ in range(20):
        try:
            status, body, _ = http_get("/api/health")
            if status == 200:
                ok(f"health {body.decode()[:80]}")
                break
        except Exception:
            time.sleep(0.5)
    else:
        fail("web server not responding on :8766")

    print("\n== HTTP: strategy export ==")
    status, body, headers = http_get(f"/api/strategies/{symbol}/export")
    if status != 200:
        fail(f"strategy export HTTP {status}: {body[:200]}")
    if "application/json" not in header_get(headers, "Content-Type"):
        fail(f"wrong content-type: {header_get(headers, 'Content-Type')!r}")
    payload = json.loads(body.decode())
    if payload.get("symbol") != symbol:
        fail(f"strategy export wrong symbol: {payload}")
    ok(f"strategy export JSON ({len(body)} bytes)")

    print("\n== HTTP: training export ==")
    status, body, headers = http_get(f"/api/training/{symbol}/export")
    if status != 200:
        fail(f"training export HTTP {status}: {body[:200]}")
    if "zip" not in header_get(headers, "Content-Type"):
        fail(f"wrong content-type: {header_get(headers, 'Content-Type')!r}")
    ok(f"training export zip ({len(body)} bytes)")

    print("\n== HTTP: training export 404 ==")
    status, body, _ = http_get("/api/training/NOSYMBOL/export", expect_status=404)
    if status != 404:
        fail(f"expected 404, got {status}")
    ok("missing checkpoint returns 404")

    print("\n== HTTP: training import ==")
    for old in (ROOT / "checkpoints").glob(f"ckpt_{symbol}_step_*.pt"):
        old.unlink()
    status, payload = http_post_multipart(
        "/api/training/import",
        zip_path,
        query=f"?symbol={symbol}",
    )
    if status != 200:
        fail(f"import HTTP {status}: {payload}")
    if payload.get("symbol") != symbol:
        fail(f"import wrong symbol: {payload}")
    ok(payload.get("message", "import ok"))

    print("\n== HTTP: import symbol mismatch ==")
    status, payload = http_post_multipart(
        "/api/training/import",
        zip_path,
        query="?symbol=WRONGSYM",
    )
    if status != 400:
        fail(f"expected 400 on mismatch, got {status}: {payload}")
    ok("rejects symbol mismatch via API")

    print("\n== HTTP: import single .pt ==")
    pt_files = list((ROOT / "checkpoints").glob(f"ckpt_{symbol}_step_*.pt"))
    if not pt_files:
        fail("no pt file for single-file import test")
    pt_copy = Path(tempfile.gettempdir()) / pt_files[0].name
    shutil.copy2(pt_files[0], pt_copy)
    for old in (ROOT / "checkpoints").glob(f"ckpt_{symbol}_step_*.pt"):
        old.unlink()
    status, payload = http_post_multipart(
        "/api/training/import",
        pt_copy,
        query=f"?symbol={symbol}",
    )
    if status != 200:
        fail(f"pt import HTTP {status}: {payload}")
    ok("single .pt import ok")

    print("\n== HTTP: overview flags ==")
    status, body, _ = http_get("/api/overview")
    if status != 200:
        fail(f"overview HTTP {status}")
    overview = json.loads(body.decode())
    ok("overview endpoint")


def cleanup(symbol: str) -> None:
    for old in (ROOT / "checkpoints").glob(f"ckpt_{symbol}_step_*.pt"):
        old.unlink(missing_ok=True)
    (ROOT / f"training_history_{symbol}.json").unlink(missing_ok=True)
    (ROOT / "strategies" / f"best_{symbol}.json").unlink(missing_ok=True)


def main() -> None:
    print("=== Training I/O live test ===")
    parquet = setup_fixture()
    zip_path = test_python_package(SYMBOL)
    test_http(SYMBOL, zip_path)
    cleanup(SYMBOL)
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
