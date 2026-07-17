"""Hands-on export/import training test via live API."""
from __future__ import annotations

import json
import mimetypes
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API = "http://127.0.0.1:8765"
SYMBOL = "ADAUSD"
WORKDIR = Path(tempfile.mkdtemp(prefix="am_train_io_"))


def step(msg: str) -> None:
    print(f"\n>> {msg}")


def ok(msg: str) -> None:
    print(f"   [OK] {msg}")


def fail(msg: str) -> None:
    print(f"   [FAIL] {msg}")
    raise SystemExit(1)


def get_json(path: str) -> dict:
    with urllib.request.urlopen(API + path, timeout=30) as r:
        return json.loads(r.read().decode())


def get_bytes(path: str) -> tuple[int, bytes, dict]:
    req = urllib.request.Request(API + path)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.read(), dict(r.headers)


def post_json(path: str, data: dict | None = None) -> dict:
    req = urllib.request.Request(
        API + path,
        json.dumps(data or {}).encode(),
        {"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def post_file(path: str, file_path: Path, query: str = "") -> tuple[int, dict]:
    boundary = "----BinanceAlphaMinerHandsOn"
    data = file_path.read_bytes()
    ctype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        API + path + query,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def ui_export_enabled() -> bool:
    ov = get_json("/api/overview")
    tr = get_json("/api/training/status")
    df = ov.get("data_file") or {}
    prog = ov.get("progress") or {}
    sym = prog.get("symbol") or df.get("symbol")
    enabled = bool(df.get("valid")) and bool(prog.get("has_checkpoint")) and not tr.get("active")
    print(
        f"   UI条件: 品种={sym} valid={df.get('valid')} "
        f"has_checkpoint={prog.get('has_checkpoint')} active={tr.get('active')} "
        f"=> 导出按钮={'可点' if enabled else '禁用'}"
    )
    return enabled


def backup_symbol_files() -> Path:
    bak = WORKDIR / "backup"
    bak.mkdir(parents=True)
    for p in (ROOT / "checkpoints").glob(f"ckpt_{SYMBOL}_step_*.pt"):
        shutil.copy2(p, bak / p.name)
    for name in [f"training_history_{SYMBOL}.json", f"strategies/best_{SYMBOL}.json"]:
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, bak / Path(name).name)
    return bak


def restore_backup(bak: Path) -> None:
    for p in (ROOT / "checkpoints").glob(f"ckpt_{SYMBOL}_step_*.pt"):
        p.unlink(missing_ok=True)
    (ROOT / f"training_history_{SYMBOL}.json").unlink(missing_ok=True)
    for p in bak.glob("ckpt_*.pt"):
        shutil.copy2(p, ROOT / "checkpoints" / p.name)
    for name in [f"training_history_{SYMBOL}.json", f"best_{SYMBOL}.json"]:
        src = bak / name
        if src.exists():
            dest = ROOT / ("strategies" if name.startswith("best_") else "") / name
            if name.startswith("best_"):
                dest = ROOT / "strategies" / name
            else:
                dest = ROOT / name
            shutil.copy2(src, dest)


def clear_symbol_files() -> None:
    for p in (ROOT / "checkpoints").glob(f"ckpt_{SYMBOL}_step_*.pt"):
        p.unlink()
    (ROOT / f"training_history_{SYMBOL}.json").unlink(missing_ok=True)


def list_symbol_files() -> dict:
    return {
        "ckpts": sorted(p.name for p in (ROOT / "checkpoints").glob(f"ckpt_{SYMBOL}_step_*.pt")),
        "history": (ROOT / f"training_history_{SYMBOL}.json").exists(),
        "strategy": (ROOT / "strategies" / f"best_{SYMBOL}.json").exists(),
    }


def main() -> None:
    print("=== 导出/导入训练 实操 ===")
    print(f"工作目录: {WORKDIR}")

    step("1. 检查服务与 UI 按钮条件")
    h = get_json("/api/health")
    ok(f"health {h}")
    if not ui_export_enabled():
        fail("按 UI 规则导出按钮应为可点，但实际不满足")

    bak = backup_symbol_files()
    ok(f"已备份到 {bak}")

    step("2. 导出训练 zip")
    status, raw, headers = get_bytes(f"/api/training/{SYMBOL}/export")
    if status != 200:
        fail(f"导出 HTTP {status}")
    zip_path = WORKDIR / f"training_{SYMBOL}.zip"
    zip_path.write_bytes(raw)
    ok(f"下载 {zip_path.name} ({len(raw)} bytes)")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    ok(f"zip 内容: {names}")
    for must in ["manifest.json", f"checkpoints/ckpt_{SYMBOL}_step_0060.pt"]:
        if must not in names:
            fail(f"zip 缺少 {must}")

    step("3. 清空本地训练文件后导入 zip")
    clear_symbol_files()
    before = list_symbol_files()
    ok(f"清空后: {before}")

    status, payload = post_file("/api/training/import", zip_path, f"?symbol={SYMBOL}")
    if status != 200 or not payload.get("ok"):
        fail(f"导入失败: {payload}")
    ok(payload.get("message", "imported"))

    after = list_symbol_files()
    ok(f"导入后: {after}")
    if not after["ckpts"]:
        fail("checkpoint 未恢复")
    if not after["history"]:
        fail("training_history 未恢复")

    step("4. 验证 overview 反映 has_checkpoint")
    ov = get_json("/api/overview")
    prog = ov.get("progress") or {}
    if not prog.get("has_checkpoint"):
        fail(f"导入后 has_checkpoint 仍为 false: {prog}")
    ok(f"step={prog.get('current_step')} has_checkpoint=True")

    step("5. 单独导入 .pt 文件")
    pt_src = ROOT / "checkpoints" / after["ckpts"][-1]
    pt_copy = WORKDIR / pt_src.name
    shutil.copy2(pt_src, pt_copy)
    clear_symbol_files()
    status, payload = post_file("/api/training/import", pt_copy, f"?symbol={SYMBOL}")
    if status != 200:
        fail(f".pt 导入失败: {payload}")
    ok(f".pt 导入 step={payload.get('step')}")

    step("6. 续训验证（启动后应出现「续训」日志）")
    post_json("/api/training/stop")
    data_file = json.loads((ROOT / "web_settings.json").read_text(encoding="utf-8"))["last_data_file"]
    post_json("/api/training/start", {"data_file": data_file})
    resume = False
    for i in range(10):
        time.sleep(4)
        st = get_json("/api/training/status")
        log = "\n".join(st.get("log_tail") or [])
        if "续训" in log and "恢复" in log:
            resume = True
            ok("日志确认断点续训")
            break
    post_json("/api/training/stop")
    if not resume:
        fail("启动训练未出现续训日志")

    step("7. 恢复测试前备份")
    restore_backup(bak)
    ok("已还原原始训练文件")

    print("\n=== 全部实操通过：导出训练 / 导入训练 / 续训 均正常 ===")


if __name__ == "__main__":
    main()
