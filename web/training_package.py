"""Export / import training checkpoints as portable zip packages."""
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from model_core.vocab import FORMULA_VOCAB
from web.progress import (
    CHECKPOINT_DIR,
    PROJECT_ROOT,
    _decode_formula,
    _load_strategy,
    checkpoint_glob,
    invalidate_checkpoint_cache,
)

_CKPT_NAME_RE = re.compile(r"^ckpt_(.+)_step_(\d+)\.pt$", re.IGNORECASE)
# 安全：品种名白名单（防止经文件名做路径穿越）
_SAFE_SYMBOL_RE = re.compile(r"^[A-Za-z0-9._-]{1,40}$")


def _history_path(symbol: str) -> Path:
    return PROJECT_ROOT / f"training_history_{symbol}.json"


def _symbol_from_ckpt_name(name: str) -> str | None:
    m = _CKPT_NAME_RE.match(Path(name).name)
    return m.group(1) if m else None


# 安全：训练包解压资源上限（防 zip 炸弹 / 内存磁盘耗尽）
_MAX_MEMBER_BYTES = 500_000_000        # 单成员 500MB
_MAX_TOTAL_UNCOMPRESSED = 2_000_000_000  # 总解压 2GB
_MAX_MEMBER_COUNT = 50                  # 最多 50 个成员
_MAX_COMPRESS_RATIO = 200               # 单成员压缩比上限


def _verify_ckpt_dict(ckpt: dict, name: str) -> dict[str, Any]:
    artifact_version = ckpt.get("vocab_version")
    if artifact_version is None:
        raise ValueError(
            f"检查点 {name} 过旧（无 vocab_version），"
            f"当前词表 {FORMULA_VOCAB.version!r}，请重新训练"
        )
    FORMULA_VOCAB.verify(artifact_version)
    return {"step": int(ckpt.get("step", 0)), "symbol": ckpt.get("symbol"),
            "best_score": ckpt.get("best_score")}


def _validate_checkpoint_file(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return _verify_ckpt_dict(ckpt, path.name)


def _validate_checkpoint_bytes(data: bytes, name: str) -> dict[str, Any]:
    """在写盘前用安全模式(weights_only=True)从内存校验检查点。"""
    ckpt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    return _verify_ckpt_dict(ckpt, name)


def build_training_export_zip(symbol: str) -> tuple[bytes, str]:
    ckpts = checkpoint_glob(symbol)
    if not ckpts:
        raise FileNotFoundError(f"未找到 {symbol} 的训练检查点，请先训练并保存 checkpoint")

    latest = ckpts[-1]
    _validate_checkpoint_file(latest)

    strategy = _load_strategy(symbol)
    history_path = _history_path(symbol)

    step = int(re.search(r"_step_(\d+)\.pt$", latest.name).group(1))
    safe = symbol.replace(".", "_")
    zip_name = f"training_{safe}_step{step:04d}.zip"

    manifest = {
        "format": "binance_alpha_miner_training_v1",
        "symbol": symbol,
        "step": step,
        "checkpoint": f"checkpoints/{latest.name}",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "files": [f"checkpoints/{latest.name}"],
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(latest, f"checkpoints/{latest.name}")

        if strategy:
            strat_name = f"strategies/best_{symbol}.json"
            payload = dict(strategy)
            if payload.get("formula") and not payload.get("formula_decoded"):
                payload["formula_decoded"] = _decode_formula(payload["formula"])
            zf.writestr(strat_name, json.dumps(payload, ensure_ascii=False, indent=2))
            manifest["files"].append(strat_name)

        if history_path.exists():
            hist_name = f"training_history_{symbol}.json"
            zf.write(history_path, hist_name)
            manifest["files"].append(hist_name)

        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    return buf.getvalue(), zip_name


def _remove_symbol_checkpoints(symbol: str) -> int:
    removed = 0
    for path in checkpoint_glob(symbol):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def import_training_package(
    content: bytes,
    filename: str,
    expected_symbol: str | None = None,
) -> dict[str, Any]:
    name = Path(filename).name.lower()
    installed: list[str] = []
    symbol: str | None = None
    step: int | None = None

    if name.endswith(".zip"):
        result = _import_zip(content, expected_symbol)
        symbol = result["symbol"]
        step = result["step"]
        installed = result["installed"]
    elif name.endswith(".pt"):
        result = _import_pt(content, filename, expected_symbol)
        symbol = result["symbol"]
        step = result["step"]
        installed = result["installed"]
    else:
        raise ValueError("仅支持 .zip 训练包或 .pt 检查点文件")

    invalidate_checkpoint_cache()
    return {
        "ok": True,
        "symbol": symbol,
        "step": step,
        "installed": installed,
        "message": f"已导入 {symbol} 的训练文件（step {step}），下次训练将从断点续训",
    }


def _import_zip(content: bytes, expected_symbol: str | None) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if "manifest.json" not in names:
            raise ValueError("训练包缺少 manifest.json")

        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest.get("format") != "binance_alpha_miner_training_v1":
            raise ValueError("不支持的训练包格式")

        symbol = manifest.get("symbol") or ""
        ckpt_rel = manifest.get("checkpoint", "")
        if not symbol or not ckpt_rel:
            raise ValueError("训练包 manifest 不完整")
        # 安全：品种名只允许安全字符，防止经 training_history_{symbol}.json 等做路径穿越
        if not _SAFE_SYMBOL_RE.match(symbol):
            raise ValueError(f"训练包品种名非法: {symbol!r}")

        if expected_symbol and symbol != expected_symbol:
            raise ValueError(
                f"训练包品种为 {symbol}，与当前选择的 {expected_symbol} 不一致"
            )

        root = PROJECT_ROOT.resolve()

        # ── 阶段 1：全部在内存里校验，通过前不删任何旧文件、不写盘 ──
        # （防数据丢失：损坏/恶意包不应先删掉旧检查点；防 zip 炸弹：总量/数量/压缩比上限）
        planned: list[tuple[str, Path, bytes]] = []  # (base, dest, data)
        total_uncompressed = 0
        member_count = 0
        for member in names:
            if member == "manifest.json":
                continue
            base = member.replace("\\", "/").split("/")[-1]
            if not base or base in (".", ".."):
                continue
            if base.startswith("ckpt_") and base.endswith(".pt"):
                dest = CHECKPOINT_DIR / base
            elif base.startswith("best_") and base.endswith(".json"):
                dest = PROJECT_ROOT / "strategies" / base
            elif base == f"training_history_{symbol}.json":
                dest = PROJECT_ROOT / base
            else:
                continue  # 不在白名单的成员一律跳过
            info = zf.getinfo(member)
            if info.file_size > _MAX_MEMBER_BYTES:
                raise ValueError(f"训练包成员过大，拒绝解压: {member}")
            if info.compress_size > 0 and info.file_size / info.compress_size > _MAX_COMPRESS_RATIO:
                raise ValueError(f"训练包成员压缩比异常(疑似 zip 炸弹): {member}")
            member_count += 1
            total_uncompressed += info.file_size
            if member_count > _MAX_MEMBER_COUNT:
                raise ValueError("训练包成员过多，拒绝导入")
            if total_uncompressed > _MAX_TOTAL_UNCOMPRESSED:
                raise ValueError("训练包总解压体积过大，拒绝导入")
            # 双保险：解析后的目标必须仍在项目根内
            dest_res = dest.resolve()
            if root != dest_res and root not in dest_res.parents:
                raise ValueError(f"训练包成员路径非法(疑似穿越): {member}")
            data = zf.read(member)
            # .pt 在写盘前就用安全模式校验（含 vocab 版本），失败则整包拒绝
            if base.endswith(".pt"):
                _validate_checkpoint_bytes(data, base)
            planned.append((base, dest, data))

        if not any(b.endswith(".pt") for b, _, _ in planned):
            raise ValueError("训练包内未找到有效检查点(.pt)")

        # ── 阶段 2：校验全通过后，才删旧检查点并落盘 ──
        _remove_symbol_checkpoints(symbol)
        installed: list[str] = []
        imported_history = False
        for base, dest, data in planned:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            installed.append(str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/"))
            if base == f"training_history_{symbol}.json":
                imported_history = True

        if not imported_history:
            try:
                _history_path(symbol).unlink(missing_ok=True)
            except OSError:
                pass

        step = int(manifest.get("step") or 0)
        return {"symbol": symbol, "step": step, "installed": installed}


def _import_pt(content: bytes, filename: str, expected_symbol: str | None) -> dict[str, Any]:
    ckpt_name = Path(filename).name
    symbol = _symbol_from_ckpt_name(ckpt_name)
    if not symbol:
        raise ValueError(f"无法从文件名识别品种: {ckpt_name}（须为 ckpt_品种_step_XXXX.pt）")

    if expected_symbol and symbol != expected_symbol:
        raise ValueError(
            f"检查点品种为 {symbol}，与当前选择的 {expected_symbol} 不一致"
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dest = CHECKPOINT_DIR / ckpt_name

    # 只导入 .pt 时，项目里可能还残留更“新”的 training_history_{symbol}.json。
    # Web 进度会优先使用该曲线文件的最后一步，导致显示步数大于 checkpoint。
    # 因此这里主动清掉旧曲线，避免「导入 60 步却显示 84」。
    try:
        _history_path(symbol).unlink(missing_ok=True)
    except OSError:
        pass

    _remove_symbol_checkpoints(symbol)
    dest.write_bytes(content)
    meta = _validate_checkpoint_file(dest)

    return {
        "symbol": symbol,
        "step": meta["step"],
        "installed": [str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/")],
    }
