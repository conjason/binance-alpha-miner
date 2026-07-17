"""File logging for the training web UI."""
from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

SERVER_LOG = LOG_DIR / "web_server.log"
ERROR_LOG = LOG_DIR / "web_errors.log"

_logger: logging.Logger | None = None
_debug_mode: bool = False


def is_debug_mode() -> bool:
    return _debug_mode


def set_debug_mode(enabled: bool) -> None:
    """Toggle verbose INFO logging. Default off."""
    global _debug_mode
    _debug_mode = bool(enabled)
    logger = get_logger()
    level = logging.INFO if _debug_mode else logging.WARNING
    for handler in logger.handlers:
        handler.setLevel(level)
    if _debug_mode:
        logger.info("Debug mode enabled")


def apply_debug_mode_from_settings() -> bool:
    from web.settings import load_settings

    enabled = bool(load_settings().get("debug_mode", False))
    set_debug_mode(enabled)
    return enabled


def setup_logging() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger("binance_alpha_miner.web")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(SERVER_LOG, encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    _logger = logger
    apply_debug_mode_from_settings()
    return logger


def get_logger() -> logging.Logger:
    return _logger or setup_logging()


def log_error(message: str, exc: BaseException | None = None) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [f"[{ts}] {message}"]
    if exc is not None:
        lines.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    block = "\n".join(lines) + "\n"

    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(block)

    get_logger().error(message, exc_info=exc)


def tail_file(path: Path, lines: int = 200) -> list[str]:
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return content.splitlines()[-lines:]


def debug_snapshot(lines: int = 200) -> dict:
    server_tail: list[str] = []
    if is_debug_mode():
        server_tail = tail_file(SERVER_LOG, lines)
    return {
        "debug_mode": is_debug_mode(),
        "server_log": str(SERVER_LOG),
        "error_log": str(ERROR_LOG),
        "server_tail": server_tail,
        "error_tail": tail_file(ERROR_LOG, lines),
    }
