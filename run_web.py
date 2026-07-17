"""
run_web.py — 启动训练 Web 控制台

用法:
    python run_web.py
    python run_web.py --port 8765

浏览器打开 http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="BinanceAlphaMiner Training Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("请先安装依赖: pip install fastapi uvicorn[standard]")
        sys.exit(1)

    debug = bool(load_settings().get("debug_mode", False))

    print(f"\n  BinanceAlphaMiner 量化因子挖掘中心")
    print(f"  → http://{args.host}:{args.port}")
    print(f"  调试模式: {'开启' if debug else '关闭（默认）'}\n")

    uvicorn.run(
        "web.app:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="debug" if debug else "warning",
        access_log=debug,
    )


if __name__ == "__main__":
    main()
