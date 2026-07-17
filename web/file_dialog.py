"""Native file picker for local training UI."""
from __future__ import annotations


def pick_parquet_file() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        raise RuntimeError("当前环境不支持图形文件选择（缺少 tkinter）")

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    path = filedialog.askopenfilename(
        title="选择 K 线 Parquet 文件",
        filetypes=[
            ("Parquet K线", "*.parquet"),
            ("所有文件", "*.*"),
        ],
    )
    root.destroy()
    return path or None


def pick_strategy_file() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        raise RuntimeError("当前环境不支持图形文件选择（缺少 tkinter）")

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    path = filedialog.askopenfilename(
        title="选择策略 JSON 文件",
        filetypes=[
            ("策略 JSON", "*.json"),
            ("所有文件", "*.*"),
        ],
    )
    root.destroy()
    return path or None
