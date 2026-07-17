"""Load training data from a single Parquet K-line file.

AlphaForge 安全 fork 的改造点（相对上游）
-------------------------------------------------
1. 时间列自适应：支持 `time / ts / timestamp / open_time / datetime / date` 等常见列名，
   自动识别毫秒/秒/（秒÷1000 的异常）并统一为 Unix 秒 int64。
2. 文件名自适应：上游强制 `{品种}_{周期}.parquet`；本 fork 允许没有周期后缀的文件
   （如币安数据的 `BTCUSDT.parquet`），此时由调用方显式传入 timeframe。
3. 另类数据透传：除 OHLCV 外，把币安合约特有的列（持仓量 oi、资金费率 funding_rate、
   多空比 topls_*/global_acc、主动买卖比 taker_bs、CVD 等）读入 raw_dict，供
   model_core.features 里新增的「另类数据因子」使用；缺列时自动跳过，不影响老数据。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger

from config import Config
from data_pipeline.data_manager import MT5DataManager
from model_core.features import MT5FeatureEngineer

# Canonical labels used across the project
_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")

# Filename suffix aliases → canonical (case-insensitive keys)
_TF_ALIASES: dict[str, str] = {
    "m1": "M1", "1m": "M1", "1min": "M1", "min1": "M1",
    "m5": "M5", "5m": "M5", "5min": "M5", "min5": "M5",
    "m15": "M15", "15m": "M15", "15min": "M15", "min15": "M15",
    "m30": "M30", "30m": "M30", "30min": "M30", "min30": "M30",
    "h1": "H1", "1h": "H1", "60m": "H1", "60min": "H1", "min60": "H1", "60": "H1",
    "h4": "H4", "4h": "H4", "240m": "H4", "240min": "H4", "min240": "H4", "240": "H4",
    "d1": "D1", "1d": "D1", "day": "D1", "daily": "D1", "1440m": "D1", "1440min": "D1",
    "w1": "W1", "1w": "W1", "week": "W1", "weekly": "W1",
    "mn1": "MN1", "1mo": "MN1", "1mon": "MN1", "month": "MN1", "monthly": "MN1",
}

# 时间列候选（按优先级）
_TIME_COL_ALIASES = ("time", "ts", "timestamp", "open_time", "datetime", "date", "datetime_bjt")
# 成交量列候选
_VOLUME_COL_ALIASES = ("volume", "tick_volume", "vol", "base_volume")

# 另类数据列：canonical 名 → (可能的原始列名, 缺失/NaN 中性填充值)
# 中性值原则：比率类填 1.0（log 后 ≈ 0），其余填 0.0。
_EXTRA_FIELD_SPECS: dict[str, tuple[tuple[str, ...], float]] = {
    "quote_vol":           (("quote_vol", "quote_volume", "qv"), 0.0),
    "oi":                  (("oi", "open_interest"), 0.0),
    "oi_value":            (("oi_value", "open_interest_value", "oi_val"), 0.0),
    "topls_pos":           (("topls_pos", "top_ls_position"), 1.0),
    "topls_acc":           (("topls_acc", "top_ls_account"), 1.0),
    "global_acc":          (("global_acc", "global_ls_account"), 1.0),
    "taker_bs":            (("taker_bs", "taker_buy_sell_ratio"), 1.0),
    "funding_rate":        (("funding_rate", "funding"), 0.0),
    "taker_buy_vol":       (("taker_buy_vol", "taker_buy_base_vol"), 0.0),
    "taker_buy_quote_vol": (("taker_buy_quote_vol",), 0.0),
    "trade_count":         (("count", "trade_count", "num_trades", "trades"), 0.0),
    "cvd_15s":             (("cvd_delta_15s", "cvd_15s"), 0.0),
    "cvd_1s":              (("cvd_delta_1s", "cvd_1s"), 0.0),
    "cvd_500ms":           (("cvd_delta_500ms", "cvd_500ms"), 0.0),
}


def normalize_timeframe_token(token: str) -> str | None:
    """Map a filename timeframe token to canonical M1/M5/.../MN1."""
    raw = (token or "").strip()
    if not raw:
        return None
    key = raw.lower().replace("-", "").replace("_", "")
    if key in _TF_ALIASES:
        return _TF_ALIASES[key]
    upper = raw.upper()
    if upper in _TIMEFRAMES:
        return upper
    return None


def parse_parquet_filename(
    path: str | Path, default_timeframe: str | None = None
) -> tuple[str, str]:
    """解析 ``{品种}_{周期}.parquet``；也支持无周期后缀的 ``{品种}.parquet``。

    - 有合法周期后缀（``BTCUSDT_5m.parquet`` / ``AAPL_H1.parquet``）→ 用后缀。
    - 无后缀（``BTCUSDT.parquet``）或后缀非法 → 用 ``default_timeframe``（如 "5m"）。
    - 两者都没有 → 抛错，提示补 --timeframe。
    """
    name = Path(path).name
    if Path(path).suffix.lower() != ".parquet":
        raise ValueError(f"请选择 .parquet 文件；当前: {name}")
    stem = Path(path).stem

    default_tf = normalize_timeframe_token(default_timeframe) if default_timeframe else None

    if "_" in stem:
        symbol, tf_raw = stem.rsplit("_", 1)
        symbol = symbol.strip()
        timeframe = normalize_timeframe_token(tf_raw)
        if symbol and timeframe is not None:
            return symbol, timeframe
        # 后缀不是合法周期（例如 1000BONK 这种），退回把整个 stem 当品种名
        if default_tf is not None:
            return stem.strip(), default_tf
        raise ValueError(
            f"文件名 {name} 的 '_' 后缀不是合法周期，且未提供 --timeframe。"
            f"请指定周期，如 --timeframe 5m。"
        )

    # 无下划线：整个 stem 是品种名，周期必须由外部给
    if default_tf is not None:
        return stem.strip(), default_tf
    raise ValueError(
        f"文件名 {name} 无周期后缀，请提供 --timeframe（如 5m/1m/H1）。"
    )


def _pick_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    return None


def _to_unix_seconds(series: pd.Series) -> np.ndarray:
    """把任意时间列转为 Unix 秒 int64。

    支持：datetime/字符串（pandas 解析）、Unix 秒、Unix 毫秒，以及被误÷1000 的秒。
    """
    if pd.api.types.is_datetime64_any_dtype(series) or series.dtype == object:
        dt = pd.to_datetime(series, utc=True, errors="coerce")
        return (dt.view("int64") // 1_000_000_000).to_numpy()

    vals = pd.to_numeric(series, errors="coerce")
    m = float(np.nanmax(vals.to_numpy())) if len(vals) else 0.0
    if m >= 1e17:        # 纳秒
        return (vals // 1_000_000_000).astype("int64").to_numpy()
    if m >= 1e14:        # 微秒
        return (vals // 1_000_000).astype("int64").to_numpy()
    if m >= 1e11:        # 毫秒（币安 ts 落这一档）
        return (vals // 1000).astype("int64").to_numpy()
    if m < 1e7 and m > 0:  # 被误÷1000 的秒（某些 A 股导出）
        return (vals * 1000).astype("int64").to_numpy()
    return vals.astype("int64").to_numpy()  # 已是秒


def inspect_parquet_file(
    path: str | Path, timeframe: str | None = None
) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    if p.suffix.lower() != ".parquet":
        raise ValueError("请选择 .parquet 文件")

    symbol, tf = parse_parquet_filename(p, default_timeframe=timeframe)
    df = pd.read_parquet(p)
    bars = len(df)
    if bars < Config.MIN_BARS:
        raise ValueError(f"数据不足: {bars} bars（至少需要 {Config.MIN_BARS}）")

    # bars/年 的估算因子（近似，仅供展示）
    bars_per_year = {
        "M1": 525600, "M5": 105120, "M15": 35040, "M30": 17520,
        "H1": 8760, "H4": 2190, "D1": 365,
    }.get(tf)
    years = round(bars / bars_per_year, 2) if bars_per_year else None

    extras = [k for k, (aliases, _) in _EXTRA_FIELD_SPECS.items()
              if _pick_column(df, aliases) is not None]

    return {
        "data_file": str(p.resolve()),
        "filename": p.name,
        "symbol": symbol,
        "timeframe": tf,
        "bars": bars,
        "years_h1": years,     # 键名保持兼容，实为按该周期估算的年数
        "extra_fields": extras,
        "valid": True,
        "message": "",
    }


class ParquetDataManager:
    """Single-symbol data manager backed by one Parquet file.

    Args:
        file_path: parquet 路径。
        timeframe: 当文件名无周期后缀时（如币安 ``BTCUSDT.parquet``）必须提供，
                   例如 "5m" / "1m" / "H1"；有后缀时可省略。
    """

    def __init__(
        self,
        file_path: str | Path,
        timeframe: str | None = None,
        max_bars: int | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.symbol, self.timeframe = parse_parquet_filename(
            self.file_path, default_timeframe=timeframe
        )
        self.max_bars = int(max_bars) if max_bars else None
        self._raw_dict: dict[str, torch.Tensor] | None = None
        self._target_ret: torch.Tensor | None = None
        self._extra_loaded: list[str] = []

    def load(self) -> None:
        df = pd.read_parquet(self.file_path)
        # 只保留最近 max_bars 根（5m 全量 40 万+ 根，训练每步逐条跑滚动算子会很慢，
        # 大周期/短验证时用它截断到近端窗口）。截断在排序前按行数近似，排序后仍是近端。
        if self.max_bars and len(df) > self.max_bars:
            df = df.iloc[-self.max_bars:].copy()
        if len(df) < Config.MIN_BARS:
            raise ValueError(f"数据不足: {len(df)} bars（至少需要 {Config.MIN_BARS}）")

        time_col = _pick_column(df, _TIME_COL_ALIASES)
        if time_col is None:
            raise ValueError(
                f"Parquet 缺少时间列（候选: {_TIME_COL_ALIASES}）；实际列: {list(df.columns)}"
            )
        volume_col = _pick_column(df, _VOLUME_COL_ALIASES)
        for c in ("open", "high", "low", "close"):
            if c not in df.columns:
                raise ValueError(f"Parquet 缺少必需列: {c}")

        work = df.copy()
        # 统一时间为 Unix 秒
        work["__t"] = _to_unix_seconds(work[time_col])
        work = work.sort_values("__t")
        work = work[~work["__t"].duplicated(keep="last")]
        work = work.reset_index(drop=True)

        # ── 核心 OHLCV ──
        ohlc = {f: work[f].astype("float64").to_numpy() for f in ("open", "high", "low", "close")}
        if volume_col is not None:
            vol = pd.to_numeric(work[volume_col], errors="coerce").fillna(0.0).to_numpy()
        else:
            vol = np.zeros(len(work), dtype="float64")
            logger.warning(f"[数据] {self.file_path.name} 无成交量列，volume 置 0")

        raw: dict[str, torch.Tensor] = {}
        for f in ("open", "high", "low", "close"):
            raw[f] = torch.tensor(np.array([ohlc[f]]), dtype=torch.float32)
        raw["volume"] = torch.tensor(np.array([vol]), dtype=torch.float32)
        raw["time"] = torch.tensor(np.array([work["__t"].to_numpy().astype("int64")]), dtype=torch.int64)

        # ── 另类数据列（缺列跳过；NaN → ffill/bfill → 中性填充）──
        self._extra_loaded = []
        for canonical, (aliases, neutral) in _EXTRA_FIELD_SPECS.items():
            col = _pick_column(work, aliases)
            if col is None:
                continue
            s = pd.to_numeric(work[col], errors="coerce")
            s = s.ffill().bfill().fillna(neutral)
            arr = s.astype("float64").to_numpy()
            raw[canonical] = torch.tensor(np.array([arr]), dtype=torch.float32)
            self._extra_loaded.append(canonical)

        self._raw_dict = raw
        self._target_ret = MT5DataManager._compute_target_ret(raw["open"])
        logger.info(
            f"[数据] 已加载 {self.symbol} {self.timeframe}，"
            f"共 {raw['open'].shape[1]} 根K线，另类列: {self._extra_loaded or '无'}，"
            f"文件 {self.file_path.name}"
        )

    @property
    def extra_fields(self) -> list[str]:
        return list(self._extra_loaded)

    @property
    def symbols(self) -> list[str]:
        return [self.symbol]

    @property
    def raw_dict(self) -> dict[str, torch.Tensor]:
        if self._raw_dict is None:
            raise RuntimeError("Call load() first")
        return self._raw_dict

    @property
    def feat_tensor(self) -> torch.Tensor:
        return MT5FeatureEngineer.compute_features(self.raw_dict)

    @property
    def target_ret(self) -> torch.Tensor:
        if self._target_ret is None:
            raise RuntimeError("Call load() first")
        return self._target_ret

    @property
    def bar_time(self) -> torch.Tensor:
        raw = self.raw_dict
        if "time" in raw:
            return raw["time"][:, -1].long()
        return torch.zeros(1, dtype=torch.int64)
