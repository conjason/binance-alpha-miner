"""
data_pipeline/single_symbol_manager.py — 单品种数据视图

将 MT5DataManager 加载的多品种数据切片为单品种视图，
供 AlphaEngine 单品种训练模式使用。

使用方式：
    with MT5DataFetcher() as fetcher:
        mgr = MT5DataManager(fetcher)
        mgr.load()
        for sym in mgr.symbols:
            single = SingleSymbolDataManager(mgr, sym)
            engine = AlphaEngine(data_manager=single)
            engine.train()
"""
from __future__ import annotations

import torch
from loguru import logger

from model_core.features import MT5FeatureEngineer


class SingleSymbolDataManager:
    """单品种数据视图，兼容 AlphaEngine 对 data_manager 的接口。

    AlphaEngine 调用的接口：
        .feat_tensor   → [1, F, T]  (N=1)
        .target_ret    → [1, T]
        .raw_dict      → {field: [1, T]}
        .symbols       → [symbol]
        .bar_time      → [1]
    """

    def __init__(self, multi_manager, symbol: str) -> None:
        """
        Args:
            multi_manager: 已 load() 的 MT5DataManager 实例。
            symbol:        要切片的品种名，必须在 multi_manager.symbols 中。
        """
        if symbol not in multi_manager.symbols:
            raise ValueError(
                f"Symbol '{symbol}' not found in multi_manager.symbols: "
                f"{multi_manager.symbols}"
            )
        self._multi  = multi_manager
        self._symbol = symbol
        self._idx    = multi_manager.symbols.index(symbol)

        logger.info(f"[SingleSymbolDataManager] symbol={symbol}  idx={self._idx}")

    # ── AlphaEngine 所需接口 ──────────────────────────────────────────────

    @property
    def symbols(self) -> list[str]:
        return [self._symbol]

    @property
    def raw_dict(self) -> dict:
        full = self._multi.raw_dict
        return {k: v[self._idx:self._idx+1] for k, v in full.items()}  # [1, T]

    @property
    def feat_tensor(self) -> torch.Tensor:
        """返回 [1, F, T] 特征张量（只含目标品种）。"""
        raw = self.raw_dict
        return MT5FeatureEngineer.compute_features(raw)   # [1, F, T]

    @property
    def target_ret(self) -> torch.Tensor:
        full = self._multi.target_ret
        return full[self._idx:self._idx+1]   # [1, T]

    @property
    def bar_time(self) -> torch.Tensor:
        full = self._multi.bar_time
        return full[self._idx:self._idx+1]   # [1]

    @property
    def symbol(self) -> str:
        return self._symbol
