"""
strategy_manager/portfolio.py — MT5 仓位管理器

管理 MT5 仓位状态，支持 JSON 持久化和 MT5 实时同步。
"""
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict

from loguru import logger

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    mt5 = None

try:
    from config import Config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    # 测试环境回退默认值
    class Config:
        PORTFOLIO_FILE = "portfolio_state.json"


@dataclass
class Position:
    """MT5 仓位数据结构。"""
    symbol: str
    ticket: int
    entry_price: float
    entry_time: float
    lot_size: float
    direction: str          # "BUY" | "SELL"
    highest_price: float    # 多头追踪止损用（最高价）
    lowest_price: float     # 空头追踪止损用（最低价）
    is_partial_closed: bool


class MT5PortfolioManager:
    """MT5 仓位管理器。

    负责记录、更新、持久化仓位状态，并与 MT5 终端实时同步。
    """

    def __init__(self) -> None:
        self.positions: Dict[str, Position] = {}
        self.state_file: str = Config.PORTFOLIO_FILE
        self.load_state()

    # ─────────────────────────────────────────────────────────────
    # 仓位增删改查
    # ─────────────────────────────────────────────────────────────

    def add_position(
        self,
        symbol: str,
        ticket: int,
        price: float,
        lot: float,
        direction: str,
    ) -> None:
        """记录一个新开仓位。

        Args:
            symbol:    交易品种
            ticket:    MT5 order ticket（由 mt5.order_send() 返回）
            price:     入场价格
            lot:       手数
            direction: "BUY" 或 "SELL"
        """
        pos = Position(
            symbol=symbol,
            ticket=ticket,
            entry_price=price,
            entry_time=time.time(),
            lot_size=lot,
            direction=direction,
            highest_price=price,
            lowest_price=price,
            is_partial_closed=False,
        )
        self.positions[symbol] = pos
        self.save_state()
        logger.info(f"[Portfolio] Position added: {symbol} {direction} lot={lot} @ {price} ticket={ticket}")

    def close_position(self, symbol: str) -> None:
        """从本地状态移除仓位（不发出 MT5 订单，仅清除记录）。

        Args:
            symbol: 要关闭的品种
        """
        if symbol in self.positions:
            pos = self.positions.pop(symbol)
            self.save_state()
            logger.info(f"[Portfolio] Position closed: {symbol} ticket={pos.ticket}")
        else:
            logger.warning(f"[Portfolio] close_position: {symbol} not found in local state")

    def get_direction(self, symbol: str) -> int:
        """返回品种当前持仓方向的整数表示。

        Returns:
            +1 多头 / -1 空头 / 0 空仓
        """
        if symbol not in self.positions:
            return 0
        d = self.positions[symbol].direction
        return 1 if d == "BUY" else -1

    def update_price(self, symbol: str, price: float) -> None:
        """更新当前价格，分别追踪多头最高价和空头最低价。"""
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        changed = False
        if pos.direction == "BUY" and price > pos.highest_price:
            pos.highest_price = price
            changed = True
        elif pos.direction == "SELL" and price < pos.lowest_price:
            pos.lowest_price = price
            changed = True
        if changed:
            self.save_state()

    def get_open_count(self) -> int:
        """返回当前持仓数量。"""
        return len(self.positions)

    # ─────────────────────────────────────────────────────────────
    # MT5 同步
    # ─────────────────────────────────────────────────────────────

    def sync_from_mt5(self) -> None:
        """与 MT5 终端同步仓位状态。

        1. 调用 mt5.positions_get() 获取当前所有持仓。
        2. 将本地记录中已不在 MT5 的仓位移除（外部平仓）。
        3. 将 MT5 中有但本地没有的仓位补录（漏记情况）。
        4. 同步 direction（以 MT5 为准）。
        """
        if not _MT5_AVAILABLE or mt5 is None:
            logger.warning("[Portfolio] MT5 not available, skipping sync")
            return

        live_positions = mt5.positions_get()
        if live_positions is None:
            logger.warning(f"[Portfolio] mt5.positions_get() failed: {mt5.last_error()}")
            return

        allowed_symbols = set(getattr(Config, "SYMBOLS", []) or [])
        excluded_symbols = set(getattr(Config, "EXCLUDED_TRADE_SYMBOLS", []) or [])

        # 以 symbol 为 key 建立 MT5 持仓索引，仅同步当前自动交易品种。
        # 兼容 mock：若 symbol 属性是字符串才使用；否则降级为按 ticket 匹配
        live_by_symbol: dict[str, object] = {}
        live_tickets: set[int] = set()
        for p in live_positions:
            sym    = getattr(p, "symbol", None)
            ticket = getattr(p, "ticket", None)
            if isinstance(sym, str):
                if sym in excluded_symbols:
                    continue
                if allowed_symbols and sym not in allowed_symbols:
                    continue
            if isinstance(sym, str):
                live_by_symbol[sym] = p
            if isinstance(ticket, int):
                live_tickets.add(ticket)

        if live_by_symbol:
            # 新逻辑：按 symbol 对账（symbol 属性为字符串时）
            to_remove = [s for s in self.positions if s not in live_by_symbol]
            for s in to_remove:
                pos = self.positions.pop(s)
                logger.info(f"[Portfolio] Externally closed, removed: {s} ticket={pos.ticket}")

            # 同步 direction 并补录漏记仓位
            for sym, p in live_by_symbol.items():
                direction = "BUY" if getattr(p, "type", 0) == 0 else "SELL"
                if sym in self.positions:
                    self.positions[sym].direction = direction
                else:
                    price = float(getattr(p, "price_open", 0.0))
                    self.positions[sym] = Position(
                        symbol=sym,
                        ticket=getattr(p, "ticket", 0),
                        entry_price=price,
                        entry_time=float(getattr(p, "time", time.time())),
                        lot_size=float(getattr(p, "volume", 0.01)),
                        direction=direction,
                        highest_price=price,
                        lowest_price=price,
                        is_partial_closed=False,
                    )
                    logger.info(f"[Portfolio] 补录MT5持仓: {sym} {direction}")
        else:
            # 降级逻辑：只有 ticket 可用时，按 ticket 移除已不存在的仓位
            to_remove = [
                s for s, pos in self.positions.items()
                if pos.ticket not in live_tickets
            ]
            for s in to_remove:
                pos = self.positions.pop(s)
                logger.info(f"[Portfolio] Externally closed (by ticket), removed: "
                            f"{s} ticket={pos.ticket}")

        if to_remove:
            self.save_state()

    # ─────────────────────────────────────────────────────────────
    # JSON 持久化
    # ─────────────────────────────────────────────────────────────

    def save_state(self) -> None:
        """将当前仓位状态保存到 JSON 文件（Config.PORTFOLIO_FILE）。"""
        data = {symbol: asdict(pos) for symbol, pos in self.positions.items()}
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"[Portfolio] State saved to {self.state_file} ({len(data)} positions)")
        except OSError as e:
            logger.error(f"[Portfolio] Failed to save state: {e}")

    def load_state(self) -> None:
        """从 JSON 文件恢复仓位状态。

        文件不存在时静默初始化为空仓位集合。
        JSON 字段不匹配时记录 WARNING 并跳过该条目。
        """
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data: dict = json.load(f)

            loaded = 0
            for symbol, fields in data.items():
                try:
                    self.positions[symbol] = Position(**fields)
                    loaded += 1
                except (TypeError, KeyError) as e:
                    logger.warning(
                        f"[Portfolio] Skipping malformed position '{symbol}': {e}"
                    )

            logger.info(
                f"[Portfolio] Loaded {loaded} position(s) from {self.state_file}"
            )

        except FileNotFoundError:
            logger.info(
                f"[Portfolio] No state file found at {self.state_file}, starting fresh"
            )
            self.positions = {}
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"[Portfolio] Failed to load state: {e}")
            self.positions = {}
