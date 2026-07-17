"""
find_aligned_symbols.py — 找出与核心品种时间对齐良好的截面特征品种

流程：
1. 先用 KlineCache 把所有候选品种的数据下载/更新到本地 D:/K线数据/
2. 然后直接读本地文件做时间戳对比（不再实时查 MT5，快很多）
3. 结果写到 strategies/aligned_symbols.json
"""
import sys, json
sys.path.insert(0, '.')

import MetaTrader5 as mt5
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from data_pipeline.kline_cache import KlineCache

OUT = Path("strategies/aligned_symbols.json")
LOG = Path("strategies/aligned_symbols_progress.txt")

CORE = ['EURUSDm', 'USDJPYm', 'XAUUSDm', 'USTECm', 'US500m']

SKIP = ['BTC', 'ETH', 'USDT', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMZN',
        'GOOG', 'META', 'BABA', 'JD', 'NIO', 'BIDU', 'NFLX', 'PYPL',
        'SBUX', 'COST', 'AMD', 'INTC', 'CSCO', 'IBM', 'ORCL', 'ADBE']
NEED = ['USD', 'EUR', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF', 'JPY', 'NOK',
        'SEK', 'DKK', 'MXN', 'ZAR', 'SGD', 'HKD', 'PLN', 'CNH',
        'XAU', 'XAG', 'XPT', 'XPD', 'XAL', 'XCU', 'XNI', 'XPB', 'XZN',
        'DXY', 'OIL', 'GAS', 'NG',
        'US30', 'US500', 'USTEC', 'UK100', 'DE30', 'FR40', 'JP225',
        'HK50', 'AUS200', 'STOXX']


def main():
    cache = KlineCache()

    # Step 1: 连 MT5 获取候选品种列表
    mt5.initialize()
    all_syms = [s.name for s in mt5.symbols_get() if s.trade_mode == 4]
    candidates = []
    for s in all_syms:
        base = s.upper().replace('M', '').replace('_X100', '').replace('_X10', '')
        if any(k in base for k in SKIP):
            continue
        if any(k in base for k in NEED):
            candidates.append(s)

    msg = f"Step 1: {len(candidates)} candidates found\n"
    print(msg, end='')
    LOG.write_text(msg)

    # Step 2: 下载/更新所有候选品种到本地（顺序，避免 MT5 并发问题）
    print("Step 2: Downloading/updating local cache (this may take a while)...\n")
    done = 0
    for s in candidates:
        cache.get(s, mt5_connected=True)
        done += 1
        if done % 10 == 0:
            p = f"  Downloaded {done}/{len(candidates)}...\n"
            print(p, end='')
            with open(LOG, 'a') as f:
                f.write(p)

    # 获取核心时间集
    core_times = None
    for s in CORE:
        df = cache.read_local(s)
        t = set(df['time'].astype(int)) if df is not None and not df.empty else set()
        core_times = t if core_times is None else core_times & t

    mt5.shutdown()

    p2 = f"\nStep 2 done. Core intersection: {len(core_times)} bars\n"
    print(p2, end='')
    with open(LOG, 'a') as f:
        f.write(p2)

    # Step 3: 读本地文件做时间对比（快，不需要 MT5）
    print("Step 3: Checking time alignment from local cache...\n")

    def check(sym):
        df = cache.read_local(sym)
        if df is None or len(df) < 3000:
            return sym, 0
        return sym, len(set(df['time'].astype(int)) & core_times)

    good = []
    done2 = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(check, s): s for s in candidates}
        for fut in as_completed(futures):
            sym, n = fut.result()
            done2 += 1
            if n >= 7000:
                good.append((sym, n))
            if done2 % 20 == 0:
                p3 = f"  Checked {done2}/{len(candidates)}, found {len(good)} good\n"
                print(p3, end='')
                with open(LOG, 'a') as f:
                    f.write(p3)

    good.sort(key=lambda x: -x[1])

    result = {
        "core_symbols": CORE,
        "core_intersection_bars": len(core_times),
        "threshold": 7000,
        "aligned_symbols": [s for s, n in good],
        "details": {s: n for s, n in good},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    summary = f"\n=== Done: {len(good)} aligned symbols (>=7000 bars overlap) ===\n"
    for s, n in good:
        summary += f"  {s}: {n}\n"
    print(summary)
    with open(LOG, 'a') as f:
        f.write(summary)

    print(f"Results -> {OUT}")
    print(f"Cache   -> D:/K线数据/ ({len(list(Path('D:/K线数据').glob('*.parquet')))} files)")


if __name__ == '__main__':
    main()
