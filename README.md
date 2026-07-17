# binance-alpha-miner

基于强化学习的**币安合约因子挖掘 / 回测**工具：从本地 Parquet K 线里自动搜索可解释的
因子公式（token 序列 = 特征 + 算子，经 StackVM 执行），支持命令行训练/回测与可选的
Web 控制台。

除常规 OHLCV 外，专门适配了币安合约的**另类数据**（持仓量、资金费率、多空比、
主动买卖比、CVD），并把它们做成可进入公式搜索空间的因子。

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

> **来源与许可**：本项目在一个采用 **AGPL-3.0** 的开源因子挖掘项目基础上二次开发，
> 依 AGPL-3.0 继续以相同许可开源。二次开发时**移除了原项目中一处会读取本机其它软件
> 登录凭证的代码**，并新增了币安数据适配。详见 [SECURITY_CHANGES.md](SECURITY_CHANGES.md)。

---

## 安装

```bash
python -m pip install -r requirements.txt
# 本地 Parquet 训练/回测无需 MT5/A股/TradingView 等可选依赖；如需实时数据源再装：
# python -m pip install -r requirements-optional.txt
```

核心依赖：`torch / pandas / pyarrow / numpy / loguru / python-dotenv`。

---

## 数据格式

加载器 `data_pipeline/parquet_manager.py` 对常见 K 线 Parquet 自适应：

| 项 | 说明 |
|---|---|
| **时间列** | 自动识别 `ts / time / timestamp / open_time / datetime`，毫秒/秒自动归一 |
| **必需列** | `open / high / low / close`（成交量列 `volume/tick_volume` 缺失则置 0） |
| **文件名** | 支持无周期后缀的 `BTCUSDT.parquet`（**无需改名**），周期自动推断（见下） |
| **坏 K 线** | 自动丢弃 OHLC 缺失的行（数据缺口/停牌），避免收益率变 NaN 拖垮训练 |
| **另类列** | 下列列存在即自动读入并生成对应因子；缺失则对应因子恒 0（不报错、不影响老数据） |

**周期自动推断（Web 与 CLI 通用，无需重命名你的数据）**，优先级从高到低：
1. 显式 `--timeframe 5m`（仅 CLI）；
2. 文件名 `{品种}_{周期}.parquet` 后缀（如 `BTCUSDT_5m.parquet`）；
3. 父目录名里的周期 token（如 `D:\币安币种数据_1m\BTCUSDT.parquet` → 1m）；
4. `Config.DEFAULT_TIMEFRAME`（环境变量 `KLINE_DEFAULT_TIMEFRAME`，默认 `5m`）。

> 所以 `D:\币安币种数据\BTCUSDT.parquet` 自动当 5m、`D:\币安币种数据_1m\BTCUSDT.parquet` 自动当 1m，**主数据一个字节都不用改**。

自动识别的另类数据列：
`quote_vol, oi, oi_value, topls_pos, topls_acc, global_acc, taker_bs, funding_rate,
taker_buy_vol, taker_buy_quote_vol, count, cvd_delta_15s/1s/500ms`。

---

## 新增的另类数据因子（12 个，类别 `crypto_alt`）

在原有 65 个价量因子上新增（词表 = 77 特征 + 62 算子 = 139 token）。
**在只有 OHLCV 的数据上这些因子恒为 0，会被评估器当常数列剪掉，不影响纯价量流程。**

| 因子 | 含义 | 因子 | 含义 |
|---|---|---|---|
| `FUNDING_Z` / `FUNDING_MOM` | 资金费率水平 / 动量 | `TOPLS_POS` / `TOPLS_ACC` | 大户持仓 / 账户多空比 |
| `OI_CHG` | 持仓量变化率 | `GLOBAL_ACC` | 全局账户多空比（散户情绪） |
| `OI_PRICE_DIV` | 价/仓背离 | `SMART_DUMB_DIV` | 大户 vs 全局背离（聪明钱） |
| `TAKER_IMBALANCE` | 主动买卖失衡 | `QUOTE_VOL_SURGE` | 成交额放量 |
| `TAKER_BUY_FRAC` | 主动买量占比 | `CVD_IMBALANCE` | CVD 订单流失衡 |

所有因子归一化统一用**因果滚动 robust 标准化**（每个 t 只用 `[t-w+1..t]`，无未来泄露）。

---

## 使用

### 训练

```bash
python train_binance.py --data-file "D:\币安币种数据\BTCUSDT.parquet" --timeframe 5m --from-scratch
```

- `--timeframe 5m`（文件名无周期后缀时必填）
- `--max-bars 60000`：只用最近 N 根（5m 全量数十万根，先用近端窗口调通）
- `--steps 3000`：覆盖训练步数（默认 9000）
- `--reward-mode standard`：默认；`ftmo`/`forex` 是外汇专属

产物：`strategies/best_{symbol}.json`（公式 token、可读公式、验证分数、词表版本）。

### 回测

```bash
python backtest_binance.py --strategy strategies\best_BTCUSDT.json \
    --data-file "D:\币安币种数据\BTCUSDT.parquet" --timeframe 5m \
    --cost 0.0005 --slippage 0.0002 --split 0.7
```

- **成本默认按币安 USDT-M 合约**：单边手续费 5bp + 滑点 2bp。
- `--split 0.7`：前 70% 样本内 / 后 30% 样本外，直接打印样本外相对样本内的夏普变化，
  **一眼看出过拟合**。
- 指标：总收益率、年化收益率、夏普、索提诺、卡玛、最大回撤、在场占比、换手、单bar胜率。

信号口径与训练一致：`仓位 = tanh(因子)`，`|仓位| < 0.05` 记空仓。

### Web 控制台（可选）

```bash
python run_web.py --port 8765     # 默认只监听 127.0.0.1，勿用 --host 0.0.0.0 暴露到公网
```

---

## 使用须知（诚实提示）

1. **单品种慎用截面算子**：`CS_RANK / CS_SCALE / REL_*` 是跨品种截面算子，单品种训练会退化
   （暖机区产生 NaN，回测按空仓兜底）。要用截面逻辑需改成多品种数据管理器。
2. **成本要如实**：默认 7bp/边已含滑点；**永续资金费未计入 pnl**，资金费敏感策略需另叠加。
3. **必须留真正的样本外**：`--split` 只是时间切分体检；严肃评估应把最近一段数据封存、
   训练时不可见，训练完只跑一次。
4. **只导入你自己训练产出的检查点**，不要导入来路不明的 `.pt`/`.zip`（安全纪律，见安全文档）。

保留的抗过拟合机制：walk-forward + gap 隔离、样本外泛化门控、成本压力测试、beta 中性惩罚、
暴露/换手惩罚。

---

## License

[GNU AGPL-3.0](LICENSE)。修改、分发或通过网络提供服务时须以相同许可公开对应源代码。
