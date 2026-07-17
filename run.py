"""
run.py — 实盘入口

启动 MT5 实盘策略主循环。

使用方式：
    python run.py

前置条件：
    1. 已运行 main.py 完成训练，生成 best_mt5_strategy.json
    2. 已配置 .env 文件，包含 MT5_LOGIN、MT5_PASSWORD、MT5_SERVER
    3. MetaTrader5 终端已启动并登录

流程：
    - 初始化 MT5StrategyRunner（加载策略公式，失败时自动 sys.exit(1)）
    - 启动同步主循环（runner.run()）
    - Ctrl+C（KeyboardInterrupt）可优雅中断
    - try/finally 确保无论何种退出方式都调用 runner.shutdown()，释放 MT5 连接

Requirements: 10.1–10.7
"""
from strategy_manager.runner import MT5StrategyRunner

if __name__ == "__main__":
    runner = MT5StrategyRunner()
    try:
        runner.run()
    except KeyboardInterrupt:
        pass
    finally:
        runner.shutdown()
