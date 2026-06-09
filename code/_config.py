# -*- coding: utf-8 -*-
"""
全局配置常量 — 所有 baseline 共用
- 数据路径 / 初始资金 / 成本 / 时间窗口
- 论文锁定参数，禁止在单个 baseline 脚本内修改
"""
import os
from pathlib import Path

# ---- 数据源路径 ----
# 本地: 设环境变量 XAU_DATA_DIR 指向原始数据目录 (见 .env.example)
# 公开复现: 把发布的 CSV 放 ./data, 不设环境变量即用默认
DATA_DIR     = Path(os.environ.get("XAU_DATA_DIR", "./data"))
V10_CSV      = DATA_DIR / os.environ.get("V10_CSV_NAME", "V10_1507笔交易.csv")
V11_OLD_CSV  = DATA_DIR / os.environ.get("V11_OLD_CSV_NAME", "V11倍率应用后.csv")
XAU_1M_CSV   = DATA_DIR / os.environ.get("XAU_1M_CSV_NAME", "原始_XAU_1m.csv")

# ---- 输出目录 (4 子目录结构) ----
CODE_DIR     = Path(__file__).resolve().parent           # v11eapaper/code/
ROOT_DIR     = CODE_DIR.parent                            # v11eapaper/
DATA_DIR_OUT = ROOT_DIR / "data"                          # csv 输出
FIG_DIR      = ROOT_DIR / "figures"                       # png 输出
DOC_DIR      = ROOT_DIR                                   # 主表 + README + report
CACHE_DIR    = ROOT_DIR / "_cache"                        # 中间缓存
for d in (DATA_DIR_OUT, FIG_DIR, CACHE_DIR):
    d.mkdir(exist_ok=True)
# 兼容别名: 旧代码默认 OUT_DIR 指 data (具体脚本应用 DATA_DIR_OUT / FIG_DIR / DOC_DIR)
OUT_DIR      = DATA_DIR_OUT

# ---- 论文锁定参数 ----
SYMBOL              = "XAUUSD"
DATE_START          = "2024-01-01"
DATE_END            = "2026-06-01"    # 扩展到 2026-06-01 (Phase 2, 1m 数据已覆盖)
INITIAL_BALANCE     = 10_000.0        # USD (匹配 V10 csv running_balance 起点)
RISK_PERCENT        = 0.36 / 100      # 0.36% per trade
SPREAD_PIP          = 0.3             # pip
COMMISSION_PER_LOT  = 7.0             # USD / lot (单边)
SLIPPAGE_PIP        = 1.0             # pip

# XAU pip 定义: 1 pip = 0.01 USD/oz, contract size = 100 oz
# → 1 lot 跑 1 pip = 100 oz × 0.01 = $1.0
# → 0.3 pip spread on 1 lot = $0.30
# → 1 pip slip on 1 lot = $1.00
PIP_VALUE_PER_LOT   = 1.0   # USD per pip per lot for XAU
CONTRACT_SIZE       = 100   # oz per lot

# ---- 子样本三段窗口 ----
SUBSAMPLE_WINDOWS = [
    ("2024",         "2024-01-01", "2024-12-31"),
    ("2025",         "2025-01-01", "2025-12-31"),
    ("2026 partial", "2026-01-01", "2026-06-01"),  # 跟主表窗口对齐
]

# ---- 蒙特卡洛 ----
MC_RUNS = 1000   # 助教定稿要求
MC_SEED = 20260608

# ---- 绘图样式 (复用 backtest_sizing.py) ----
STYLE = dict(
    bg     = "#0f1318",
    ax     = "#1a1f26",
    text   = "#d8e0e8",
    grid   = "#2a323d",
    win    = "#26a69a",
    loss   = "#ef5350",
    blue   = "#42a5f5",
    purple = "#ab47bc",
    orange = "#ff9800",
    yellow = "#ffd54f",
)

# ---- 一致性: 计算每笔交易的总成本 ----
def round_trip_cost_usd(lot: float) -> float:
    """单笔交易往返成本 (开 + 平)。spread + 2×commission + 2×slippage"""
    spread_cost     = SPREAD_PIP * PIP_VALUE_PER_LOT * lot
    commission_cost = 2 * COMMISSION_PER_LOT * lot
    slip_cost       = 2 * SLIPPAGE_PIP * PIP_VALUE_PER_LOT * lot
    return spread_cost + commission_cost + slip_cost
