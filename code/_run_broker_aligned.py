# -*- coding: utf-8 -*-
"""
Broker-time-aligned Python control run (时区变量从设计上消除)
========================================================================
把 Python 引擎搬到 broker 时间 (Europe/Athens = EET/EEST, 真 DST) 跑 V10/V11,
让 4H/D1 桶边界跟 MT5 broker time 精确对齐。这样 Python vs MT5 残差 gap 只剩
data vendor (HistData vs ECMarkets) + tick 执行精度 —— 即 contribution (d) 的目标。

输出:
  data/python_v10_trades_broker.csv
  data/python_v11_trades_broker.csv
  data/python_v10_v11_metrics_broker.csv
并打印 broker-aligned vs UTC 的对比。
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import INITIAL_BALANCE, OUT_DIR
from _metrics import compute_metrics
from v11_engine import run
from python_v10_v11_metrics import to_daily_equity, compute_rr

BROKER_TZ = "Europe/Athens"   # EET/EEST = GMT+2 冬 / +3 夏, 匹配 ECMarkets 服务器


def metrics_for(df, label):
    eq, pnl = to_daily_equity(df)
    rr = compute_rr(df)
    return compute_metrics(eq, trades_pnl=pnl, trades_rr=rr,
                           initial_balance=INITIAL_BALANCE, label=label)


def main():
    rows = []
    for ver, dyn in [("V10", False), ("V11", True)]:
        print("\n" + "=" * 70)
        print(f"RUN broker-aligned: {ver} (broker_tz={BROKER_TZ})")
        print("=" * 70)
        df = run(use_dynamic_sizing=dyn, broker_tz=BROKER_TZ, log_progress=False)
        out = OUT_DIR / f"python_{ver.lower()}_trades_broker.csv"
        df.to_csv(out, index=False, encoding="utf-8")
        m = metrics_for(df, f"Python {ver} (broker)")
        m["n_trades"] = len(df)
        rows.append(m)
        print(f"  {ver} broker: n={len(df)}  Sharpe={m.get('sharpe'):.3f}  "
              f"Return={m.get('total_return_pct'):.1f}%  Calmar={m.get('calmar'):.3f}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "python_v10_v11_metrics_broker.csv",
                             index=False, encoding="utf-8")

    # 对比 UTC 旧数字
    print("\n" + "=" * 70)
    print("对比: UTC (原) vs Broker-aligned (新)")
    print("=" * 70)
    for ver in ["v10", "v11"]:
        try:
            utc = pd.read_csv(OUT_DIR / f"python_{ver}_trades.csv")
            mu = metrics_for(utc, f"{ver} utc")
            brk = pd.read_csv(OUT_DIR / f"python_{ver}_trades_broker.csv")
            mb = metrics_for(brk, f"{ver} broker")
            print(f"\n{ver.upper()}:")
            print(f"  n_trades : UTC {len(utc):>5}  ->  broker {len(brk):>5}  "
                  f"(Δ {100*(len(brk)-len(utc))/len(utc):+.1f}%)")
            for k in ["sharpe", "total_return_pct", "calmar", "sortino"]:
                a, b = mu.get(k), mb.get(k)
                if a is not None and b is not None:
                    print(f"  {k:<16}: UTC {a:>8.3f}  ->  broker {b:>8.3f}  "
                          f"(Δ {100*(b-a)/abs(a):+.1f}%)")
        except Exception as e:
            print(f"  {ver}: compare ERR {e}")


if __name__ == "__main__":
    main()
