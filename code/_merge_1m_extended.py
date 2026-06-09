# -*- coding: utf-8 -*-
"""
合并 HistData 2026 Q1+ 增量 1m 数据到现有 原始_XAU_1m.csv
- 旧: 2004-06-11 ~ 2025-12-31, 格式 "YYYY.MM.DD HH:MM;O;H;L;C;V"
- 新: 5 个 HistData ASCII 月文件 (2026-01 ~ 2026-05), 格式 "YYYYMMDD HHMMSS;O;H;L;C;V"
- 输出: 原始_XAU_1m.csv 覆盖, 范围 2004-06 ~ 2026-05
- 清理: 删 _cache/xau_1m_array.npz 强制下次重建
"""
import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import XAU_1M_CSV, CACHE_DIR

# 本地: 设环境变量 HISTDATA_DIR 指向 HistData ASCII 月文件目录
DOWNLOAD_DIR = Path(os.environ.get("HISTDATA_DIR", "./histdata"))
NEW_FILES = [
    DOWNLOAD_DIR / "DAT_ASCII_XAUUSD_M1_202601.csv",
    DOWNLOAD_DIR / "DAT_ASCII_XAUUSD_M1_202602.csv",
    DOWNLOAD_DIR / "DAT_ASCII_XAUUSD_M1_202603.csv",
    DOWNLOAD_DIR / "DAT_ASCII_XAUUSD_M1_202604.csv",
    DOWNLOAD_DIR / "DAT_ASCII_XAUUSD_M1_202605.csv",
]
BACKUP = XAU_1M_CSV.with_suffix(".bak.csv")


def parse_histdata_new(path: Path) -> pd.DataFrame:
    """读 HistData ASCII 新格式, 返回与旧 csv 同 schema 的 DataFrame"""
    df = pd.read_csv(path, sep=";", header=None,
                     names=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d %H%M%S")
    # 转回旧字符串格式 "YYYY.MM.DD HH:MM"
    df["Date"] = df["Date"].dt.strftime("%Y.%m.%d %H:%M")
    return df


def main():
    print(f"[1/4] 备份原文件 -> {BACKUP.name}")
    if not BACKUP.exists():
        BACKUP.write_bytes(XAU_1M_CSV.read_bytes())
    else:
        print(f"      .bak 已存在, 跳过")

    print(f"[2/4] 读旧文件 {XAU_1M_CSV.name}")
    # 旧文件首行是表头: Date;Open;High;Low;Close;Volume
    old = pd.read_csv(XAU_1M_CSV, sep=";", encoding="utf-8")
    print(f"      旧文件 {len(old):,} 行, 范围 {old['Date'].iloc[0]} ~ {old['Date'].iloc[-1]}")

    print(f"[3/4] 读 5 个 HistData 新文件并转格式")
    new_parts = []
    for f in NEW_FILES:
        df = parse_histdata_new(f)
        new_parts.append(df)
        print(f"      {f.name}: {len(df):,} 行, {df['Date'].iloc[0]} ~ {df['Date'].iloc[-1]}")
    new = pd.concat(new_parts, ignore_index=True)
    print(f"      合并新数据 {len(new):,} 行")

    # 拼接 + 去重 (按 Date)
    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date"], keep="first")
    print(f"      拼接 + 去重后 {len(combined):,} 行")

    print(f"[4/4] 写回 {XAU_1M_CSV.name} (覆盖)")
    combined.to_csv(XAU_1M_CSV, sep=";", index=False, encoding="utf-8")
    print(f"      新文件范围 {combined['Date'].iloc[0]} ~ {combined['Date'].iloc[-1]}")

    # 清缓存
    npz = CACHE_DIR / "xau_1m_array.npz"
    if npz.exists():
        npz.unlink()
        print(f"      清理 _cache/{npz.name} (下次会自动重建)")
    npz_daily = CACHE_DIR / "xau_daily.csv"
    if npz_daily.exists():
        npz_daily.unlink()
        print(f"      清理 _cache/{npz_daily.name}")
    npz_ext = CACHE_DIR / "xau_daily_extended.csv"
    if npz_ext.exists():
        npz_ext.unlink()
        print(f"      清理 _cache/{npz_ext.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
