# -*- coding: utf-8 -*-
"""扫描 MT5 tester log, 列出每个 session 的 EA 名 + modeling 模式"""
import os
import re
from pathlib import Path

# 本地: 设环境变量 MT5_LOG 指向 tester 日志, 见 .env.example
LOG = Path(os.environ.get("MT5_LOG", "./tester_log.log"))

text = LOG.read_bytes().decode("utf-16-le", errors="replace")
lines = text.splitlines()

# Find session starts
sess_starts = []
for i, l in enumerate(lines):
    m = re.search(r"testing of Experts\\(\S+\.ex5)", l)
    if m:
        sess_starts.append((i, m.group(1)))

print(f"{len(sess_starts)} sessions found in 20260609.log\n")

for idx, (line_no, ea) in enumerate(sess_starts):
    end_line = sess_starts[idx + 1][0] if idx + 1 < len(sess_starts) else len(lines)
    # Look for modeling mode in first 60 lines
    window = lines[line_no:line_no + 60]
    mode = ""
    final_bal = ""
    for w in window:
        wl = w.lower()
        if "every tick" in wl or "1 minute" in wl or "open prices" in wl:
            if not mode:
                mode = w.strip()[:140]
    # Look for final balance near end of session
    tail = lines[max(line_no, end_line - 30):end_line]
    for w in tail:
        m = re.search(r"final balance\s+([\d.]+)", w)
        if m:
            final_bal = m.group(1)
    print(f"  #{idx}  行 {line_no:>6}-{end_line:>6}  EA={ea}")
    print(f"         模式: {mode if mode else '(未找到 modeling 标识)'}")
    print(f"         final balance: ${final_bal}")
    print()
