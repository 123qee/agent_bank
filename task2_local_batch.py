#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地批量回放 Agent：默认加载 run120.py，也可用 --agent 指定 run.py 等任意入口。

用法（PowerShell）:
  set TASK2_DEBUG_LOG=1
  set TASK2_DEBUG_LOG_DIR=.\\logs
  set TASK2_DEBUG_CTX=1
  python task2_local_batch.py fixtures\\task2_sample.jsonl
  python task2_local_batch.py fixtures\\task2_sample.jsonl --agent run.py

环境与所加载脚本一致（数据库、ONE_API_*）。调试日志只写文件，不写 stdout。

调试含义：把题干批量送进 mod.run(question)，可选对照 expect；配合 TASK2_DEBUG_LOG
可读每条题的 route_channel。对错摘要是「相对你写的 expect」，不是评测官方 hidden。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path


def _load_agent(py_path: Path):
    spec = importlib.util.spec_from_file_location("task2_agent_impl", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compact(s: str) -> str:
    return "".join(str(s).split())


def _extract_numbers(s: str) -> list[float]:
    out: list[float] = []
    for m in re.finditer(r"-?\d+(?:\.\d+)?", str(s)):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            continue
    return out


def expect_match(expect: str, got: str, numeric_tol: float) -> tuple[bool, str]:
    """宽松比对：整段包含 / 去空白相等 / 数值集合相对误差。"""
    if expect is None or expect == "":
        return True, "skip"
    ex = expect.strip()
    g = got.strip()
    if ex in g or _compact(ex) == _compact(g):
        return True, "substring_or_compact"
    nums_e = _extract_numbers(ex)
    nums_g = _extract_numbers(g)
    if len(nums_e) == len(nums_g) and nums_e:
        ok = True
        for a, b in zip(nums_e, nums_g):
            if abs(a) < 1e-12:
                if abs(b) > 1e-9:
                    ok = False
                    break
            elif abs(a - b) / abs(a) > numeric_tol:
                ok = False
                break
        return ok, "numeric_tuple"
    return False, "mismatch"


def main() -> int:
    ap = argparse.ArgumentParser(description="Task2 Agent JSONL 批量回放")
    ap.add_argument("jsonl", type=Path, help="UTF-8 JSONL，字段 question 必填")
    ap.add_argument(
        "--agent",
        type=Path,
        default=Path(__file__).resolve().parent / "run120.py",
        help="Agent 入口脚本路径（默认 run120.py）",
    )
    ap.add_argument(
        "--group-by-scenario",
        action="store_true",
        help="按每行 scenario 分组：仅当 scenario 变化时 reset_agent_session_state",
    )
    ap.add_argument(
        "--numeric-tol",
        type=float,
        default=0.001,
        help="expect 与 got 均含数值时的最大相对误差（默认 0.1%%）",
    )
    args = ap.parse_args()

    mod = _load_agent(args.agent.resolve())
    run = getattr(mod, "run")
    reset = getattr(mod, "reset_agent_session_state", None)
    if reset is None:

        def reset() -> None:
            """兼容未实现 reset 的脚本（如部分 run.py）；无法在行间清空会话。"""

        print(
            "warning: 模块无 reset_agent_session_state，行间无法清空会话；"
            "逐题隔离/多轮测试请优先用 run120.py",
            file=sys.stderr,
        )

    lines = args.jsonl.read_text(encoding="utf-8").splitlines()
    prev_scenario: object | None = object()
    ok_n = fail_n = skip_n = 0
    t0_all = time.perf_counter()

    for idx, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        q = row.get("question") or row.get("q") or row.get("text")
        if not q:
            print(f"[{idx}] skip: no question field", file=sys.stderr)
            continue
        exp = row.get("expect") or row.get("answer")

        if args.group_by_scenario:
            sc = row.get("scenario")
            if sc != prev_scenario:
                reset()
                prev_scenario = sc
        else:
            reset()

        t0 = time.perf_counter()
        got = run(q)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        matched, how = expect_match(str(exp) if exp is not None else "", got, args.numeric_tol)
        if exp is None or exp == "":
            flag = "?"
            skip_n += 1
        elif matched:
            flag = "OK"
            ok_n += 1
        else:
            flag = "FAIL"
            fail_n += 1

        meta = json.dumps(row.get("meta") or {}, ensure_ascii=False)
        preview = (got.replace("\n", "\\n"))[:160]
        print(f"[{idx}] {flag} {how} line_ms={elapsed_ms} meta={meta}")
        print(f"    Q: {q[:120]}{'…' if len(q) > 120 else ''}")
        print(f"    A: {preview}{'…' if len(got) > 160 else ''}")
        if exp is not None and exp != "" and not matched:
            print(f"    E: {str(exp)[:200]}")

    total_ms = int((time.perf_counter() - t0_all) * 1000)
    print(f"--- done ok={ok_n} skip={skip_n} fail={fail_n} total_ms={total_ms} ---")
    return 1 if fail_n else 0


if __name__ == "__main__":
    sys.exit(main())
