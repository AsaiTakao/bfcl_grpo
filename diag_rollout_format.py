# -*- coding: utf-8 -*-
"""diag_rollout_format.py — ロールアウト本文の <tool_call> を分類する。

目的
----
val 報酬が 0 に張り付く原因を「モデルの実力」と「配線(ツール名の不一致)」に
切り分ける。verl の generation dump(trainer.rollout_data_dir /
trainer.validation_data_dir)の jsonl を読み、応答本文中の <tool_call> を

  wrapper : {"name": "bfcl_multi_turn_call", ...}   -> 報酬計算に載る
  direct  : {"name": "cd", ...} など実関数を直呼び   -> sglang が破棄 -> 報酬 0
  broken  : JSON として壊れている
  none    : そもそも tool_call が出ていない(散文だけ)

に分類する。direct が支配的なら報酬関数のバグではなく
「ラッパツール経由を守らせる設計」の問題。

使い方:
    python diag_rollout_format.py <dump.jsonl> [<dump.jsonl> ...] [--show 3]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys

TOOL_NAME = "bfcl_multi_turn_call"
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# 応答本文から取り出す候補キー(verl のバージョンで揺れる)。
_TEXT_KEYS = ("output", "response", "responses", "generation", "text", "output_str")


def response_text(rec: dict) -> str:
    for k in _TEXT_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, list) and v and isinstance(v[0], str):
            return "\n".join(v)
    return ""


def classify(text: str) -> collections.Counter:
    c = collections.Counter()
    blocks = _TOOL_CALL_RE.findall(text)
    if not blocks:
        c["none"] += 1
        return c
    for b in blocks:
        try:
            obj = json.loads(b)
        except Exception:  # noqa: BLE001
            c["broken"] += 1
            continue
        name = obj.get("name") if isinstance(obj, dict) else None
        if name == TOOL_NAME:
            args = obj.get("arguments")
            if isinstance(args, str):
                c["wrapper_args_is_string"] += 1   # _parse_tool_calls が拾える形か要確認
            elif isinstance(args, dict) and isinstance(args.get("tool_calls"), list):
                c["wrapper_ok"] += 1
            else:
                c["wrapper_bad_args"] += 1
        else:
            c["direct:" + str(name)] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--show", type=int, default=2, help="生本文をこの件数だけ出す")
    args = ap.parse_args()

    files = [p for pat in args.paths for p in sorted(glob.glob(pat))]
    if not files:
        print("dump が見つかりません", file=sys.stderr)
        return 2

    total = collections.Counter()
    n_rows = 0
    shown = 0
    scores = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                n_rows += 1
                txt = response_text(rec)
                total.update(classify(txt))
                if isinstance(rec.get("score"), (int, float)):
                    scores.append(float(rec["score"]))
                if shown < args.show:
                    shown += 1
                    print(f"===== sample {shown} ({path}) score={rec.get('score')} =====")
                    print(txt[:3000])
                    print("=" * 60)

    print(f"\n行数: {n_rows}")
    if scores:
        print(f"score 平均: {sum(scores)/len(scores):.4f} / 非ゼロ: "
              f"{sum(1 for s in scores if s > 0)}/{len(scores)}")
    direct = sum(v for k, v in total.items() if k.startswith("direct:"))
    wrapper = sum(v for k, v in total.items() if k.startswith("wrapper"))
    print(f"tool_call 分類: wrapper={wrapper} direct={direct} "
          f"broken={total['broken']} 本文にtool_call無し={total['none']}")
    for k, v in total.most_common():
        print(f"  {k}: {v}")
    print("\n判定の目安:")
    print("  direct >> wrapper       -> 配線/プロンプト設計の問題(報酬関数は無罪)")
    print("  wrapper_ok が多いのに score=0 -> 報酬関数か GT デコードのバグ")
    print("  none が支配的           -> チャットテンプレート/生成長/停止条件の問題")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
