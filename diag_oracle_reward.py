# -*- coding: utf-8 -*-
"""diag_oracle_reward.py — 「完璧なモデル」を流して報酬関数の上限を測る。

目的
----
報酬 0 の原因がモデル側か報酬側かを 1 回で切り分ける。GT(正解呼び出し列)を
そのままモデル出力に見立てて報酬パイプラインへ流し、満点が出るか確認する。

  満点が出る  -> 報酬関数と GT デコードは健全。問題はロールアウト側
                 (ツール名不一致 / テンプレート / 生成長)。
  0 が出る    -> 報酬関数・GT デコード・バックエンド初期化のどれかがバグ。
                 どの段で落ちたかは下の内訳出力で分かる。

3 経路すべてを測る:
  A) tool 経路   : BFCLMultiTurnTool.create/execute/calc_reward
                   (= 実際に GRPO の報酬になる経路)
  B) text 経路   : bfcl_reward.compute_score(ラッパ形式のテキスト)
                   (= reward_scores 欠落時のフォールバック)
  C) 直呼び形式  : 実関数名を直接 <tool_call> した場合のテキスト採点
                   (= 実機ログで頻出する形。ここが 0 なのは仕様どおりか確認)

使い方(bfcl_eval が入った環境 = 学習用イメージ内で実行):
    python diag_oracle_reward.py --parquet /workspace/data/bfcl_mt_val.parquet -n 5
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
from typing import Any, Dict, List

import bfcl_reward
from bfcl_dataset_and_interaction import decode_field
from bfcl_verl_tool import BFCLMultiTurnTool

TOOL_NAME = bfcl_reward.TOOL_NAME


def gt_string_to_call(call_str: str) -> Dict[str, Any]:
    """GT の "func(a=1, b='x')" を {"name","arguments"} に変換する。"""
    tree = ast.parse(call_str.strip(), mode="eval")
    assert isinstance(tree.body, ast.Call), call_str
    name = tree.body.func.id if isinstance(tree.body.func, ast.Name) else \
        tree.body.func.attr
    args = {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords}
    return {"name": name, "arguments": args}


def ast_gt_to_calls(ast_gt: List[dict]) -> List[Dict[str, Any]]:
    """possible_answer の AST 形式から「許容値の先頭」を選んで呼び出しを作る。"""
    calls = []
    for item in ast_gt:
        if not isinstance(item, dict) or len(item) != 1:
            continue
        (name, arg_spec), = item.items()
        args = {}
        for k, acceptable in (arg_spec or {}).items():
            vals = acceptable if isinstance(acceptable, list) else [acceptable]
            vals = [v for v in vals if v != ""] or vals
            if vals:
                args[k] = vals[0]
        calls.append({"name": name, "arguments": args})
    return calls


def wrap(calls: List[dict]) -> str:
    payload = {"name": TOOL_NAME, "arguments": {"tool_calls": calls}}
    return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"


def wrap_direct(calls: List[dict]) -> str:
    return "\n".join(
        f"<tool_call>\n{json.dumps(c, ensure_ascii=False)}\n</tool_call>" for c in calls)


async def tool_path_reward(create_kwargs: dict, turns: List[List[dict]]) -> float:
    """実際に GRPO の報酬になる経路(tool の create/execute/calc_reward)。"""
    tool = BFCLMultiTurnTool(config={}, tool_schema=None)
    iid = await tool.create(**create_kwargs)
    for calls in turns:
        obs, r, m = await tool.execute(iid, {"tool_calls": calls})
        print(f"    execute: process={r:.3f} metrics={m} obs={obs[:120]!r}")
    reward = await tool.calc_reward(iid)
    await tool.release(iid)
    return reward


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("-n", type=int, default=5)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.parquet)

    for i in range(min(args.n, len(df))):
        rec = df.iloc[i]
        extra = dict(rec["extra_info"])
        ck = dict(dict(extra["tools_kwargs"])[TOOL_NAME]["create_kwargs"])
        ds = rec["data_source"]
        print(f"\n=== row {i} data_source={ds} id={ck.get('id')} mode={ck.get('mode')} ===")

        if ck.get("mode") == "ast":
            gt = decode_field(ck.get("ast_ground_truth"), [])
            print(f"  ast_ground_truth 件数: {len(gt)}  (0 なら GT デコードのバグ)")
            calls = ast_gt_to_calls(gt)
            turns = [calls]
        else:
            gt_turns = decode_field(ck.get("ground_truth"), [])
            print(f"  ground_truth ターン数: {len(gt_turns)} "
                  f"involved={decode_field(ck.get('involved_classes'), [])}")
            turns = []
            for t in gt_turns:
                try:
                    turns.append([gt_string_to_call(s) for s in t])
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] GT 文字列を構造化できません: {e}")
                    turns.append([])

        r_tool = asyncio.run(tool_path_reward(ck, turns))
        text_wrapper = "\n".join(wrap(t) for t in turns if t)
        text_direct = "\n".join(wrap_direct(t) for t in turns if t)
        r_text = bfcl_reward.compute_score(ds, text_wrapper,
                                           rec["reward_model"]["ground_truth"], extra)
        r_direct = bfcl_reward.compute_score(ds, text_direct,
                                             rec["reward_model"]["ground_truth"], extra)
        print(f"  A) tool 経路(GRPO 実経路) : {r_tool:.4f}   <- ここが 0 なら報酬側のバグ")
        print(f"  B) text 経路(ラッパ形式)   : {r_text:.4f}")
        print(f"  C) text 経路(実関数直呼び) : {r_direct:.4f}   <- 0 が仕様。実機出力がこの形なら報酬は永遠に 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
