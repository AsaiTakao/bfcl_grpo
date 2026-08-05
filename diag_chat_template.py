# -*- coding: utf-8 -*-
"""diag_chat_template.py — 学習で実際に使われるプロンプト文字列を目で見る。

目的
----
「モデルにツールが見えているか」「tool 応答が正しいロールで戻るか」を
GPU も verl も動かさずに確認する。verl(sglang rollout)は
  tokenizer.apply_chat_template(messages, tools=[...], add_generation_prompt=True)
で 1 ターン目を組み立て、以降 assistant の tool_call と role="tool" の
observation を追記していく。ここではその流れを手で再現する。

チェック項目:
  1) 描画結果に bfcl_multi_turn_call の schema(# Tools セクション)が入るか
     -> 入らなければモデルはツールを知らないまま。ツール名不一致の主因になる。
  2) system prompt の関数一覧(BFCL の実関数)と tools セクションが両立し、
     モデルが「どちらの名前で呼ぶべきか」区別できる書き方になっているか。
  3) assistant(tool_call) -> role="tool" の observation が
     <tool_response> で包まれて user ロールに入るか(Qwen3 hermes の仕様)。
  4) 末尾が <|im_start|>assistant で終わり、think が強制無効化されていないか。

使い方:
    python diag_chat_template.py --model Qwen/Qwen3-8B \
        --parquet /workspace/data/bfcl_mt_val.parquet --row 0
"""
from __future__ import annotations

import argparse
import json

import yaml
from transformers import AutoTokenizer


def load_prompt_from_parquet(path: str, row: int):
    import pandas as pd

    df = pd.read_parquet(path)
    rec = df.iloc[row]
    prompt = rec["prompt"]
    prompt = [dict(m) for m in prompt]
    extra = rec.get("extra_info")
    return prompt, (dict(extra) if extra is not None else {}), rec.get("data_source")


def tool_schemas(tool_config_path: str):
    with open(tool_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [{"type": "function", "function": t["tool_schema"]["function"]}
            for t in cfg["tools"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--tool-config", default="bfcl_tool_config.yaml")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("chat_template 有無:", bool(getattr(tok, "chat_template", None)))

    prompt, extra, data_source = load_prompt_from_parquet(args.parquet, args.row)
    tools = tool_schemas(args.tool_config)
    print(f"data_source={data_source} tools={[t['function']['name'] for t in tools]}")

    rendered = tok.apply_chat_template(
        prompt, tools=tools, add_generation_prompt=True, tokenize=False)
    print("\n########## 1 ターン目にモデルが見る文字列 ##########")
    print(rendered)
    print("###################################################")
    print("tools セクションに wrapper が入っている:",
          "bfcl_multi_turn_call" in rendered)
    print("末尾:", repr(rendered[-80:]))
    print("トークン長:", len(tok(rendered)["input_ids"]))

    # --- 2 ターン目(tool 応答を挟んだ継続)の描画 -------------------------
    call = {"name": "bfcl_multi_turn_call",
            "arguments": {"tool_calls": [{"name": "ls", "arguments": {}}]}}
    cont = prompt + [
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function", "function": {
             "name": call["name"],
             "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}]},
        {"role": "tool", "content": "ls -> ['a.txt']"},
    ]
    rendered2 = tok.apply_chat_template(
        cont, tools=tools, add_generation_prompt=True, tokenize=False)
    print("\n########## 2 ターン目(tool 応答つき) ##########")
    print(rendered2[len(rendered) - 40:])
    print("###############################################")
    print("<tool_response> で包まれている:", "<tool_response>" in rendered2)
    print("1 ターン目が prefix として保存されている(トークン整合):",
          rendered2.startswith(rendered[: rendered.rfind("<|im_start|>assistant")]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
