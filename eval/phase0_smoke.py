# -*- coding: utf-8 -*-
"""
phase0_smoke.py

フェーズ0(mock + dummy_user)。外部 API 不要・高速。CP ごと毎回回す。

目的(設計書「確認すべき技術的ポイント」):
  Qwen3 の thinking を評価でも発火させたうえで、<think> が hermes パーサの
  tool_call 抽出を壊さないかを検出する。壊れていれば非ゼロ終了し、
  vLLM の --reasoning-parser 追加を促す。

やること:
  - vLLM OpenAI 互換エンドポイントに、BFCL 風のツール定義付きで問い合わせる。
  - dummy_user の 2 ターンを回し、各ターンで
      (a) tool_calls が構造化されて取れているか(arguments が正しい JSON か)
      (b) think が本文に漏れて tool_call 抽出を壊していないか
    を判定する。
  - 破損パターン(<think> が content に残り tool_calls が空)を検出したら FAIL。

終了コード: 0=健全 / 1=破損検出 / 2=接続不可等。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from dummy_user import default_dummy_user, STOP_TOKEN

# BFCL 風の最小ツール定義(名前だけ合わせた mock。実実行はしない)。
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "cat",
            "description": "指定ファイルの内容を返す。",
            "parameters": {
                "type": "object",
                "properties": {"file_name": {"type": "string"}},
                "required": ["file_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "指定の相手にメッセージを送る。",
            "parameters": {
                "type": "object",
                "properties": {
                    "receiver": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["receiver", "message"],
            },
        },
    },
]

SYSTEM = (
    "あなたはツールを使うエージェントです。必要な操作はツール呼び出しで行い、"
    "推測で答えないでください。"
)


def _has_think_leak(content: str | None) -> bool:
    """本文に <think> タグが残っている(=reasoning が分離されていない)か。"""
    if not content:
        return False
    return "<think>" in content or "</think>" in content


def _check_tool_calls(msg: Any) -> tuple[bool, str]:
    """message.tool_calls が取れて arguments が JSON として妥当かを見る。"""
    tool_calls = getattr(msg, "tool_calls", None) or []
    if not tool_calls:
        return False, "tool_calls が空"
    for tc in tool_calls:
        try:
            args = tc.function.arguments
            json.loads(args) if isinstance(args, str) else args
        except Exception as e:  # noqa: BLE001
            return False, f"arguments が不正 JSON: {e}"
    return True, f"tool_calls={len(tool_calls)} 件 OK"


def run(base_url: str, model: str, api_key: str, max_turns: int) -> int:
    try:
        from openai import OpenAI
    except Exception as e:  # noqa: BLE001
        print(f"[phase0] openai パッケージが必要です: {e}", file=sys.stderr)
        return 2

    client = OpenAI(base_url=base_url, api_key=api_key)
    user = default_dummy_user()
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM}]

    ok_all = True
    broken_parse = False
    turn = 0
    # dummy_user のターンを供給しながらエージェントの応答パースを検査する。
    next_user = user.respond()
    while next_user != STOP_TOKEN and turn < max_turns:
        messages.append({"role": "user", "content": next_user})
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS,
                temperature=0.6, max_tokens=1024,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[phase0] 推論リクエスト失敗: {e}", file=sys.stderr)
            return 2

        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None)
        tc_ok, tc_info = _check_tool_calls(msg)
        leaked = _has_think_leak(content)

        print(f"[phase0] turn={turn} think={'有' if reasoning else '無/本文'} "
              f"leak={leaked} tool={tc_info}")

        # 破損パターン: think が本文に漏れ、かつ tool_calls が取れていない。
        if leaked and not tc_ok:
            broken_parse = True
            ok_all = False
            print("[phase0] 破損検出: <think> が本文に残り tool_call 抽出が失敗。")
        elif not tc_ok:
            ok_all = False  # ツールを呼ぶべき場面で呼べていない(パース以外の要因も)

        # 会話を進める(assistant 応答を積み、次の user ターンへ)。
        # tool_calls が無いときはキー自体を省く(null を送ると弾く互換サーバがある)。
        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content}
        tc_list = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name,
                          "arguments": tc.function.arguments}}
            for tc in (getattr(msg, "tool_calls", None) or [])
        ]
        if tc_list:
            assistant_msg["tool_calls"] = tc_list
        messages.append(assistant_msg)
        # mock なのでツール結果はダミーを返して次へ。
        for tc in (getattr(msg, "tool_calls", None) or []):
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": "(mock result)",
            })
        turn += 1
        next_user = user.respond()

    if broken_parse:
        print("[phase0] FAIL: think と tool_call のパースが干渉しています。")
        print("         → vLLM 起動に --reasoning-parser qwen3 を付けて再確認してください。")
        return 1
    if not ok_all:
        print("[phase0] WARN: tool_call が期待どおり取れないターンがありました"
              "(パース以外の要因の可能性)。")
        return 1
    print("[phase0] PASS: think 発火下でも tool_call 抽出は健全です。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.getenv("AGENT_BASE_URL",
                                                    "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default=os.getenv("AGENT_MODEL", "qwen3-8b-bfcl"))
    ap.add_argument("--api-key", default=os.getenv("AGENT_API_KEY", "EMPTY"))
    ap.add_argument("--max-turns", type=int, default=4)
    args = ap.parse_args()
    return run(args.base_url, args.model, args.api_key, args.max_turns)


if __name__ == "__main__":
    raise SystemExit(main())
