# -*- coding: utf-8 -*-
"""
bfcl_dataset_and_interaction.py

(1) build_verl_dataset : BFCL v3 multi_turn の question + possible_answer を結合し、
    verl の multi_turn(tool + interaction)ロールアウトが読む行を作る。
(2) BFCLInteraction     : 事前定義された user 発話を順に供給する verl の user 役。

BFCL multi_turn の 1 エントリ:
  {
    "id": "multi_turn_base_0",
    "question": [[{"role":"user","content":"..."}], [...], ...],  # ターン列
    "involved_classes": ["GorillaFileSystem", ...],
    "initial_config": {"GorillaFileSystem": {...}, ...},
  }
possible_answer 側:
  {"id": "...", "ground_truth": [["func(a=1)", ...], [...], ...]}  # ターンごとの正解呼び出し
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    from verl.interactions.base import BaseInteraction  # type: ignore
except Exception:  # noqa: BLE001  ローカル単体テスト用フォールバック
    class BaseInteraction:  # minimal stub
        def __init__(self, config):
            self.config = config
            self.name = config.get("name", "bfcl") if isinstance(config, dict) else "bfcl"


VERSION_PREFIX = "BFCL_v3"
TOOL_NAME = "bfcl_multi_turn_call"       # bfcl_tool_config.yaml の function.name と一致
INTERACTION_NAME = "bfcl"                # interaction_kwargs["name"] と一致させる


# ============================================================ データセット構築

def _read_json_or_jsonl(path: str) -> List[dict]:
    """BFCL のデータは JSONL のことが多いが、JSON 配列にも耐えるよう両対応。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _load_func_docs(bfcl_data_dir: str, involved_classes: List[str]) -> List[dict]:
    """involved_classes の関数ドキュメント(OpenAI 互換 schema)を結合して返す。

    見つからないクラスは黙ってスキップ(バージョン差で場所が変わるため)。
    """
    docs: List[dict] = []
    doc_dir = os.path.join(bfcl_data_dir, "multi_turn_func_doc")
    for cls in involved_classes:
        p = os.path.join(doc_dir, f"{cls}.json")
        if os.path.exists(p):
            try:
                docs.extend(_read_json_or_jsonl(p))
            except Exception:  # noqa: BLE001
                continue
    return docs


def _system_prompt(func_docs: List[dict]) -> str:
    """利用可能な BFCL 関数一覧を明示し、ラッパツールでの呼び出し方を指示する。"""
    lines = [
        "あなたはツールを使ってユーザーの要求を段階的に達成するエージェントです。",
        "利用可能なバックエンド関数は以下です(名前と引数):",
        "",
    ]
    for d in func_docs:
        name = d.get("name", "?")
        desc = (d.get("description", "") or "").strip().replace("\n", " ")
        params = d.get("parameters", {}).get("properties", {})
        arg_names = ", ".join(params.keys())
        lines.append(f"- {name}({arg_names}): {desc}")
    lines += [
        "",
        f"1 ターンで必要な呼び出しを、ツール `{TOOL_NAME}` の引数",
        '`tool_calls`(= [{"name": 関数名, "arguments": {引数}} , ...])として渡してください。',
        "情報が不足している場合は勝手に仮定せず、ユーザーに質問してください。",
    ]
    return "\n".join(lines)


def _turn_user_text(turn: Any) -> str:
    """question の 1 ターン(メッセージ list)から user 発話テキストを取り出す。"""
    if isinstance(turn, str):
        return turn
    if isinstance(turn, list):
        parts = [
            m.get("content", "")
            for m in turn
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        return "\n".join(p for p in parts if p)
    if isinstance(turn, dict):
        return turn.get("content", "")
    return ""


def is_multi_turn_category(category: str) -> bool:
    """multi_turn 系(状態ベース採点)か、シングルターン系(AST 採点)かを判定する。"""
    return category.startswith("multi_turn")


def build_verl_dataset(category: str, bfcl_data_dir: str) -> List[dict]:
    """指定カテゴリの verl 行 list を返す。prepare_bfcl_data.py から呼ばれる。

    multi_turn 系は状態ベース採点、それ以外(simple / multiple / parallel / live_*)は
    AST 採点の行を作る。どちらも同じラッパツール(TOOL_NAME)を使うため、
    モデルから見た出力フォーマットは 1 つに保たれる。
    """
    if not is_multi_turn_category(category):
        return _build_single_turn_dataset(category, bfcl_data_dir)

    q_path = os.path.join(bfcl_data_dir, f"{VERSION_PREFIX}_{category}.json")
    a_path = os.path.join(
        bfcl_data_dir, "possible_answer", f"{VERSION_PREFIX}_{category}.json"
    )
    questions = _read_json_or_jsonl(q_path)
    answers = {a["id"]: a for a in _read_json_or_jsonl(a_path)}

    rows: List[dict] = []
    for entry in questions:
        eid = entry["id"]
        turns = entry.get("question", [])
        involved = entry.get("involved_classes", []) or []
        initial_config = entry.get("initial_config", {}) or {}
        long_context = "long_context" in category
        gt = answers.get(eid, {}).get("ground_truth", [])

        func_docs = _load_func_docs(bfcl_data_dir, involved)
        first_user = _turn_user_text(turns[0]) if turns else ""
        remaining_turns = [_turn_user_text(t) for t in turns[1:]]

        prompt = [
            {"role": "system", "content": _system_prompt(func_docs)},
            {"role": "user", "content": first_user},
        ]

        create_kwargs = {
            "mode": "state",             # 状態ベース採点(BFCLMultiTurnTool が分岐)
            "involved_classes": involved,
            "initial_config": initial_config,
            "ground_truth": gt,
            "long_context": long_context,
            "id": eid,
        }
        rows.append(
            {
                "data_source": "bfcl_multi_turn",
                "prompt": prompt,
                "ability": "tool_use",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
                "extra_info": {
                    "index": eid,
                    "category": category,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        TOOL_NAME: {"create_kwargs": create_kwargs},
                    },
                    "interaction_kwargs": {
                        "name": INTERACTION_NAME,
                        "user_turns": remaining_turns,
                        "id": eid,
                    },
                },
            }
        )
    return rows


def _build_single_turn_dataset(category: str, bfcl_data_dir: str) -> List[dict]:
    """シングルターン(AST 採点)カテゴリの verl 行 list を返す。

    multi_turn との違い:
      - 関数 schema は involved_classes ではなくエントリ内の "function" に入っている
        (実行可能な実装は存在せず、schema だけが与えられる架空 API)。
      - possible_answer は状態ではなく期待呼び出しの AST
        ([{"func_name": {"arg": [許容値, ...]}}, ...])。
      - user 発話は 1 ターンのみ → interaction_kwargs.user_turns は空。

    シングルターンを学習/検証に混ぜる狙いは、multi_turn 特化 RL による
    既存のシングルターン function calling 能力の退行(破滅的忘却)を
    抑制・検知すること。
    """
    q_path = os.path.join(bfcl_data_dir, f"{VERSION_PREFIX}_{category}.json")
    a_path = os.path.join(
        bfcl_data_dir, "possible_answer", f"{VERSION_PREFIX}_{category}.json"
    )
    questions = _read_json_or_jsonl(q_path)
    answers = {a["id"]: a for a in _read_json_or_jsonl(a_path)}

    rows: List[dict] = []
    for entry in questions:
        eid = entry["id"]
        turns = entry.get("question", [])
        func_docs = entry.get("function", []) or []
        if isinstance(func_docs, dict):      # 単一関数が dict で来る版に耐える
            func_docs = [func_docs]

        gt = answers.get(eid, {}).get("ground_truth", [])
        if not gt:
            # 正解が引けないエントリは報酬が常に 0 になるので学習に混ぜない。
            continue

        first_user = _turn_user_text(turns[0]) if turns else ""
        if not first_user:
            continue

        prompt = [
            {"role": "system", "content": _system_prompt(func_docs)},
            {"role": "user", "content": first_user},
        ]

        create_kwargs = {
            "mode": "ast",               # AST 採点(BFCLMultiTurnTool が分岐)
            "ast_ground_truth": gt,
            "id": eid,
        }
        rows.append(
            {
                "data_source": "bfcl_single_turn",   # val 指標を MT と分離するため別名
                "prompt": prompt,
                "ability": "tool_use",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
                "extra_info": {
                    "index": eid,
                    "category": category,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        TOOL_NAME: {"create_kwargs": create_kwargs},
                    },
                    "interaction_kwargs": {
                        "name": INTERACTION_NAME,
                        "user_turns": [],     # 追加の user 発話は無い(1 ターン)
                        "id": eid,
                    },
                },
            }
        )
    return rows


# ================================================================ user 役

class BFCLInteraction(BaseInteraction):
    """事前定義された user ターンを 1 つずつ供給する。ターンが尽きたら終了。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.name = (config or {}).get("name", INTERACTION_NAME)
        self._state: Dict[str, Dict[str, Any]] = {}

    async def start_interaction(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> str:
        from uuid import uuid4

        instance_id = instance_id or str(uuid4())
        self._state[instance_id] = {
            "user_turns": list(kwargs.get("user_turns", []) or []),
            "idx": 0,
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """次の user 発話を返す。(should_terminate, response, score, metadata)"""
        st = self._state.get(instance_id)
        if st is None:
            return True, "", 0.0, {}

        turns = st["user_turns"]
        idx = st["idx"]
        if idx >= len(turns):
            # 供給すべき user 発話が尽きた → 対話終了。
            return True, "", 0.0, {"turns_consumed": idx}

        st["idx"] = idx + 1
        response = turns[idx]
        # 最後のターンを渡したらこれで終了フラグを立てる。
        should_terminate = st["idx"] >= len(turns)
        return should_terminate, response, 0.0, {"turn": idx}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        # 報酬は BFCLMultiTurnTool.calc_reward 側で計上するため、ここでは 0。
        return 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._state.pop(instance_id, None)


__all__ = ["build_verl_dataset", "is_multi_turn_category", "BFCLInteraction"]
