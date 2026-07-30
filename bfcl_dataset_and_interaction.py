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
import re
from typing import Any, Dict, List, Optional, Tuple

import bfcl_backend as backend

try:
    from verl.interactions.base import BaseInteraction  # type: ignore
except Exception:  # noqa: BLE001  ローカル単体テスト用フォールバック
    class BaseInteraction:  # minimal stub
        def __init__(self, config):
            self.config = config
            self.name = config.get("name", "bfcl") if isinstance(config, dict) else "bfcl"


VERSION_PREFIX = "BFCL_v3"               # 検出できなかった場合のフォールバック
TOOL_NAME = "bfcl_multi_turn_call"       # bfcl_tool_config.yaml の function.name と一致
INTERACTION_NAME = "bfcl"                # interaction_kwargs["name"] と一致させる

_PREFIX_RE = re.compile(r"^(BFCL_v\d+)_.+\.json$")


class FuncDocNotFoundError(RuntimeError):
    """involved class の関数ドキュメントが解決できない。

    カテゴリ欠損(バージョン差)とは違い、これは黙って飛ばしてはいけない。
    prepare_bfcl_data._load は OSError / KeyError / ValueError を「そのカテゴリは
    存在しない」として警告付きスキップするため、その網に掛からない型にしている。
    """


def detect_version_prefix(bfcl_data_dir: str) -> str:
    """データディレクトリ実物から BFCL_vN プレフィックスを検出する。

    bfcl_eval のバージョンでファイル名が BFCL_v3_* / BFCL_v4_* と変わるため、
    決め打ちにすると全カテゴリが「ファイルなし」で静かにスキップされる。
    複数バージョンが混在していれば新しい方を採用する。
    """
    found = set()
    try:
        for fn in os.listdir(bfcl_data_dir):
            m = _PREFIX_RE.match(fn)
            if m:
                found.add(m.group(1))
    except OSError:
        return VERSION_PREFIX
    if not found:
        return VERSION_PREFIX
    return max(found, key=lambda p: int(p.rsplit("v", 1)[1]))


def available_categories(bfcl_data_dir: str, version_prefix: str) -> List[str]:
    """そのディレクトリで実際に読めるカテゴリ名の一覧(エラーメッセージ用)。"""
    out = []
    try:
        entries = os.listdir(bfcl_data_dir)
    except OSError:
        return out
    head = f"{version_prefix}_"
    for fn in sorted(entries):
        if fn.startswith(head) and fn.endswith(".json"):
            out.append(fn[len(head):-len(".json")])
    return out


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


_FUNC_DOC_DIRNAME = "multi_turn_func_doc"

# 読み込んだ func doc のキャッシュ(パス -> schema list)。同じクラスが数百
# エントリで再登場するため、毎回 JSON を読み直すのを避ける。
_FUNC_DOC_CACHE: Dict[str, List[dict]] = {}


def _camel_to_snake(name: str) -> str:
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def _doc_basenames(cls_name: str) -> List[str]:
    """involved class 名から func doc ファイル名(拡張子なし)の候補を優先順に返す。

    BFCL の multi_turn_func_doc は「クラス名」ではなく「実装モジュール名」で
    置かれている:
        GorillaFileSystem  -> gorilla_file_system.json
        TwitterAPI         -> posting_api.json
        TravelAPI          -> travel_booking.json
        VehicleControlAPI  -> vehicle_control.json
    以前は `{クラス名}.json` を探して必ず外し、しかも欠損を黙ってスキップして
    いたため、全エントリが「関数一覧が空」の system prompt で学習していた
    (モデルは関数名を推測するしかなく undefined function を量産する)。

    候補の優先順:
      1) bfcl_eval の CLASS_FILE_PATH_MAPPING のモジュール名(実データと一致)
      2) クラス名の機械的な snake_case 化(将来の新クラス向け)
      3) クラス名そのまま
    """
    cands = []
    module_path = backend.class_module_mapping().get(cls_name)
    if module_path:
        # 値は dotted module path("...func_source_code.gorilla_file_system")だが、
        # バージョンによってファイルパス表記のこともあるので両方から名前を取る。
        stem = os.path.basename(str(module_path).replace("\\", "/"))
        if stem.endswith(".py"):
            stem = stem[: -len(".py")]
        cands.append(stem.rsplit(".", 1)[-1])
    cands.append(_camel_to_snake(cls_name))
    cands.append(cls_name)
    # 重複を除きつつ順序は維持する。
    return list(dict.fromkeys(c for c in cands if c))


def func_doc_dir(bfcl_data_dir: str) -> Optional[str]:
    """multi_turn_func_doc ディレクトリの実パス。見つからなければ None。"""
    parent = os.path.dirname(os.path.normpath(bfcl_data_dir))
    for cand in (os.path.join(bfcl_data_dir, _FUNC_DOC_DIRNAME),
                 os.path.join(parent, _FUNC_DOC_DIRNAME)):
        if os.path.isdir(cand):
            return cand
    return None


def _load_func_docs(bfcl_data_dir: str,
                    involved_classes: List[str]) -> Tuple[List[dict], List[str]]:
    """involved_classes の関数ドキュメント(OpenAI 互換 schema)を結合する。

    戻り値は (schema list, 解決できなかったクラス名 list)。
    """
    doc_dir = func_doc_dir(bfcl_data_dir)
    docs: List[dict] = []
    missing: List[str] = []
    for cls in involved_classes:
        loaded: Optional[List[dict]] = None
        for base in ([] if doc_dir is None else _doc_basenames(cls)):
            path = os.path.join(doc_dir, f"{base}.json")
            if path in _FUNC_DOC_CACHE:
                loaded = _FUNC_DOC_CACHE[path]
                break
            if not os.path.exists(path):
                continue
            try:
                loaded = [d for d in _read_json_or_jsonl(path) if isinstance(d, dict)]
            except (OSError, json.JSONDecodeError):
                continue
            _FUNC_DOC_CACHE[path] = loaded
            break
        if loaded:
            docs.extend(loaded)
        else:
            missing.append(cls)
    return docs, missing


def _require_func_docs(bfcl_data_dir: str, involved_classes: List[str],
                       entry_id: str) -> List[dict]:
    """func doc を読む。1 クラスでも解決できなければ即座に落とす。

    黙ってスキップすると「関数一覧が空のプロンプト」で学習が完走してしまい、
    undefined function だらけの報酬 0 ロールアウトに GPU 時間を丸ごと溶かす。
    データ準備の時点で止める方が安い。
    """
    docs, missing = _load_func_docs(bfcl_data_dir, involved_classes)
    if not missing:
        return docs

    doc_dir = func_doc_dir(bfcl_data_dir)
    tried = [f"{b}.json" for c in missing for b in _doc_basenames(c)]
    if doc_dir is None:
        where = (f"{os.path.join(bfcl_data_dir, _FUNC_DOC_DIRNAME)} "
                 "(ディレクトリ自体が見つかりません)")
        listing: List[str] = []
    else:
        where = doc_dir
        listing = sorted(f for f in os.listdir(doc_dir) if f.endswith(".json"))
    raise FuncDocNotFoundError(
        f"[{entry_id}] 関数ドキュメントを解決できないクラスがあります: {missing}\n"
        f"  探した場所: {where}\n"
        f"  試したファイル名: {tried}\n"
        f"  実在するファイル: {listing}\n"
        "関数一覧が空の system prompt で学習すると、モデルは関数名を推測するしか"
        "なく undefined function を量産します。--bfcl-data-dir を確認してください。"
    )


# 1 関数あたりの description の上限。BFCL の description は引数の説明まで含んで
# 長いものがあり、そのまま全関数を並べると max_prompt_length を超える。超えた行は
# filter_overlong_prompts に静かに落とされて学習から消えるため、ここで抑える。
_MAX_DESC_CHARS = 240


def _format_func_doc(d: dict) -> str:
    """関数 schema を 1 行のシグネチャに落とす。

    引数名だけでなく型と必須/省略可も出す。名前だけだと引数の当てずっぽうが
    残り、実行時例外(= process 報酬 0)になる。
    """
    name = d.get("name", "?")
    params = d.get("parameters") if isinstance(d.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    required = set(params.get("required") or [])

    args = []
    for arg_name, spec in (props or {}).items():
        typ = (spec.get("type") if isinstance(spec, dict) else None) or "any"
        # 省略可能な引数は "名前?" と表記する(表記の意味は system prompt 冒頭で説明)。
        args.append(f"{arg_name}{'' if arg_name in required else '?'}: {typ}")

    desc = " ".join((d.get("description") or "").split())
    if len(desc) > _MAX_DESC_CHARS:
        desc = desc[:_MAX_DESC_CHARS].rstrip() + "…"
    sig = f"- {name}({', '.join(args)})"
    return f"{sig}: {desc}" if desc else sig


def _system_prompt(func_docs: List[dict]) -> str:
    """利用可能な BFCL 関数一覧を明示し、ラッパツールでの呼び出し方を指示する。"""
    if not func_docs:
        # ここを通してしまうと「関数一覧が空」のまま学習が走る。呼び出し側
        # (_require_func_docs / シングルターンの空 schema スキップ)で防ぐ前提。
        raise ValueError("func_docs が空です: 関数一覧なしのプロンプトは作れません")

    lines = [
        "あなたはツールを使ってユーザーの要求を段階的に達成するエージェントです。",
        "利用可能なバックエンド関数は以下です(`引数名?` は省略可能な引数):",
        "",
    ]
    lines += [_format_func_doc(d) for d in func_docs]
    lines += [
        "",
        f"1 ターンで必要な呼び出しを、ツール `{TOOL_NAME}` の引数",
        '`tool_calls`(= [{"name": 関数名, "arguments": {引数}} , ...])として渡してください。',
        "関数名は必ず上の一覧から選んでください(一覧に無い名前はエラーになります)。",
        "情報が不足している場合は勝手に仮定せず、ユーザーに質問してください。",
        # 応答長の打ち切り対策。think が長いと tool_call まで到達せず、
        # 壊れた出力は報酬 0 になって学習を濁す。
        "思考は簡潔に(数文以内)。迷い続けず、まず必要な関数呼び出しを出してください。",
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


def encode_field(value: Any) -> str:
    """parquet に載せられない自由形式の値を JSON 文字列にする。

    なぜ必要か:
      initial_config はバックエンドクラスごとに構造が違い、ast_ground_truth は
      関数名がキーになる。行ごとにスキーマが変わるため pyarrow は単一の struct
      型に落とせず "cannot mix list and non-list" で落ちる。さらに parquet から
      読み戻したリストは numpy 配列になり、`x or []` のような真偽評価が
      例外になる。文字列で持てば両方回避できる。
    """
    return json.dumps(value, ensure_ascii=False)


def decode_field(value: Any, default: Any) -> Any:
    """encode_field で書いた値を戻す。既に構造体ならそのまま返す。

    parquet 経由(文字列)と、テストや直接呼び出し(生の dict/list)の
    両方を受け付ける。
    """
    if value is None:
        return default
    if isinstance(value, str):
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def is_multi_turn_category(category: str) -> bool:
    """multi_turn 系(状態ベース採点)か、シングルターン系(AST 採点)かを判定する。"""
    return category.startswith("multi_turn")


def build_verl_dataset(category: str, bfcl_data_dir: str,
                       version_prefix: Optional[str] = None) -> List[dict]:
    """指定カテゴリの verl 行 list を返す。prepare_bfcl_data.py から呼ばれる。

    multi_turn 系は状態ベース採点、それ以外(simple_* / multiple / parallel / live_*)は
    AST 採点の行を作る。どちらも同じラッパツール(TOOL_NAME)を使うため、
    モデルから見た出力フォーマットは 1 つに保たれる。

    version_prefix を省略するとデータディレクトリから自動検出する。
    """
    prefix = version_prefix or detect_version_prefix(bfcl_data_dir)
    if not is_multi_turn_category(category):
        return _build_single_turn_dataset(category, bfcl_data_dir, prefix)

    q_path = os.path.join(bfcl_data_dir, f"{prefix}_{category}.json")
    a_path = os.path.join(
        bfcl_data_dir, "possible_answer", f"{prefix}_{category}.json"
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

        func_docs = _require_func_docs(bfcl_data_dir, involved, eid)
        first_user = _turn_user_text(turns[0]) if turns else ""
        remaining_turns = [_turn_user_text(t) for t in turns[1:]]

        prompt = [
            {"role": "system", "content": _system_prompt(func_docs)},
            {"role": "user", "content": first_user},
        ]

        # parquet のスキーマを両モードで揃えるため、キー構成は state/ast で共通にし、
        # 自由形式の値はすべて JSON 文字列で持つ(encode_field の docstring 参照)。
        create_kwargs = {
            "mode": "state",             # 状態ベース採点(BFCLMultiTurnTool が分岐)
            "involved_classes": encode_field(involved),
            "initial_config": encode_field(initial_config),
            "ground_truth": encode_field(gt),
            "ast_ground_truth": encode_field([]),
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
                        "user_turns": encode_field(remaining_turns),
                        "id": eid,
                    },
                },
            }
        )
    return rows


def _build_single_turn_dataset(category: str, bfcl_data_dir: str,
                               version_prefix: str) -> List[dict]:
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
    q_path = os.path.join(bfcl_data_dir, f"{version_prefix}_{category}.json")
    a_path = os.path.join(
        bfcl_data_dir, "possible_answer", f"{version_prefix}_{category}.json"
    )
    questions = _read_json_or_jsonl(q_path)
    answers = {a["id"]: a for a in _read_json_or_jsonl(a_path)}

    rows: List[dict] = []
    n_no_docs = 0
    for entry in questions:
        eid = entry["id"]
        turns = entry.get("question", [])
        func_docs = entry.get("function", []) or []
        if isinstance(func_docs, dict):      # 単一関数が dict で来る版に耐える
            func_docs = [func_docs]
        func_docs = [d for d in func_docs if isinstance(d, dict) and d.get("name")]
        if not func_docs:
            # 関数一覧が空のプロンプトはモデルに名前を推測させるだけで、
            # 報酬 0 の undefined function しか生まない。学習に混ぜない。
            n_no_docs += 1
            continue

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

        # キー構成は multi_turn 側と揃える(parquet のスキーマ統一のため)。
        create_kwargs = {
            "mode": "ast",               # AST 採点(BFCLMultiTurnTool が分岐)
            "involved_classes": encode_field([]),
            "initial_config": encode_field({}),
            "ground_truth": encode_field([]),
            "ast_ground_truth": encode_field(gt),
            "long_context": False,
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
                        # 追加の user 発話は無い(1 ターン)
                        "user_turns": encode_field([]),
                        "id": eid,
                    },
                },
            }
        )
    if n_no_docs:
        print(f"[warn] {category}: 関数 schema が無い {n_no_docs} 件を除外しました")
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
            # parquet 経由では JSON 文字列で届く(encode_field 参照)。
            "user_turns": list(decode_field(kwargs.get("user_turns"), [])),
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


__all__ = ["build_verl_dataset", "is_multi_turn_category",
           "detect_version_prefix", "available_categories", "func_doc_dir",
           "FuncDocNotFoundError", "BFCLInteraction"]
