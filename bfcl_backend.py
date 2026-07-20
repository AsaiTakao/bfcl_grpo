# -*- coding: utf-8 -*-
"""
bfcl_backend.py

BFCL multi_turn のバックエンド API クラス(GorillaFileSystem, MessageAPI, ...)を
動的にロード・実行・状態比較するアダプタ層。

bfcl_eval の内部構造(モジュールパス、_load_scenario 等)に依存する箇所を
ここ 1 ファイルに隔離する。bfcl_eval のバージョンで差異が出たら、まずここを直す。

BFCL の実行モデル:
  1) test_entry の involved_classes を初期化し、initial_config を _load_scenario で流し込む。
  2) 各クラスの public メソッド名は 1 シナリオ内で一意 → メソッド名 -> bound method の
     namespace を作る。
  3) モデル/GT の関数呼び出しはこの namespace 上で実行する
     (モデル: name+arguments を直接 dispatch、GT: "func(a=1)" 文字列を eval)。
  4) 全ターン後、stateful クラスの内部状態(__dict__)を GT 実行結果と比較 = outcome。
"""
from __future__ import annotations

import ast
import copy
import importlib
from typing import Any, Dict, List, Tuple

# --- bfcl_eval の involved class -> 実装モジュール対応(フォールバック) -------------
# bfcl_eval.constants から取れれば優先。取れない場合の既知マッピング。
_FUNC_SRC = "bfcl_eval.eval_checker.multi_turn_eval.func_source_code"
_FALLBACK_CLASS_MODULE = {
    "GorillaFileSystem": f"{_FUNC_SRC}.gorilla_file_system",
    "MathAPI": f"{_FUNC_SRC}.math_api",
    "MessageAPI": f"{_FUNC_SRC}.message_api",
    "TwitterAPI": f"{_FUNC_SRC}.posting_api",
    "TicketAPI": f"{_FUNC_SRC}.ticket_api",
    "TradingBot": f"{_FUNC_SRC}.trading_bot",
    "TravelAPI": f"{_FUNC_SRC}.travel_booking",
    "VehicleControlAPI": f"{_FUNC_SRC}.vehicle_control",
}

# 状態を持たない(状態比較の対象外)クラス。BFCL の慣例に合わせる。
STATELESS_CLASSES = {"MathAPI"}


def _class_module_mapping() -> Dict[str, str]:
    """bfcl_eval が提供するマッピングを優先し、無ければフォールバックを返す。"""
    for path, attr in [
        ("bfcl_eval.constants.default_prompts", "CLASS_FILE_PATH_MAPPING"),
        ("bfcl_eval.constants.category_mapping", "CLASS_FILE_PATH_MAPPING"),
    ]:
        try:
            mod = importlib.import_module(path)
            m = getattr(mod, attr, None)
            if isinstance(m, dict) and m:
                return m
        except Exception:
            continue
    return dict(_FALLBACK_CLASS_MODULE)


def load_involved_classes(
    involved_classes: List[str],
    initial_config: Dict[str, Any] | None,
    long_context: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """involved_classes を初期化し、(instances, namespace) を返す。

    instances : {ClassName: instance}
    namespace : {method_name: bound_method}  (呼び出し実行用)
    """
    mapping = _class_module_mapping()
    initial_config = initial_config or {}
    instances: Dict[str, Any] = {}
    namespace: Dict[str, Any] = {}

    for cls_name in involved_classes:
        module_path = mapping.get(cls_name)
        if module_path is None:
            raise KeyError(f"未知の involved class: {cls_name}(マッピング未登録)")
        module = importlib.import_module(module_path)
        cls = getattr(module, cls_name)
        inst = cls()

        # BFCL の各クラスは _load_scenario(config) で初期状態を注入する。
        scenario = copy.deepcopy(initial_config.get(cls_name, {}))
        loader = getattr(inst, "_load_scenario", None)
        if callable(loader):
            try:
                loader(scenario, long_context=long_context)
            except TypeError:
                loader(scenario)  # long_context 引数の無い版

        instances[cls_name] = inst

        # public メソッド(先頭が _ でない callable)を namespace へ。
        for attr_name in dir(inst):
            if attr_name.startswith("_"):
                continue
            attr = getattr(inst, attr_name)
            if callable(attr):
                namespace[attr_name] = attr

    return instances, namespace


def execute_structured_call(
    namespace: Dict[str, Any], name: str, arguments: Dict[str, Any] | None
) -> Tuple[bool, str]:
    """モデルの構造化呼び出し {name, arguments} を実行。(success, rendered) を返す。"""
    arguments = arguments or {}
    fn = namespace.get(name)
    if fn is None:
        return False, f"[error] 未定義の関数: {name}"
    try:
        result = fn(**arguments)
        return True, _render(result)
    except Exception as e:  # noqa: BLE001  実行時例外はそのまま失敗として扱う
        return False, f"[error] {name}: {type(e).__name__}: {e}"


def execute_gt_string(namespace: Dict[str, Any], call_str: str) -> Tuple[bool, str]:
    """GT の呼び出し文字列 'func(a=1, b="x")' を namespace 上で実行する。

    eval は namespace(bound method のみ)を globals に限定して行う。
    リテラル引数のみ想定(BFCL の ground_truth はリテラル)。
    """
    call_str = call_str.strip()
    if not call_str:
        return True, ""
    try:
        # 構文検証: Call 式単体であることを確認してから eval する。
        tree = ast.parse(call_str, mode="eval")
        if not isinstance(tree.body, ast.Call):
            return False, f"[error] Call 式ではない: {call_str}"
        result = eval(  # noqa: S307  namespace 限定・GT はデータ側の信頼入力
            compile(tree, "<gt_call>", "eval"),
            {"__builtins__": {}},
            namespace,
        )
        return True, _render(result)
    except Exception as e:  # noqa: BLE001
        return False, f"[error] gt '{call_str}': {type(e).__name__}: {e}"


def _render(result: Any) -> str:
    """ツール実行結果を文字列化(モデルに返す observation)。"""
    if result is None:
        return "None"
    try:
        return str(result)
    except Exception:  # noqa: BLE001
        return repr(result)


def extract_state(instance: Any) -> Dict[str, Any]:
    """状態比較用にインスタンスの内部状態を取り出す。

    __dict__ から volatile/内部用の属性を除いた deep copy を返す。
    """
    raw = getattr(instance, "__dict__", {})
    state: Dict[str, Any] = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        try:
            state[k] = copy.deepcopy(v)
        except Exception:  # noqa: BLE001  deepcopy 不能なものは repr で近似
            state[k] = repr(v)
    return state


def compare_states(
    model_instances: Dict[str, Any],
    gt_instances: Dict[str, Any],
) -> float:
    """stateful クラスごとに状態一致を判定し、一致割合 [0,1] を返す(= outcome)。"""
    targets = [
        c
        for c in model_instances
        if c not in STATELESS_CLASSES and c in gt_instances
    ]
    if not targets:
        return 0.0
    matched = 0
    for cls_name in targets:
        a = extract_state(model_instances[cls_name])
        b = extract_state(gt_instances[cls_name])
        if a == b:
            matched += 1
    return matched / len(targets)


__all__ = [
    "STATELESS_CLASSES",
    "load_involved_classes",
    "execute_structured_call",
    "execute_gt_string",
    "extract_state",
    "compare_states",
]
