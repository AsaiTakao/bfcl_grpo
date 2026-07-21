# -*- coding: utf-8 -*-
"""
ast_match.py

BFCL シングルターン(AST)カテゴリ用の照合器。

マルチターンは「実行後の状態一致」で採点できる(bfcl_backend.compare_states)が、
シングルターンの simple/multiple/parallel/live_* は実行可能なバックエンドを持たない
(架空 API の schema だけが与えられる)。BFCL 公式もこれらは AST 照合で採点する。

possible_answer の形:
  {"id": "simple_0",
   "ground_truth": [{"func_name": {"arg1": [許容値, ...], "arg2": [...]}}, ...]}
  - 1 要素 = 期待される 1 呼び出し(parallel 系は複数要素)。
  - 各引数の値は「許容値のリスト」。リストに "" を含む引数は省略可(optional)。

注意: これは公式 checker の簡約版。ネスト型の厳密検査や型強制など公式の細目までは
再現していない。目的は「シングルターン能力の退行検知」であり、絶対スコアを公式
リーダーボードと比較することではない。
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

# 引数の許容値リストにこれが含まれていれば「省略可」を意味する(BFCL の慣例)。
_OPTIONAL_MARKER = ""


def normalize_func_name(name: str) -> str:
    """関数名の表記ゆれを吸収する。

    BFCL の live_* は "api.v1.get_weather" のようなドット付き名を含み、
    モデル側/データ側でドットとアンダースコアが揺れる。比較前に正規化する。
    """
    return str(name or "").strip().replace(".", "_").replace("-", "_").lower()


def _values_equal(actual: Any, expected: Any) -> bool:
    """許容値との緩い一致判定。

    - 完全一致
    - 数値は int/float をまたいで比較(1 と 1.0 を同一視)
    - 文字列は前後空白と大文字小文字を無視
    - list/dict は要素ごとに再帰
    """
    # bool は int のサブクラスで True == 1 が成立してしまう。引数の型退行を
    # 見逃さないよう、素の等値比較より先に弾く。
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected

    if actual == expected:
        return True

    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return float(actual) == float(expected)

    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip().lower() == expected.strip().lower()

    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _values_equal(a, e) for a, e in zip(actual, expected)
        )

    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual.keys()) == set(expected.keys()) and all(
            _values_equal(actual[k], expected[k]) for k in actual
        )

    return False


def _arg_matches(actual: Any, acceptable: Any) -> bool:
    """1 引数が許容値リストのいずれかに一致するか。"""
    if not isinstance(acceptable, list):
        # データ側がリストで包んでいない場合にも耐える。
        acceptable = [acceptable]
    return any(_values_equal(actual, exp) for exp in acceptable)


def call_matches(predicted: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """予測呼び出し 1 件が期待呼び出し 1 件に一致するか。

    predicted: {"name": str, "arguments": {...}}
    expected : {"func_name": {"arg": [許容値, ...]}}   (1 キーの dict)
    """
    if not isinstance(expected, dict) or len(expected) != 1:
        return False
    (exp_name, exp_args), = expected.items()

    if normalize_func_name(predicted.get("name")) != normalize_func_name(exp_name):
        return False

    args = predicted.get("arguments")
    if not isinstance(args, dict):
        return False
    if not isinstance(exp_args, dict):
        return False

    # 期待される引数がすべて満たされているか。
    for key, acceptable in exp_args.items():
        if key not in args:
            # 許容値に "" を含むものは optional 扱いで省略を許す。
            if isinstance(acceptable, list) and _OPTIONAL_MARKER in acceptable:
                continue
            return False
        if not _arg_matches(args[key], acceptable):
            return False

    # 期待に無い引数を勝手に足していないか(幻覚引数の検出)。
    if set(args.keys()) - set(exp_args.keys()):
        return False

    return True


def match_score(
    predicted_calls: Sequence[Dict[str, Any]],
    ground_truth: Sequence[Dict[str, Any]],
) -> float:
    """AST 一致スコアを [0.0, 1.0] で返す。

    - 期待呼び出しごとに、未使用の予測呼び出しから一致するものを 1 つ割り当てる
      (parallel 系は順不同のため貪欲マッチング)。
    - スコア = 一致した期待呼び出し数 / 期待呼び出し総数。
    - 予測が期待より多い(余計な呼び出し)場合は、その分だけ減点する。
    """
    gt = [g for g in (ground_truth or []) if isinstance(g, dict)]
    if not gt:
        return 0.0

    pred = [p for p in (predicted_calls or []) if isinstance(p, dict)]
    if not pred:
        return 0.0

    used: set[int] = set()
    matched = 0
    for expected in gt:
        for i, p in enumerate(pred):
            if i in used:
                continue
            if call_matches(p, expected):
                used.add(i)
                matched += 1
                break

    score = matched / len(gt)

    # 余計な呼び出しへのペナルティ(期待数を分母に、超過分を割り引く)。
    extra = len(pred) - len(used)
    if extra > 0:
        score -= extra / len(gt)

    return max(0.0, min(1.0, score))


__all__ = ["match_score", "call_matches", "normalize_func_name"]
