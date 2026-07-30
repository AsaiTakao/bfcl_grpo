# -*- coding: utf-8 -*-
"""
system prompt の関数一覧と、応答長の予算まわりのユニットテスト。

なぜ: SMOKE テストで以下の 2 点が判明した。
  1. 関数一覧が空のまま学習していた。multi_turn_func_doc は「クラス名」ではなく
     「実装モジュール名」でファイルが置かれている(GorillaFileSystem ->
     gorilla_file_system.json, TwitterAPI -> posting_api.json)のに
     `{クラス名}.json` を探し、欠損を黙ってスキップしていた。モデルは関数名を
     推測するしかなく undefined function が 114 件出ていた。
  2. 応答が 19〜47% 打ち切られていた(tool 応答も応答長予算を食う)。

固定する仕様:
  - クラス名 -> 実ファイル名の解決(SMOKE で確認した実配置)
  - 解決できないクラスは「静かに空プロンプト」ではなく即エラー
  - プロンプトに関数名・引数型・必須/省略可・一覧外禁止の指示が載る
  - observation は上限文字数で切り、切ったことを明示する

実行: python -m pytest tests/test_func_doc_prompt.py
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import bfcl_backend as backend
import bfcl_dataset_and_interaction as di


# SMOKE に記載された実配置(involved_classes -> 実ファイル名)。
SMOKE_DOC_FILES = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "TwitterAPI": "posting_api.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


class DocFileNameTest(unittest.TestCase):
    def test_resolves_smoke_filenames(self) -> None:
        # ここが外れていたのが第1の問題の根本原因。
        for cls, filename in SMOKE_DOC_FILES.items():
            cands = [f"{b}.json" for b in di._doc_basenames(cls)]
            self.assertIn(filename, cands, f"{cls} の候補に {filename} が無い")

    def test_module_name_is_the_first_candidate(self) -> None:
        # bfcl_eval のマッピング由来の名前を最優先に試す(実データと一致する)。
        self.assertEqual(di._doc_basenames("TwitterAPI")[0], "posting_api")

    def test_camel_to_snake_fallback_for_unknown_class(self) -> None:
        # マッピングに無い将来のクラスでも機械変換で拾えるようにしておく。
        self.assertIn("new_shiny_api", di._doc_basenames("NewShinyAPI"))
        self.assertIn("NewShinyAPI", di._doc_basenames("NewShinyAPI"))

    def test_all_mapped_classes_have_candidates(self) -> None:
        for cls in backend.class_module_mapping():
            self.assertTrue(di._doc_basenames(cls), f"{cls} の候補が空")


class LoadFuncDocsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        di._FUNC_DOC_CACHE.clear()
        self.addCleanup(di._FUNC_DOC_CACHE.clear)

    def _doc(self, basename: str, docs) -> None:
        _write_json(
            os.path.join(self.tmp.name, "multi_turn_func_doc", f"{basename}.json"),
            docs,
        )

    def test_loads_docs_placed_under_module_name(self) -> None:
        self._doc("gorilla_file_system", [{"name": "cd"}, {"name": "ls"}])
        docs, missing = di._load_func_docs(self.tmp.name, ["GorillaFileSystem"])
        self.assertEqual([d["name"] for d in docs], ["cd", "ls"])
        self.assertEqual(missing, [])

    def test_merges_multiple_classes(self) -> None:
        self._doc("gorilla_file_system", [{"name": "cd"}])
        self._doc("posting_api", [{"name": "post_tweet"}])
        docs, missing = di._load_func_docs(
            self.tmp.name, ["GorillaFileSystem", "TwitterAPI"])
        self.assertEqual([d["name"] for d in docs], ["cd", "post_tweet"])
        self.assertEqual(missing, [])

    def test_reports_missing_class(self) -> None:
        self._doc("gorilla_file_system", [{"name": "cd"}])
        docs, missing = di._load_func_docs(
            self.tmp.name, ["GorillaFileSystem", "TravelAPI"])
        self.assertEqual(missing, ["TravelAPI"])
        self.assertEqual(len(docs), 1)

    def test_jsonl_docs_are_accepted(self) -> None:
        path = os.path.join(self.tmp.name, "multi_turn_func_doc",
                            "gorilla_file_system.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"name": "cd"}\n{"name": "ls"}\n')
        docs, missing = di._load_func_docs(self.tmp.name, ["GorillaFileSystem"])
        self.assertEqual([d["name"] for d in docs], ["cd", "ls"])
        self.assertEqual(missing, [])

    def test_missing_doc_dir_reports_all_classes(self) -> None:
        docs, missing = di._load_func_docs(self.tmp.name, ["GorillaFileSystem"])
        self.assertEqual((docs, missing), ([], ["GorillaFileSystem"]))
        self.assertIsNone(di.func_doc_dir(self.tmp.name))


class MultiTurnDatasetDocTest(unittest.TestCase):
    """関数一覧が引けないなら、学習を始める前に落とす。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        di._FUNC_DOC_CACHE.clear()
        self.addCleanup(di._FUNC_DOC_CACHE.clear)

        _write_json(
            os.path.join(self.tmp.name, "BFCL_v3_multi_turn_base.json"),
            [{
                "id": "multi_turn_base_0",
                "question": [[{"role": "user", "content": "ファイルを見せて"}],
                             [{"role": "user", "content": "削除して"}]],
                "involved_classes": ["GorillaFileSystem"],
                "initial_config": {"GorillaFileSystem": {"root": {}}},
            }],
        )
        _write_json(
            os.path.join(self.tmp.name, "possible_answer",
                         "BFCL_v3_multi_turn_base.json"),
            [{"id": "multi_turn_base_0", "ground_truth": [["ls()"], ["rm(f='a')"]]}],
        )

    def _write_docs(self) -> None:
        _write_json(
            os.path.join(self.tmp.name, "multi_turn_func_doc",
                         "gorilla_file_system.json"),
            [
                {"name": "ls", "description": "ディレクトリを一覧する",
                 "parameters": {"properties": {"a": {"type": "boolean"}},
                                "required": []}},
                {"name": "rm", "description": "ファイルを削除する",
                 "parameters": {"properties": {"file_name": {"type": "string"}},
                                "required": ["file_name"]}},
            ],
        )

    def test_prompt_lists_involved_class_functions(self) -> None:
        self._write_docs()
        system = di.build_verl_dataset(
            "multi_turn_base", self.tmp.name)[0]["prompt"][0]["content"]
        self.assertIn("- ls(", system)
        self.assertIn("- rm(", system)
        self.assertIn("ファイルを削除する", system)

    def test_prompt_marks_required_and_optional_args(self) -> None:
        self._write_docs()
        system = di.build_verl_dataset(
            "multi_turn_base", self.tmp.name)[0]["prompt"][0]["content"]
        self.assertIn("file_name: string", system)   # 必須
        self.assertIn("a?: boolean", system)         # 省略可は "?" 付き
        self.assertNotIn("file_name?", system)

    def test_prompt_forbids_names_outside_the_list(self) -> None:
        # undefined function を減らすための明示的な制約。
        self._write_docs()
        system = di.build_verl_dataset(
            "multi_turn_base", self.tmp.name)[0]["prompt"][0]["content"]
        self.assertIn("一覧", system)
        self.assertIn(di.TOOL_NAME, system)

    def test_missing_docs_raise_instead_of_empty_prompt(self) -> None:
        # doc を書かずに構築 → 黙って空一覧のプロンプトを作らせない。
        with self.assertRaises(di.FuncDocNotFoundError) as ctx:
            di.build_verl_dataset("multi_turn_base", self.tmp.name)
        msg = str(ctx.exception)
        self.assertIn("GorillaFileSystem", msg)
        self.assertIn("gorilla_file_system.json", msg)   # 探した名前を示す

    def test_missing_docs_are_not_swallowed_as_a_missing_category(self) -> None:
        # prepare_bfcl_data._load は「存在しないカテゴリ」を警告付きで飛ばす。
        # func doc 欠損がその網に掛かると、また関数一覧なしで学習が走ってしまう。
        import prepare_bfcl_data as prep

        with self.assertRaises(di.FuncDocNotFoundError):
            prep._load("multi_turn_base", self.tmp.name, "BFCL_v3")

    def test_empty_func_docs_never_becomes_a_prompt(self) -> None:
        with self.assertRaises(ValueError):
            di._system_prompt([])


class FuncDocFormatTest(unittest.TestCase):
    def test_long_description_is_truncated(self) -> None:
        doc = {"name": "f", "description": "あ" * 5000,
               "parameters": {"properties": {}}}
        line = di._format_func_doc(doc)
        self.assertLess(len(line), di._MAX_DESC_CHARS + 60)
        self.assertTrue(line.endswith("…"))

    def test_newlines_are_flattened(self) -> None:
        # 1 関数 1 行を保つ(改行が混ざると一覧が読めなくなる)。
        doc = {"name": "f", "description": "1 行目\n2 行目",
               "parameters": {"properties": {}}}
        self.assertNotIn("\n", di._format_func_doc(doc))

    def test_untyped_and_missing_parameters_are_tolerated(self) -> None:
        self.assertEqual(di._format_func_doc({"name": "f"}), "- f()")
        line = di._format_func_doc(
            {"name": "g", "parameters": {"properties": {"x": None}}})
        self.assertEqual(line, "- g(x?: any)")


class UndefinedFunctionHintTest(unittest.TestCase):
    """存在しない関数を呼ばれたら、次のターンで直せるヒントを返す。"""

    NS = {"post_tweet": lambda: None, "retweet": lambda: None,
          "comment": lambda: None}

    def test_close_name_is_suggested(self) -> None:
        hint = backend.undefined_function_hint(self.NS, "post_tweets")
        self.assertIn("post_tweet", hint)

    def test_available_names_listed_when_nothing_is_close(self) -> None:
        hint = backend.undefined_function_hint(self.NS, "zzz_launch_rocket")
        for name in self.NS:
            self.assertIn(name, hint)

    def test_execute_reports_undefined_with_hint(self) -> None:
        ok, rendered = backend.execute_structured_call(self.NS, "post_tweets", {})
        self.assertFalse(ok)
        self.assertIn("未定義の関数", rendered)
        self.assertIn("post_tweet", rendered)

    def test_empty_namespace_does_not_crash(self) -> None:
        ok, rendered = backend.execute_structured_call({}, "f", {})
        self.assertFalse(ok)
        self.assertIn("f", rendered)


class ObservationBudgetTest(unittest.TestCase):
    """tool 応答も max_response_length を食う(第2の問題)。"""

    def test_short_observation_is_untouched(self) -> None:
        import bfcl_verl_tool

        self.assertEqual(bfcl_verl_tool._truncate_obs("abc"), "abc")

    def test_long_observation_is_truncated_and_marked(self) -> None:
        import bfcl_verl_tool

        text = "x" * (bfcl_verl_tool.MAX_OBS_CHARS + 500)
        out = bfcl_verl_tool._truncate_obs(text)
        self.assertLess(len(out), len(text))
        self.assertIn("省略", out)   # 切ったことを隠さない


if __name__ == "__main__":
    unittest.main()
