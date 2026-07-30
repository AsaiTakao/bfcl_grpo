# -*- coding: utf-8 -*-
"""
シングルターン(既存能力)の維持・退行検知まわりのユニットテスト。

なぜ: 学習目的はマルチターン精度の向上だが、multi_turn 専用データだけで RL を
回すとシングルターン function calling が破滅的忘却で壊れる。壊れても
EarlyStopping / 最良 CP 選定はマルチターン指標しか見ないため気付けない。

固定する仕様:
  1. ast_match が BFCL の possible_answer 形式(許容値リスト・optional 引数・
     順不同の parallel)を正しく採点する
  2. BFCLMultiTurnTool が mode="ast" のセッションを実行系なしで採点できる
  3. TrainMonitor が multi_turn を主指標に取り、single_turn の退行を
     combined score へペナルティとして反映する

実行: python -m unittest test_single_turn_guard
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import unittest

import ast_match
import train_monitor
from bfcl_dataset_and_interaction import available_categories as di_available


# --------------------------------------------------------------- ast_match
class AstMatchTest(unittest.TestCase):
    def test_exact_single_call(self) -> None:
        gt = [{"calc_area": {"base": [10], "height": [5]}}]
        pred = [{"name": "calc_area", "arguments": {"base": 10, "height": 5}}]
        self.assertEqual(ast_match.match_score(pred, gt), 1.0)

    def test_alternative_accepted_value(self) -> None:
        gt = [{"get_weather": {"city": ["Tokyo", "tokyo"], "unit": ["celsius"]}}]
        pred = [{"name": "get_weather", "arguments": {"city": "tokyo", "unit": "celsius"}}]
        self.assertEqual(ast_match.match_score(pred, gt), 1.0)

    def test_optional_arg_may_be_omitted(self) -> None:
        # 許容値に "" を含む引数は省略してよい(BFCL の optional 表現)。
        gt = [{"get_weather": {"city": ["Tokyo"], "unit": ["celsius", ""]}}]
        pred = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
        self.assertEqual(ast_match.match_score(pred, gt), 1.0)

    def test_required_arg_missing_fails(self) -> None:
        gt = [{"get_weather": {"city": ["Tokyo"], "unit": ["celsius"]}}]
        pred = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
        self.assertEqual(ast_match.match_score(pred, gt), 0.0)

    def test_wrong_value_fails(self) -> None:
        gt = [{"get_weather": {"city": ["Tokyo"]}}]
        pred = [{"name": "get_weather", "arguments": {"city": "Osaka"}}]
        self.assertEqual(ast_match.match_score(pred, gt), 0.0)

    def test_hallucinated_arg_fails(self) -> None:
        gt = [{"get_weather": {"city": ["Tokyo"]}}]
        pred = [{"name": "get_weather", "arguments": {"city": "Tokyo", "foo": 1}}]
        self.assertEqual(ast_match.match_score(pred, gt), 0.0)

    def test_parallel_is_order_insensitive(self) -> None:
        gt = [
            {"get_weather": {"city": ["Tokyo"]}},
            {"get_weather": {"city": ["Osaka"]}},
        ]
        pred = [
            {"name": "get_weather", "arguments": {"city": "Osaka"}},
            {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        ]
        self.assertEqual(ast_match.match_score(pred, gt), 1.0)

    def test_partial_parallel_match(self) -> None:
        gt = [
            {"get_weather": {"city": ["Tokyo"]}},
            {"get_weather": {"city": ["Osaka"]}},
        ]
        pred = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
        self.assertEqual(ast_match.match_score(pred, gt), 0.5)

    def test_extra_call_is_penalized(self) -> None:
        gt = [{"get_weather": {"city": ["Tokyo"]}}]
        pred = [
            {"name": "get_weather", "arguments": {"city": "Tokyo"}},
            {"name": "get_weather", "arguments": {"city": "Osaka"}},
        ]
        # 1 件一致 - 余計な 1 件 = 0.0
        self.assertEqual(ast_match.match_score(pred, gt), 0.0)

    def test_dotted_name_normalized(self) -> None:
        gt = [{"api.v1.get_weather": {"city": ["Tokyo"]}}]
        pred = [{"name": "api_v1_get_weather", "arguments": {"city": "Tokyo"}}]
        self.assertEqual(ast_match.match_score(pred, gt), 1.0)

    def test_int_float_equivalence(self) -> None:
        gt = [{"f": {"x": [1]}}]
        pred = [{"name": "f", "arguments": {"x": 1.0}}]
        self.assertEqual(ast_match.match_score(pred, gt), 1.0)

    def test_bool_not_confused_with_int(self) -> None:
        # True == 1 を素通しすると引数の型退行を見逃す。
        gt = [{"f": {"x": [1]}}]
        pred = [{"name": "f", "arguments": {"x": True}}]
        self.assertEqual(ast_match.match_score(pred, gt), 0.0)

    def test_no_prediction_scores_zero(self) -> None:
        gt = [{"f": {"x": [1]}}]
        self.assertEqual(ast_match.match_score([], gt), 0.0)


# ------------------------------------------------------------------- tool
class ToolNameTest(unittest.TestCase):
    """verl の BaseTool.__init__ は self.name へ代入する。

    name を read-only property にしていたため、実機の
    ``initialize_tools_from_config`` が
    "property 'name' of 'BFCLMultiTurnTool' object has no setter" で落ちた。
    代入可能であることと、名前が取れない schema での既定名を固定する。
    """

    def _tool(self, schema):
        import bfcl_verl_tool

        return bfcl_verl_tool.BFCLMultiTurnTool({}, schema)

    def test_name_is_assignable(self) -> None:
        tool = self._tool(type("S", (), {})())
        tool.name = "assigned_by_verl"  # ここで AttributeError にならないこと
        self.assertEqual(tool.name, "assigned_by_verl")

    def test_name_falls_back_when_schema_has_no_function(self) -> None:
        tool = self._tool(type("S", (), {})())
        self.assertEqual(tool.name, "bfcl_multi_turn_call")

    def test_name_comes_from_schema_function(self) -> None:
        function = type("F", (), {"name": "bfcl_multi_turn_call_v2"})()
        schema = type("S", (), {"function": function})()
        self.assertEqual(self._tool(schema).name, "bfcl_multi_turn_call_v2")


class AstModeToolTest(unittest.TestCase):
    """mode="ast" は実行系(bfcl_eval)なしで完結することを固定する。"""

    def _tool(self):
        import bfcl_verl_tool

        schema = type("S", (), {})()
        return bfcl_verl_tool.BFCLMultiTurnTool({}, schema)

    def test_ast_session_scores_without_backend(self) -> None:
        tool = self._tool()
        gt = [{"get_weather": {"city": ["Tokyo"]}}]

        async def run() -> float:
            iid = await tool.create(mode="ast", ast_ground_truth=gt)
            obs, process, _ = await tool.execute(
                iid,
                {"tool_calls": [{"name": "get_weather",
                                 "arguments": {"city": "Tokyo"}}]},
            )
            self.assertEqual(process, 1.0)
            # observation に正解を漏らさない(報酬 hacking 防止)。
            self.assertNotIn("Tokyo", obs.replace("get_weather", ""))
            return await tool.calc_reward(iid)

        self.assertEqual(asyncio.run(run()), 1.0)

    def test_ast_session_wrong_call_scores_zero(self) -> None:
        tool = self._tool()
        gt = [{"get_weather": {"city": ["Tokyo"]}}]

        async def run() -> float:
            iid = await tool.create(mode="ast", ast_ground_truth=gt)
            await tool.execute(
                iid,
                {"tool_calls": [{"name": "send_mail", "arguments": {"to": "x"}}]},
            )
            return await tool.calc_reward(iid)

        self.assertEqual(asyncio.run(run()), 0.0)

    def test_empty_tool_calls_scores_zero(self) -> None:
        tool = self._tool()
        gt = [{"get_weather": {"city": ["Tokyo"]}}]

        async def run() -> float:
            iid = await tool.create(mode="ast", ast_ground_truth=gt)
            _, process, _ = await tool.execute(iid, {"tool_calls": []})
            self.assertEqual(process, 0.0)
            return await tool.calc_reward(iid)

        self.assertEqual(asyncio.run(run()), 0.0)


# ---------------------------------------------------------------- dataset
class SingleTurnDatasetTest(unittest.TestCase):
    """シングルターンカテゴリの行構造を合成データで固定する。"""

    def setUp(self) -> None:
        import json

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.makedirs(os.path.join(self.tmp.name, "possible_answer"))

        questions = [{
            "id": "simple_0",
            "question": [[{"role": "user", "content": "東京の天気を教えて"}]],
            "function": [{
                "name": "get_weather",
                "description": "天気を取得",
                "parameters": {"properties": {"city": {"type": "string"}}},
            }],
        }]
        answers = [{"id": "simple_0",
                    "ground_truth": [{"get_weather": {"city": ["Tokyo"]}}]}]
        with open(os.path.join(self.tmp.name, "BFCL_v3_simple.json"),
                  "w", encoding="utf-8") as f:
            json.dump(questions, f)
        with open(os.path.join(self.tmp.name, "possible_answer",
                               "BFCL_v3_simple.json"), "w", encoding="utf-8") as f:
            json.dump(answers, f)

    def _rows(self) -> list:
        import bfcl_dataset_and_interaction as di

        return di.build_verl_dataset("simple", self.tmp.name)

    def test_category_dispatch(self) -> None:
        import bfcl_dataset_and_interaction as di

        self.assertTrue(di.is_multi_turn_category("multi_turn_base"))
        self.assertFalse(di.is_multi_turn_category("simple"))
        self.assertFalse(di.is_multi_turn_category("live_simple"))

    def test_row_uses_separate_data_source(self) -> None:
        # data_source が分かれていないと verl が val 指標を分離せず、
        # 退行ガードが効かない。
        self.assertEqual(self._rows()[0]["data_source"], "bfcl_single_turn")

    def test_row_is_ast_mode_with_no_extra_user_turns(self) -> None:
        import bfcl_dataset_and_interaction as di

        info = self._rows()[0]["extra_info"]
        ck = info["tools_kwargs"]["bfcl_multi_turn_call"]["create_kwargs"]
        self.assertEqual(ck["mode"], "ast")
        self.assertEqual(di.decode_field(ck["ast_ground_truth"], None),
                         [{"get_weather": {"city": ["Tokyo"]}}])
        self.assertEqual(
            di.decode_field(info["interaction_kwargs"]["user_turns"], None), [])

    def test_prompt_lists_the_entry_function_docs(self) -> None:
        # シングルターンの schema は involved_classes ではなくエントリ内 "function"。
        system = self._rows()[0]["prompt"][0]["content"]
        self.assertIn("get_weather", system)

    def test_entry_without_ground_truth_is_dropped(self) -> None:
        import json

        with open(os.path.join(self.tmp.name, "possible_answer",
                               "BFCL_v3_simple.json"), "w", encoding="utf-8") as f:
            json.dump([{"id": "simple_0", "ground_truth": []}], f)
        # 常に報酬 0 になる行を学習に混ぜない。
        self.assertEqual(self._rows(), [])


# ---------------------------------------------------------------- parquet
class ParquetSchemaTest(unittest.TestCase):
    """parquet に載る形かどうかを固定する。

    なぜ: initial_config はバックエンドクラスごとに、ast_ground_truth は
    関数名がキーになるため行ごとに構造が変わる。そのまま DataFrame に入れると
    pyarrow が単一スキーマに落とせず
    "cannot mix list and non-list, non-null values" で落ちる。
    自由形式の値は JSON 文字列にし、両モードでキー構成を揃える。
    """

    def setUp(self) -> None:
        import json

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.makedirs(os.path.join(self.tmp.name, "possible_answer"))

        # multi_turn: initial_config がクラスごとに違う構造を持つ 2 件。
        mt_q = [
            {"id": "multi_turn_base_0",
             "question": [[{"role": "user", "content": "a"}],
                          [{"role": "user", "content": "b"}]],
             "involved_classes": ["GorillaFileSystem"],
             "initial_config": {"GorillaFileSystem": {"root": {"f": {}}}}},
            {"id": "multi_turn_base_1",
             "question": [[{"role": "user", "content": "c"}]],
             "involved_classes": ["TradingBot"],
             "initial_config": {"TradingBot": {"balance": 100, "orders": []}}},
        ]
        mt_a = [{"id": "multi_turn_base_0", "ground_truth": [["f(a=1)"], ["g()"]]},
                {"id": "multi_turn_base_1", "ground_truth": [["h(x=2)"]]}]
        st_q = [{"id": "simple_0",
                 "question": [[{"role": "user", "content": "d"}]],
                 "function": [{"name": "w", "description": "",
                               "parameters": {"properties": {"c": {}}}}]}]
        st_a = [{"id": "simple_0", "ground_truth": [{"w": {"c": ["x"]}}]}]

        for name, obj in [("BFCL_v4_multi_turn_base.json", mt_q),
                          ("BFCL_v4_simple_python.json", st_q)]:
            with open(os.path.join(self.tmp.name, name), "w", encoding="utf-8") as f:
                json.dump(obj, f)
        for name, obj in [("BFCL_v4_multi_turn_base.json", mt_a),
                          ("BFCL_v4_simple_python.json", st_a)]:
            with open(os.path.join(self.tmp.name, "possible_answer", name),
                      "w", encoding="utf-8") as f:
                json.dump(obj, f)

    def _all_rows(self) -> list:
        import bfcl_dataset_and_interaction as di

        return (di.build_verl_dataset("multi_turn_base", self.tmp.name)
                + di.build_verl_dataset("simple_python", self.tmp.name))

    def test_create_kwargs_keys_identical_across_modes(self) -> None:
        import bfcl_dataset_and_interaction as di

        keysets = {
            frozenset(r["extra_info"]["tools_kwargs"][di.TOOL_NAME]["create_kwargs"])
            for r in self._all_rows()
        }
        self.assertEqual(len(keysets), 1, f"キー構成が揃っていない: {keysets}")

    def test_free_form_values_are_json_strings(self) -> None:
        import bfcl_dataset_and_interaction as di

        volatile = ["involved_classes", "initial_config",
                    "ground_truth", "ast_ground_truth"]
        for r in self._all_rows():
            ck = r["extra_info"]["tools_kwargs"][di.TOOL_NAME]["create_kwargs"]
            for k in volatile:
                self.assertIsInstance(ck[k], str, f"{k} が文字列でない")
            self.assertIsInstance(
                r["extra_info"]["interaction_kwargs"]["user_turns"], str)

    def test_round_trip_restores_structure(self) -> None:
        import bfcl_dataset_and_interaction as di

        rows = self._all_rows()
        mt = rows[0]["extra_info"]["tools_kwargs"][di.TOOL_NAME]["create_kwargs"]
        self.assertEqual(di.decode_field(mt["initial_config"], None),
                         {"GorillaFileSystem": {"root": {"f": {}}}})
        self.assertEqual(di.decode_field(mt["ground_truth"], None),
                         [["f(a=1)"], ["g()"]])
        self.assertEqual(di.decode_field(mt["involved_classes"], None),
                         ["GorillaFileSystem"])

    def test_decode_passes_through_raw_structures(self) -> None:
        import bfcl_dataset_and_interaction as di

        # テストや直接呼び出しでは生の dict/list が来る。両方受け付ける。
        self.assertEqual(di.decode_field(["a"], None), ["a"])
        self.assertEqual(di.decode_field(None, []), [])
        self.assertEqual(di.decode_field("", []), [])
        self.assertEqual(di.decode_field("not json", []), [])


# --------------------------------------------------------------- version
class VersionPrefixTest(unittest.TestCase):
    """データファイルの接頭辞検出。

    なぜ: bfcl_eval のバージョンで BFCL_v3_* / BFCL_v4_* とファイル名が変わる。
    決め打ちにすると全カテゴリが「ファイルなし」で静かにスキップされ、
    最終的に 0 件で止まるまで原因が分からない。
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _touch(self, name: str) -> None:
        with open(os.path.join(self.tmp.name, name), "w", encoding="utf-8") as f:
            f.write("[]")

    def test_detects_v4(self) -> None:
        import bfcl_dataset_and_interaction as di

        self._touch("BFCL_v4_multi_turn_base.json")
        self.assertEqual(di.detect_version_prefix(self.tmp.name), "BFCL_v4")

    def test_detects_v3(self) -> None:
        import bfcl_dataset_and_interaction as di

        self._touch("BFCL_v3_multi_turn_base.json")
        self.assertEqual(di.detect_version_prefix(self.tmp.name), "BFCL_v3")

    def test_prefers_newer_when_mixed(self) -> None:
        import bfcl_dataset_and_interaction as di

        self._touch("BFCL_v3_simple.json")
        self._touch("BFCL_v4_simple_python.json")
        self.assertEqual(di.detect_version_prefix(self.tmp.name), "BFCL_v4")

    def test_missing_dir_falls_back(self) -> None:
        import bfcl_dataset_and_interaction as di

        self.assertEqual(
            di.detect_version_prefix(os.path.join(self.tmp.name, "nope")),
            di.VERSION_PREFIX,
        )

    def test_v3_name_aliases_to_v4_on_load(self) -> None:
        import json
        import prepare_bfcl_data as prep

        # v4 環境で旧名 "simple" を指定しても simple_python を読む。
        # 黙って 400 件を取り逃すのを防ぐ。
        os.makedirs(os.path.join(self.tmp.name, "possible_answer"))
        q = [{"id": "simple_0", "question": [[{"role": "user", "content": "a"}]],
              "function": [{"name": "f", "description": "",
                            "parameters": {"properties": {}}}]}]
        a = [{"id": "simple_0", "ground_truth": [{"f": {}}]}]
        with open(os.path.join(self.tmp.name, "BFCL_v4_simple_python.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(q, fh)
        with open(os.path.join(self.tmp.name, "possible_answer",
                               "BFCL_v4_simple_python.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(a, fh)

        rows = prep._load("simple", self.tmp.name, "BFCL_v4")
        self.assertEqual(len(rows), 1)

    def test_existing_category_is_not_aliased(self) -> None:
        import prepare_bfcl_data as prep

        # 実在するカテゴリ名は読み替えない(v3 環境で simple はそのまま)。
        self._touch("BFCL_v3_simple.json")
        self.assertIn("simple", di_available(self.tmp.name, "BFCL_v3"))
        rows = prep._load("simple", self.tmp.name, "BFCL_v3")
        self.assertEqual(rows, [])   # 中身は空 JSON なので 0 件(読み替えは起きない)

    def test_available_categories_listed(self) -> None:
        import bfcl_dataset_and_interaction as di

        self._touch("BFCL_v4_simple_python.json")
        self._touch("BFCL_v4_multi_turn_base.json")
        self._touch("BFCL_v3_ignored.json")
        self.assertEqual(
            di.available_categories(self.tmp.name, "BFCL_v4"),
            ["multi_turn_base", "simple_python"],
        )


# ---------------------------------------------------------------- holdout
class HoldoutSplitTest(unittest.TestCase):
    """API クラス単位の holdout 分割。

    なぜ: BFCL multi_turn のバックエンドクラスは 8 個しかなく、ランダム分割では
    val が train と同じ API を使うため未知 API への汎化を測れない。最終評価
    (τ-bench retail/airline)は完全に別ドメインなので OOD 転移の指標が要る。
    """

    @staticmethod
    def _row(classes: list) -> dict:
        import bfcl_dataset_and_interaction as di

        return {
            "data_source": "bfcl_multi_turn",
            "extra_info": {
                "tools_kwargs": {
                    di.TOOL_NAME: {"create_kwargs": {"involved_classes": classes}},
                }
            },
        }

    def test_row_classes_extracted(self) -> None:
        import prepare_bfcl_data as prep

        self.assertEqual(prep._row_classes(self._row(["TravelAPI"])), {"TravelAPI"})

    def test_single_turn_row_has_no_classes(self) -> None:
        import prepare_bfcl_data as prep

        # シングルターン行は involved_classes を持たない → holdout の影響を受けない。
        row = {"data_source": "bfcl_single_turn", "extra_info": {"tools_kwargs": {}}}
        self.assertEqual(prep._row_classes(row), set())

    def test_holdout_class_goes_to_val(self) -> None:
        import prepare_bfcl_data as prep

        rows = [self._row(["TravelAPI"]), self._row(["MessageAPI"])]
        val, train = prep._split_by_holdout(rows, {"TravelAPI"})
        self.assertEqual(len(val), 1)
        self.assertEqual(len(train), 1)
        self.assertEqual(prep._row_classes(val[0]), {"TravelAPI"})

    def test_mixed_class_row_goes_to_val_not_train(self) -> None:
        import prepare_bfcl_data as prep

        # holdout クラスを 1 つでも含む行を train に残すとリークする。
        rows = [self._row(["MessageAPI", "TravelAPI"])]
        val, train = prep._split_by_holdout(rows, {"TravelAPI"})
        self.assertEqual(len(val), 1)
        self.assertEqual(train, [])

    def test_train_side_never_touches_holdout_class(self) -> None:
        import prepare_bfcl_data as prep

        rows = [
            self._row(["TravelAPI"]),
            self._row(["MessageAPI", "TravelAPI"]),
            self._row(["MessageAPI"]),
            self._row(["GorillaFileSystem", "MathAPI"]),
        ]
        holdout = {"TravelAPI"}
        _, train = prep._split_by_holdout(rows, holdout)
        for r in train:
            self.assertFalse(prep._row_classes(r) & holdout)

    def test_multiple_holdout_classes(self) -> None:
        import prepare_bfcl_data as prep

        rows = [
            self._row(["TravelAPI"]),
            self._row(["VehicleControlAPI"]),
            self._row(["MessageAPI"]),
        ]
        val, train = prep._split_by_holdout(rows, {"TravelAPI", "VehicleControlAPI"})
        self.assertEqual(len(val), 2)
        self.assertEqual(len(train), 1)


# ---------------------------------------------------------------- monitor
class RegressionGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # 早期終了で SIGTERM を飛ばさないよう無効化しておく。
        for k, v in [("EARLY_STOP", "0"), ("ST_REGRESSION_WEIGHT", "1.0"),
                     ("ST_REGRESSION_TOL", "0.05")]:
            os.environ[k] = v
            self.addCleanup(os.environ.pop, k, None)

    def _monitor(self) -> train_monitor.TrainMonitor:
        return train_monitor.TrainMonitor(
            train_log=os.path.join(self.tmp.name, "train.log"),
            ckpt_dir=self.tmp.name,
            train_pid=os.getpid(),
        )

    @staticmethod
    def _line(step: int, mt: float, st: float | None) -> str:
        parts = [f"step:{step}",
                 f"val-core/bfcl_multi_turn/reward/mean@1:{mt}"]
        if st is not None:
            parts.append(f"val-core/bfcl_single_turn/reward/mean@1:{st}")
        return " - ".join(parts)

    def test_multi_turn_is_the_primary_metric(self) -> None:
        m = self._monitor()
        m.handle_line(self._line(5, 0.40, 0.80))
        self.assertIn("multi_turn", m.chosen_key)
        self.assertIn("single_turn", m.st_key)

    def test_baseline_is_first_single_turn_value(self) -> None:
        m = self._monitor()
        m.handle_line(self._line(5, 0.40, 0.80))
        self.assertEqual(m.st_baseline, 0.80)

    def test_no_penalty_when_single_turn_holds(self) -> None:
        m = self._monitor()
        m.handle_line(self._line(5, 0.40, 0.80))
        m.handle_line(self._line(10, 0.50, 0.79))   # tol 5% 以内の低下
        self.assertEqual(m.best_value, 0.50)
        self.assertEqual(m.best_step, 10)

    def test_single_turn_regression_is_penalized(self) -> None:
        m = self._monitor()
        m.handle_line(self._line(5, 0.40, 0.80))
        # mt は伸びたが st が 0.80 -> 0.30 に崩壊。
        # threshold = 0.80*0.95 = 0.76、penalty = 0.30-0.76 = -0.46
        # combined = 0.60 - 0.46 = 0.14 < 0.40 なので best は更新されない。
        m.handle_line(self._line(10, 0.60, 0.30))
        self.assertEqual(m.best_step, 5)
        self.assertEqual(m.bad_evals, 1)

    def test_single_turn_gain_does_not_inflate_score(self) -> None:
        # 主目的はマルチターン。st が伸びても加点しない。
        m = self._monitor()
        m.handle_line(self._line(5, 0.40, 0.80))
        m.handle_line(self._line(10, 0.40, 0.99))
        self.assertEqual(m.bad_evals, 1)      # mt が同じなら改善なし扱い

    def test_falls_back_to_single_metric_when_no_single_turn(self) -> None:
        # シングルターン val が無い場合も従来どおり動く(後方互換)。
        m = self._monitor()
        m.handle_line("step:5 - val-core/bfcl/reward/mean@1:0.42")
        self.assertEqual(m.best_value, 0.42)
        self.assertIsNone(m.st_baseline)

    def test_detail_is_recorded_for_uploader(self) -> None:
        import json

        m = self._monitor()
        m.handle_line(self._line(5, 0.40, 0.80))
        with open(os.path.join(self.tmp.name, "val_metrics.jsonl"),
                  encoding="utf-8") as f:
            rec = json.loads(f.readline())
        # uploader は value(combined)で最良 CP を選ぶ。内訳も残す。
        self.assertEqual(rec["value"], 0.40)
        self.assertEqual(rec["mt"], 0.40)
        self.assertEqual(rec["st"], 0.80)


if __name__ == "__main__":
    unittest.main()
