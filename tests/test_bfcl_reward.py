# -*- coding: utf-8 -*-
"""
bfcl_reward(verl 報酬フック)のユニットテスト。

なぜ: BFCLMultiTurnTool.calc_reward の報酬は rollout の
non_tensor_batch["reward_scores"] に入るが、verl 0.4.1 の naive reward
manager はそれを読まず default_compute_score に落ちて
NotImplementedError(data_source='bfcl_multi_turn')で学習が停止した(実機)。
bfcl_reward は (1) reward_scores を最優先で報酬にし、(2) 欠落行だけ
テキストから再採点する。ここでは判定・解析・フォールバック採点を固定する。

実行: python -m pytest tests/test_bfcl_reward.py
"""
from __future__ import annotations

import json
import unittest

import bfcl_reward


def wrap(calls):
    """bfcl_multi_turn_call の <tool_call> ブロック 1 ターン分を組み立てる。"""
    payload = {"name": bfcl_reward.TOOL_NAME,
               "arguments": {"tool_calls": calls}}
    return f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"


class ParseTurnCallsTest(unittest.TestCase):
    def test_two_turns_in_order(self) -> None:
        text = (
            "<think>x</think>\n" + wrap([{"name": "f", "arguments": {"a": 1}}]) +
            "\n<tool_response>ok</tool_response>\n" +
            wrap([{"name": "g", "arguments": {}}, {"name": "h", "arguments": {}}])
        )
        turns = bfcl_reward.parse_turn_calls(text)
        self.assertEqual([len(t) for t in turns], [1, 2])
        self.assertEqual(turns[0][0]["name"], "f")

    def test_broken_json_counts_as_empty_turn(self) -> None:
        text = "<tool_call>\n{not json\n</tool_call>"
        self.assertEqual(bfcl_reward.parse_turn_calls(text), [[]])

    def test_direct_call_of_unknown_tool_is_empty_turn(self) -> None:
        # ラッパを介さず架空関数を直接呼んだブロック(実機ログで頻出)。
        text = '<tool_call>{"name": "book_flight", "arguments": {}}</tool_call>'
        self.assertEqual(bfcl_reward.parse_turn_calls(text), [[]])

    def test_no_blocks(self) -> None:
        self.assertEqual(bfcl_reward.parse_turn_calls("I cannot do that."), [])


class ToolRewardOfTest(unittest.TestCase):
    def test_reads_tool_entry(self) -> None:
        nt = {"reward_scores": {bfcl_reward.TOOL_NAME: 0.75,
                                "user_turn_rewards": [0.0]}}
        self.assertEqual(bfcl_reward._tool_reward_of(nt), 0.75)

    def test_missing_or_invalid_returns_none(self) -> None:
        self.assertIsNone(bfcl_reward._tool_reward_of({}))
        self.assertIsNone(bfcl_reward._tool_reward_of({"reward_scores": {}}))
        self.assertIsNone(bfcl_reward._tool_reward_of(
            {"reward_scores": {bfcl_reward.TOOL_NAME: True}}))  # bool は報酬でない


class SingleTurnFallbackTest(unittest.TestCase):
    def _extra_info(self, gt) -> dict:
        return {"tools_kwargs": {bfcl_reward.TOOL_NAME: {
            "create_kwargs": {"mode": "ast",
                              "ast_ground_truth": json.dumps(gt)}}}}

    def test_exact_match_scores_one(self) -> None:
        gt = [{"get_weather": {"city": ["Tokyo"]}}]
        text = wrap([{"name": "get_weather", "arguments": {"city": "Tokyo"}}])
        score = bfcl_reward.compute_score(
            "bfcl_single_turn", text, json.dumps(gt), self._extra_info(gt))
        self.assertEqual(score, 1.0)

    def test_wrong_call_scores_zero(self) -> None:
        gt = [{"get_weather": {"city": ["Tokyo"]}}]
        text = wrap([{"name": "send_mail", "arguments": {"to": "x"}}])
        score = bfcl_reward.compute_score(
            "bfcl_single_turn", text, json.dumps(gt), self._extra_info(gt))
        self.assertEqual(score, 0.0)

    def test_ground_truth_string_used_when_extra_info_missing(self) -> None:
        gt = [{"f": {"x": [1]}}]
        text = wrap([{"name": "f", "arguments": {"x": 1}}])
        score = bfcl_reward.compute_score(
            "bfcl_single_turn", text, json.dumps(gt), extra_info=None)
        self.assertEqual(score, 1.0)


class FakeBackend:
    """bfcl_eval 非依存の決定的な擬似バックエンド。

    execute_structured_call: name が "ok_call" のときだけ成功。
    compare_states: pred 側で ok_call が GT と同数実行されたら 1.0。
    """

    def __init__(self) -> None:
        self.loads = 0

    def load_involved_classes(self, involved, initial_config, long_context=False):
        self.loads += 1
        counter = {"ok": 0}
        return {"counter": counter}, counter

    def execute_structured_call(self, ns, name, args):
        if name == "ok_call":
            ns["ok"] += 1
            return True, "done"
        return False, "error"

    def execute_gt_string(self, ns, call_str):
        ns["ok"] += 1
        return True, "done"

    def compare_states(self, pred_instances, gt_instances):
        return 1.0 if (pred_instances["counter"]["ok"]
                       == gt_instances["counter"]["ok"]) else 0.0


class MultiTurnFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self._real_backend = bfcl_reward.backend
        bfcl_reward.backend = FakeBackend()

    def tearDown(self) -> None:
        bfcl_reward.backend = self._real_backend

    def _extra_info(self, gt_turns) -> dict:
        return {"tools_kwargs": {bfcl_reward.TOOL_NAME: {
            "create_kwargs": {
                "mode": "state",
                "involved_classes": json.dumps(["Fake"]),
                "initial_config": json.dumps({}),
                "ground_truth": json.dumps(gt_turns),
                "long_context": False,
            }}}}

    def test_perfect_trajectory_scores_positive(self) -> None:
        gt = [["ok_call()"], ["ok_call()"]]
        text = (wrap([{"name": "ok_call", "arguments": {}}]) +
                wrap([{"name": "ok_call", "arguments": {}}]))
        score = bfcl_reward.compute_score(
            "bfcl_multi_turn", text, json.dumps(gt), self._extra_info(gt))
        # process=1.0 が 2 ターン + outcome=1.0 -> 二層報酬の総和は正で、
        # 全失敗ケースより厳密に大きい。
        self.assertGreater(score, 0.0)

    def test_failed_calls_score_less_than_perfect(self) -> None:
        gt = [["ok_call()"]]
        good = wrap([{"name": "ok_call", "arguments": {}}])
        bad = wrap([{"name": "bad_call", "arguments": {}}])
        s_good = bfcl_reward.compute_score(
            "bfcl_multi_turn", good, json.dumps(gt), self._extra_info(gt))
        s_bad = bfcl_reward.compute_score(
            "bfcl_multi_turn", bad, json.dumps(gt), self._extra_info(gt))
        self.assertGreater(s_good, s_bad)

    def test_missing_create_kwargs_returns_zero(self) -> None:
        score = bfcl_reward.compute_score(
            "bfcl_multi_turn", "no calls", "[]", extra_info=None)
        self.assertEqual(score, 0.0)


class UnknownSourceTest(unittest.TestCase):
    def test_raises_for_unknown_data_source(self) -> None:
        with self.assertRaises(NotImplementedError):
            bfcl_reward.compute_score("gsm8k", "x", "y")


class PatchTest(unittest.TestCase):
    def test_patch_skips_gracefully_without_verl(self) -> None:
        # ローカルには verl が無いので False で戻る(例外を出さない)ことを固定。
        # verl がある実機では True になり __call__ が差し替わる。
        result = bfcl_reward.patch_naive_reward_manager()
        try:
            import verl  # noqa: F401
            has_verl = True
        except ImportError:
            has_verl = False
        if not has_verl:
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
