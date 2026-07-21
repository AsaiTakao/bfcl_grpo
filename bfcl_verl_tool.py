# -*- coding: utf-8 -*-
"""
bfcl_verl_tool.py

BFCL multi_turn を verl の BaseTool として実装する。

bfcl_tool_config.yaml で
  class_name: "bfcl_verl_tool.BFCLMultiTurnTool"
として登録される単一ラッパツール `bfcl_multi_turn_call`。
モデルは 1 ターンで `tool_calls: [{name, arguments}, ...]` を渡し、
本ツールがそれらを BFCL バックエンド上で実行して observation を返す。

create_kwargs の "mode" で 2 系統を切り替える(データ側が指定):
  mode="state" : multi_turn。実バックエンド上で実行し、最終状態を GT と比較。
  mode="ast"   : シングルターン(simple/multiple/parallel/live_*)。実行可能な
                 実装が無いため、予測呼び出しを ast_match で GT と照合する。
                 multi_turn 特化 RL による既存能力の退行を抑える目的で混ぜる。

報酬:
  mode="state" (mt_reward の二層報酬)
  - execute(): そのターンの process 報酬(実行成立率)を即時値として返す。
  - calc_reward(): GT を実行して最終状態を比較 → outcome。
                   process 履歴 + outcome を mt_reward で結合し、
                   トラジェクトリのスカラ報酬を返す。
  mode="ast"
  - execute(): 呼び出しを記録するだけ(observation に正解のヒントは出さない)。
  - calc_reward(): ast_match.match_score による一致率 [0,1]。

verl の BaseTool import はバージョン差に備えて defensive に行う。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import ast_match
import bfcl_backend as backend
from bfcl_dataset_and_interaction import decode_field
from mt_reward import RewardConfig, trajectory_reward

try:  # verl 本体があるとき
    from verl.tools.base_tool import BaseTool  # type: ignore
except Exception:  # noqa: BLE001  ローカル単体テスト用フォールバック
    class BaseTool:  # minimal stub
        def __init__(self, config, tool_schema):
            self.config = config
            self.tool_schema = tool_schema

        def get_openai_tool_schema(self):
            return self.tool_schema


class BFCLMultiTurnTool(BaseTool):
    """1 ターン分の BFCL 関数呼び出しをまとめて実行するツール。"""

    def __init__(self, config: dict, tool_schema: Any):
        super().__init__(config, tool_schema)
        self.reward_cfg = RewardConfig.from_dict(config)
        # instance_id -> ロールアウト状態
        self._sessions: Dict[str, Dict[str, Any]] = {}

    # verl は name 属性で openai schema の関数名を参照する。
    @property
    def name(self) -> str:
        try:
            return self.tool_schema.function.name
        except Exception:  # noqa: BLE001
            return "bfcl_multi_turn_call"

    def get_openai_tool_schema(self):
        return self.tool_schema

    # ------------------------------------------------------------------ create
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """test_entry のコンテキスト(create_kwargs 由来)で BFCL バックエンドを初期化。"""
        instance_id = instance_id or str(uuid4())
        mode = kwargs.get("mode") or "state"

        if mode == "ast":
            # シングルターン(AST 採点)。実行可能なバックエンドは存在しないため
            # インスタンスは作らず、予測呼び出しを溜めて calc_reward で照合する。
            self._sessions[instance_id] = {
                "mode": "ast",
                "ast_ground_truth": decode_field(kwargs.get("ast_ground_truth"), []),
                "predicted_calls": [],
                "process_rewards": [],
                "step": self._current_step(kwargs.get("global_step")),
            }
            return instance_id

        # parquet 経由では自由形式の値が JSON 文字列で届く(encode_field 参照)。
        involved = decode_field(kwargs.get("involved_classes"), [])
        initial_config = decode_field(kwargs.get("initial_config"), {})
        ground_truth = decode_field(kwargs.get("ground_truth"), [])
        long_context = bool(kwargs.get("long_context", False))

        instances, namespace = backend.load_involved_classes(
            involved, initial_config, long_context=long_context
        )
        self._sessions[instance_id] = {
            "mode": "state",
            "involved": involved,
            "initial_config": initial_config,
            "ground_truth": ground_truth,       # per-turn list[list[str]]
            "long_context": long_context,
            "instances": instances,
            "namespace": namespace,
            "process_rewards": [],              # 各ターンの process 報酬
            "step": self._current_step(kwargs.get("global_step")),
        }
        return instance_id

    # ----------------------------------------------------------------- execute
    async def execute(
        self, instance_id: str, parameters: Dict[str, Any], **kwargs
    ) -> Tuple[str, float, Dict[str, Any]]:
        """このターンの tool_calls を実行し、(observation, process_reward, metrics)。"""
        sess = self._sessions.get(instance_id)
        if sess is None:
            return "[error] セッション未初期化", 0.0, {"error": "no_session"}

        tool_calls = self._parse_tool_calls(parameters)
        if not tool_calls:
            # 呼び出しが無い/パース不能: process 0、observation で軽く矯正。
            sess["process_rewards"].append(0.0)
            return (
                "[warn] tool_calls が空、または JSON として解釈できませんでした。",
                0.0,
                {"n_calls": 0, "n_ok": 0},
            )

        if sess["mode"] == "ast":
            # 実行系が無いので照合用に記録するだけ。observation は「受理した」ことを
            # 返すに留め、正解のヒントは一切与えない(報酬 hacking を防ぐ)。
            sess["predicted_calls"].extend(tool_calls)
            # 形式として成立した呼び出しであれば process 報酬 1.0
            # (正誤は calc_reward の AST 照合で評価する)。
            sess["process_rewards"].append(1.0)
            names = ", ".join(str(c.get("name", "?")) for c in tool_calls)
            return (
                f"[ok] {len(tool_calls)} 件の呼び出しを受理しました: {names}",
                1.0,
                {"n_calls": len(tool_calls), "mode": "ast"},
            )

        ns = sess["namespace"]
        outputs: List[str] = []
        n_ok = 0
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {}
            ok, rendered = backend.execute_structured_call(ns, name, args)
            n_ok += int(ok)
            outputs.append(f"{name} -> {rendered}")

        process = n_ok / len(tool_calls)
        sess["process_rewards"].append(process)

        observation = "\n".join(outputs)
        metrics = {"n_calls": len(tool_calls), "n_ok": n_ok, "process": process}
        return observation, process, metrics

    # -------------------------------------------------------------- calc_reward
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """GT を実行して outcome を求め、二層報酬のトラジェクトリ総和を返す。"""
        sess = self._sessions.get(instance_id)
        if sess is None:
            return 0.0

        if sess["mode"] == "ast":
            # シングルターンは AST 一致率をそのまま [0,1] の報酬とする。
            # multi_turn の二層報酬とはスケールが違うが、GRPO の advantage は
            # 同一プロンプトの group 内で正規化されるため混在して問題ない。
            return float(
                ast_match.match_score(
                    sess["predicted_calls"], sess["ast_ground_truth"]
                )
            )

        outcome = self._compute_outcome(sess)
        step = self._current_step(kwargs.get("global_step"), sess.get("step", 0))
        return trajectory_reward(
            sess["process_rewards"], outcome, step, self.reward_cfg
        )

    # ------------------------------------------------------------------ release
    async def release(self, instance_id: str, **kwargs) -> None:
        self._sessions.pop(instance_id, None)

    # -------------------------------------------------------------- 内部ヘルパ
    @staticmethod
    def _current_step(explicit: Any = None, fallback: int = 0) -> int:
        """λ 減衰に使う学習 step を解決する。

        verl はツールへ global_step を渡さないため、優先順は
          1) kwargs の global_step(将来 verl が渡すようになった場合)
          2) 環境変数 VERL_GLOBAL_STEP
          3) train_monitor が train.log から抽出して書く global_step.txt
             (GLOBAL_STEP_FILE、既定 ${CKPT_DIR}/global_step.txt)
          4) fallback
        これが無いと step が常に 0 になり λ が lam_start に張り付く
        (= outcome 報酬が永遠に効かない)ので注意。
        """
        if explicit is not None:
            try:
                return int(explicit)
            except (TypeError, ValueError):
                pass
        env_step = os.getenv("VERL_GLOBAL_STEP")
        if env_step:
            try:
                return int(env_step)
            except ValueError:
                pass
        step_file = os.getenv(
            "GLOBAL_STEP_FILE",
            os.path.join(os.getenv("CKPT_DIR", ""), "global_step.txt"),
        )
        try:
            with open(step_file, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return int(fallback or 0)

    def _compute_outcome(self, sess: Dict[str, Any]) -> float:
        """GT を新規インスタンス上で全ターン実行し、最終状態一致率を返す。"""
        gt_turns = sess["ground_truth"]
        if not gt_turns:
            return 0.0
        gt_instances, gt_ns = backend.load_involved_classes(
            sess["involved"], sess["initial_config"], long_context=sess["long_context"]
        )
        for turn in gt_turns:
            for call_str in turn:
                backend.execute_gt_string(gt_ns, call_str)
        return backend.compare_states(sess["instances"], gt_instances)

    @staticmethod
    def _parse_tool_calls(parameters: Any) -> List[Dict[str, Any]]:
        """parameters から呼び出しリストを取り出す。複数の入力形状に耐える。

        受け付ける形:
          {"tool_calls": [{"name":..,"arguments":..}, ...]}
          {"name":.., "arguments":..}                       (単一呼び出し)
          "[{...}]" / "{...}"                                (JSON 文字列)
        """
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except Exception:  # noqa: BLE001
                return []
        if isinstance(parameters, dict):
            if "tool_calls" in parameters:
                calls = parameters["tool_calls"]
                return list(calls) if isinstance(calls, list) else []
            if "name" in parameters:
                return [parameters]
            return []
        if isinstance(parameters, list):
            return [c for c in parameters if isinstance(c, dict)]
        return []


__all__ = ["BFCLMultiTurnTool"]
