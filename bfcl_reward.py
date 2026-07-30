# -*- coding: utf-8 -*-
"""bfcl_reward.py — verl の報酬フックに BFCL の二層報酬を配線する。

なぜ必要か
----------
BFCLMultiTurnTool.calc_reward が計算した報酬(mt_reward の二層報酬 /
シングルターンの AST 一致率)は、verl 0.4.1 の sglang rollout によって
``non_tensor_batch["reward_scores"]``(dict: tool 名 -> float)に格納される。
ところが naive reward manager はこれを読まず、``data_source`` をキーに
verl 同梱の ``default_compute_score`` を引くため、

    NotImplementedError: Reward function is not implemented for
    data_source='bfcl_multi_turn'

で学習が落ちる(実機で val_before_train の報酬計算時に発生)。

当て方
------
verl の公式フック ``custom_reward_function.path`` にこのファイルを渡す。
``get_custom_reward_fn`` がこのモジュールを driver プロセスで import する
副作用で ``patch_naive_reward_manager()`` が走り、NaiveRewardManager.__call__
を「rollout が格納済みの reward_scores を最優先で使う」実装に差し替える。
これによりツールが実際のバックエンド状態から計算した報酬がそのまま
GRPO の報酬になる(報酬の二重計算も再実行も不要)。

reward_scores が無い行(理論上は起きないが防御)のみ、本モジュールの
``compute_score`` がテキストから再採点するフォールバックに落ちる:
  - bfcl_single_turn: 応答中の <tool_call> を全て集めて ast_match
  - bfcl_multi_turn : 予測呼び出しを新規バックエンドで再実行し、
                      GT 実行後の状態と比較して trajectory_reward を再構成

注意: 差し替え対象は verl 0.4.1 の NaiveRewardManager(このリポジトリの
既定 reward_manager=naive)。verl を上げる場合は reward_scores の受け渡しが
公式対応されていないか先に確認すること。
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import ast_match
import bfcl_backend as backend
from bfcl_dataset_and_interaction import decode_field
from bfcl_verl_tool import BFCLMultiTurnTool
from mt_reward import RewardConfig, trajectory_reward

#: rollout の reward_scores を引くキー(bfcl_tool_config.yaml の関数名)。
TOOL_NAME = BFCLMultiTurnTool.DEFAULT_NAME

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def log(msg: str) -> None:
    print(f"[bfcl-reward] {msg}", flush=True)


# ---------------------------------------------------------- 応答テキストの解析
def parse_turn_calls(solution_str: str) -> List[List[Dict[str, Any]]]:
    """応答から <tool_call> ブロックを順に取り、ターンごとの呼び出しリストへ。

    ブロックが JSON として壊れている / ラッパツール以外を直接呼んでいる場合は
    空ターン扱い(rollout 時のツール実行でも失敗になるのと同じ扱い)。
    """
    turns: List[List[Dict[str, Any]]] = []
    for block in _TOOL_CALL_RE.findall(solution_str):
        try:
            obj = json.loads(block)
        except Exception:  # noqa: BLE001
            turns.append([])
            continue
        if not isinstance(obj, dict) or obj.get("name") != TOOL_NAME:
            turns.append([])
            continue
        args = obj.get("arguments", {})
        # ツール本体と同じ寛容なパースを再利用する(挙動を揃えるため)。
        turns.append(BFCLMultiTurnTool._parse_tool_calls(args))
    return turns


def _create_kwargs_of(extra_info: Any) -> Optional[Dict[str, Any]]:
    """extra_info から tools_kwargs.<TOOL_NAME>.create_kwargs を取り出す。"""
    if not isinstance(extra_info, dict):
        return None
    tools_kwargs = extra_info.get("tools_kwargs") or {}
    tool_cfg = tools_kwargs.get(TOOL_NAME) or {}
    ck = tool_cfg.get("create_kwargs")
    return ck if isinstance(ck, dict) else None


# ------------------------------------------------------ フォールバック採点本体
def _score_single_turn(solution_str: str, ground_truth: Any,
                       extra_info: Any) -> float:
    """AST 照合。GT は create_kwargs.ast_ground_truth 優先、無ければ ground_truth。"""
    gt: List[Any] = []
    ck = _create_kwargs_of(extra_info)
    if ck is not None:
        gt = decode_field(ck.get("ast_ground_truth"), [])
    if not gt and ground_truth:
        try:
            gt = json.loads(ground_truth) if isinstance(ground_truth, str) else list(ground_truth)
        except Exception:  # noqa: BLE001
            gt = []
    calls = [c for turn in parse_turn_calls(solution_str) for c in turn]
    return float(ast_match.match_score(calls, gt))


def _score_multi_turn(solution_str: str, ground_truth: Any,
                      extra_info: Any) -> float:
    """予測呼び出しを再実行して二層報酬を再構成する(reward_scores 欠落時のみ)。

    BFCL バックエンドは決定的なので、rollout 時と同じ呼び出し列を実行すれば
    同じ最終状態(= 同じ outcome)になる。process 報酬もツールと同じ定義
    (ターン内の成功呼び出し率)で再計算する。
    """
    ck = _create_kwargs_of(extra_info)
    if ck is None:
        log("extra_info に create_kwargs が無くフォールバック採点不可 -> 0.0")
        return 0.0

    involved = decode_field(ck.get("involved_classes"), [])
    initial_config = decode_field(ck.get("initial_config"), {})
    gt_turns = decode_field(ck.get("ground_truth"), [])
    if not gt_turns and ground_truth:
        try:
            gt_turns = json.loads(ground_truth)
        except Exception:  # noqa: BLE001
            gt_turns = []
    long_context = bool(ck.get("long_context", False))

    # 予測呼び出しを新規バックエンドで再実行(process 報酬を再構成)。
    pred_instances, pred_ns = backend.load_involved_classes(
        involved, initial_config, long_context=long_context)
    process_rewards: List[float] = []
    for turn_calls in parse_turn_calls(solution_str):
        if not turn_calls:
            process_rewards.append(0.0)
            continue
        n_ok = 0
        for call in turn_calls:
            args = call.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {}
            ok, _ = backend.execute_structured_call(pred_ns, call.get("name", ""), args)
            n_ok += int(ok)
        process_rewards.append(n_ok / len(turn_calls))
    # ツール未呼び出し(ターン 0)は combine_turn_rewards が outcome のみで
    # 評価する — ツール本体の calc_reward と同じ挙動。

    # GT を別インスタンスで全ターン実行し、最終状態一致率 = outcome。
    outcome = 0.0
    if gt_turns:
        gt_instances, gt_ns = backend.load_involved_classes(
            involved, initial_config, long_context=long_context)
        for turn in gt_turns:
            for call_str in turn:
                backend.execute_gt_string(gt_ns, call_str)
        outcome = backend.compare_states(pred_instances, gt_instances)

    step = BFCLMultiTurnTool._current_step()
    return float(trajectory_reward(process_rewards, outcome, step,
                                   RewardConfig.from_dict(None)))


def compute_score(data_source: str, solution_str: str, ground_truth: Any,
                  extra_info: Any = None, **kwargs) -> float:
    """verl custom_reward_function 用のエントリポイント(フォールバック採点)。

    通常は patch_naive_reward_manager が rollout 済みの reward_scores を使う
    ため、ここへは reward_scores を持たない行だけが落ちてくる。
    """
    if data_source == "bfcl_single_turn":
        return _score_single_turn(solution_str, ground_truth, extra_info)
    if data_source == "bfcl_multi_turn":
        return _score_multi_turn(solution_str, ground_truth, extra_info)
    raise NotImplementedError(f"bfcl_reward: 未対応の data_source={data_source!r}")


# --------------------------------------------- NaiveRewardManager の差し替え
def _tool_reward_of(non_tensor: Dict[str, Any]) -> Optional[float]:
    """rollout が格納した reward_scores からツール報酬を取り出す(無ければ None)。"""
    scores = non_tensor.get("reward_scores")
    if not isinstance(scores, dict):
        return None
    value = scores.get(TOOL_NAME)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def patch_naive_reward_manager() -> bool:
    """NaiveRewardManager.__call__ を reward_scores 優先の実装に差し替える。"""
    try:
        from verl.workers.reward_manager.naive import NaiveRewardManager
    except Exception as e:  # noqa: BLE001  verl の無い環境(単体テスト等)
        log(f"NaiveRewardManager を import できずパッチをスキップ: {e}")
        return False
    if getattr(NaiveRewardManager, "_bfcl_reward_patched", False):
        return False

    import torch

    def __call__(self, data, return_dict=False):  # noqa: N807
        # 本家と同じ early return(外部 RM を使う構成のため)。
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        printed: Dict[str, int] = {}

        for i in range(len(data)):
            item = data[i]
            prompt_length = item.batch["prompts"].shape[-1]
            valid_response_length = int(
                item.batch["attention_mask"][prompt_length:].sum())
            data_source = item.non_tensor_batch[self.reward_fn_key]

            reward = _tool_reward_of(item.non_tensor_batch)
            from_tool = 1.0
            if reward is None:
                # rollout が報酬を残さなかった行のみテキストから再採点。
                from_tool = 0.0
                response_ids = item.batch["responses"][:valid_response_length]
                response_str = self.tokenizer.decode(
                    response_ids, skip_special_tokens=True)
                ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
                extra_info = item.non_tensor_batch.get("extra_info", None)
                reward = float(self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                ))

            reward_tensor[i, max(valid_response_length - 1, 0)] = reward
            reward_extra_info["reward_from_tool"].append(from_tool)

            shown = printed.get(data_source, 0)
            if shown < self.num_examine:
                printed[data_source] = shown + 1
                log(f"source={data_source} reward={reward:.4f} "
                    f"from_tool={bool(from_tool)}")

        if return_dict:
            return {"reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info}
        return reward_tensor

    NaiveRewardManager.__call__ = __call__
    NaiveRewardManager._bfcl_reward_patched = True
    log("適用: NaiveRewardManager は rollout の reward_scores を報酬に使います")
    return True


# custom_reward_function.path 経由の import(= driver / 報酬ワーカー)で当てる。
patch_naive_reward_manager()
