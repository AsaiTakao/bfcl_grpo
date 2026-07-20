# -*- coding: utf-8 -*-
"""
mt_reward.py

BFCL multi_turn 用の「二層報酬(two-layer reward)」。

設計(BFCL学習τ2-bench評価.txt / MUA-RL 参考):
  - outcome 報酬: 全ターン終了後の状態一致(state-based checker)。
                  gamma で各ターンへ後方伝播(後のターンほど重い)。
  - process 報酬: 各ターンの実行妥当性(呼び出しがエラーせず成立したか等)。
                  学習が進むほど重み lambda を下げ、outcome に収束させる。

    r_t = lam(step) * process[t] + (1 - lam(step)) * gamma^(T-1-t) * outcome

  lam は学習 step に対して lam_start -> lam_end へ線形減衰。
  初期は密な shaping(lam_start=1.0)、終盤は outcome 主導(lam_end=0.2)。

この二層値の総和 sum_t r_t を 1 トラジェクトリのスカラ報酬として返す。
(verl の GRPO は group 内で相対化するため、絶対スケールより形が重要。)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class RewardConfig:
    """bfcl_tool_config.yaml の config ブロックから流し込むパラメータ。"""

    gamma: float = 0.9          # outcome 後方伝播の割引
    total_steps: int = 1000     # lambda 減衰の総ステップ(trainer 総更新数に合わせる)
    lam_start: float = 1.0      # process 重み初期値(密な shaping)
    lam_end: float = 0.2        # process 重み終値(outcome に収束)
    model_name: str = "rl_policy"

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "RewardConfig":
        cfg = cfg or {}
        known = {f: cfg[f] for f in cls.__dataclass_fields__ if f in cfg}
        return cls(**known)

    def lam_at(self, step: int) -> float:
        """学習 step における process 重み。lam_start -> lam_end へ線形減衰。"""
        if self.total_steps <= 0:
            return self.lam_end
        frac = min(max(step, 0), self.total_steps) / float(self.total_steps)
        return self.lam_start + (self.lam_end - self.lam_start) * frac


def outcome_backprop(outcome: float, n_turns: int, gamma: float) -> List[float]:
    """outcome を各ターンへ後方伝播。最終ターン=outcome、遡るほど gamma 倍で減衰。

    turn t (0-indexed, T=n_turns) への寄与 = gamma^(T-1-t) * outcome
    """
    if n_turns <= 0:
        return []
    return [outcome * (gamma ** (n_turns - 1 - t)) for t in range(n_turns)]


def combine_turn_rewards(
    process_rewards: Sequence[float],
    outcome: float,
    step: int,
    cfg: RewardConfig,
) -> List[float]:
    """各ターンの二層報酬 r_t を返す。len == len(process_rewards)。

    ターンが 0 の場合(1度もツールを呼ばなかった)は outcome のみを 1 ターン扱いで返す。
    """
    n = len(process_rewards)
    if n == 0:
        # ツール未使用でも outcome は評価する(たいてい 0 だが、稀に不要ケースがある)。
        return [outcome]

    lam = cfg.lam_at(step)
    oc = outcome_backprop(outcome, n, cfg.gamma)
    return [lam * process_rewards[t] + (1.0 - lam) * oc[t] for t in range(n)]


def trajectory_reward(
    process_rewards: Sequence[float],
    outcome: float,
    step: int,
    cfg: RewardConfig,
) -> float:
    """1 トラジェクトリのスカラ報酬(= 二層報酬の総和)。verl に返す最終値。"""
    return float(sum(combine_turn_rewards(process_rewards, outcome, step, cfg)))


__all__ = [
    "RewardConfig",
    "outcome_backprop",
    "combine_turn_rewards",
    "trajectory_reward",
]
