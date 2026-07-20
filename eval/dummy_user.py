# -*- coding: utf-8 -*-
"""
dummy_user.py

phase0 用の決定的ユーザーシミュレータ(外部 API 不要)。
事前定義した発話を順に返すだけ。LLM ユーザーの当たり外れを排除し、
「エージェント出力のパースが壊れていないか」の検出に集中するために使う。

τ-bench / τ2-bench の user strategy に "dummy" が無い環境でも、
phase0_smoke.py がこれを直接使って multi_turn を再現できる。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# 会話終了の合図(τ-bench 慣例に合わせた文字列)。
STOP_TOKEN = "###STOP###"


@dataclass
class DummyUser:
    """canned な user 発話を idx 順に供給する。尽きたら STOP を返す。"""

    turns: List[str] = field(default_factory=list)
    idx: int = 0

    def reset(self) -> None:
        self.idx = 0

    def respond(self, agent_message: str = "") -> str:
        """次の user 発話を返す。agent_message は使わない(決定的)。"""
        if self.idx >= len(self.turns):
            return STOP_TOKEN
        msg = self.turns[self.idx]
        self.idx += 1
        return msg

    @property
    def done(self) -> bool:
        return self.idx >= len(self.turns)


# phase0 の既定シナリオ:ファイル操作 → メッセージ送信、の 2 ターン。
# BFCL のツール群を模した最小タスクで、think + tool_call のパース健全性を見る。
DEFAULT_SCENARIO_TURNS = [
    "作業ディレクトリにあるファイル 'report.txt' の中身を見せて。",
    "その内容を Alice に 'ここに要約を送ります' というメッセージで送っておいて。",
]


def default_dummy_user() -> DummyUser:
    return DummyUser(turns=list(DEFAULT_SCENARIO_TURNS))
