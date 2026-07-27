# -*- coding: utf-8 -*-
"""
GPU / verl / Docker に依存しない純 Python ロジックのローカルテスト。

対象:
  - mt_reward.py         二層報酬(lambda 減衰・outcome 後方伝播・総和)
  - train_monitor.py     train.log 解析・val 指標抽出・EarlyStopping
  - eval/dummy_user.py   決定的ユーザーシミュレータ
  - eval/run_tau_eval.py parse_results(pass 率 / 平均 reward 集計)
"""
import json
import os

import pytest


# --------------------------------------------------------------- mt_reward
from mt_reward import (
    RewardConfig,
    combine_turn_rewards,
    outcome_backprop,
    trajectory_reward,
)


def test_rewardconfig_from_dict_ignores_unknown():
    cfg = RewardConfig.from_dict({"gamma": 0.5, "unknown": 123})
    assert cfg.gamma == 0.5
    assert cfg.lam_start == 1.0  # 既定のまま


def test_lam_decay_endpoints_and_clamp():
    cfg = RewardConfig(total_steps=100, lam_start=1.0, lam_end=0.2)
    assert cfg.lam_at(0) == pytest.approx(1.0)
    assert cfg.lam_at(100) == pytest.approx(0.2)
    assert cfg.lam_at(50) == pytest.approx(0.6)
    # 範囲外は clamp される
    assert cfg.lam_at(-10) == pytest.approx(1.0)
    assert cfg.lam_at(9999) == pytest.approx(0.2)


def test_lam_at_zero_total_steps():
    cfg = RewardConfig(total_steps=0, lam_end=0.2)
    assert cfg.lam_at(5) == pytest.approx(0.2)


def test_outcome_backprop_shape_and_decay():
    out = outcome_backprop(1.0, 3, gamma=0.9)
    assert out == pytest.approx([0.81, 0.9, 1.0])  # 最終ターンが最大
    assert outcome_backprop(1.0, 0, 0.9) == []


def test_combine_zero_turns_returns_outcome_only():
    cfg = RewardConfig()
    assert combine_turn_rewards([], outcome=0.7, step=0, cfg=cfg) == [0.7]


def test_combine_early_step_is_process_dominated():
    # step=0 では lam=1.0 → process のみ
    cfg = RewardConfig(total_steps=100, lam_start=1.0, lam_end=0.2)
    r = combine_turn_rewards([0.5, 0.5], outcome=1.0, step=0, cfg=cfg)
    assert r == pytest.approx([0.5, 0.5])


def test_combine_final_step_is_outcome_dominated():
    # step=total では lam=lam_end=0.2 → outcome 主導
    cfg = RewardConfig(total_steps=100, lam_start=1.0, lam_end=0.2, gamma=0.9)
    r = combine_turn_rewards([0.0, 0.0], outcome=1.0, step=100, cfg=cfg)
    # lam=0.2, (1-lam)=0.8, oc=[0.9,1.0]
    assert r == pytest.approx([0.8 * 0.9, 0.8 * 1.0])


def test_trajectory_reward_is_sum():
    cfg = RewardConfig(total_steps=100, lam_start=1.0, lam_end=0.2)
    rewards = [0.5, 0.5]
    total = trajectory_reward(rewards, outcome=1.0, step=0, cfg=cfg)
    assert total == pytest.approx(1.0)
    assert isinstance(total, float)


# ----------------------------------------------------------- train_monitor
from train_monitor import TrainMonitor


def _mk_monitor(tmp_path, **env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        m = TrainMonitor(
            train_log=str(tmp_path / "train.log"),
            ckpt_dir=str(tmp_path),
            train_pid=os.getpid(),
        )
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return m


def test_monitor_writes_global_step(tmp_path):
    m = _mk_monitor(tmp_path)
    m.handle_line("step:50 - actor/loss:0.1")
    assert (tmp_path / "global_step.txt").read_text() == "50"
    assert m.last_step == 50


def test_monitor_extracts_val_metric(tmp_path):
    m = _mk_monitor(tmp_path)
    m.handle_line("step:50 - val-core/bfcl/reward/mean@1:0.42 - other:1")
    lines = (tmp_path / "val_metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["step"] == 50
    assert rec["value"] == pytest.approx(0.42)
    assert "reward" in rec["key"]


def test_monitor_ignores_non_val_lines(tmp_path):
    m = _mk_monitor(tmp_path)
    m.handle_line("step:1 - actor/reward/mean:0.9")  # 'val' を含まない
    assert not (tmp_path / "val_metrics.jsonl").exists()


def test_monitor_custom_regex(tmp_path):
    m = _mk_monitor(tmp_path, VAL_LINE_REGEX=r"mymetric=(?P<value>[0-9.]+)")
    m.handle_line("step:3 - mymetric=0.77")
    rec = json.loads((tmp_path / "val_metrics.jsonl").read_text().strip())
    assert rec["value"] == pytest.approx(0.77)


def test_monitor_early_stopping_marker(tmp_path, monkeypatch):
    killed = {}

    def fake_killpg(pgid, sig):
        killed["pgid"] = pgid

    monkeypatch.setattr("train_monitor.os.killpg", fake_killpg)
    monkeypatch.setattr("train_monitor.os.getpgid", lambda pid: 12345)

    m = _mk_monitor(tmp_path, EARLY_STOP=1, EARLY_STOP_PATIENCE=2)
    # 改善 → その後 patience 回連続で改善なし → 早期終了
    m.handle_line("step:10 - val/reward:0.9")   # best=0.9
    m.handle_line("step:20 - val/reward:0.5")   # bad 1
    assert not (tmp_path / "early_stopped.json").exists()
    m.handle_line("step:30 - val/reward:0.4")   # bad 2 == patience → 発火
    marker = json.loads((tmp_path / "early_stopped.json").read_text())
    assert marker["best_step"] == 10
    assert marker["best_value"] == pytest.approx(0.9)
    assert killed["pgid"] == 12345


def test_monitor_no_early_stop_when_disabled(tmp_path):
    m = _mk_monitor(tmp_path, EARLY_STOP=0, EARLY_STOP_PATIENCE=1)
    m.handle_line("step:10 - val/reward:0.9")
    m.handle_line("step:20 - val/reward:0.1")
    assert not (tmp_path / "early_stopped.json").exists()


# -------------------------------------------------------------- dummy_user
from dummy_user import DummyUser, STOP_TOKEN, default_dummy_user


def test_dummy_user_sequence_then_stop():
    u = DummyUser(turns=["a", "b"])
    assert u.respond() == "a"
    assert u.respond() == "b"
    assert u.done
    assert u.respond() == STOP_TOKEN


def test_dummy_user_reset():
    u = default_dummy_user()
    first = u.respond()
    u.reset()
    assert u.idx == 0
    assert u.respond() == first


# ------------------------------------------------------------ parse_results
from run_tau_eval import parse_results


def test_parse_results_list_format(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(
        [{"reward": 1.0}, {"reward": 0.0}, {"reward": 1.0}]))
    s = parse_results(str(tmp_path))
    assert s["num_tasks_scored"] == 3
    assert s["avg_reward"] == pytest.approx(2 / 3)
    assert s["pass_rate"] == pytest.approx(2 / 3)


def test_parse_results_dict_results_key(tmp_path):
    (tmp_path / "b.json").write_text(json.dumps(
        {"results": [{"reward": 1.0}, {"reward": 0.5}]}))
    s = parse_results(str(tmp_path))
    assert s["num_tasks_scored"] == 2
    assert s["pass_rate"] == pytest.approx(0.5)


def test_parse_results_empty_and_malformed(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "norewards.json").write_text(json.dumps([{"foo": 1}]))
    s = parse_results(str(tmp_path))
    assert s["num_tasks_scored"] == 0
    assert s["avg_reward"] is None
    assert s["pass_rate"] is None
