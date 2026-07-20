# -*- coding: utf-8 -*-
"""
run_tau_eval.py

phase1 / phase2 の実行オーケストレータ。ベンチ本体は再実装せず、
公式 τ-bench / τ2-bench のランナーを subprocess で呼び、結果を集計する。

設計書:
  phase1: retail + ローカル user(API不要、傾向確認)
  phase2: retail/airline + gemini-3.5-flash(公式条件準拠、最終CPのみ)

前提: エージェントは vLLM OpenAI 互換で起動済み(serve_agent_vllm.sh)。
  - tau-bench は litellm 経由でモデルを呼ぶ。
  - エージェント(評価対象)= OpenAI 互換ローカル vLLM
      → --model-provider openai、接続は OPENAI_API_BASE / OPENAI_API_KEY。
  - user simulator:
      phase1 ローカル = 同じ vLLM(--user-model-provider openai)
      phase2 Gemini   = --user-model-provider google(GEMINI_API_KEY)

公式ランナーの CLI 名/引数はバージョン依存。TAU_RUN_CMD 等の環境変数で
丸ごと差し替えられるようにしてある。既定は tau-bench(sierra-research)の CLI。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
from typing import Any, Dict, List

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_secret_file(path: str) -> str:
    """secret ファイルを読み、前後空白・改行を除去して返す。読めなければ空。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError as e:  # noqa: BLE001 — 存在しない/権限なしは空扱いでフォールバック
        print(f"[eval] 警告: secret ファイルを読めません: {path} ({e})",
              file=sys.stderr)
        return ""


def _resolve_api_key(user_cfg: Dict[str, Any]) -> str:
    """user の api_key を解決する。

    HF_TOKEN と同じく「ファイル渡し(Docker/BuildKit secret)」を最優先にする。
    キー値をコマンドライン・設定ファイル・広い環境変数に置かず、ファイル1枚に
    閉じ込められるようにするのが狙い。優先順位:
      1. api_key_file_env が指す環境変数のパス先ファイル(例 GEMINI_API_KEY_FILE)
      2. api_key_file の既定パス(例 /run/secrets/gemini_api_key)
      3. api_key_env の環境変数の値(従来互換)
      4. 設定ファイルの api_key リテラル(既定 "EMPTY")
    """
    # 1) パスを持つ環境変数(<NAME>_FILE 慣例)。DOK で secret をマウントする場合。
    file_env = user_cfg.get("api_key_file_env")
    if file_env:
        path = os.getenv(file_env, "").strip()
        if path:
            key = _read_secret_file(path)
            if key:
                return key

    # 2) 既定の secret ファイルパス(HF_TOKEN_FILE=/run/secrets/hf_token と同思想)。
    default_path = user_cfg.get("api_key_file")
    if default_path and os.path.isfile(default_path):
        key = _read_secret_file(default_path)
        if key:
            return key

    # 3) 環境変数から直接(後方互換)。
    if "api_key_env" in user_cfg:
        key = os.getenv(user_cfg["api_key_env"], "")
        if not key:
            print(f"[eval] 警告: {user_cfg['api_key_env']} も secret ファイルも "
                  f"未設定です。", file=sys.stderr)
        return key

    # 4) リテラル(ローカル vLLM のダミーキー等)。
    return user_cfg.get("api_key", "EMPTY")


def build_tau_command(
    cfg: Dict[str, Any],
    phase_cfg: Dict[str, Any],
    out_dir: str,
    bench: str,
) -> List[str]:
    """公式ランナーの起動コマンドを組む(tau-bench の実 CLI 準拠)。

    TAU_RUN_CMD(空白区切り)が指定されていればそれをベースにする。
    """
    agent = cfg["agent"]
    tau = cfg.get("tau", {})
    domain = phase_cfg.get("domain", "retail")
    user = phase_cfg.get("user", {})
    num_tasks = int(phase_cfg.get("num_tasks", 0) or 0)

    override = os.getenv("TAU_RUN_CMD", "").strip()
    if override:
        base = shlex.split(override)
    elif bench == "tau2":
        base = [sys.executable, "-m", "tau2.run"]      # τ2-bench(要バージョン確認)
    else:
        base = [sys.executable, "-m", "tau_bench.run"]  # τ-bench(sierra-research)

    flags = [
        "--agent-strategy", agent.get("strategy", "tool-calling"),
        "--env", domain,
        "--model", agent["served_name"],
        "--model-provider", agent.get("provider", "openai"),
        "--user-model", user.get("model", agent["served_name"]),
        "--user-model-provider", user.get("provider", "openai"),
        "--user-strategy", user.get("strategy", "llm"),
        "--task-split", tau.get("task_split", "test"),
        "--log-dir", out_dir,
    ]
    if num_tasks > 0:
        # 先頭 num_tasks 件だけ評価(phase1 の傾向確認用)。
        flags += ["--start-index", "0", "--end-index", str(num_tasks)]

    return base + flags


def build_env(cfg: Dict[str, Any], phase_cfg: Dict[str, Any]) -> Dict[str, str]:
    """subprocess に渡す環境変数(litellm がプロバイダ別に読む)。"""
    agent = cfg["agent"]
    user = phase_cfg.get("user", {})
    env = dict(os.environ)

    # エージェント = OpenAI 互換ローカル vLLM。litellm/openai は API_BASE を見る。
    agent_base = agent.get("base_url", "http://127.0.0.1:8000/v1")
    env["OPENAI_API_BASE"] = agent_base
    env["OPENAI_BASE_URL"] = agent_base       # SDK 版差の両対応
    env["OPENAI_API_KEY"] = agent.get("api_key", "EMPTY")

    # user simulator のプロバイダに応じてキーを差す。
    provider = user.get("provider", "openai")
    key = _resolve_api_key(user)
    if provider in ("google", "gemini"):
        # litellm google/gemini(Google AI Studio)は GEMINI_API_KEY を読む。
        env["GEMINI_API_KEY"] = key
        env["GOOGLE_API_KEY"] = key
    elif provider == "openai":
        # ローカル user も同じ vLLM を流用(OPENAI_API_BASE を共有)。
        # 別エンドポイントを使う場合は user.base_url を優先。
        if user.get("base_url"):
            env["OPENAI_API_BASE"] = user["base_url"]
            env["OPENAI_BASE_URL"] = user["base_url"]
        if key and key != "EMPTY":
            env["OPENAI_API_KEY"] = key
    return env


def parse_results(out_dir: str) -> Dict[str, Any]:
    """公式ランナーが吐く結果 JSON を拾って pass 率 / 平均 reward を集計する。

    tau-bench は log-dir に EnvRunResult のリスト(各 dict に "reward")を書く。
    フォーマット差に備え defensive に "reward" を再帰探索する。
    """
    files = sorted(glob.glob(os.path.join(out_dir, "**", "*.json"), recursive=True))
    rewards: List[float] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        records = data if isinstance(data, list) else data.get("results", [])
        if not isinstance(records, list):
            continue
        for r in records:
            if isinstance(r, dict) and "reward" in r:
                try:
                    rewards.append(float(r["reward"]))
                except (TypeError, ValueError):
                    pass

    n = len(rewards)
    return {
        "num_tasks_scored": n,
        "avg_reward": (sum(rewards) / n) if n else None,
        # τ-bench の pass は reward>=1(=完全成功)を成功とみなす慣例。
        "pass_rate": (sum(1 for x in rewards if x >= 1.0) / n) if n else None,
        "result_files": files,
    }


def run(config_path: str, phase: str, out_dir: str) -> int:
    cfg = load_config(config_path)
    phase_cfg = cfg["phases"][phase]
    bench = phase_cfg.get("bench", "tau")
    os.makedirs(out_dir, exist_ok=True)

    if bench == "mock":
        print("[eval] phase0(mock)は phase0_smoke.py を使ってください。",
              file=sys.stderr)
        return 2

    cmd = build_tau_command(cfg, phase_cfg, out_dir, bench)
    env = build_env(cfg, phase_cfg)
    print(f"[eval] {phase} ({phase_cfg.get('desc','')})")
    print(f"[eval] $ {' '.join(shlex.quote(c) for c in cmd)}")

    try:
        proc = subprocess.run(cmd, env=env, check=False)
    except FileNotFoundError as e:
        print(f"[eval] 公式ランナーが見つかりません: {e}\n"
              f"       TAU_RUN_CMD で正しい起動コマンドを指定してください。",
              file=sys.stderr)
        return 2

    summary = parse_results(out_dir)
    summary.update({
        "phase": phase,
        "bench": bench,
        "domain": phase_cfg.get("domain"),
        "runner_exit_code": proc.returncode,
    })
    summary_path = os.path.join(out_dir, f"summary_{phase}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[eval] 集計: pass_rate={summary['pass_rate']} "
          f"avg_reward={summary['avg_reward']} (n={summary['num_tasks_scored']})")
    print(f"[eval] -> {summary_path}")
    return 0 if proc.returncode == 0 else 1


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(here, "eval_config.yaml"))
    ap.add_argument("--phase", required=True, choices=["phase1", "phase2"])
    ap.add_argument("--out-dir", default=os.path.join(here, "results"))
    args = ap.parse_args()
    return run(args.config, args.phase, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
