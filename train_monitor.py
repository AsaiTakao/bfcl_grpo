# -*- coding: utf-8 -*-
"""
train_monitor.py

train.log を tail して学習を監視する常駐プロセス(entrypoint.sh が背景起動)。
設計書(BFCL学習τ2-bench評価.txt)対応:
  - 「eval_steps ごとの train.log を記録」
      verl console logger の validation 行から (step, val 指標) を抽出し、
      ${CKPT_DIR}/val_metrics.jsonl に追記する。
      → hf_checkpoint_uploader が「最良 eval 3 つ」の選定と
        「merge 版にどの CP か・eval の数字を記録」に使う。
  - 「EarlyStoppingCallBack(patience=3)」
      val 指標(既定: reward 系、大きいほど良い)が patience 回連続で
      改善しなければ学習プロセス(プロセスグループ)へ SIGTERM を送り停止。
      ${CKPT_DIR}/early_stopped.json を残す(entrypoint が正常終了扱いにする)。
  - 二層報酬 λ 減衰への step 供給
      verl はツール側へ global_step を渡さないため、ログの step を
      ${CKPT_DIR}/global_step.txt に書き出し、BFCLMultiTurnTool が読む。

verl console 出力の想定形式(Tracking logger=console):
  step:50 - ... - val-core/.../reward/mean@1:0.42 - ...
バージョン差で形式が違う場合は環境変数で調整する:
  VAL_METRIC_KEY   val 指標キーに含まれるべき部分文字列(既定 "reward")
  VAL_LINE_REGEX   丸ごと差し替え用 regex(named group: step, value)

環境変数:
  TRAIN_LOG            監視対象ログ(必須。--log でも指定可)
  CKPT_DIR             出力先(val_metrics.jsonl / global_step.txt / early_stopped.json)
  EARLY_STOP           1 で早期終了を有効(既定 1)
  EARLY_STOP_PATIENCE  改善なし許容回数(既定 3)
  MONITOR_POLL_SECONDS tail の polling 間隔秒(既定 5)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from typing import Optional, TextIO

_STEP_RE = re.compile(r"\bstep:(\d+)\b")
# "key:value" ペア(verl console logger の " - " 区切り形式)。
_PAIR_RE = re.compile(
    r"([A-Za-z0-9_@/.\-]+):(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?=\s|$|\s*-\s)"
)


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def log(msg: str) -> None:
    print(f"[monitor] {msg}", flush=True)


class TrainMonitor:
    def __init__(self, train_log: str, ckpt_dir: str, train_pid: int,
                 start_offset: int = 0) -> None:
        self.train_log = train_log
        self.ckpt_dir = ckpt_dir
        self.train_pid = train_pid
        # resume 時、HF から復元した過去ログを再処理すると val_metrics.jsonl が
        # 重複するため、今回の学習が書き始める位置(バイト)から読む。
        self.start_offset = max(int(start_offset), 0)
        self.metric_key_sub = _env("VAL_METRIC_KEY", "reward")
        self.patience = int(_env("EARLY_STOP_PATIENCE", "3"))
        self.early_stop = _env("EARLY_STOP", "1") == "1"
        self.poll = max(int(_env("MONITOR_POLL_SECONDS", "5")), 1)
        custom = os.getenv("VAL_LINE_REGEX", "")
        self.custom_re = re.compile(custom) if custom else None

        self.metrics_path = os.path.join(ckpt_dir, "val_metrics.jsonl")
        self.step_path = os.path.join(ckpt_dir, "global_step.txt")
        self.marker_path = os.path.join(ckpt_dir, "early_stopped.json")

        self.chosen_key: Optional[str] = None  # 一度選んだ指標キーに固定する
        self.best_value: Optional[float] = None
        self.best_step: int = -1
        self.bad_evals: int = 0
        self.last_step: int = -1

    # ------------------------------------------------------------- 行の解釈
    def handle_line(self, line: str) -> None:
        step = self._extract_step(line)
        if step is not None and step != self.last_step:
            self.last_step = step
            self._write_step(step)

        val = self._extract_val_metric(line)
        if val is None:
            return
        key, value = val
        step = self.last_step if step is None else step
        self._record_metric(step if step is not None else -1, key, value)

    def _extract_step(self, line: str) -> Optional[int]:
        m = _STEP_RE.search(line)
        return int(m.group(1)) if m else None

    def _extract_val_metric(self, line: str) -> Optional[tuple[str, float]]:
        """行から val 指標 (key, value) を 1 つ選ぶ。無ければ None。"""
        if self.custom_re is not None:
            m = self.custom_re.search(line)
            if m:
                return ("custom", float(m.group("value")))
            return None
        if "val" not in line:
            return None
        candidates: list[tuple[str, float]] = []
        for key, raw in _PAIR_RE.findall(line):
            if "val" not in key:
                continue
            if self.metric_key_sub and self.metric_key_sub not in key:
                continue
            try:
                candidates.append((key, float(raw)))
            except ValueError:
                continue
        if not candidates:
            return None
        if self.chosen_key is None:
            # 最初に見つけたキーへ固定(以後、同じ指標で一貫して比較する)。
            self.chosen_key = candidates[0][0]
            log(f"val 指標キーを選択: {self.chosen_key}")
        for key, value in candidates:
            if key == self.chosen_key:
                return (key, value)
        return None

    # ------------------------------------------------------------- 記録と判定
    def _write_step(self, step: int) -> None:
        try:
            tmp = self.step_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(str(step))
            os.replace(tmp, self.step_path)
        except OSError as e:
            log(f"global_step.txt 書き込み失敗: {e}")

    def _record_metric(self, step: int, key: str, value: float) -> None:
        rec = {"step": step, "key": key, "value": value, "ts": time.time()}
        try:
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            log(f"val_metrics.jsonl 追記失敗: {e}")
        log(f"val step={step} {key}={value}")

        # EarlyStopping(大きいほど良い指標を前提: reward / pass 率)。
        if self.best_value is None or value > self.best_value:
            self.best_value = value
            self.best_step = step
            self.bad_evals = 0
            return
        self.bad_evals += 1
        log(f"改善なし {self.bad_evals}/{self.patience} "
            f"(best={self.best_value} @step {self.best_step})")
        if self.early_stop and self.bad_evals >= self.patience:
            self._trigger_early_stop(step)

    def _trigger_early_stop(self, step: int) -> None:
        info = {
            "stopped_at_step": step,
            "best_value": self.best_value,
            "best_step": self.best_step,
            "patience": self.patience,
            "metric_key": self.chosen_key,
        }
        try:
            with open(self.marker_path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
        except OSError as e:
            log(f"early_stopped.json 書き込み失敗: {e}")
        log(f"EarlyStopping 発火: {info}")
        # entrypoint は setsid で学習を独立プロセスグループにしている前提。
        try:
            os.killpg(os.getpgid(self.train_pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                os.kill(self.train_pid, signal.SIGTERM)
            except OSError:
                pass

    # --------------------------------------------------------------- tail 本体
    def _train_alive(self) -> bool:
        try:
            os.kill(self.train_pid, 0)
            return True
        except OSError:
            return False

    def run(self) -> int:
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # ログファイルが作られるまで待つ。
        while not os.path.exists(self.train_log):
            if not self._train_alive():
                log("学習プロセスが終了済み。監視を終了します。")
                return 0
            time.sleep(self.poll)

        f: TextIO = open(self.train_log, "r", encoding="utf-8", errors="replace")
        if self.start_offset > 0:
            # 復元後にログが短くなっているケースでも安全に(末尾へクランプ)。
            size = os.fstat(f.fileno()).st_size
            f.seek(min(self.start_offset, size))
        log(f"tail 開始: {self.train_log} (train_pid={self.train_pid}, "
            f"early_stop={self.early_stop}, patience={self.patience}, "
            f"offset={self.start_offset})")
        try:
            while True:
                line = f.readline()
                if line:
                    self.handle_line(line.rstrip("\n"))
                    continue
                if not self._train_alive():
                    # 取り残しを読み切ってから終了。
                    for rest in f:
                        self.handle_line(rest.rstrip("\n"))
                    log("学習プロセス終了を検知。監視を終了します。")
                    return 0
                time.sleep(self.poll)
        finally:
            f.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.getenv("TRAIN_LOG"))
    ap.add_argument("--ckpt-dir", default=os.getenv("CKPT_DIR"))
    ap.add_argument("--train-pid", type=int, required=True)
    ap.add_argument("--start-offset", type=int,
                    default=int(os.getenv("TRAIN_LOG_OFFSET", "0") or "0"))
    args = ap.parse_args()
    if not args.log or not args.ckpt_dir:
        print("[monitor] TRAIN_LOG / CKPT_DIR 未設定のため終了します。",
              file=sys.stderr, flush=True)
        return 0
    return TrainMonitor(args.log, args.ckpt_dir, args.train_pid,
                        start_offset=args.start_offset).run()


if __name__ == "__main__":
    raise SystemExit(main())
