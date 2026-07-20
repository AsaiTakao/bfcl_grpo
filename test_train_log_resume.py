# -*- coding: utf-8 -*-
"""
train.log の resume 追記対応のユニットテスト(TDD)。

なぜ: コンテナ再起動を挟む resume では workspace が消えるため、
train.log を HF へ push/restore しないと過去の学習経過が失われる。
また復元したログを train_monitor が先頭から再処理すると
val_metrics.jsonl に重複が生じるため、開始オフセットで新規行だけを読む。

固定する仕様:
  1. push_checkpoint は checkpoint と一緒に train.log も push する
  2. mode_restore は train.log も CKPT_DIR へ復元する
  3. TrainMonitor は start_offset より前(=復元した過去分)を処理しない

実行: python -m unittest test_train_log_resume(HF への実通信なし)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from argparse import Namespace


# ------------------------------------------------------------------ uploader
class FakeApi:
    """HfApi の記録用スタブ(実通信しない)。"""

    def __init__(self) -> None:
        self.uploaded_files: list[str] = []   # path_in_repo
        self.uploaded_folders: list[str] = []

    def create_repo(self, *a, **k) -> None:
        pass

    def upload_folder(self, *, path_in_repo: str, **k) -> None:
        self.uploaded_folders.append(path_in_repo)

    def upload_file(self, *, path_in_repo: str, **k) -> None:
        self.uploaded_files.append(path_in_repo)

    def list_repo_files(self, *a, **k) -> list[str]:
        return ["checkpoints/global_step_100/model.pt"]


class PushTrainLogTest(unittest.TestCase):
    def test_push_includes_train_log(self) -> None:
        """train.log があれば checkpoint push と同時に push される。"""
        import hf_checkpoint_uploader as up

        with tempfile.TemporaryDirectory() as d:
            step_dir = os.path.join(d, "global_step_100")
            os.makedirs(step_dir)
            for name in ("latest_checkpointed_iteration.txt", "train.log"):
                with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                    f.write("x")
            api = FakeApi()
            up.push_checkpoint(api, "user/repo", "checkpoints", 100, step_dir, d)
            self.assertIn("checkpoints/train.log", api.uploaded_files)

    def test_push_without_train_log_is_fine(self) -> None:
        """train.log が無くても push は失敗しない(存在チェック)。"""
        import hf_checkpoint_uploader as up

        with tempfile.TemporaryDirectory() as d:
            step_dir = os.path.join(d, "global_step_100")
            os.makedirs(step_dir)
            api = FakeApi()
            up.push_checkpoint(api, "user/repo", "checkpoints", 100, step_dir, d)
            self.assertNotIn("checkpoints/train.log", api.uploaded_files)


class RestoreTrainLogTest(unittest.TestCase):
    def test_restore_brings_back_train_log(self) -> None:
        """mode_restore は checkpoint と一緒に train.log も復元する。"""
        # huggingface_hub をスタブ化(実通信なし・未インストールでも動く)。
        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = lambda token=None: FakeApi()

        def fake_snapshot_download(*, local_dir: str, allow_patterns, **k) -> None:
            # restore が要求するパターンに応じて repo 内レイアウトを再現する。
            os.makedirs(os.path.join(local_dir, "checkpoints", "global_step_100"),
                        exist_ok=True)
            with open(os.path.join(local_dir, "checkpoints", "global_step_100",
                                   "model.pt"), "w", encoding="utf-8") as f:
                f.write("weights")
            for name, body in (("val_metrics.jsonl", '{"step":100}\n'),
                               ("train.log", "step:100 - loss:0.5\n")):
                if any(name in p for p in allow_patterns):
                    with open(os.path.join(local_dir, "checkpoints", name),
                              "w", encoding="utf-8") as f:
                        f.write(body)

        fake_hub.snapshot_download = fake_snapshot_download
        saved = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = fake_hub
        try:
            import hf_checkpoint_uploader as up

            with tempfile.TemporaryDirectory() as d:
                args = Namespace(ckpt_dir=d, repo_id="user/repo",
                                 prefix="checkpoints")
                rc = up.mode_restore(args, token=None)
                self.assertEqual(rc, 0)
                log_path = os.path.join(d, "train.log")
                self.assertTrue(os.path.isfile(log_path),
                                "train.log が復元されていない")
                with open(log_path, encoding="utf-8") as f:
                    self.assertIn("step:100", f.read())
        finally:
            if saved is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = saved


# ------------------------------------------------------------------- monitor
class MonitorOffsetTest(unittest.TestCase):
    def _dead_pid(self) -> int:
        """既に終了した実 PID を返す(monitor が即座に読み切りモードになる)。"""
        p = subprocess.Popen(["true"])
        p.wait()
        return p.pid

    def test_offset_skips_restored_history(self) -> None:
        """start_offset より前の val 行は val_metrics.jsonl に記録されない。"""
        from train_monitor import TrainMonitor

        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "train.log")
            old = "step:50 - val-core/x/reward/mean@1:0.9\n"
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(old)
            offset = os.path.getsize(log_path)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("step:100 - val-core/x/reward/mean@1:0.5\n")

            mon = TrainMonitor(log_path, d, self._dead_pid(),
                               start_offset=offset)
            self.assertEqual(mon.run(), 0)

            with open(os.path.join(d, "val_metrics.jsonl"),
                      encoding="utf-8") as f:
                recs = [json.loads(l) for l in f if l.strip()]
            self.assertEqual([r["step"] for r in recs], [100],
                             "復元済みの過去分(step=50)が再記録されている")

    def test_offset_beyond_size_is_safe(self) -> None:
        """ログが想定より短くても(切詰め等)クラッシュしない。"""
        from train_monitor import TrainMonitor

        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "train.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("short\n")
            mon = TrainMonitor(log_path, d, self._dead_pid(),
                               start_offset=10_000)
            self.assertEqual(mon.run(), 0)

    def test_default_offset_reads_from_start(self) -> None:
        """offset 未指定(0)なら従来どおり先頭から読む(後方互換)。"""
        from train_monitor import TrainMonitor

        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "train.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("step:50 - val-core/x/reward/mean@1:0.9\n")
            mon = TrainMonitor(log_path, d, self._dead_pid())
            self.assertEqual(mon.run(), 0)
            with open(os.path.join(d, "val_metrics.jsonl"),
                      encoding="utf-8") as f:
                recs = [json.loads(l) for l in f if l.strip()]
            self.assertEqual([r["step"] for r in recs], [50])


if __name__ == "__main__":
    unittest.main()
