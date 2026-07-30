# -*- coding: utf-8 -*-
"""
学習ログの HF push のユニットテスト(実通信なし)。

なぜ: DOK のコンソールログは先頭が切り落とされ、起動時の設定・データ準備・
最初の step という一番読みたい部分が読めない。よってログは HF repo 側に必ず
残す必要がある。checkpoint push に相乗りさせるだけでは
  - 最初の checkpoint(SAVE_FREQ step 目)より前に落ちた
  - HF_PUSH_CONTENTS=lora で adapter が無く push をスキップした
  - push が失敗した
のいずれでもログが 1 バイトも残らない。

固定する仕様:
  1. CKPT_DIR 直下の *.log を(checkpoint とは独立に)push する
  2. HF_LOG_FILES で CKPT_DIR 外のログ(起動ログ)も対象にできる
  3. 変化していないファイルは送らない(空コミットを積まない)
  4. 巨大ログは「中央」を中略する(先頭は必ず残す)
  5. mode=logs / mode=final は checkpoint が無くてもログを push する
  6. mode=restore は checkpoint を戻さないケースでも train.log を戻す
     (repo 側は固定名なので、戻さないと今回分の短いログで前回分を上書きする)

実行: python -m pytest tests/test_log_upload.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from argparse import Namespace

import hf_checkpoint_uploader as up


class FakeApi:
    """HfApi の記録用スタブ。upload_file の中身も控える。"""

    def __init__(self) -> None:
        self.files: list[tuple[str, bytes | str]] = []   # (path_in_repo, payload)
        self.folders: list[str] = []
        self.created = 0

    def create_repo(self, *a, **k) -> None:
        self.created += 1

    def upload_folder(self, *, path_in_repo: str, **k) -> None:
        self.folders.append(path_in_repo)

    def upload_file(self, *, path_in_repo: str, path_or_fileobj=None, **k) -> None:
        self.files.append((path_in_repo, path_or_fileobj))

    def list_repo_files(self, *a, **k) -> list[str]:
        return []

    def paths(self) -> list[str]:
        return [p for p, _ in self.files]


def _args(ckpt_dir: str, **over) -> Namespace:
    base = dict(ckpt_dir=ckpt_dir, repo_id="user/repo", prefix="checkpoints",
                every=10, keep_best=3, keep_latest=3, poll=60,
                log_every=300, log_max_mb=50, reason="manual",
                contents="lora")
    base.update(over)
    return Namespace(**base)


class LogFileDiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.environ.pop("HF_LOG_FILES", None)
        self.addCleanup(os.environ.pop, "HF_LOG_FILES", None)

    def _touch(self, name: str, body: str = "x", d: str | None = None) -> str:
        path = os.path.join(d or self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        return path

    def test_picks_up_logs_in_ckpt_dir(self) -> None:
        self._touch("train.log")
        self._touch("entrypoint-20260730T000000Z.log")
        self._touch("val_metrics.jsonl")          # ログではない
        names = [os.path.basename(p) for p in up.log_files(self.tmp.name)]
        self.assertEqual(names, ["entrypoint-20260730T000000Z.log", "train.log"])

    def test_env_override_can_point_outside_ckpt_dir(self) -> None:
        # 起動ログは CODE_DIR/CKPT_DIR の外に置く(clone を壊さないため)。
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        boot = self._touch("entrypoint-x.log", d=other.name)
        train = self._touch("train.log")
        os.environ["HF_LOG_FILES"] = f"{train}:{boot}"
        self.assertEqual(up.log_files(self.tmp.name), [train, boot])

    def test_env_override_accepts_comma(self) -> None:
        train = self._touch("train.log")
        os.environ["HF_LOG_FILES"] = f"{train},{train}"
        self.assertEqual(up.log_files(self.tmp.name), [train])   # 同名は 1 つ

    def test_missing_files_are_skipped(self) -> None:
        os.environ["HF_LOG_FILES"] = os.path.join(self.tmp.name, "nope.log")
        self.assertEqual(up.log_files(self.tmp.name), [])

    def test_empty_dir_is_empty_list(self) -> None:
        self.assertEqual(up.log_files(self.tmp.name), [])

    def test_glob_is_reevaluated_each_call(self) -> None:
        # watch 起動時点では train.log がまだ存在しない(学習はこの後で始まる)。
        self.assertEqual(up.log_files(self.tmp.name), [])
        self._touch("train.log")
        self.assertEqual(len(up.log_files(self.tmp.name)), 1)


class CappedLogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, body: bytes) -> str:
        path = os.path.join(self.tmp.name, "train.log")
        with open(path, "wb") as f:
            f.write(body)
        return path

    def test_small_log_is_verbatim(self) -> None:
        path = self._write(b"hello\n")
        self.assertEqual(up.capped_log_bytes(path, 1000), b"hello\n")

    def test_zero_limit_means_no_cap(self) -> None:
        path = self._write(b"hello\n")
        self.assertEqual(up.capped_log_bytes(path, 0), b"hello\n")

    def test_large_log_keeps_head_and_tail(self) -> None:
        # 先頭を捨てる実装だと DOK コンソールと同じ「先頭が切れたログ」になる。
        path = self._write(b"HEAD" + b"." * 5000 + b"TAIL")
        out = up.capped_log_bytes(path, 1000)
        self.assertTrue(out.startswith(b"HEAD"))
        self.assertTrue(out.endswith(b"TAIL"))
        self.assertIn("中略".encode(), out)
        self.assertLess(len(out), 1300)   # 上限 + 中略マーカー程度


class LogSyncTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.environ.pop("HF_LOG_FILES", None)
        self.addCleanup(os.environ.pop, "HF_LOG_FILES", None)
        self.api = FakeApi()

    def _sync(self, **over) -> up.LogSync:
        return up.LogSync(self.api, "user/repo", "checkpoints", self.tmp.name,
                          **over)

    def _write(self, name: str, body: str) -> str:
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        return path

    def test_uploads_under_prefix_with_basename(self) -> None:
        self._write("train.log", "step:1\n")
        self.assertEqual(self._sync().sync(force=True), 1)
        self.assertEqual(self.api.paths(), ["checkpoints/train.log"])

    def test_unchanged_file_is_not_reuploaded(self) -> None:
        self._write("train.log", "step:1\n")
        s = self._sync()
        self.assertEqual(s.sync(force=True), 1)
        self.assertEqual(s.sync(force=True), 0)   # 空コミットを積まない
        self.assertEqual(len(self.api.files), 1)

    def test_appended_file_is_reuploaded(self) -> None:
        path = self._write("train.log", "step:1\n")
        s = self._sync()
        s.sync(force=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write("step:2\n")
        self.assertEqual(s.sync(force=True), 1)
        self.assertEqual(len(self.api.files), 2)
        self.assertIn(b"step:2", self.api.files[-1][1])

    def test_interval_gates_periodic_sync(self) -> None:
        self._write("train.log", "step:1\n")
        s = self._sync(interval=3600)
        self.assertEqual(s.sync(), 1)            # 初回は即送る
        self._write("train.log", "step:1\nstep:2\n")
        self.assertEqual(s.sync(), 0)            # interval 内は送らない
        self.assertEqual(s.sync(force=True), 1)  # force なら送る

    def test_upload_failure_does_not_raise(self) -> None:
        # ログ送信の失敗で学習を止めてはいけない。
        class Boom(FakeApi):
            def upload_file(self, **k):
                raise RuntimeError("network down")

        self._write("train.log", "step:1\n")
        s = up.LogSync(Boom(), "user/repo", "checkpoints", self.tmp.name)
        self.assertEqual(s.sync(force=True), 0)

    def test_failed_upload_is_retried_next_time(self) -> None:
        class Flaky(FakeApi):
            def __init__(self) -> None:
                super().__init__()
                self.fail = True

            def upload_file(self, **k):
                if self.fail:
                    raise RuntimeError("temporary")
                super().upload_file(**k)

        api = Flaky()
        self._write("train.log", "step:1\n")
        s = up.LogSync(api, "user/repo", "checkpoints", self.tmp.name)
        s.sync(force=True)
        api.fail = False
        self.assertEqual(s.sync(force=True), 1)   # 失敗を「送信済み」にしない

    def test_multiple_logs_are_all_sent(self) -> None:
        self._write("train.log", "a")
        self._write("entrypoint-1.log", "b")
        self.assertEqual(self._sync().sync(force=True), 2)
        self.assertEqual(sorted(self.api.paths()),
                         ["checkpoints/entrypoint-1.log", "checkpoints/train.log"])


class ModeTest(unittest.TestCase):
    """checkpoint が無くてもログは push される。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.environ.pop("HF_LOG_FILES", None)
        self.addCleanup(os.environ.pop, "HF_LOG_FILES", None)
        self.api = FakeApi()
        self._orig = up._api
        up._api = lambda token: self.api
        self.addCleanup(setattr, up, "_api", self._orig)
        with open(os.path.join(self.tmp.name, "train.log"), "w",
                  encoding="utf-8") as f:
            f.write("step:1 - loss:0.5\n")

    def test_mode_logs_pushes_without_checkpoint(self) -> None:
        self.assertEqual(up.mode_logs(_args(self.tmp.name), token=None), 0)
        self.assertEqual(self.api.paths(), ["checkpoints/train.log"])

    def test_mode_logs_reason_is_in_commit_message(self) -> None:
        captured = {}

        def upload_file(*, path_in_repo, commit_message=None, **k):
            captured["msg"] = commit_message

        self.api.upload_file = upload_file  # type: ignore[assignment]
        up.mode_logs(_args(self.tmp.name, reason="exit=137"), token=None)
        self.assertIn("exit=137", captured["msg"])

    def test_mode_final_pushes_logs_even_without_checkpoint(self) -> None:
        # 学習が最初の save 前に落ちたケース。ここでログを送らないと何も残らない。
        self.assertEqual(up.mode_final(_args(self.tmp.name), token=None), 0)
        self.assertEqual(self.api.paths(), ["checkpoints/train.log"])

    def test_mode_logs_with_no_log_files_is_ok(self) -> None:
        os.remove(os.path.join(self.tmp.name, "train.log"))
        self.assertEqual(up.mode_logs(_args(self.tmp.name), token=None), 0)
        self.assertEqual(self.api.paths(), [])


class RestoreLogTest(unittest.TestCase):
    """checkpoint を戻さないモードでも train.log は戻す。

    HF_PUSH_CONTENTS の既定は lora で、そのとき mode_restore は checkpoint を
    復元しない(optimizer 状態が無く resume できないため)。しかしログを
    戻さないまま学習を始めると、LogSync が checkpoints/train.log を今回起動分の
    短いログで上書きし、前回までの学習経過が HF から消える。
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.requested: list[list[str]] = []   # snapshot_download の要求パターン
        self.remote = {"train.log": "step:1 - 前回起動分\n",
                       "val_metrics.jsonl": '{"step":1,"value":0.5}\n'}

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = lambda token=None: FakeApi()

        def fake_snapshot_download(*, local_dir: str, allow_patterns, **k) -> None:
            self.requested.append(list(allow_patterns))
            os.makedirs(os.path.join(local_dir, "checkpoints"), exist_ok=True)
            for name, body in self.remote.items():
                if not any(name in p for p in allow_patterns):
                    continue
                with open(os.path.join(local_dir, "checkpoints", name),
                          "w", encoding="utf-8") as f:
                    f.write(body)

        fake_hub.snapshot_download = fake_snapshot_download
        self._saved = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = fake_hub
        self.addCleanup(self._restore_hub)

    def _restore_hub(self) -> None:
        if self._saved is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = self._saved

    def _read(self, name: str) -> str:
        with open(os.path.join(self.tmp.name, name), encoding="utf-8") as f:
            return f.read()

    def test_lora_restore_brings_back_train_log(self) -> None:
        args = _args(self.tmp.name, contents="lora")
        self.assertEqual(up.mode_restore(args, token=None), 0)
        self.assertIn("前回起動分", self._read("train.log"))
        self.assertIn("step", self._read("val_metrics.jsonl"))

    def test_lora_restore_does_not_bring_back_checkpoint_pointer(self) -> None:
        # latest_checkpointed_iteration.txt を戻すと resume_mode=auto が
        # 存在しない checkpoint を掴んで壊れる。
        up.mode_restore(_args(self.tmp.name, contents="lora"), token=None)
        self.assertFalse(os.path.isfile(
            os.path.join(self.tmp.name, "latest_checkpointed_iteration.txt")))
        for pats in self.requested:
            self.assertFalse(any("latest_checkpointed" in p for p in pats))

    def test_existing_local_log_is_not_overwritten(self) -> None:
        with open(os.path.join(self.tmp.name, "train.log"), "w",
                  encoding="utf-8") as f:
            f.write("今回起動分\n")
        up.mode_restore(_args(self.tmp.name, contents="lora"), token=None)
        self.assertEqual(self._read("train.log"), "今回起動分\n")

    def test_no_download_when_nothing_missing(self) -> None:
        for name in ("train.log", "val_metrics.jsonl"):
            with open(os.path.join(self.tmp.name, name), "w",
                      encoding="utf-8") as f:
                f.write("x")
        up.mode_restore(_args(self.tmp.name, contents="lora"), token=None)
        self.assertEqual(self.requested, [])   # 無駄な DL をしない

    def test_download_failure_is_not_fatal(self) -> None:
        # 初回起動では repo 自体が存在しない。ここで落ちたら学習が始まらない。
        def boom(**k):
            raise RuntimeError("404 repo not found")

        sys.modules["huggingface_hub"].snapshot_download = boom
        self.assertEqual(
            up.mode_restore(_args(self.tmp.name, contents="lora"), token=None), 0)
        self.assertFalse(os.path.isfile(os.path.join(self.tmp.name, "train.log")))

    def test_tmp_dir_is_cleaned_up(self) -> None:
        # 残骸が CKPT_DIR に残ると *.log の glob や checkpoint 列挙を汚す。
        up.mode_restore(_args(self.tmp.name, contents="lora"), token=None)
        self.assertEqual(
            sorted(os.listdir(self.tmp.name)), ["train.log", "val_metrics.jsonl"])

    def test_full_contents_restores_log_when_no_remote_checkpoint(self) -> None:
        # 最初の save より前に落ちた → checkpoint は無いがログはある。
        self.assertEqual(
            up.mode_restore(_args(self.tmp.name, contents="full"), token=None), 0)
        self.assertIn("前回起動分", self._read("train.log"))


if __name__ == "__main__":
    unittest.main()
