# -*- coding: utf-8 -*-
"""
HF へ上げる checkpoint を LoRA adapter だけに絞る挙動のユニットテスト。

なぜ: フル checkpoint は 8B だと FSDP shard + optimizer 状態で ~80GB/個 になり、
SAVE_FREQ ごとの HF push が現実的でない。LoRA 学習なら adapter は ~200MB なので、
adapter と tokenizer/config だけを上げる。

固定する仕様:
  1. contents="lora" では upload_folder に adapter/tokenizer の allow_patterns が渡る
  2. contents="full" では allow_patterns を渡さない(従来どおり全部上げる)
  3. lora 指定なのに lora_adapter が無い checkpoint は push しない
     (黙って 80GB を上げてしまわないため)
  4. lora 指定のとき mode_restore は何もしない(optimizer 状態が無く resume 不可)

実行: python -m pytest tests/test_lora_only_push.py(HF への実通信なし)
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RecordingApi:
    """upload_folder に渡された allow_patterns を記録するスタブ。"""

    def __init__(self) -> None:
        self.folder_calls: list[dict] = []
        self.uploaded_files: list[str] = []

    def create_repo(self, *a, **k) -> None:
        pass

    def upload_folder(self, **k) -> None:
        self.folder_calls.append(k)

    def upload_file(self, *, path_in_repo: str, **k) -> None:
        self.uploaded_files.append(path_in_repo)

    def list_repo_files(self, *a, **k) -> list[str]:
        return []


def _make_ckpt(root: str, step: int, with_lora: bool) -> str:
    """verl のレイアウトを模した checkpoint を作る。"""
    step_dir = os.path.join(root, f"global_step_{step}")
    actor = os.path.join(step_dir, "actor")
    os.makedirs(os.path.join(actor, "huggingface"), exist_ok=True)
    with open(os.path.join(actor, "model_world_size_8_rank_0.pt"), "w") as f:
        f.write("shard")
    if with_lora:
        lora = os.path.join(actor, "lora_adapter")
        os.makedirs(lora, exist_ok=True)
        with open(os.path.join(lora, "adapter_model.safetensors"), "w") as f:
            f.write("adapter")
    return step_dir


class LoraOnlyPushTest(unittest.TestCase):
    def test_lora_contents_sets_allow_patterns(self) -> None:
        import hf_checkpoint_uploader as up

        with tempfile.TemporaryDirectory() as d:
            step_dir = _make_ckpt(d, 10, with_lora=True)
            api = RecordingApi()
            up.push_checkpoint(api, "user/repo", "checkpoints", 10, step_dir, d,
                               contents="lora")
            self.assertEqual(len(api.folder_calls), 1)
            allow = api.folder_calls[0]["allow_patterns"]
            self.assertIn("*/lora_adapter/*", allow)
            self.assertIn("*/huggingface/*", allow)

    def test_full_contents_has_no_allow_patterns(self) -> None:
        import hf_checkpoint_uploader as up

        with tempfile.TemporaryDirectory() as d:
            step_dir = _make_ckpt(d, 10, with_lora=True)
            api = RecordingApi()
            up.push_checkpoint(api, "user/repo", "checkpoints", 10, step_dir, d,
                               contents="full")
            self.assertIsNone(api.folder_calls[0]["allow_patterns"])

    def test_has_lora_adapter_detection(self) -> None:
        import hf_checkpoint_uploader as up

        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(up.has_lora_adapter(_make_ckpt(d, 1, True)))
            self.assertFalse(up.has_lora_adapter(_make_ckpt(d, 2, False)))

    def test_skips_push_when_no_adapter(self) -> None:
        """フル FT の checkpoint を lora 指定で上げようとしたら何もしない。"""
        import hf_checkpoint_uploader as up

        with tempfile.TemporaryDirectory() as d:
            step_dir = _make_ckpt(d, 10, with_lora=False)
            api = RecordingApi()
            args = Namespace(ckpt_dir=d, repo_id="user/repo", prefix="checkpoints",
                             keep_best=3, keep_latest=3, contents="lora")
            self.assertFalse(up._push_once(api, args, 10, step_dir))
            self.assertEqual(api.folder_calls, [])


class LoraRestoreTest(unittest.TestCase):
    def test_restore_is_noop_for_lora_contents(self) -> None:
        """adapter しか無い repo から resume を試みない。"""
        fake_hub = types.ModuleType("huggingface_hub")

        def _boom(*a, **k):  # 呼ばれたら失敗させる
            raise AssertionError("lora push では restore を試みてはいけない")

        fake_hub.HfApi = _boom
        fake_hub.snapshot_download = _boom
        saved = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = fake_hub
        try:
            import hf_checkpoint_uploader as up

            with tempfile.TemporaryDirectory() as d:
                args = Namespace(ckpt_dir=d, repo_id="user/repo",
                                 prefix="checkpoints", contents="lora")
                self.assertEqual(up.mode_restore(args, token=None), 0)
        finally:
            if saved is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = saved


if __name__ == "__main__":
    unittest.main()
