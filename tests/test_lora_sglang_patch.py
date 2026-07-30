# -*- coding: utf-8 -*-
"""
verl_lora_sglang_patch の単体テスト。

なぜ: verl 0.4.1 の sglang 経路は LoRA を扱えないため、rollout エンジンへ渡す
直前に LoRA を base へマージするパッチを当てている。ここが誤っていると
「学習は進むが rollout は base 重みのまま」という気付きにくい壊れ方をするので、
名前の変換とマージ計算を実機に投げる前に固定しておく。

固定する仕様:
  1. PEFT のキーが素の HF キーに戻る(base_model.model. と .base_layer を除去)
  2. lora_A / lora_B は出力に残らない
  3. W' = W + (B @ A) * (alpha / r) が実際に加算される
  4. use_rslora=True では scaling が alpha / sqrt(r) になる
  5. DoRA は明示的に未対応エラー
  6. 対応する lora_B / base 重みが欠けていたら黙って進まない
  7. asyncio.get_event_loop の uvloop 互換フォールバック(ループが無ければ作る /
     同じループを使い続ける / 閉じたループは作り直す)

実行: python -m pytest tests/test_lora_sglang_patch.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import verl_lora_sglang_patch as patch  # noqa: E402

try:
    import torch
except ImportError:  # CI ではインストールしない(重いため)
    torch = None

_Q = ("base_model.model.model.layers.0.self_attn.q_proj")


class NameTest(unittest.TestCase):
    def test_strip_peft_name(self) -> None:
        self.assertEqual(
            patch.strip_peft_name(f"{_Q}.base_layer.weight"),
            "model.layers.0.self_attn.q_proj.weight")

    def test_strip_keeps_non_lora_modules(self) -> None:
        """LoRA 対象外(layernorm 等)は接頭辞を外すだけ。"""
        self.assertEqual(
            patch.strip_peft_name(
                "base_model.model.model.layers.0.input_layernorm.weight"),
            "model.layers.0.input_layernorm.weight")
        self.assertEqual(
            patch.strip_peft_name("base_model.model.lm_head.weight"),
            "lm_head.weight")

    def test_strip_handles_fsdp_prefix(self) -> None:
        """FSDP のラップ接頭辞が残っている経路でも素のキーに戻す。"""
        self.assertEqual(
            patch.strip_peft_name(
                "_fsdp_wrapped_module.base_model.model.model.layers.0."
                "self_attn.q_proj.base_layer.weight"),
            "model.layers.0.self_attn.q_proj.weight")

    def test_is_lora_param(self) -> None:
        self.assertTrue(patch.is_lora_param(f"{_Q}.lora_A.default.weight"))
        self.assertTrue(patch.is_lora_param(f"{_Q}.lora_B.default.weight"))
        self.assertFalse(patch.is_lora_param(f"{_Q}.base_layer.weight"))


class ScalingTest(unittest.TestCase):
    class _Cfg:
        def __init__(self, r, alpha, rslora=False):
            self.r, self.lora_alpha, self.use_rslora = r, alpha, rslora

    def test_plain_scaling(self) -> None:
        self.assertAlmostEqual(
            patch.scaling_for({"default": self._Cfg(32, 32)}, "default"), 1.0)
        self.assertAlmostEqual(
            patch.scaling_for({"default": self._Cfg(16, 32)}, "default"), 2.0)

    def test_rslora_scaling(self) -> None:
        import math
        self.assertAlmostEqual(
            patch.scaling_for({"default": self._Cfg(16, 32, True)}, "default"),
            32 / math.sqrt(16))


@unittest.skipIf(torch is None, "torch 未インストール")
class MergeTest(unittest.TestCase):
    def _state_dict(self, r=4, in_f=8, out_f=6, dtype=None):
        dtype = dtype or torch.float32
        torch.manual_seed(0)
        return {
            f"{_Q}.base_layer.weight": torch.randn(out_f, in_f, dtype=dtype),
            f"{_Q}.lora_A.default.weight": torch.randn(r, in_f, dtype=dtype),
            f"{_Q}.lora_B.default.weight": torch.randn(out_f, r, dtype=dtype),
            "base_model.model.model.layers.0.input_layernorm.weight":
                torch.ones(in_f, dtype=dtype),
        }

    def test_merge_matches_reference(self) -> None:
        """W + (B @ A) * scaling が PEFT の定義どおり加算される。"""
        sd = self._state_dict()
        w0 = sd[f"{_Q}.base_layer.weight"].clone()
        a = sd[f"{_Q}.lora_A.default.weight"].clone()
        b = sd[f"{_Q}.lora_B.default.weight"].clone()

        out = patch.merge_lora_state_dict(sd, {"default": 2.0})

        expected = w0 + (b @ a) * 2.0
        got = out["model.layers.0.self_attn.q_proj.weight"]
        self.assertTrue(torch.allclose(got, expected, atol=1e-6),
                        f"max diff={(got - expected).abs().max()}")

    def test_lora_params_are_dropped(self) -> None:
        out = patch.merge_lora_state_dict(self._state_dict(), {"default": 1.0})
        self.assertNotIn(f"{_Q}.lora_A.default.weight", out)
        self.assertFalse([k for k in out if "lora_" in k])
        # LoRA 対象外の重みはそのまま残る
        self.assertIn("model.layers.0.input_layernorm.weight", out)

    def test_bf16_weights_stay_bf16(self) -> None:
        """rollout へ渡す dtype を勝手に変えない(fp32 で計算して戻す)。"""
        sd = self._state_dict(dtype=torch.bfloat16)
        out = patch.merge_lora_state_dict(sd, {"default": 1.0})
        self.assertEqual(out["model.layers.0.self_attn.q_proj.weight"].dtype,
                         torch.bfloat16)

    def test_missing_lora_b_raises(self) -> None:
        sd = self._state_dict()
        del sd[f"{_Q}.lora_B.default.weight"]
        with self.assertRaises(KeyError):
            patch.merge_lora_state_dict(sd, {"default": 1.0})

    def test_missing_base_weight_raises(self) -> None:
        sd = self._state_dict()
        del sd[f"{_Q}.base_layer.weight"]
        with self.assertRaises(KeyError):
            patch.merge_lora_state_dict(sd, {"default": 1.0})

    def test_no_lora_raises(self) -> None:
        """LoRA が1つも無い state_dict は異常(パッチ対象を取り違えている)。"""
        with self.assertRaises(RuntimeError):
            patch.merge_lora_state_dict(
                {"base_model.model.lm_head.weight": torch.ones(2, 2)},
                {"default": 1.0})

    def test_dora_raises(self) -> None:
        sd = self._state_dict()
        sd[f"{_Q}.lora_magnitude_vector.default"] = torch.ones(6)
        with self.assertRaises(NotImplementedError):
            patch.merge_lora_state_dict(sd, {"default": 1.0})

    def test_unknown_adapter_raises(self) -> None:
        sd = self._state_dict()
        with self.assertRaises(KeyError):
            patch.merge_lora_state_dict(sd, {"other": 1.0})


@unittest.skipIf(torch is None, "torch 未インストール")
class PeftEquivalenceTest(unittest.TestCase):
    """peft が入っていれば、本物の merge_and_unload と一致することを確かめる。"""

    def test_matches_peft_merge_and_unload(self) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest("peft 未インストール")

        torch.manual_seed(0)
        base = torch.nn.Sequential()
        base.add_module("q_proj", torch.nn.Linear(8, 6, bias=False))
        model = get_peft_model(
            base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"]))
        # LoRA B は 0 初期化なので、差が出るよう乱数を入れる。
        for n, p in model.named_parameters():
            if "lora_B" in n:
                torch.nn.init.normal_(p)

        sd = {f"base_model.model.{k}": v.clone()
              for k, v in model.base_model.model.state_dict().items()}
        merged = patch.merge_lora_state_dict(sd, {"default": 8 / 4})

        reference = model.merge_and_unload().state_dict()["q_proj.weight"]
        self.assertTrue(
            torch.allclose(merged["q_proj.weight"], reference, atol=1e-6),
            f"max diff={(merged['q_proj.weight'] - reference).abs().max()}")


class EventLoopFallbackTest(unittest.TestCase):
    """asyncio.get_event_loop の uvloop 互換フォールバック。

    verl 0.4.1 は worker のメインスレッドで get_event_loop + run_until_complete
    を使うが、sglang が入れる uvloop の get_event_loop は current が無いと
    RuntimeError を投げる(実機で初回 rollout が
    "There is no current event loop in thread 'MainThread'" で停止した)。
    """

    def setUp(self) -> None:
        # プロセスの実際のループ状態を触らずに済むよう、get/set とも差し替える。
        self._saved_get = asyncio.get_event_loop
        self._saved_set = asyncio.set_event_loop
        self._current = None
        self._created = []
        asyncio.set_event_loop = self._record_current

    def tearDown(self) -> None:
        asyncio.get_event_loop = self._saved_get
        asyncio.set_event_loop = self._saved_set
        for loop in self._created:
            if not loop.is_closed():
                loop.close()

    def _record_current(self, loop) -> None:
        self._current = loop

    def _install_uvloop_like(self) -> None:
        """current が無いと必ず RuntimeError を投げる uvloop 相当に差し替える。"""
        def strict_get_event_loop():
            if self._current is None:
                raise RuntimeError("There is no current event loop in thread 'MainThread'.")
            return self._current

        asyncio.get_event_loop = strict_get_event_loop
        patch.ensure_event_loop_available()

    def test_creates_loop_when_missing(self) -> None:
        self._install_uvloop_like()
        loop = asyncio.get_event_loop()
        self._created.append(loop)
        self.assertFalse(loop.is_closed())
        self.assertEqual(loop.run_until_complete(_answer()), 42)

    def test_reuses_the_same_loop(self) -> None:
        self._install_uvloop_like()
        first = asyncio.get_event_loop()
        self._created.append(first)
        # 作ったループは current に残るので、呼び直しても同じものが返る
        # (sglang エンジンの非同期状態と齟齬が出ないことが重要)。
        self.assertIs(asyncio.get_event_loop(), first)

    def test_replaces_closed_loop(self) -> None:
        closed = asyncio.new_event_loop()
        closed.close()
        asyncio.get_event_loop = lambda: closed
        patch.ensure_event_loop_available()

        loop = asyncio.get_event_loop()
        self._created.append(loop)
        self.assertIsNot(loop, closed)
        self.assertFalse(loop.is_closed())

    def test_is_idempotent(self) -> None:
        self._install_uvloop_like()
        patched = asyncio.get_event_loop
        self.assertFalse(patch.ensure_event_loop_available())
        self.assertIs(asyncio.get_event_loop, patched)


async def _answer() -> int:
    return 42


if __name__ == "__main__":
    unittest.main()
