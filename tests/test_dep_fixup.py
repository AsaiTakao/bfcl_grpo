# -*- coding: utf-8 -*-
"""
dep_fixup(実行時の依存ピン検査/自己修復)のユニットテスト。

なぜ: sglang 0.4.6.post5 は torchao / transformers を厳密に固定するのに対し、
peft と compressed-tensors は上限なしの推移依存で、ビルド時期によって
2026 年の最新版が乗る。実機ではこれで 2 回学習が落ちている
(peft>=0.19 → torchao>=0.16 要求 / compressed-tensors>=0.16 →
transformers.masking_utils 参照)。pip を叩く部分は実機依存なので、
「どれをズレと判定し、何を入れ直すか」の判断だけを固定する。

実行: python -m unittest test_dep_fixup
"""
from __future__ import annotations

import unittest

import dep_fixup


class ResolvePinsTest(unittest.TestCase):
    def test_defaults_cover_the_four_skew_prone_packages(self) -> None:
        pins = dep_fixup.resolve_pins({})
        self.assertEqual(
            set(pins),
            {"transformers", "torchao", "peft", "compressed-tensors"},
        )

    def test_env_override(self) -> None:
        pins = dep_fixup.resolve_pins({"DEP_PIN_PEFT": "0.16.0"})
        self.assertEqual(pins["peft"], "0.16.0")

    def test_env_override_with_dash_in_name(self) -> None:
        pins = dep_fixup.resolve_pins({"DEP_PIN_COMPRESSED_TENSORS": "0.10.2"})
        self.assertEqual(pins["compressed-tensors"], "0.10.2")

    def test_empty_env_value_disables_the_pin(self) -> None:
        pins = dep_fixup.resolve_pins({"DEP_PIN_TRANSFORMERS": ""})
        self.assertNotIn("transformers", pins)


class MismatchedTest(unittest.TestCase):
    def test_all_pinned_versions_are_ok(self) -> None:
        pins = {"peft": "0.15.2", "torchao": "0.9.0"}
        self.assertEqual(mismatch_names(pins, {"peft": "0.15.2", "torchao": "0.9.0"}), [])

    def test_reports_the_observed_skew(self) -> None:
        # 実機で踏んだ組み合わせ。両方とも入れ直し対象になる。
        pins = dep_fixup.resolve_pins({})
        versions = {
            "transformers": "4.51.1",
            "torchao": "0.9.0",
            "peft": "0.20.0",
            "compressed-tensors": "0.17.1",
        }
        self.assertEqual(mismatch_names(pins, versions), ["peft", "compressed-tensors"])

    def test_missing_package_counts_as_mismatch(self) -> None:
        pins = {"peft": "0.15.2"}
        bad = dep_fixup.mismatched(pins, {"peft": None})
        self.assertEqual(bad, [("peft", None, "0.15.2")])


def mismatch_names(pins, versions):
    return [package for package, _, _ in dep_fixup.mismatched(pins, versions)]


if __name__ == "__main__":
    unittest.main()
