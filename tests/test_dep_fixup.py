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


class OtelPinsTest(unittest.TestCase):
    """otel は固定値ではなく「入っている sdk の Requires-Dist」から導く。

    なぜ: ray の dashboard agent が opentelemetry.exporter.prometheus を
    import できないとノード起動が timeout する。正解のリリース列は
    イメージに焼かれた sdk で変わるので、判断ロジックだけを固定する。
    """

    SDK_REQUIRES = [
        "opentelemetry-api==1.37.0",
        "opentelemetry-semantic-conventions==0.58b0",
        "typing-extensions>=4.5.0",
        'importlib-metadata<8.8.0,>=6.0; python_version < "3.13"',
    ]

    def test_exporter_follows_the_semconv_release_train(self) -> None:
        want = dep_fixup.otel_pins(self.SDK_REQUIRES)
        self.assertEqual(want, {
            "opentelemetry-api": "1.37.0",
            "opentelemetry-semantic-conventions": "0.58b0",
            "opentelemetry-exporter-prometheus": "0.58b0",
        })

    def test_loose_requirements_and_markers_are_ignored(self) -> None:
        pins = dep_fixup.parse_pinned_requires(self.SDK_REQUIRES)
        self.assertEqual(set(pins),
                         {"opentelemetry-api", "opentelemetry-semantic-conventions"})

    def test_extras_are_stripped(self) -> None:
        pins = dep_fixup.parse_pinned_requires(["opentelemetry-exporter-otlp[grpc]==0.58b0"])
        self.assertEqual(pins, {"opentelemetry-exporter-otlp": "0.58b0"})

    def test_without_a_semconv_pin_the_exporter_is_left_alone(self) -> None:
        # 列が決まらないのに exporter を動かすと、直すつもりで壊す。
        want = dep_fixup.otel_pins(["opentelemetry-api>=1.30"])
        self.assertEqual(want, {})

    def test_observed_skew_is_reported(self) -> None:
        # 実機で踏んだ形: semconv だけ古い列に残り、
        # OtelComponentTypeValues が無くて ray が起動しない。
        want = dep_fixup.otel_pins(self.SDK_REQUIRES)
        versions = {
            "opentelemetry-api": "1.37.0",
            "opentelemetry-semantic-conventions": "0.55b1",
            "opentelemetry-exporter-prometheus": "0.58b0",
        }
        self.assertEqual(mismatch_names(want, versions),
                         ["opentelemetry-semantic-conventions"])

    def test_env_can_disable_the_otel_alignment(self) -> None:
        self.assertEqual(dep_fixup.resolve_otel_pins({"DEP_OTEL_FIX": "0"}), {})


class OtelFloorTest(unittest.TestCase):
    """揃える先は「入っている sdk」ではなく「ray が要求する下限」。

    なぜ: 実機イメージは api/sdk=1.26.0 に対し exporter=0.65b0 だった。
    sdk 基準で揃えると exporter が 0.47b0 まで落ち、ray の
    create_histogram(explicit_bucket_boundaries_advisory=...) が
    TypeError になってノード起動が timeout する(1 敗)。
    """

    RAY_REQUIRES = [
        "opentelemetry-sdk>=1.30.0; extra == 'default'",
        "opentelemetry-exporter-prometheus; extra == 'default'",
        "opentelemetry-proto; extra == 'default'",
        "click>=7.0",
    ]

    def test_floor_is_read_from_ray_metadata(self) -> None:
        mins = dep_fixup.parse_min_requires(self.RAY_REQUIRES)
        self.assertEqual(mins["opentelemetry-sdk"], "1.30.0")

    def test_upper_bound_only_requirements_are_ignored(self) -> None:
        mins = dep_fixup.parse_min_requires(["opentelemetry-sdk<2"])
        self.assertEqual(mins, {})

    def test_highest_floor_wins_across_extras(self) -> None:
        mins = dep_fixup.parse_min_requires([
            "opentelemetry-sdk>=1.30.0; extra == 'default'",
            "opentelemetry-sdk>=1.9.0; extra == 'observability'",
        ])
        self.assertEqual(mins["opentelemetry-sdk"], "1.30.0")

    def test_prerelease_maps_back_to_its_stable_train(self) -> None:
        # 実機の exporter=0.65b0 は 1.44 列。ray の下限 1.30.0 より新しいので
        # こちらが揃える先になる(古い方へ引き戻すと ray が動かない)。
        self.assertEqual(dep_fixup.stable_from_prerelease("0.65b0"), "1.44.0")
        self.assertEqual(dep_fixup.stable_from_prerelease("0.47b0"), "1.26.0")
        self.assertEqual(dep_fixup.stable_from_prerelease("0.51b0"), "1.30.0")

    def test_stable_versions_are_not_treated_as_prerelease(self) -> None:
        self.assertIsNone(dep_fixup.stable_from_prerelease("1.44.0"))
        self.assertIsNone(dep_fixup.stable_from_prerelease(None))

    def test_installed_sdk_below_ray_floor_is_detected(self) -> None:
        # 実機の値。1.26.0 < 1.30.0 なので sdk を持ち上げる側に倒す。
        self.assertLess(dep_fixup.version_key("1.26.0"),
                        dep_fixup.version_key("1.30.0"))

    def test_prerelease_versions_compare_without_crashing(self) -> None:
        # semconv は 0.65b0 のようなプレリリース表記で来る。
        self.assertLess(dep_fixup.version_key("0.47b0"),
                        dep_fixup.version_key("0.65b0"))

    def test_env_can_pin_the_floor(self) -> None:
        self.assertEqual(dep_fixup.otel_sdk_floor({"DEP_OTEL_SDK_MIN": "1.44.0"}),
                         "1.44.0")

    def test_empty_env_disables_the_floor_step(self) -> None:
        self.assertIsNone(dep_fixup.otel_sdk_floor({"DEP_OTEL_SDK_MIN": ""}))


def mismatch_names(pins, versions):
    return [package for package, _, _ in dep_fixup.mismatched(pins, versions)]


if __name__ == "__main__":
    unittest.main()
