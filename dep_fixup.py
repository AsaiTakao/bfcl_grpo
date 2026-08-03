#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dep_fixup.py — 実行時に依存バージョンのズレを検査し、必要なら直す。

なぜ必要か
----------
学習イメージの土台は sglang 0.4.6.post5 で、これは ``torchao==0.9.0`` /
``transformers==4.51.1`` を厳密に固定する。一方 ``peft`` と
``compressed-tensors`` は上限なしの推移依存なので、ビルドした時期によって
2026 年の最新版が 2025 年のスタックの上に乗る。実際に踏んだ不整合:

- ``peft>=0.19`` は ``torchao>=0.16`` を要求する。LoRA 注入
  (``get_peft_model``)が "Found an incompatible version of torchao" で落ちる。
- ``compressed-tensors>=0.16`` は import 時に ``transformers.masking_utils``
  (transformers 4.52 以降)を参照する。sglang の
  ``layers/quantization/__init__.py`` が compressed_tensors を掴むため、
  ``import sglang`` 系がすべて ModuleNotFoundError になる
  (= verl_lora_sglang_patch が当たらず、LoRA 未マージのまま rollout が回る)。
- opentelemetry 一式のリリース列がバラけると ray が起動しない。詳しくは
  下の OTEL_* 定数のコメント(``ImportError: cannot import name
  'OtelComponentTypeValues'`` → "The current node timed out during startup.")。

本来は Dockerfile 側でピンするのが正しい(そちらも修正済み)。ただし
イメージの再ビルドは時間がかかり、古いイメージで起動した場合に
GPU 課金中に初めて発覚するのが最も高い。学習コードは実行時 clone なので、
この自己修復はイメージを作り直さずに効く。

使い方
------
    python dep_fixup.py            # ズレていれば pip で直し、import 検査まで行う
    python dep_fixup.py --check    # 検査のみ(直さない)。ズレていたら exit 1

env:
    DEP_FIXUP=0            run script 側でこの処理自体をスキップ
    DEP_PIN_PEFT=0.15.2    個別のピンを上書き(空文字にするとその項目を無視)
    DEP_PIN_COMPRESSED_TENSORS / DEP_PIN_TRANSFORMERS / DEP_PIN_TORCHAO
    DEP_OTEL_FIX=0         opentelemetry の列揃え(下記)だけを止める
    DEP_PIN_OPENTELEMETRY_SEMANTIC_CONVENTIONS=0.58b0
                           otel 側も DEP_PIN_* で個別に上書きできる
                           (パッケージ名の - と . を _ にして大文字化)
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
from importlib import metadata
from typing import Dict, Iterable, List, Optional, Tuple

# sglang 0.4.6.post5 と同時期の版に揃える(= sglang 自身が固定する値、および
# その時期の transformers 4.51.1 で import できる版)。
DEFAULT_PINS: Dict[str, str] = {
    "transformers": "4.51.1",
    "torchao": "0.9.0",
    "peft": "0.15.2",
    "compressed-tensors": "0.9.4",
}

# 依存の欠落ではなくバージョンだけを直したいので --no-deps で入れる。
# 上記のピンはいずれも torch/transformers/pydantic 等が既に入っている前提で
# 動く版なので、新しい torch を引きずり込まないことを優先する。
PIP_INSTALL_FLAGS = ["--no-deps", "--disable-pip-version-check"]

# パッチが実際に掴むモジュール。ここが import できるかが最終的な合否。
IMPORT_CHECK = "verl.workers.sharding_manager.fsdp_sglang"

# --- opentelemetry: ray が起動できるように「同じリリース列」へ揃える ----------
# ray は各ノードで dashboard agent を立て、その中で
# ray/dashboard/modules/aggregator/aggregator_agent.py が
# opentelemetry.exporter.prometheus を import する。ここで失敗すると agent が
# 死に、raylet が
#   Exception: The current node timed out during startup.
# で落ちる = ray.init() が返らず、学習が 1 step も進まない。実機で踏んだ形:
#   ImportError: cannot import name 'OtelComponentTypeValues'
#       from 'opentelemetry.semconv._incubating.attributes.otel_attributes'
#
# 原因はバグではなくバージョンのズレ。opentelemetry-python は
#   - 安定版(api / sdk)   … 1.X.Y
#   - プレリリース版(semantic-conventions / exporter 群) … 0.(X+21)bZ
# を必ず同時にリリースし、sdk のメタデータは仲間を ``==`` で厳密固定する
# (例: sdk 1.37.0 → api 1.37.0 / semantic-conventions 0.58b0)。
# Dockerfile は pip を何レイヤーにも分けて叩くので、後のレイヤーが
# sdk か exporter か semconv のどれか 1 つだけを上げ下げすると列がバラける。
# 新しい sdk/exporter が古い semconv に無いシンボルを参照して上の ImportError。
#
# ここに固定値を書かないのは、正解の列がイメージのビルド時期(= 焼かれた sdk)
# で変わるため。入っている sdk の Requires-Dist を正とし、api /
# semantic-conventions / exporter をその列へ引き戻す。
OTEL_SDK = "opentelemetry-sdk"
# sdk が == で固定する仲間(値は sdk のメタデータから読む)。
OTEL_SDK_PINNED = ("opentelemetry-api", "opentelemetry-semantic-conventions")
# semantic-conventions と同じ番号で出るプレリリース群。ray が掴むのは
# prometheus exporter だけなので、揃えるのもそれだけにする。
OTEL_PRERELEASE = ("opentelemetry-exporter-prometheus",)
# ray の agent が最初に踏む import。ここが通れば node は起動する。
OTEL_IMPORT_CHECK = "opentelemetry.exporter.prometheus"

# "opentelemetry-semantic-conventions == 0.58b0" のような固定だけを拾う。
_PINNED_REQUIREMENT = re.compile(r"^([A-Za-z0-9._-]+)\s*==\s*([^\s,;()\[\]]+)$")


def log(msg: str) -> None:
    print(f"[dep-fixup] {msg}", flush=True)


def env_key(package: str) -> str:
    return "DEP_PIN_" + package.upper().replace("-", "_").replace(".", "_")


def resolve_pins(environ: Optional[Dict[str, str]] = None,
                 defaults: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """既定のピンに env の上書きを反映する(空文字はその項目を無効化)。"""
    environ = os.environ if environ is None else environ
    defaults = DEFAULT_PINS if defaults is None else defaults
    pins: Dict[str, str] = {}
    for package, default in defaults.items():
        value = environ.get(env_key(package), default)
        if value:
            pins[package] = value
    return pins


def parse_pinned_requires(requires: Optional[Iterable[str]]) -> Dict[str, str]:
    """Requires-Dist から ``==`` の厳密固定だけを取り出す。

    環境マーカー(``; python_version < "3.13"``)と extra 指定は落とす。
    ``>=`` などの緩い指定は「列が決まらない」ので拾わない。
    """
    pins: Dict[str, str] = {}
    for raw in requires or []:
        spec = raw.split(";")[0]
        spec = re.sub(r"\[[^\]]*\]", "", spec).strip()
        matched = _PINNED_REQUIREMENT.match(spec)
        if matched:
            name = matched.group(1).lower().replace("_", "-")
            pins[name] = matched.group(2)
    return pins


def otel_pins(sdk_requires: Optional[Iterable[str]]) -> Dict[str, str]:
    """sdk の固定を正として、otel 一式の期待バージョンを組み立てる。

    exporter 群は semantic-conventions と同じプレリリース番号で出るため、
    semconv のピンをそのまま流用する。
    """
    pinned = parse_pinned_requires(sdk_requires)
    want = {package: pinned[package]
            for package in OTEL_SDK_PINNED if package in pinned}
    train = want.get("opentelemetry-semantic-conventions")
    if train:
        for package in OTEL_PRERELEASE:
            want[package] = train
    return want


def resolve_otel_pins(environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """実機の otel 期待値。sdk が無い/揃える材料が無いなら何もしない。

    入っていないパッケージは足さない: exporter を新規導入すると ray が
    使わない依存まで引き込むうえ、そもそもズレていない環境で pip を
    叩く理由が無い。
    """
    environ = os.environ if environ is None else environ
    if environ.get("DEP_OTEL_FIX", "1") != "1":
        return {}
    if installed_version(OTEL_SDK) is None:
        return {}
    try:
        requires = metadata.requires(OTEL_SDK)
    except metadata.PackageNotFoundError:
        return {}
    want = {package: version
            for package, version in otel_pins(requires).items()
            if installed_version(package) is not None}
    return resolve_pins(environ, want)


def installed_version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def mismatched(pins: Dict[str, str],
               versions: Dict[str, Optional[str]]) -> List[Tuple[str, Optional[str], str]]:
    """(package, 実際, 期待) のリスト。未インストールも直す対象に含める。"""
    out = []
    for package, want in pins.items():
        got = versions.get(package)
        if got != want:
            out.append((package, got, want))
    return out


def current_versions(pins: Dict[str, str]) -> Dict[str, Optional[str]]:
    return {package: installed_version(package) for package in pins}


def pip_install(specs: List[str]) -> int:
    cmd = [sys.executable, "-m", "pip", "install", *PIP_INSTALL_FLAGS, *specs]
    log("実行: " + " ".join(cmd))
    rc = subprocess.call(cmd)
    # 入れ直した直後の metadata 読み直しが古い値を返すと、直っているのに
    # 「入れ直したのに一致しません」で学習を止めてしまう。importlib の
    # キャッシュは site-packages の mtime 基準(解像度 1 秒)なので捨てておく。
    importlib.invalidate_caches()
    return rc


def import_check(module: str = IMPORT_CHECK) -> int:
    """別プロセスで import を試す(このプロセスの import 状態を汚さない)。"""
    code = f"import {module}  # noqa: F401"
    return subprocess.call([sys.executable, "-c", code])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="検査のみ。ズレていたら exit 1(pip を叩かない)")
    parser.add_argument("--skip-import-check", action="store_true",
                        help=f"{IMPORT_CHECK} の import 検査を省略する")
    args = parser.parse_args(argv)

    pins = resolve_pins()
    # otel は固定値ではなく実機の sdk から導く(OTEL_* のコメント参照)。
    pins.update(resolve_otel_pins())
    versions = current_versions(pins)
    log("現状: " + ", ".join(f"{p}={versions[p] or '未導入'}" for p in pins))

    bad = mismatched(pins, versions)
    if args.check:
        if bad:
            for package, got, want in bad:
                log(f"ズレ: {package} {got or '未導入'} != {want}")
            return 1
        log("ピン一致")
        return 0

    if bad:
        specs = [f"{package}=={want}" for package, _, want in bad]
        log("ズレを検出したので入れ直します: " + ", ".join(
            f"{p} {g or '未導入'} -> {w}" for p, g, w in bad))
        rc = pip_install(specs)
        if rc != 0:
            log(f"pip install が失敗しました (rc={rc})。"
                "ネットワークが無い場合はイメージを再ビルドしてください。")
            return rc
        versions = current_versions(pins)
        still = mismatched(pins, versions)
        if still:
            for package, got, want in still:
                log(f"入れ直したのに一致しません: {package} {got or '未導入'} != {want}")
            return 1
        log("修復後: " + ", ".join(f"{p}={versions[p]}" for p in pins))
    else:
        log("ピン一致(修復不要)")

    if args.skip_import_check:
        return 0

    rc = import_check()
    if rc != 0:
        log(f"{IMPORT_CHECK} を import できません。"
            "上の traceback の一番奥にある ModuleNotFoundError の出所が"
            "ピンすべき依存です(DEP_PIN_* で上書きできます)。")
        return rc
    log(f"{IMPORT_CHECK} import OK")

    # ray の dashboard agent が最初に踏む import。ここで落ちる状態のまま
    # 学習を始めると ray.init() が「The current node timed out during
    # startup.」で失敗する = GPU を掴んだ後に発覚するので、先に見ておく。
    if installed_version(OTEL_PRERELEASE[0]) is not None:
        rc = import_check(OTEL_IMPORT_CHECK)
        if rc != 0:
            log(f"{OTEL_IMPORT_CHECK} を import できません。"
                "opentelemetry のリリース列がまだ揃っていません"
                "(DEP_PIN_OPENTELEMETRY_* で明示指定できます)。"
                "このままだと ray のノード起動が timeout します。")
            return rc
        log(f"{OTEL_IMPORT_CHECK} import OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
