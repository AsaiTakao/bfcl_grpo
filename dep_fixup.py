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
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib import metadata
from typing import Dict, List, Optional, Tuple

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


def log(msg: str) -> None:
    print(f"[dep-fixup] {msg}", flush=True)


def env_key(package: str) -> str:
    return "DEP_PIN_" + package.upper().replace("-", "_")


def resolve_pins(environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """既定のピンに env の上書きを反映する(空文字はその項目を無効化)。"""
    environ = os.environ if environ is None else environ
    pins: Dict[str, str] = {}
    for package, default in DEFAULT_PINS.items():
        value = environ.get(env_key(package), default)
        if value:
            pins[package] = value
    return pins


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
    return subprocess.call(cmd)


def import_check() -> int:
    """別プロセスで import を試す(このプロセスの import 状態を汚さない)。"""
    code = f"import {IMPORT_CHECK}  # noqa: F401"
    return subprocess.call([sys.executable, "-c", code])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="検査のみ。ズレていたら exit 1(pip を叩かない)")
    parser.add_argument("--skip-import-check", action="store_true",
                        help=f"{IMPORT_CHECK} の import 検査を省略する")
    args = parser.parse_args(argv)

    pins = resolve_pins()
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
