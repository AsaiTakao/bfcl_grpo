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
    DEP_OTEL_SDK_MIN=1.44.0
                           opentelemetry-sdk の下限を明示する(既定は ray の
                           メタデータから読む)。空文字でこの段階を止める
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

# --- opentelemetry: ray が起動できる列へ揃える --------------------------------
# ray は起動時に dashboard(と各ノードの agent)を立て、その中で
# ray/dashboard/modules/aggregator/... が opentelemetry を使ってメトリクスを
# 登録する。ここで例外が出るとプロセスが死に、raylet が
#   Exception: The current node timed out during startup.
# で落ちる = ray.init() が返らず、学習が 1 step も進まない。実機で 2 形踏んだ:
#   ImportError: cannot import name 'OtelComponentTypeValues'
#       from 'opentelemetry.semconv._incubating.attributes.otel_attributes'
#   TypeError: Meter.create_histogram() got an unexpected keyword argument
#       'explicit_bucket_boundaries_advisory'
#
# 前提となる otel の作り: opentelemetry-python は
#   - 安定版(api / sdk)   … 1.X.Y
#   - プレリリース版(semantic-conventions / exporter 群) … 0.(X+21)bZ
# を必ず同時にリリースし、sdk のメタデータは仲間を ``==`` で厳密固定する
# (例: sdk 1.37.0 → api 1.37.0 / semantic-conventions 0.58b0)。
# Dockerfile は pip を何レイヤーにも分けて叩くので、後のレイヤーが
# sdk か exporter か semconv のどれか 1 つだけを上げ下げすると列がバラける。
#
# 揃える先を「入っている sdk」にしてはいけない(1 敗)。実機のイメージは
# api/sdk=1.26.0(2024 年)に対し exporter=0.65b0(= 1.44 列)で、sdk 基準に
# 揃えると exporter が 0.47b0 まで落ちて上の TypeError になった。ray が
#   setup.py extras["default"]: opentelemetry-sdk >= 1.30.0
# と宣言している通り、ray 側が要求する下限の方が sdk の実際より新しいことが
# ある。そこで順序を 2 段にする:
#   1) ray が宣言する下限を下回っていたら sdk をそこまで上げる
#      (deps 込みで入れ、api / semantic-conventions を == で追従させる)
#   2) その上で exporter を semantic-conventions と同じ番号へ揃える
# 下限は ray のメタデータから読む(ray を上げたら自動で追従する)。
#
# 副作用として、vllm/sglang が入れる opentelemetry-exporter-otlp-* が古い列に
# 取り残される可能性がある。これらは tracing を有効にした時しか import されず、
# 本学習では使わない。実害が出るなら DEP_OTEL_FIX=0 で止めて手当てする。
OTEL_SDK = "opentelemetry-sdk"
# sdk が == で固定する仲間(値は sdk のメタデータから読む)。
OTEL_SDK_PINNED = ("opentelemetry-api", "opentelemetry-semantic-conventions")
# semantic-conventions と同じ番号で出るプレリリース群。ray が掴むのは
# prometheus exporter だけなので、揃えるのもそれだけにする。
OTEL_PRERELEASE = ("opentelemetry-exporter-prometheus",)
# 安定版 1.X.Y ↔ プレリリース版 0.(X+21)bZ の差。1.26↔0.47 / 1.30↔0.51 /
# 1.37↔0.58 / 1.44↔0.65 で一貫している。
OTEL_TRAIN_OFFSET = 21
# ray 自身の import 経路で検査する。opentelemetry.exporter.prometheus だけでは
# 足りない(上の TypeError は ray がメトリクスを登録する時に初めて出る)。
OTEL_IMPORT_CHECK = "ray.dashboard.modules.aggregator.aggregator_agent"

# "opentelemetry-semantic-conventions == 0.58b0" のような固定だけを拾う。
_PINNED_REQUIREMENT = re.compile(r"^([A-Za-z0-9._-]+)\s*==\s*([^\s,;()\[\]]+)$")
# "opentelemetry-sdk >= 1.30.0" のような下限だけを拾う。
_MIN_REQUIREMENT = re.compile(r"^([A-Za-z0-9._-]+)\s*>=\s*([^\s,;()\[\]]+)")


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


def parse_min_requires(requires: Optional[Iterable[str]]) -> Dict[str, str]:
    """Requires-Dist から ``>=`` の下限だけを取り出す。

    ray は同じパッケージを複数の extra で宣言することがあるので、
    見つかった中で一番高い下限を採る。
    """
    mins: Dict[str, str] = {}
    for raw in requires or []:
        spec = raw.split(";")[0]
        spec = re.sub(r"\[[^\]]*\]", "", spec).strip()
        matched = _MIN_REQUIREMENT.match(spec)
        if matched:
            name = matched.group(1).lower().replace("_", "-")
            found = matched.group(2)
            if name not in mins or version_key(found) > version_key(mins[name]):
                mins[name] = found
    return mins


def version_key(version: str):
    """バージョン比較用のキー。packaging が無い環境でも桁比較で妥協する。"""
    try:
        from packaging.version import Version

        return Version(version)
    except Exception:
        return tuple(int(part) for part in re.findall(r"\d+", version))


def stable_from_prerelease(version: Optional[str]) -> Optional[str]:
    """``0.65b0`` → ``1.44.0``。プレリリース列に対応する安定版を返す。

    otel は安定版 1.X.Y とプレリリース版 0.(X+21)bZ を同時に出す。
    パッチは 0 を仮定する(各列に .0 は必ずある)。
    """
    if not version:
        return None
    matched = re.match(r"^0\.(\d+)b\d+$", version)
    if not matched:
        return None
    minor = int(matched.group(1)) - OTEL_TRAIN_OFFSET
    return f"1.{minor}.0" if minor >= 0 else None


def otel_sdk_floor(environ: Optional[Dict[str, str]] = None) -> Optional[str]:
    """揃える先の opentelemetry-sdk。候補の中で一番新しい列を採る。

    候補は 2 つ:

    - ray がメタデータで宣言する下限(現行 ray は extras["default"] に
      ``opentelemetry-sdk >= 1.30.0``)。ray を上げれば自動で追従する。
    - 既に入っているプレリリース版(exporter / semantic-conventions)の列。
      イメージのビルド時に pip が ray のために解決した列がここに残る
      (実機は exporter=0.65b0 = 1.44 列だった)。

    常に**新しい方へ**揃えるのが肝。古い方へ引き戻すと、ビルド時に
    pip が選んだ ray 向けの列を壊す(それで TypeError を踏んだ)。
    DEP_OTEL_SDK_MIN で明示指定でき、空文字なら段階 1) を丸ごと止める。
    """
    environ = os.environ if environ is None else environ
    override = environ.get("DEP_OTEL_SDK_MIN")
    if override is not None:
        return override or None

    candidates: List[str] = []
    if installed_version("ray") is not None:
        try:
            floor = parse_min_requires(metadata.requires("ray")).get(OTEL_SDK)
        except metadata.PackageNotFoundError:
            floor = None
        if floor:
            candidates.append(floor)
    for package in OTEL_PRERELEASE + ("opentelemetry-semantic-conventions",):
        stable = stable_from_prerelease(installed_version(package))
        if stable:
            candidates.append(stable)
    if not candidates:
        return None
    return max(candidates, key=version_key)


def ensure_otel_sdk(environ: Optional[Dict[str, str]] = None) -> int:
    """段階 1): 揃える先の列より古い sdk を持ち上げる。

    ここだけは ``--no-deps`` にしない。api / semantic-conventions は sdk が
    ``==`` で固定しているので、pip に解決させるのが最も確実
    (プレリリース番号の b0/b1 を自前で当てにいかなくて済む)。
    """
    environ = os.environ if environ is None else environ
    if environ.get("DEP_OTEL_FIX", "1") != "1":
        return 0
    floor = otel_sdk_floor(environ)
    got = installed_version(OTEL_SDK)
    if not floor or got is None:
        return 0
    if version_key(got) >= version_key(floor):
        return 0
    log(f"{OTEL_SDK} {got} は揃える先の列 {floor} より古いので上げます"
        "(api / semantic-conventions も追従)")
    rc = subprocess.call([sys.executable, "-m", "pip", "install",
                          "--disable-pip-version-check", f"{OTEL_SDK}=={floor}"])
    importlib.invalidate_caches()
    if rc != 0:
        log(f"{OTEL_SDK} の導入に失敗しました (rc={rc})")
    return rc


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


def import_check(module: str = IMPORT_CHECK, optional: bool = False) -> int:
    """別プロセスで import を試す(このプロセスの import 状態を汚さない)。

    optional=True のとき、モジュール自体が無い場合(ray のバージョン差で
    パスが変わった等)は成功扱いにする。中で起きた ImportError /
    TypeError は当然失敗のままにする。
    """
    if optional:
        code = (
            "import importlib, sys\n"
            f"name = {module!r}\n"
            "try:\n"
            "    importlib.import_module(name)\n"
            "except ModuleNotFoundError as e:\n"
            "    missing = e.name or ''\n"
            "    if name == missing or name.startswith(missing + '.'):\n"
            "        print(f'[dep-fixup] {name} は無いので検査を省略します')\n"
            "        sys.exit(0)\n"
            "    raise\n"
        )
    else:
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
    # otel は 2 段階(OTEL_* のコメント参照)。
    # 1) ray の下限を満たす sdk を入れる(--check では触らずに報告だけ)。
    if not args.check:
        rc = ensure_otel_sdk()
        if rc != 0:
            return rc
    elif otel_sdk_floor() and installed_version(OTEL_SDK) and \
            version_key(installed_version(OTEL_SDK)) < version_key(otel_sdk_floor()):
        log(f"ズレ: {OTEL_SDK} {installed_version(OTEL_SDK)} "
            f"< 揃える先の列 {otel_sdk_floor()}")
        return 1
    # 2) api / semantic-conventions / exporter をその sdk の列へ揃える。
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

    # ray の dashboard / agent が踏むのと同じ経路。ここが通らない状態で学習を
    # 始めると ray.init() が「The current node timed out during startup.」で
    # 失敗する = GPU を掴んだ後に発覚するので、先に見ておく。
    # 検査を import で行うのが重要: メトリクス登録はモジュール読み込み時に
    # 走るため、バージョン不一致は import 時の TypeError として現れる。
    if installed_version("ray") is not None:
        rc = import_check(OTEL_IMPORT_CHECK, optional=True)
        if rc != 0:
            log(f"{OTEL_IMPORT_CHECK} を import できません。"
                "opentelemetry と ray のバージョンが噛み合っていません"
                "(DEP_OTEL_SDK_MIN=1.44.0 のように列を明示指定できます)。"
                "このままだと ray のノード起動が timeout します。")
            return rc
        log(f"{OTEL_IMPORT_CHECK} import OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
