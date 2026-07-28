# -*- coding: utf-8 -*-
"""
verl_lora_sglang_patch.py

verl 0.4.1 の sglang rollout で LoRA 学習を成立させるパッチ。

なぜ必要か
----------
verl 0.4.1 の LoRA 配線は vLLM 経路にしか無い。sglang 経路の
``FSDPSGLangShardingManager`` は PEFT ラップを解かず adapter のマージもせずに
``module.state_dict()`` をそのまま ``update_weights_from_tensor`` へ流すため、
``base_model.model.….lora_A.default.weight`` のようなキーが sglang へ渡り、
最初の重み同期で落ちる(仮に落ちなくても rollout は base 重みのままになる)。

sglang 0.4.6 には「adapter をテンソルで受け取る」API が無い(verl 0.8.0 が使う
``load_lora_adapter_from_tensor`` は後年の追加)。そこで rollout エンジンへ渡す
直前に LoRA を base へマージし、PEFT の名前を素の HF 名へ戻す。

  W' = W + (B @ A) * scaling      scaling = lora_alpha / r(rslora なら /√r)

学習側は LoRA のまま(optimizer は adapter だけ)で、rollout がマージ済み重みを
使うだけなので、数学的には通常の LoRA 学習と同じ。checkpoint も verl が
``actor/lora_adapter/`` に adapter を書く挙動が変わらない。

当て方
------
verl の公式フックである ``actor_rollout_ref.model.external_lib`` を使う。
これは各 worker の ``init_model()`` 内、rollout/sharding manager を組む前に
``importlib.import_module`` される。run_bfcl_grpo_h100x8.sh が LoRA 有効時に
``actor_rollout_ref.model.external_lib=verl_lora_sglang_patch`` を渡す。

メモリ
------
マージのため全 base 重みを一度 full tensor 化する(8B bf16 ≒ 16GB/rank)。
LoRA 学習では optimizer 状態が小さいので H100 80GB なら
sglang 40GB + shard 2GB + 16GB ≒ 58GB で収まる見込み。足りない場合は
``actor_rollout_ref.rollout.gpu_memory_utilization`` を下げる。
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Optional

# PEFT が付ける接頭辞。get_peft_model(PeftModelForCausalLM) → .base_model
# (LoraModel) → .model(素の CausalLM)なので state_dict のキーは
# "base_model.model." + 元のキー になる。
_PEFT_PREFIX = "base_model.model."
_FSDP_PREFIX = "_fsdp_wrapped_module."
_LORA_A_RE = re.compile(r"^(?P<base>.+)\.lora_A\.(?P<adapter>[^.]+)\.weight$")
# マージ後に落とす LoRA 側のパラメータ。
_LORA_MARKERS = (".lora_A.", ".lora_B.", ".lora_embedding_A", ".lora_embedding_B")
# DoRA は W' = m * (W + BA)/||W + BA|| で、単純な加算では再現できない。
_DORA_MARKER = "lora_magnitude_vector"


def log(msg: str) -> None:
    print(f"[lora-patch] {msg}", flush=True)


# ------------------------------------------------------------------ 名前の変換
def strip_peft_name(name: str) -> str:
    """PEFT の state_dict キーを素の HF モデルのキーに戻す。

    base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight
      -> model.layers.0.self_attn.q_proj.weight

    verl は state_dict_type を設定してから state_dict() を呼ぶので通常は
    `_fsdp_wrapped_module.` は付かないが、付く経路もある(verl 自身の
    layered_summon_lora_params が両方を想定している)ため落としておく。
    """
    name = name.replace(_FSDP_PREFIX, "")
    if name.startswith(_PEFT_PREFIX):
        name = name[len(_PEFT_PREFIX):]
    return name.replace(".base_layer.", ".")


def is_lora_param(name: str) -> bool:
    """マージ後に rollout エンジンへ渡してはいけないキーか。"""
    return any(m in name for m in _LORA_MARKERS)


# -------------------------------------------------------------------- 係数
def scaling_for(peft_config: Mapping[str, Any], adapter: str) -> float:
    """adapter 名に対応する LoRA scaling(alpha / r、rslora なら alpha / √r)。"""
    cfg = peft_config[adapter]
    r = getattr(cfg, "r", None)
    alpha = getattr(cfg, "lora_alpha", None)
    if not r or alpha is None:
        raise ValueError(f"adapter={adapter} の r/lora_alpha を取得できません: {cfg!r}")
    if getattr(cfg, "use_rslora", False):
        return alpha / math.sqrt(r)
    return alpha / r


# ------------------------------------------------------------------ マージ本体
def _to_full(tensor):
    """FSDP の sharded state_dict(DTensor)なら全体を集める。"""
    full = getattr(tensor, "full_tensor", None)
    return full() if callable(full) else tensor


def merge_lora_state_dict(params: Dict[str, Any],
                          scalings: Mapping[str, float]) -> Dict[str, Any]:
    """PEFT 名の state_dict を、LoRA をマージした素の state_dict にする。

    Args:
        params: state_dict(値は通常のテンソルでも DTensor でもよい)
        scalings: adapter 名 -> scaling(alpha/r)

    Returns:
        lora_* を除き、`base_model.model.` と `.base_layer` を取り除いた dict。
        base 重みには (B @ A) * scaling が加算済み。
    """
    dora = [k for k in params if _DORA_MARKER in k]
    if dora:
        raise NotImplementedError(
            f"DoRA(use_dora=True)はこのマージに未対応です: 例 {dora[0]}")

    # 1) full tensor 化しつつ、マージ対象の base 重みを特定する。
    full: Dict[str, Any] = {}
    for name, tensor in params.items():
        full[name] = _to_full(tensor)

    merged = 0
    for name in list(full):
        m = _LORA_A_RE.match(name)
        if m is None:
            continue
        base, adapter = m.group("base"), m.group("adapter")
        b_name = f"{base}.lora_B.{adapter}.weight"
        w_name = f"{base}.base_layer.weight"
        if b_name not in full:
            raise KeyError(f"{name} に対応する lora_B が見つかりません: {b_name}")
        if w_name not in full:
            raise KeyError(f"{name} に対応する base 重みが見つかりません: {w_name}")
        if adapter not in scalings:
            raise KeyError(f"adapter={adapter} の scaling が不明です "
                           f"(既知: {sorted(scalings)})")
        _add_lora_delta(full[w_name], full[name], full[b_name], scalings[adapter])
        merged += 1

    if merged == 0:
        raise RuntimeError(
            "LoRA パラメータが state_dict に見つかりませんでした。"
            "external_lib で当てるパッチの対象が想定と違う可能性があります。")

    out: Dict[str, Any] = {}
    for name, tensor in full.items():
        if is_lora_param(name):
            continue
        out[strip_peft_name(name)] = tensor
    log(f"LoRA マージ: {merged} 層 -> rollout へ {len(out)} テンソル")
    return out


def _add_lora_delta(weight, lora_a, lora_b, scaling: float) -> None:
    """weight += (B @ A) * scaling を in-place で行う。

    in-place にするのは、8B 分の base 重みをもう一組作らないため。
    FSDP の state_dict は元のパラメータとは別に確保されたテンソルを返すので、
    ここでの書き換えが学習側の重みを壊すことはない。
    計算は fp32 で行ってから weight の dtype へ戻す(bf16 の丸め誤差対策)。
    """
    import torch

    delta = (lora_b.to(torch.float32) @ lora_a.to(torch.float32)) * scaling
    weight.add_(delta.to(dtype=weight.dtype, device=weight.device))


# ------------------------------------------------------------------ パッチ適用
def _peft_model_of(module) -> Optional[Any]:
    """FSDP でラップされたモジュールから PeftModel を取り出す(無ければ None)。"""
    inner = getattr(module, "_fsdp_wrapped_module", module)
    return inner if hasattr(inner, "peft_config") else None


def apply_patch() -> bool:
    """FSDPSGLangShardingManager.update_weights に LoRA マージを挟む。"""
    try:
        from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager
    except Exception as e:  # noqa: BLE001  driver 側など sglang を import できない環境
        log(f"FSDPSGLangShardingManager を import できずスキップ: {e}")
        return False

    if getattr(FSDPSGLangShardingManager, "_bfcl_lora_patched", False):
        return False

    original = FSDPSGLangShardingManager.update_weights

    async def update_weights(self, params):
        peft_model = _peft_model_of(self.module)
        if peft_model is not None:
            scalings = {name: scaling_for(peft_model.peft_config, name)
                        for name in peft_model.peft_config}
            params = merge_lora_state_dict(params, scalings)
        return await original(self, params)

    FSDPSGLangShardingManager.update_weights = update_weights
    FSDPSGLangShardingManager._bfcl_lora_patched = True
    log("適用: sglang への重み同期前に LoRA を base へマージします")
    return True


apply_patch()
