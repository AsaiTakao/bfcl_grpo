#!/usr/bin/env bash
# =============================================================================
# merge_checkpoint.sh — verl の FSDP checkpoint を HF 形式へマージする。
#
# 設計書の評価フロー:
#   RL チェックポイント(verl 保存)→ マージ(LoRA なら base に統合)
#   → vLLM で OpenAI 互換エンドポイントとして起動
#
# verl の checkpoint は global_step_N/actor 配下に FSDP シャードで保存される。
# vLLM で読めるよう、単一の HF モデルディレクトリへ統合する。
#
# 使い方:
#   CKPT_STEP_DIR=/workspace/code/checkpoints/.../global_step_600 \
#   MERGED_DIR=/workspace/merged/step600 \
#   bash eval/merge_checkpoint.sh
# =============================================================================
set -euo pipefail

log() { echo "[merge] $*"; }

# verl の checkpoint(global_step_N ディレクトリ)。actor サブディレクトリを見る。
: "${CKPT_STEP_DIR:?CKPT_STEP_DIR(例 .../global_step_600)を指定してください}"
: "${MERGED_DIR:?MERGED_DIR(出力先)を指定してください}"
: "${MERGE_BACKEND:=fsdp}"

ACTOR_DIR="${CKPT_STEP_DIR}/actor"
if [[ ! -d "${ACTOR_DIR}" ]]; then
  # actor サブディレクトリが無い構成にもフォールバック。
  ACTOR_DIR="${CKPT_STEP_DIR}"
fi

mkdir -p "${MERGED_DIR}"
log "merge ${ACTOR_DIR} -> ${MERGED_DIR} (backend=${MERGE_BACKEND})"

# verl.model_merger のインタフェースはバージョン依存。まず新 CLI、失敗したら旧 CLI。
# LoRA checkpoint では model_merger 非対応のバージョンがあるため、失敗しても
# 即終了せず、下の LoRA 統合(peft)で救えるか試す。
MERGER_OK=1
if python -m verl.model_merger merge \
      --backend "${MERGE_BACKEND}" \
      --local_dir "${ACTOR_DIR}" \
      --target_dir "${MERGED_DIR}"; then
  log "merge 完了(verl.model_merger merge)"
elif python -m verl.model_merger \
      --backend "${MERGE_BACKEND}" \
      --local_dir "${ACTOR_DIR}" \
      --target_dir "${MERGED_DIR}"; then
  log "merge 完了(verl.model_merger 旧CLI)"
else
  log "verl.model_merger が失敗。LoRA adapter があれば peft 統合を試みます。" >&2
  MERGER_OK=0
fi

# --- LoRA なら base に統合(設計書「マージ(LoRA なら base に統合)」)---------
# verl は LoRA 学習時、actor 配下に lora_adapter を保存する。model_merger が
# adapter を統合しないバージョンに備え、peft で merge_and_unload して
# フルウェイトの HF モデルを MERGED_DIR に出力する。LORA_MERGE=0 でスキップ可。
: "${LORA_MERGE:=1}"
LORA_DIR=""
for cand in "${ACTOR_DIR}/lora_adapter" "${MERGED_DIR}/lora_adapter"; do
  if [[ -d "${cand}" ]]; then LORA_DIR="${cand}"; break; fi
done
if [[ "${LORA_MERGE}" == "1" && -n "${LORA_DIR}" ]]; then
  : "${BASE_MODEL_PATH:?LoRA 統合には BASE_MODEL_PATH(base モデルのパス)が必要です}"
  log "LoRA adapter を base に統合: ${LORA_DIR} + ${BASE_MODEL_PATH} -> ${MERGED_DIR}"
  python - "${BASE_MODEL_PATH}" "${LORA_DIR}" "${MERGED_DIR}" <<'PY'
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print(f"[merge] LoRA merged model saved -> {out}")
PY
elif [[ "${MERGER_OK}" == "0" ]]; then
  log "model_merger 失敗かつ LoRA adapter も無し。バージョンに合わせて CLI を確認してください。" >&2
  exit 1
fi

# tokenizer 等がマージ出力に含まれない場合、base からコピー(vLLM 起動に必要)。
if [[ -n "${BASE_MODEL_PATH:-}" && ! -f "${MERGED_DIR}/tokenizer_config.json" ]]; then
  log "tokenizer を ${BASE_MODEL_PATH} から補完コピー"
  cp -n "${BASE_MODEL_PATH}"/tokenizer* "${MERGED_DIR}/" 2>/dev/null || true
  cp -n "${BASE_MODEL_PATH}"/*.json "${MERGED_DIR}/" 2>/dev/null || true
fi

log "出力: ${MERGED_DIR}"
