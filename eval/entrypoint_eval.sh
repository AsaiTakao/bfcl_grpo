#!/usr/bin/env bash
# =============================================================================
# entrypoint_eval.sh — 評価コンテナ(vLLM サービング)の bootstrap。
#
# 設計書の評価フロー:
#   RL checkpoint → マージ → vLLM OpenAI 互換起動 → τ2-bench(phase0/1/2)
#
# 既定運用(推奨・軽量):
#   マージは学習イメージ側(verl あり)で済ませ、マージ済み HF モデルを HF へ上げる。
#   この評価コンテナはそれを DL して serve するだけ(verl 不要)。
#     MERGED_MODEL_REPO=user/qwen3-8b-bfcl-merged-step600 → DL → MERGE=0 で serve
#
# コンテナ内マージ(WITH_VERL=1 でビルドした場合のみ):
#   CKPT を DL/マウントし、eval/merge_checkpoint.sh を回してから serve。
#     CKPT_MODEL_REPO or CKPT_STEP_DIR + MERGE=1
#
# デバッグ: DOK のタスク作成画面で ENTRYPOINT を bash に差し替え可能。
# =============================================================================
set -euo pipefail
log() { echo "[eval-entry] $*"; }

# --- HF トークン(ファイルから。履歴・レイヤーに残さない)--------------------
: "${HF_TOKEN_FILE:=/run/secrets/hf_token}"
if [[ -z "${HF_TOKEN:-}" && -f "${HF_TOKEN_FILE}" ]]; then
  HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"; export HF_TOKEN
fi
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- 評価コードを実行時取得(学習側と同じ方針)-----------------------------
: "${CODE_DIR:=/workspace/code}"
: "${CODE_REF:=main}"
if [[ -n "${CODE_REPO:-}" ]]; then
  if [[ -d "${CODE_DIR}/.git" ]]; then
    git -C "${CODE_DIR}" fetch --depth 1 origin "${CODE_REF}"
    git -C "${CODE_DIR}" reset --hard "origin/${CODE_REF}"
  else
    git clone --depth 1 --branch "${CODE_REF}" "${CODE_REPO}" "${CODE_DIR}"
  fi
else
  log "CODE_REPO 未設定。${CODE_DIR} の既存コードを使用。"
  mkdir -p "${CODE_DIR}"
fi
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/eval:${PYTHONPATH:-}"

# --- モデル取得 --------------------------------------------------------------
: "${MERGED_DIR:=/workspace/merged/current}"
: "${MERGE:=0}"
mkdir -p "$(dirname "${MERGED_DIR}")"

if [[ "${MERGE}" == "1" ]]; then
  # コンテナ内マージ(WITH_VERL=1 イメージ前提)。
  : "${MERGED_DIR:?}"
  if [[ -n "${CKPT_MODEL_REPO:-}" ]]; then
    : "${CKPT_STEP_DIR:=/workspace/ckpt}"
    log "checkpoint DL: ${CKPT_MODEL_REPO} -> ${CKPT_STEP_DIR}"
    huggingface-cli download "${CKPT_MODEL_REPO}" \
      --local-dir "${CKPT_STEP_DIR}" --local-dir-use-symlinks False
    export CKPT_STEP_DIR
  fi
  : "${CKPT_STEP_DIR:?MERGE=1 には CKPT_STEP_DIR か CKPT_MODEL_REPO が必要}"
else
  # マージ済み HF モデルを DL して serve。
  if [[ -n "${MERGED_MODEL_REPO:-}" ]]; then
    log "merged model DL: ${MERGED_MODEL_REPO} -> ${MERGED_DIR}"
    huggingface-cli download "${MERGED_MODEL_REPO}" \
      --local-dir "${MERGED_DIR}" --local-dir-use-symlinks False
  else
    log "MERGED_MODEL_REPO 未設定。${MERGED_DIR} に既にモデルがある前提で進む。"
  fi
fi

# --- 評価パイプライン起動 ----------------------------------------------------
: "${PHASES:=phase0}"
log "評価開始 PHASES='${PHASES}' MERGE=${MERGE} MERGED_DIR=${MERGED_DIR}"
MERGED_DIR="${MERGED_DIR}" MERGE="${MERGE}" PHASES="${PHASES}" \
  bash eval/run_eval_pipeline.sh
