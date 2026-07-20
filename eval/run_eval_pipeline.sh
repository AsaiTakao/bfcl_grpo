#!/usr/bin/env bash
# =============================================================================
# run_eval_pipeline.sh — 1 チェックポイントの評価パイプライン。
#
# 設計書の評価フロー:
#   RL checkpoint → マージ → vLLM で OpenAI 互換起動 → τ2-bench
#   phase0(壊れ検出、毎回)→ phase1(傾向)→ phase2(公式、最終CPのみ)
#
# 流れ:
#   1) merge_checkpoint.sh で HF 形式へマージ(MERGE=0 でスキップ可)
#   2) serve_agent_vllm.sh を背景起動し /health を待つ
#   3) phase0_smoke.py(必須。壊れ検出で以降を止める)
#   4) PHASES に含めた phase1 / phase2 を実行
#   5) サーバ停止
#
# 使い方:
#   CKPT_STEP_DIR=.../global_step_600 MERGED_DIR=/workspace/merged/step600 \
#   PHASES="phase0 phase1" bash eval/run_eval_pipeline.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log() { echo "[pipeline] $*"; }

: "${MERGED_DIR:?MERGED_DIR(マージ済みモデル出力/読込先)を指定してください}"
: "${SERVED_NAME:=qwen3-8b-bfcl}"
: "${PORT:=8000}"
: "${AGENT_BASE_URL:=http://127.0.0.1:${PORT}/v1}"
: "${PHASES:=phase0}"
: "${MERGE:=1}"
: "${HEALTH_TIMEOUT:=1200}"     # 起動待ちの上限秒(8B ロードは数分かかる)

# --- 1) マージ ---------------------------------------------------------------
if [[ "${MERGE}" == "1" ]]; then
  : "${CKPT_STEP_DIR:?MERGE=1 のときは CKPT_STEP_DIR を指定してください}"
  MERGED_DIR="${MERGED_DIR}" CKPT_STEP_DIR="${CKPT_STEP_DIR}" \
    bash "${HERE}/merge_checkpoint.sh"
else
  log "MERGE=0: 既存の ${MERGED_DIR} をそのまま使用"
fi

# --- 2) vLLM 背景起動 + health 待ち -----------------------------------------
log "vLLM 起動: ${MERGED_DIR} port=${PORT}"
MODEL_DIR="${MERGED_DIR}" SERVED_NAME="${SERVED_NAME}" PORT="${PORT}" \
  bash "${HERE}/serve_agent_vllm.sh" &
SERVER_PID=$!

cleanup() {
  log "vLLM 停止 (pid=${SERVER_PID})"
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

log "health 待ち(最大 ${HEALTH_TIMEOUT}s): ${AGENT_BASE_URL%/v1}/health"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
until curl -sf "${AGENT_BASE_URL%/v1}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "vLLM プロセスが起動に失敗しました。" >&2
    exit 1
  fi
  if [[ $(date +%s) -ge ${deadline} ]]; then
    log "health タイムアウト。" >&2
    exit 1
  fi
  sleep 5
done
log "vLLM ready"

# --- 3/4) フェーズ実行 -------------------------------------------------------
export AGENT_BASE_URL AGENT_MODEL="${SERVED_NAME}" AGENT_API_KEY="EMPTY"
RC=0
for phase in ${PHASES}; do
  if [[ "${phase}" == "phase0" ]]; then
    log "phase0 smoke(壊れ検出)"
    if ! python "${HERE}/phase0_smoke.py"; then
      log "phase0 で破損検出。以降のフェーズを中止します。" >&2
      RC=1
      break
    fi
  else
    log "${phase} 実行"
    python "${HERE}/run_tau_eval.py" --phase "${phase}" || RC=1
  fi
done

log "パイプライン完了 (rc=${RC})"
exit "${RC}"
