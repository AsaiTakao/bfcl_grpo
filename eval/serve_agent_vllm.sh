#!/usr/bin/env bash
# =============================================================================
# serve_agent_vllm.sh — マージ済みモデルを vLLM OpenAI 互換で起動する。
#
# 設計書の技術ポイント(phase0 で洗い出す):
#   think と tool_call パースの整合。学習と同じく評価でも Qwen3 の thinking を
#   発火させる。τ2-bench はエージェント出力をパースするので、<think> が hermes
#   パーサの tool_call 抽出を壊さないか確認。壊れる場合は --reasoning-parser を追加。
#
#   → ここでは既定で --reasoning-parser qwen3 を付け、<think> を reasoning_content
#     として本文から分離する。これで hermes の <tool_call> 抽出が think に汚染されない。
#   → parser 名/挙動は vLLM バージョン依存。空にすると無効化して比較検証できる。
#
# 使い方(バックグラウンド起動は run_eval_pipeline.sh が行う):
#   MODEL_DIR=/workspace/merged/step600 PORT=8000 bash eval/serve_agent_vllm.sh
# =============================================================================
set -euo pipefail

log() { echo "[serve] $*"; }

: "${MODEL_DIR:?MODEL_DIR(マージ済みモデルのパス)を指定してください}"
: "${SERVED_NAME:=qwen3-8b-bfcl}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${MAX_MODEL_LEN:=16384}"
: "${GPU_UTIL:=0.90}"
: "${TP_SIZE:=1}"                       # 評価は 1GPU で十分なことが多い
: "${TOOL_CALL_PARSER:=hermes}"         # 学習と同じ hermes
: "${REASONING_PARSER:=qwen3}"          # 空文字で think 分離を無効化(比較用)

ARGS=(
  serve "${MODEL_DIR}"
  --served-model-name "${SERVED_NAME}"
  --host "${HOST}" --port "${PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_UTIL}"
  --tensor-parallel-size "${TP_SIZE}"
  --enable-auto-tool-choice
  --tool-call-parser "${TOOL_CALL_PARSER}"
)

# reasoning-parser は指定があるときだけ付ける(バージョン非対応なら空にして回避)。
if [[ -n "${REASONING_PARSER}" ]]; then
  ARGS+=( --reasoning-parser "${REASONING_PARSER}" )
  log "reasoning-parser=${REASONING_PARSER}(think を reasoning_content へ分離)"
else
  log "reasoning-parser 無効。<think> は本文に残る(tool_call 抽出への影響を要確認)。"
fi

log "vllm ${ARGS[*]}"
exec vllm "${ARGS[@]}"
