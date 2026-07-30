#!/bin/bash
# BFCL multi_turn GRPO 学習 (H100 x8, 単一ノード)
# base Qwen3-8B から RL。think プレフィル前提、二層報酬(mt_reward)。
set -ex
export HYDRA_FULL_ERROR=1
# 非特権コンテナでは失敗し得るため、失敗しても続行(set -e 対策)。
ulimit -n 65535 || true

# --- パス ---
# 値は環境変数で上書き可能(entrypoint / DOK コントロールパネルから)。
PROJECT_DIR="$(pwd)"
# BFCLMultiTurnTool / BFCLInteraction / mt_reward 等が import できるように
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
# config-name(bfcl_multiturn_grpo.yaml)と tool/interaction 用 yaml の在り処。
# 既定は PROJECT_DIR 直下(このリポジトリのフラット構成)。
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_DIR}}"
TOOL_CONFIG="${TOOL_CONFIG:-${PROJECT_DIR}/bfcl_tool_config.yaml}"
INTERACTION_CONFIG="${INTERACTION_CONFIG:-${PROJECT_DIR}/bfcl_interaction_config.yaml}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"   # base から RL (SFT 起点ではない)
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_DIR}/data/bfcl_mt_train.parquet}"
VAL_FILE="${VAL_FILE:-${PROJECT_DIR}/data/bfcl_mt_val.parquet}"
# checkpoint 出力先。hf_checkpoint_uploader の監視先(CKPT_DIR)と必ず揃える。
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/checkpoints}"

# --- GPU 配分 (H100 x8) ---
# sglang rollout と FSDP 学習は verl が同一 GPU 群で時分割する(colocate)。
# 8枚を trainer に渡し、rollout の GPU メモリ比率で棲み分ける。
N_GPUS="${N_GPUS:-8}"
# rollout エンジン。multi_turn ツール実行は verl 0.4.1 では sglang 前提。
ROLLOUT_NAME="${ROLLOUT_NAME:-sglang}"
# OOM 時に env だけで offload へ倒せるようにする(True で step 時間は延びる)。
# LoRA なら optimizer 状態は小さいが、rollout 同期のマージで base 重みを一度
# full tensor 化するぶん(8B bf16 ≒ 16GB)ピークが上がる点に注意。
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
ACTOR_OPTIM_OFFLOAD="${ACTOR_OPTIM_OFFLOAD:-False}"

# --- バッチ/シーケンス ---
# multi_turn は 1 rollout が複数生成。長め context に備える。
TRAIN_BATCH="${TRAIN_BATCH:-128}"                 # プロンプト数/step
PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"
MICRO_BATCH_PER_GPU="${MICRO_BATCH_PER_GPU:-4}"   # 8B + multi_turn は控えめに
# log_prob 再計算(ref / rollout)の micro batch。verl は actor.use_dynamic_bsz=False の
# とき ref と rollout の両方に log_prob_micro_batch_size(_per_gpu) を要求する
# (未設定だと _validate_config が ValueError で落ちる)。前向き計算のみなので
# 学習側より大きくできるが、既定は MICRO_BATCH_PER_GPU に揃えて OOM を避ける。
LOG_PROB_MICRO_BATCH_PER_GPU="${LOG_PROB_MICRO_BATCH_PER_GPU:-${MICRO_BATCH_PER_GPU}}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-4096}"          # 初期 config。long_context は別枠で拡張。
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-2048}"      # 1ターンの think+tool_call 長
GROUP_SIZE="${GROUP_SIZE:-8}"                     # GRPO の G。H100 x8 なら 8〜16 可
# 学習スケジュール(env 上書き可)。
# 総 step 数は小さい: BFCL multi_turn 3 カテゴリ = 600 件、val 10% を引いて
# train 540 件。TRAIN_BATCH=128 なら 4 step/epoch × 15 epoch ≒ 60 step。
# save/test の間隔はこの総 step 数を基準に決める(200/50 では eval が 1 回しか
# 走らず EarlyStopping の patience に到達せず、checkpoint も 1 つも残らない)。
LR="${LR:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
SAVE_FREQ="${SAVE_FREQ:-10}"                      # ≒6 CP(HF upload の UPLOAD_EVERY と揃える)
TEST_FREQ="${TEST_FREQ:-5}"                       # ≒12 回 eval。patience=3 が機能する粒度

# --- FP8 rollout(設計書: rollout-only FP8)---------------------------------
# 学習側(FSDP actor)は bf16 のまま、sglang 推論エンジンのみ FP8 量子化する。
# verl は毎 step、bf16 の actor 重みを rollout エンジンへ同期する。FP8 量子化は
# その同期後にエンジン側で online に行われる(重み更新のたびに再量子化)。
# ここは verl/sglang のバージョン依存が強いので既定 OFF。phase0 で実機確認して有効化。
#   ROLLOUT_FP8=1        FP8 rollout を有効化
#   ROLLOUT_FP8_QUANT    sglang の quantization(既定 fp8。fp8 は online W8A8)
#   ROLLOUT_FP8_KV       KV cache dtype(auto=bf16維持 / fp8_e5m2 / fp8_e4m3)
#   ROLLOUT_EXTRA_ARGS   キー名がバージョンで違う場合の逃げ道(空白区切りで丸ごと追加)
ROLLOUT_FP8="${ROLLOUT_FP8:-0}"
ROLLOUT_FP8_QUANT="${ROLLOUT_FP8_QUANT:-fp8}"
ROLLOUT_FP8_KV="${ROLLOUT_FP8_KV:-auto}"

EXTRA_ARGS=()
if [[ "${ROLLOUT_FP8}" == "1" ]]; then
  # verl 0.4.1 の sglang rollout は engine_kwargs.sglang.* を一切読まない
  # (SGLangRollout が AsyncEngine を固定引数で構築するため。quantization /
  #  kv_cache_dtype は Engine へ渡らない)。加えて struct モードの config に
  # 無いキーなので hydra override 自体が ConfigKeyError で落ちる。
  # 黙って無視されるより明示的に止める。有効化するには verl の更新が要る。
  echo "[run] ERROR: ROLLOUT_FP8=1 は verl ${VERL_VERSION_HINT:-0.4.1} + sglang では未対応です。" >&2
  echo "[run]        engine_kwargs.sglang.* は rollout エンジンへ渡らないため、" >&2
  echo "[run]        FP8 は verl 更新後に対応してください(ROLLOUT_FP8=0 で実行)。" >&2
  exit 1
fi
# --- LoRA(設計書: LoRA r=32)------------------------------------------------
# actor を LoRA で学習する。LORA_RANK=0 でフル FT。
#
# verl 0.4.1 の LoRA 配線は vLLM 経路にしか無く、sglang 側の
# FSDPSGLangShardingManager は PEFT ラップも adapter も解かずに state_dict を
# rollout エンジンへ流してしまう。そこで verl_lora_sglang_patch を
# external_lib(各 worker の init_model で import される公式フック)で当て、
# sglang へ渡す直前に LoRA を base へマージする。詳細はパッチ側の docstring。
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGETS="${LORA_TARGETS:-all-linear}"
LORA_PATCH_MODULE="${LORA_PATCH_MODULE:-verl_lora_sglang_patch}"
if [[ "${LORA_RANK}" != "0" ]]; then
  EXTRA_ARGS+=( "actor_rollout_ref.model.lora_rank=${LORA_RANK}" )
  EXTRA_ARGS+=( "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}" )
  EXTRA_ARGS+=( "actor_rollout_ref.model.target_modules=${LORA_TARGETS}" )
  if [[ "${ROLLOUT_NAME}" == "sglang" ]]; then
    # PYTHONPATH に PROJECT_DIR が入っているので module 名で import できる。
    EXTRA_ARGS+=( "actor_rollout_ref.model.external_lib=${LORA_PATCH_MODULE}" )
    echo "[run] sglang 向け LoRA マージパッチを external_lib で適用: ${LORA_PATCH_MODULE}"
  else
    # vLLM 経路は verl 本体が adapter を engine へ渡す。その要件。
    EXTRA_ARGS+=( "actor_rollout_ref.rollout.load_format=safetensors" )
  fi
  echo "[run] LoRA 有効: r=${LORA_RANK} alpha=${LORA_ALPHA} targets=${LORA_TARGETS}"
else
  echo "[run] LoRA 無効: フル FT で学習します。"
fi

# バージョン差でキー名が違う場合の逃げ道(例: ROLLOUT_EXTRA_ARGS="a.b=c d.e=f")。
if [[ -n "${ROLLOUT_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=( ${ROLLOUT_EXTRA_ARGS} )
fi

# --- 依存バージョンの自己修復 -------------------------------------------------
# 古いイメージ(peft / compressed-tensors が最新版のまま焼かれたもの)で起動しても
# ここで直る。学習コードは実行時 clone なので、イメージ再ビルドを待たずに効く。
# 直せなかった場合は GPU を掴む前に落とす(rollout が base 重みで回る事故を防ぐ)。
# 詳細は dep_fixup.py の docstring。DEP_FIXUP=0 で無効化。
if [[ "${DEP_FIXUP:-1}" == "1" ]]; then
  python3 "${PROJECT_DIR}/dep_fixup.py" || {
    echo "[run] ERROR: 依存バージョンの整合に失敗しました。" >&2
    echo "[run]        Dockerfile の『依存の整合』レイヤーを含めて再ビルドしてください。" >&2
    exit 1
  }
fi

python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_DIR}" \
    --config-name='bfcl_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=${TRAIN_BATCH} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIM_OFFLOAD} \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=${GROUP_SIZE} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=15 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=15 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=False \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="${INTERACTION_CONFIG}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_PER_GPU} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=[console,wandb] \
    trainer.project_name=bfcl_mt_grpo \
    trainer.experiment_name=qwen3_8b_base_mt_grpo_n${GROUP_SIZE} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=auto \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    "${EXTRA_ARGS[@]}"
