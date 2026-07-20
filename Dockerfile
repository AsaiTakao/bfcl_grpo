# syntax=docker/dockerfile:1.7
# =============================================================================
# BFCL multi_turn GRPO 学習イメージ(さくら高火力 DOK / H100 x8 想定)
#
# ビルドは安価な CPU Linux インスタンスで行う(設計書「開発・ビルド用の作業部屋」)。
# 方針:
#   - 学習コードはイメージに焼かず、entrypoint が実行時に git clone/pull する
#     → コード修正のたびに再ビルド不要。
#   - ハイパラ等は環境変数で渡す(DOK コントロールパネルで上書き)。
#   - HF トークンはビルド時レイヤーに残さない(BuildKit secret + ファイル渡し)。
#   - モデル重み(Qwen3-8B)だけは例外的に焼き込む(DL 時間も課金対象のため)。
#
# ビルド例(CPU インスタンス上、BuildKit 必須):
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=hf_token,src=./hf_token.txt \
#     -t <registry>/bfcl-mt-grpo:latest .
# =============================================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# --- verl+rollout のバージョン整合(要実機検証。DOK 側 ARG で上書き可) ---------
# 設計書: vllm/verl/torchao の具体バージョンを固定する。
# 学習 rollout は run script に合わせて sglang を使用(vLLM はフェーズ2の
# 評価サービング用の別イメージで使う)。
ARG VERL_VERSION=0.4.1
ARG SGLANG_VERSION=0.4.6.post5
ARG TORCHAO_VERSION=0.11.0
ARG FLASH_ATTN_VERSION=2.7.4.post1

# --- OS パッケージ ------------------------------------------------------------
RUN set -e; \
    apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential ninja-build && \
    rm -rf /var/lib/apt/lists/*

# --- Python 依存 --------------------------------------------------------------
# hf_transfer で並列・高速ダウンロード。verl は sglang extra で導入。
RUN set -e; \
    pip install --upgrade pip setuptools wheel packaging && \
    pip install "huggingface_hub[hf_transfer]" hf_transfer && \
    pip install "torchao==${TORCHAO_VERSION}" && \
    pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation && \
    pip install "sglang[all]==${SGLANG_VERSION}" && \
    pip install "verl==${VERL_VERSION}" && \
    pip install "bfcl-eval" wandb pandas pyarrow

# --- NCCL を最新へ手動アップグレード -----------------------------------------
# 設計書: 8GPU 全部で all_reduce が通るか / 実際に速くなるかを実機ベンチで確認。
# ここでは pip 版 nvidia-nccl を最新へ上げるだけ。検証は entrypoint の
# NCCL_SMOKE_TEST=1 で実行(下記 entrypoint 参照)。
RUN set -e; pip install --upgrade nvidia-nccl-cu12

# --- モデル重みだけ焼き込む(secret でトークンを渡し、レイヤーに残さない)------
ARG HF_MODEL=Qwen/Qwen3-8B
ENV BASE_MODEL_PATH=/models/Qwen3-8B
RUN --mount=type=secret,id=hf_token \
    set -e; \
    if [ -f /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi; \
    huggingface-cli download "${HF_MODEL}" \
        --local-dir "${BASE_MODEL_PATH}" --local-dir-use-symlinks False; \
    unset HF_TOKEN

# --- 実行時のハイパラ既定値(DOK パネルで上書き可能)-------------------------
# entrypoint / run script が os.getenv 相当で読む。
ENV CODE_REPO="" \
    CODE_REF="main" \
    CODE_DIR=/workspace/code \
    HF_REPO_ID="" \
    HF_CKPT_PREFIX="checkpoints" \
    HF_MERGED_PREFIX="merged-final" \
    UPLOAD_EVERY=200 \
    KEEP_BEST=3 \
    KEEP_LATEST=3 \
    RESUME_FROM_HF=1 \
    MERGE_FINAL=1 \
    EARLY_STOP=1 \
    EARLY_STOP_PATIENCE=3 \
    FINAL_BUFFER_SECONDS=300 \
    N_GPUS=8 \
    TRAIN_BATCH=128 \
    PPO_MINI_BATCH=32 \
    MICRO_BATCH_PER_GPU=4 \
    MAX_PROMPT_LEN=4096 \
    MAX_RESPONSE_LEN=2048 \
    GROUP_SIZE=8 \
    LORA_RANK=32 \
    LORA_ALPHA=32 \
    LR=1e-6 \
    TOTAL_EPOCHS=15 \
    SAVE_FREQ=200 \
    TEST_FREQ=50 \
    NCCL_SMOKE_TEST=0 \
    ROLLOUT_FP8=0 \
    ROLLOUT_FP8_QUANT=fp8 \
    ROLLOUT_FP8_KV=auto

WORKDIR /workspace

# bootstrap の entrypoint だけイメージに焼く(学習コード本体は焼かない)。
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# DOK のタスク作成画面で ENTRYPOINT/CMD は上書き可能。
# デバッグ時は entrypoint を bash に差し替える運用。
ENTRYPOINT ["/workspace/entrypoint.sh"]
