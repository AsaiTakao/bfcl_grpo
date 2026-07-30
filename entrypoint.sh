#!/usr/bin/env bash
# =============================================================================
# entrypoint.sh — DOK 起動時の bootstrap(イメージに焼く唯一のコード)
#
# 流れ(設計書「entrypoint.sh の設計」):
#   1) HF トークンをファイルから読む(シェル履歴に残さない)
#   2) 学習コードを git clone/pull(イメージに焼かず実行時取得)
#   3) resume: HF 上の最新 checkpoint を復元(中断からの再開)
#      + 200 step ごとに checkpoint を HF へ push(背景監視、
#        保持は「最良 eval 3 + 直近 3 = 最大 6」)
#   4) 学習を起動(ハイパラは環境変数、DOK パネルで上書き可能)。
#      出力は train.log に記録し、train_monitor が eval 抽出と
#      EarlyStopping(patience=3)を行う。
#      ログ(起動ログ + train.log)は HF repo へも定期 push する
#      = DOK コンソールは先頭が切れるため、そちらを一次資料にしない。
#   5) 終了後に「最新 checkpoint を同期 push」+ merge 版を HF へ push
#      (どの CP か・eval の数字を記録)+ バッファ待機
#      → 非同期アップロード中の強制終了で取りこぼすのを防ぐ
#
# デバッグ: DOK のタスク作成画面で ENTRYPOINT を bash に差し替えれば素の shell。
# =============================================================================
set -euo pipefail

log() { echo "[entrypoint] $*"; }

# --- 0) 自分自身の出力をファイルにも残す(DOK コンソールは先頭が切れる)------
# DOK のログは先頭が切り落とされ、起動時の設定・データ準備・resume 判定という
# 一番読みたい部分が消える。そこでここから下の全出力を起動ログへ tee し、
# hf_checkpoint_uploader が HF repo(${HF_CKPT_PREFIX}/)へ push する。
#
# 起動ごとに別ファイル名にするのは、コンテナ再起動でローカルが消えるため
# 固定名にすると HF 上の前回起動分を上書きしてしまうから
# (train.log は逆に resume で復元して追記するので固定名のまま)。
#
# 置き場所は CODE_DIR の外にする: CODE_DIR は git clone の対象で、
# 事前にファイルを作ると「destination path already exists and is not empty」
# で clone が失敗する。
: "${CODE_DIR:=/workspace/code}"
: "${CKPT_DIR:=${CODE_DIR}/checkpoints}"
: "${TRAIN_LOG:=${CKPT_DIR}/train.log}"
: "${LOG_DIR:=/workspace/logs}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
: "${BOOT_LOG:=${LOG_DIR}/entrypoint-${RUN_ID}.log}"
mkdir -p "$(dirname "${BOOT_LOG}")"
# uploader が HF へ上げるログ(既定は CKPT_DIR 直下の *.log なので、
# CKPT_DIR 外に置く起動ログは明示的に渡す)。
: "${HF_LOG_FILES:=${TRAIN_LOG}:${BOOT_LOG}}"
export CODE_DIR CKPT_DIR TRAIN_LOG LOG_DIR RUN_ID BOOT_LOG HF_LOG_FILES
# fd 3 = 素のコンソール。train.log の tail はこちらへ流す
# (tee に流すと起動ログに train.log 全体が二重に入る)。
exec 3>&1
exec > >(tee -a "${BOOT_LOG}") 2>&1
log "起動ログ: ${BOOT_LOG} → HF ${HF_REPO_ID:-(未設定)}/${HF_CKPT_PREFIX:-checkpoints}/$(basename "${BOOT_LOG}")"

# --- 1) HF トークンをファイルから取得(履歴・レイヤーに残さない)-------------
# 優先: HF_TOKEN_FILE で指定されたパス → DOK のシークレット既定パス。
: "${HF_TOKEN_FILE:=/run/secrets/hf_token}"
if [[ -z "${HF_TOKEN:-}" && -f "${HF_TOKEN_FILE}" ]]; then
  HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
  export HF_TOKEN
fi
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- 2) 学習コードを実行時に取得 ---------------------------------------------
# CODE_DIR は 0) で解決済み。
: "${CODE_REF:=main}"
if [[ -n "${CODE_REPO:-}" ]]; then
  if [[ -d "${CODE_DIR}/.git" ]]; then
    log "git pull ${CODE_DIR} (${CODE_REF})"
    git -C "${CODE_DIR}" fetch --depth 1 origin "${CODE_REF}"
    git -C "${CODE_DIR}" checkout -f "${CODE_REF}"
    git -C "${CODE_DIR}" reset --hard "origin/${CODE_REF}" || true
  else
    log "git clone ${CODE_REPO} -> ${CODE_DIR} (${CODE_REF})"
    git clone --depth 1 --branch "${CODE_REF}" "${CODE_REPO}" "${CODE_DIR}"
  fi
else
  log "CODE_REPO 未設定。${CODE_DIR} に既にあるコードを使用します。"
  mkdir -p "${CODE_DIR}"
fi
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

# --- 2.5) 学習データ準備(parquet が無ければ生成)-----------------------------
# コードは実行時 clone のため、fresh 起動ではまだ parquet が無い。
# bfcl_eval はイメージに焼いてあるので、その同梱データから生成する。
: "${TRAIN_FILE:=${CODE_DIR}/data/bfcl_mt_train.parquet}"
: "${VAL_FILE:=${CODE_DIR}/data/bfcl_mt_val.parquet}"
: "${BFCL_CATEGORIES:=multi_turn_base,multi_turn_miss_func,multi_turn_miss_param}"
# シングルターンを train に混ぜて既存能力の忘却を抑え、val にも入れて退行を検知する。
# BFCL_ST_CATEGORIES="" で無効化(= 従来のマルチターン専用構成に戻す)。
: "${BFCL_ST_CATEGORIES:=simple_python,multiple,parallel,live_simple}"
: "${ST_TRAIN_RATIO:=0.3}"
: "${VAL_ST_SIZE:=0}"
# multi_turn の val を API クラス単位で切り出し、未知 API への汎化(OOD 転移)を測る。
# 空にするとランダム分割に戻る(= 同一 API 内の未知シナリオしか測れない)。
: "${HOLDOUT_CLASSES:=TravelAPI,VehicleControlAPI}"
if [[ ! -f "${TRAIN_FILE}" ]]; then
  log "学習 parquet が無いため生成します(mt=${BFCL_CATEGORIES} st=${BFCL_ST_CATEGORIES:-なし} holdout=${HOLDOUT_CLASSES:-なし})"
  BFCL_DATA_DIR="$(python -c "import bfcl_eval,os;print(os.path.join(os.path.dirname(bfcl_eval.__file__),'data'))")"
  python prepare_bfcl_data.py \
    --bfcl-data-dir "${BFCL_DATA_DIR}" \
    --out-dir "$(dirname "${TRAIN_FILE}")" \
    --categories "${BFCL_CATEGORIES}" \
    --st-categories "${BFCL_ST_CATEGORIES}" \
    --st-train-ratio "${ST_TRAIN_RATIO}" \
    --val-st-size "${VAL_ST_SIZE}" \
    --holdout-classes "${HOLDOUT_CLASSES}"
fi
export TRAIN_FILE VAL_FILE

# --- NCCL 環境チューニング(設計書: NCCL_ALGO / NCCL_P2P_LEVEL 等)-----------
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# 任意: 8GPU all_reduce のスモークテスト(設計書「実際に通るか確認」)。
# torchrun は stdin からスクリプトを読めないため一時ファイルに書き出して起動。
if [[ "${NCCL_SMOKE_TEST:-0}" == "1" ]]; then
  log "NCCL all_reduce スモークテスト実行(${N_GPUS:-8} GPU)"
  cat > /tmp/nccl_test.py <<'PY'
import time, torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(r % torch.cuda.device_count())
t = torch.ones(1 << 24, device="cuda") * (r + 1)  # ~16M float
torch.cuda.synchronize(); s = time.time()
for _ in range(20): dist.all_reduce(t)
torch.cuda.synchronize()
if r == 0: print(f"[nccl] all_reduce ok, 20 iters {time.time()-s:.3f}s", flush=True)
dist.destroy_process_group()
PY
  torchrun --nproc_per_node="${N_GPUS:-8}" /tmp/nccl_test.py || log "NCCL テスト失敗(継続)"
fi

# --- 3) resume(HF から最新 checkpoint を復元)+ 背景アップローダ起動 --------
# CKPT_DIR は 0) で解決済み(run script の trainer.default_local_dir と揃える)。
# 実体の作成はここ: CODE_DIR の git clone 後でなければ clone が失敗する。
mkdir -p "${CKPT_DIR}"

# resume 採用(設計書): 途中中断時、HF 上の最新 checkpoint をローカルへ戻し、
# trainer.resume_mode=auto(run script 側で指定)で再開する。
: "${RESUME_FROM_HF:=1}"
if [[ -n "${HF_REPO_ID:-}" && "${RESUME_FROM_HF}" == "1" ]]; then
  log "resume 確認: HF 上の最新 checkpoint を復元(あれば)"
  python hf_checkpoint_uploader.py --mode restore || log "restore 失敗(fresh start で継続)"
fi

UPLOADER_PID=""
if [[ -n "${HF_REPO_ID:-}" ]]; then
  log "checkpoint uploader 起動: ${HF_REPO_ID}/${HF_CKPT_PREFIX:-checkpoints} every=${UPLOAD_EVERY:-${SAVE_FREQ:-10}} 保持=最良${KEEP_BEST:-3}+直近${KEEP_LATEST:-3} 中身=${HF_PUSH_CONTENTS:-lora}"
  log "ログ push: ${HF_LOG_FILES} を ${LOG_UPLOAD_SECONDS:-300}s ごとに送信"
  python hf_checkpoint_uploader.py --mode watch &
  UPLOADER_PID=$!
else
  log "HF_REPO_ID 未設定。checkpoint / ログのアップロードは無効。"
fi

# 学習が失敗/終了しても必ず最終 push とアップローダ停止を行う。
final_sync() {
  local code=$?
  # EarlyStopping による停止は正常終了として扱う(monitor がマーカーを残す)。
  if [[ ${code} -ne 0 && -f "${CKPT_DIR}/early_stopped.json" ]]; then
    log "EarlyStopping による停止を検知。exit=0 として扱います。"
    code=0
  fi
  log "学習プロセス終了 (exit=${code})。最終 checkpoint を同期 push します。"
  if [[ -n "${HF_REPO_ID:-}" ]]; then
    python hf_checkpoint_uploader.py --mode final || log "最終 push 失敗"
    # 設計書: 学習終了後、merge 版 checkpoint を HF へアップロード
    # (どの CP か・eval の数字は eval_record.json に記録される)。
    if [[ "${MERGE_FINAL:-1}" == "1" ]]; then
      python hf_checkpoint_uploader.py --mode merged || log "merged push 失敗"
    fi
  fi
  if [[ -n "${UPLOADER_PID}" ]]; then
    kill "${UPLOADER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${TAIL_PID:-}" ]]; then
    kill "${TAIL_PID}" 2>/dev/null || true
  fi
  # 背景 uploader を止めた後にログだけ同期 push する。
  # ここが最後の砦: checkpoint が 1 つも無くても(= 学習が最初の save 前に
  # 落ちても)train.log と起動ログは HF に残る。終了理由が載る最後の数十行も
  # このタイミングで初めて確定する。
  if [[ -n "${HF_REPO_ID:-}" ]]; then
    python hf_checkpoint_uploader.py --mode logs --reason "exit=${code}" \
      || log "ログ push 失敗"
  fi
  # 非同期送信の完了待ちバッファ(強制終了での取りこぼし防止)。
  log "バッファ待機 ${FINAL_BUFFER_SECONDS:-300}s"
  sleep "${FINAL_BUFFER_SECONDS:-300}"
  exit "${code}"
}
trap final_sync EXIT

# --- 4) 学習起動(train.log 記録 + 監視)-------------------------------------
# ハイパラは環境変数として run script に渡る(run script 側が os.getenv 相当で読む)。
: "${BASE_MODEL:=${BASE_MODEL_PATH:-Qwen/Qwen3-8B}}"
export BASE_MODEL
# TRAIN_LOG は 0) で解決・export 済み(uploader へ HF_LOG_FILES で渡すため)。
log "学習開始 BASE_MODEL=${BASE_MODEL} N_GPUS=${N_GPUS:-8} GROUP_SIZE=${GROUP_SIZE:-8} log=${TRAIN_LOG}"

# 設計書: eval_steps ごとの train.log を記録。
# 学習は setsid で独立プロセスグループにし(EarlyStopping の killpg 用)、
# 出力は train.log へ集約。DOK コンソールには tail で転送する。
#
# resume 対応: train.log は checkpoint と一緒に HF へ push/restore されるため
# (hf_checkpoint_uploader)、再開時は復元済みの過去分の末尾へ追記される。
# 区切り行で再開位置を明示し、開始時サイズを train_monitor に渡して
# 過去分の再処理(val_metrics.jsonl の重複記録)を防ぐ。
touch "${TRAIN_LOG}"
if [[ -s "${TRAIN_LOG}" ]]; then
  printf '\n==== [entrypoint] resume %s : 以降を追記 ====\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${TRAIN_LOG}"
fi
TRAIN_LOG_OFFSET="$(stat -c %s "${TRAIN_LOG}")"
export TRAIN_LOG_OFFSET
setsid bash run_bfcl_grpo_h100x8.sh >>"${TRAIN_LOG}" 2>&1 &
TRAIN_PID=$!
# コンソール(fd 3)へ直接流す。tee 経由にすると起動ログに train.log が
# 丸ごと二重で入り、HF へ上げるログが無駄に膨らむ。
tail -n 0 -F "${TRAIN_LOG}" >&3 &
TAIL_PID=$!

# train.log から val 指標を抽出し EarlyStopping(patience=3)と
# 二層報酬 λ 減衰用の global_step.txt 更新を行う監視プロセス。
# --start-offset で復元済み過去分をスキップ(今回の学習分だけを読む)。
python train_monitor.py --log "${TRAIN_LOG}" --ckpt-dir "${CKPT_DIR}" \
  --train-pid "${TRAIN_PID}" --start-offset "${TRAIN_LOG_OFFSET}" &
MONITOR_PID=$!

TRAIN_RC=0
wait "${TRAIN_PID}" || TRAIN_RC=$?
log "run_bfcl_grpo_h100x8.sh 終了 (rc=${TRAIN_RC})"
exit "${TRAIN_RC}"
# exit で trap final_sync が発火する(early_stopped.json があれば rc=0 扱い)。
