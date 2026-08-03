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
# sglang 0.4.6.post5 は torchao==0.9.0 / transformers==4.51.1 を厳密に固定する。
# ここを別値にしても後の sglang インストールで上書きされるだけなので合わせる。
ARG TORCHAO_VERSION=0.9.0
ARG TRANSFORMERS_VERSION=4.51.1
# peft / compressed-tensors は上限なしの推移依存で、放置すると最新版が入る。
# 下の「依存の整合」レイヤーの説明を参照(どちらも実機で学習を落とした)。
ARG PEFT_VERSION=0.15.2
# vllm 0.8.5.post1 が compressed-tensors==0.9.3 を厳密要求するため 0.9.3 に揃える。
# ここで 0.9.4 を強制すると「依存の整合」層が vllm のピンを事後的に破る。
# この固定の目的は 0.16 以降(transformers.masking_utils を参照する版)を防ぐこと
# なので、0.9.3 でも目的は達成される。
ARG COMPRESSED_TENSORS_VERSION=0.9.3
ARG FLASH_ATTN_VERSION=2.7.4.post1
# vllm は rollout には使わないが、インストール自体は必須(下の vllm レイヤー参照)。
# torch を動かさない版を選ぶこと(torch 2.6.0 系 → vllm 0.8.5.post1)。
# 未検証の組み合わせを踏んだら --build-arg VLLM_VERSION=... で差し替える。
ARG VLLM_VERSION=0.8.5.post1
# sglang/verl の依存解決がベースの torch 2.5.1 を 2.6.0+cu124 へ引き上げる
# (CUDA 12.4 のままなので許容して進める)。実機で問題が出た場合の退避策として、
# TORCH_PIN=2.5.1 のように指定すると全依存の導入後に torch を強制ピンする。
ARG TORCH_PIN=""

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
    pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation && \
    pip install "sglang[all]==${SGLANG_VERSION}" && \
    pip install "verl==${VERL_VERSION}" && \
    pip install "bfcl-eval" wandb pandas pyarrow && \
    if [ -n "${TORCH_PIN}" ]; then pip install "torch==${TORCH_PIN}"; fi

# --- vllm(rollout では使わないが verl の import に必要)------------------------
# verl 0.4.1 の verl.workers.sharding_manager は fsdp_vllm を無条件に import する
# ため、rollout に sglang を使う構成でも vllm が無いと
#   verl.workers.sharding_manager.fsdp_sglang → No module named 'vllm'
# になり、verl_lora_sglang_patch が LoRA を rollout へマージできない
# (= base 重みのまま rollout する = 学習が無意味)。実行時に
# apply_patch() が RuntimeError で中断する設計なので、ここで必ず入れておく。
#
# 独立したレイヤーにしているのは、上の pip レイヤーが flash-attn の
# コンパイルで 200 秒超かかるため。vllm のバージョンを試行錯誤しても
# そこを作り直さずに済むようにする。
#
# import ではなくメタデータで確認する: CPU ビルド機では vllm の import が
# CUDA 不在で失敗し得る(ビルド時 import 検査も ModuleNotFoundError 以外は
# WARN 扱いにしてある)。ここで見たいのは「入っているか」だけ。
RUN set -e; \
    pip install "vllm==${VLLM_VERSION}" && \
    python -c "import importlib.metadata as m; print('vllm', m.version('vllm'))"

# --- 依存の整合(上限なしの推移依存が最新版へ解決されるのを戻す)---------------
# sglang / verl は peft と compressed-tensors をバージョン指定なしで要求するため、
# ビルド時期によっては 2025 年の torch/transformers の上に 2026 年の最新版が乗る。
# 実機で踏んだ不整合は 2 つ:
#   - peft>=0.19 は torchao>=0.16 を要求する。sglang が固定する torchao 0.9.0 では
#     LoRA 注入(get_peft_model)が
#     "Found an incompatible version of torchao" で ImportError になる。
#   - compressed-tensors>=0.16 は import 時に transformers.masking_utils
#     (transformers 4.52 以降)を参照する。sglang が固定する 4.51.1 では
#     sglang を掴む import 経路が ModuleNotFoundError になり、
#     verl_lora_sglang_patch がパッチを当てられない(= LoRA 未マージのまま
#     rollout が base 重みで走る)。
# どちらも sglang 0.4.6.post5 と同時期のバージョンへ固定して回避する。
# 必ず sglang/verl/bfcl-eval の後に置くこと(前だと上書きされる)。
RUN set -e; \
    pip install \
        "torchao==${TORCHAO_VERSION}" \
        "transformers==${TRANSFORMERS_VERSION}" \
        "peft==${PEFT_VERSION}" \
        "compressed-tensors==${COMPRESSED_TENSORS_VERSION}"

# --- NCCL を最新へ手動アップグレード -----------------------------------------
# 設計書: 8GPU 全部で all_reduce が通るか / 実際に速くなるかを実機ベンチで確認。
# ここでは pip 版 nvidia-nccl を最新へ上げるだけ。検証は entrypoint の
# NCCL_SMOKE_TEST=1 で実行(下記 entrypoint 参照)。
RUN set -e; pip install --upgrade nvidia-nccl-cu12

# --- setuptools の永続化(verl の pkg_resources 依存)--------------------------
# sglang/verl の依存解決の過程で setuptools が抜け、実行時に
# 「ModuleNotFoundError: No module named 'pkg_resources'」で verl が import
# 不可になった。実行中コンテナへの pip install は --rm で破棄されるため、
# 必ずイメージ側(最後の pip レイヤーの後)で入れて永続化する。
# setuptools 81 以降は pkg_resources を削除予定のため上限を掛ける。
RUN set -e; \
    pip install --force-reinstall "setuptools<81" && \
    python -c "import pkg_resources; print('pkg_resources OK')"

# --- opentelemetry を ray が動く列へ揃える(ノード起動の前提条件)-------------
# ray は起動時に dashboard(と各ノードの agent)を立て、その中で
# ray/dashboard/modules/aggregator/... が opentelemetry でメトリクスを登録する。
# ここで例外が出るとプロセスが死に、raylet が
# 「Exception: The current node timed out during startup.」で落ちる
# = ray.init() が返らず学習が 1 step も進まない。実機で 2 形踏んだ:
#   ImportError: cannot import name 'OtelComponentTypeValues'
#       from 'opentelemetry.semconv._incubating.attributes.otel_attributes'
#   TypeError: Meter.create_histogram() got an unexpected keyword argument
#       'explicit_bucket_boundaries_advisory'   (← api<1.30 だと出る)
#
# opentelemetry-python は安定版(api/sdk = 1.X.Y)とプレリリース版
# (semantic-conventions / exporter 群 = 0.(X+21)bZ)を必ず同時に出し、
# sdk のメタデータが仲間を == で厳密固定する(sdk 1.37.0 → semconv 0.58b0)。
# 上の pip レイヤー群(sglang / verl / vllm / 依存の整合)はそれぞれ独立に
# 依存解決するため、どれか 1 つが otel を上げ下げすると列が割れる。
#
# 揃える先を「入っている sdk」にしてはいけない(1 敗)。実機のイメージは
# api/sdk=1.26.0(2024 年)に対し exporter=0.65b0(= 1.44 列)で、sdk 基準に
# 揃えたら exporter が 0.47b0 へ落ちて上の TypeError になった。ray は
# setup.py の extras["default"] で opentelemetry-sdk >= 1.30.0 を宣言しており、
# そちらが実際の下限になる。よって 2 段階で揃える:
#   1) ray が宣言する下限を下回っていたら sdk をそこまで上げる(deps 込み)
#   2) その上で exporter を semantic-conventions と同じ番号へ揃える
# 必ず全ての pip レイヤーの後に置くこと(前だと後続に壊される)。
#
# 列を明示したい場合は --build-arg OTEL_SDK_VERSION=1.44.0(下限として効く)。
# 副作用: vllm/sglang の opentelemetry-exporter-otlp-* が古い列に取り残される
# ことがある。tracing を有効にした時しか import されないので学習には無害。
ARG OTEL_SDK_VERSION=""
RUN OTEL_SDK_VERSION="${OTEL_SDK_VERSION}" python - <<'PY'
import importlib, os, re, subprocess, sys
from importlib import metadata

def ver(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None

def pip(*specs):
    # バージョンだけ動かしたいので --no-deps(新しい torch 等を引き込ませない)。
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", *specs])
    # 直後の metadata 読み直しが古い値を返さないようにする
    # (importlib のキャッシュは site-packages の mtime 基準で、解像度は 1 秒)。
    importlib.invalidate_caches()

def key(version):
    from packaging.version import Version
    return Version(version)

def requirements(package, pattern):
    out = {}
    for raw in metadata.requires(package) or []:
        spec = re.sub(r"\[[^\]]*\]", "", raw.split(";")[0]).strip()
        matched = re.match(pattern, spec)
        if matched:
            name = matched.group(1).lower().replace("_", "-")
            if name not in out or key(matched.group(2)) > key(out[name]):
                out[name] = matched.group(2)
    return out

if ver("opentelemetry-sdk") is None:
    print("[otel] opentelemetry-sdk 未導入。ray が otel を使わない構成として続行")
    raise SystemExit(0)

# 1) 揃える先(= 一番新しい列)まで sdk を上げる。候補は ray が宣言する下限と、
#    既に入っているプレリリース版の列(pip が ray のために解決した結果)。
#    常に新しい方へ寄せるのが肝で、古い方へ引き戻すと ray 向けの列を壊す。
#    api / semantic-conventions は sdk が == で固定しているので pip に解決させる
#    (プレリリースの b0/b1 を自前で当てにいかなくて済む)。
def stable_of(prerelease):
    matched = re.match(r"^0\.(\d+)b\d+$", prerelease or "")
    if not matched:
        return None
    minor = int(matched.group(1)) - 21   # 1.26↔0.47 / 1.37↔0.58 / 1.44↔0.65
    return f"1.{minor}.0" if minor >= 0 else None

floor = os.environ.get("OTEL_SDK_VERSION", "").strip()
if not floor:
    candidates = []
    if ver("ray") is not None:
        ray_min = requirements(
            "ray", r"^([A-Za-z0-9._-]+)\s*>=\s*([^\s,;()\[\]]+)").get("opentelemetry-sdk")
        if ray_min:
            candidates.append(ray_min)
    for package in ("opentelemetry-exporter-prometheus",
                    "opentelemetry-semantic-conventions"):
        stable = stable_of(ver(package))
        if stable:
            candidates.append(stable)
    floor = max(candidates, key=key) if candidates else ""
if floor and key(ver("opentelemetry-sdk")) < key(floor):
    print(f"[otel] sdk {ver('opentelemetry-sdk')} は揃える先 {floor} より古い。上げます")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           f"opentelemetry-sdk=={floor}"])
    importlib.invalidate_caches()

sdk = ver("opentelemetry-sdk")
# 2) sdk の == ピンを正として仲間を揃える。
pinned = requirements("opentelemetry-sdk",
                      r"^([A-Za-z0-9._-]+)\s*==\s*([^\s,;()\[\]]+)$")

want = {p: pinned[p] for p in ("opentelemetry-api",
                               "opentelemetry-semantic-conventions") if p in pinned}
# exporter 群は semantic-conventions と同じ番号で出る。ray が掴むのは
# prometheus exporter だけなので、揃えるのもそれだけにする。
train = want.get("opentelemetry-semantic-conventions")
if train and ver("opentelemetry-exporter-prometheus") is not None:
    want["opentelemetry-exporter-prometheus"] = train

print(f"[otel] sdk={sdk} 期待={want}")
bad = [(p, v) for p, v in want.items() if ver(p) != v]
if bad:
    print("[otel] 列のズレを検出: " + ", ".join(f"{p} {ver(p)} -> {v}" for p, v in bad))
    pip(*[f"{p}=={v}" for p, v in bad])
print("[otel] 揃え後: " + ", ".join(f"{p}={ver(p)}" for p in want))

# ray の dashboard / agent が踏むのと同じ経路で検査する。メトリクス登録は
# モジュール読み込み時に走るので、バージョン不一致は import 時に露見する
# (opentelemetry.exporter.prometheus 単体の import では上の TypeError を
#  素通りしてしまう)。ここが通らないイメージは出荷しない。
if ver("ray") is not None:
    subprocess.check_call([sys.executable, "-c", (
        "import importlib, sys\n"
        "name = 'ray.dashboard.modules.aggregator.aggregator_agent'\n"
        "try:\n"
        "    importlib.import_module(name)\n"
        "except ModuleNotFoundError as e:\n"
        "    missing = e.name or ''\n"
        "    if name == missing or name.startswith(missing + '.'):\n"
        "        print('[otel] ' + name + ' は無いので検査を省略')\n"
        "        sys.exit(0)\n"
        "    raise\n"
        "print('[otel] ' + name + ' import OK')\n")])
PY

# --- ビルド時 import 検査 ------------------------------------------------------
# 実行時(= GPU 課金中)に初めて import 失敗が発覚するのを防ぐ。ここで落ちたら
# 依存解決が壊れている(例: torch 引き上げによる flash-attn の ABI 不整合)。
RUN python - <<'PY'
import pkg_resources
import torch, torchao, sglang, verl, flash_attn, transformers, peft, compressed_tensors
print("torch             :", torch.__version__)
print("torchao           :", torchao.__version__)
print("sglang            :", sglang.__version__)
print("verl              :", getattr(verl, "__version__", "?"))
print("flash_attn        :", flash_attn.__version__)
print("transformers      :", transformers.__version__)
print("peft              :", peft.__version__)
print("compressed_tensors:", compressed_tensors.__version__)

# peft>=0.19 + torchao<0.16 だと LoRA 注入時にここで ImportError になる。
# 実行時(GPU 課金中)ではなくビルド時に落とす。
from peft.import_utils import is_torchao_available
is_torchao_available()

# verl_lora_sglang_patch が掴む本体。ここが import できないと LoRA が sglang へ
# マージされず、rollout が base 重みのまま回る(学習が無意味になる)。
# CPU ビルド機では GPU 不在に起因する失敗もあり得るので、依存欠落
# (ModuleNotFoundError)だけをビルド失敗として扱う。
try:
    from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager  # noqa: F401
    print("fsdp_sglang       : OK")
except ModuleNotFoundError as e:
    raise SystemExit(f"[build-check] 依存の欠落で fsdp_sglang を import できません: {e}")
except Exception as e:
    print(f"[build-check] WARN: fsdp_sglang import は GPU 環境で要再確認: {type(e).__name__}: {e}")
PY

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
# 注意: ここで値を焼くと run script 側の既定(${VAR:-default})より優先される。
# 既定を 1 箇所に保つため、run script と同じ値だけを書くこと。
# UPLOAD_EVERY は意図的に焼かない(未設定なら SAVE_FREQ に追従する設計)。
#
# 同期漏れの実例(再発させないこと): MAX_PROMPT_LEN / MAX_RESPONSE_LEN を
# run script 側だけ 4096/2048 → 8192/8192 に直したことがあり、ここに焼いた
# 古い値が勝って修正が無効化された。プロンプトへ関数一覧を載せる変更や
# 応答長の調整を入れたら、必ずこのブロックも同じ値へ更新すること
# (症状は静かで、filter_overlong_prompts による行の欠落と
#  response_length/clip_ratio の上昇としてしか現れない)。
ENV CODE_REPO="" \
    CODE_REF="main" \
    CODE_DIR=/workspace/code \
    HF_REPO_ID="" \
    HF_CKPT_PREFIX="checkpoints" \
    HF_MERGED_PREFIX="merged-final" \
    HF_PUSH_CONTENTS="lora" \
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
    MAX_PROMPT_LEN=8192 \
    MAX_RESPONSE_LEN=8192 \
    GROUP_SIZE=8 \
    LORA_RANK=32 \
    LORA_ALPHA=32 \
    LR=1e-6 \
    TOTAL_EPOCHS=15 \
    SAVE_FREQ=10 \
    TEST_FREQ=5 \
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
