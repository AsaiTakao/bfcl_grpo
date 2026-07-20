# bfcl_grpo — BFCL multi_turn を GRPO で学習し τ-bench/τ2-bench で評価

MUA-RL(Multi-turn User-interacting Agent RL)を参考に、**BFCL v3 multi_turn で
学習(verl/GRPO)**し、**独立ベンチマーク τ-bench / τ2-bench で評価**する。
「一方で学習し、複数の独立ベンチマークで評価する」ことを実践する。

- 学習: BFCL `multi_turn`(base + miss_func + miss_param)/ base = Qwen/Qwen3-8B
- 評価: τ-bench(retail/airline)または τ2-bench(telecom)を **公式評価コード**で測る
- 実行環境: さくら高火力 DOK(H100 × 8、専有)。ビルドは安価な CPU インスタンス。

---

## 全体フロー

```
[学習イメージ: sglang]                       [評価イメージ: vLLM]
BFCL multi_turn ──GRPO(verl/LoRA r=32)──▶ checkpoint ──▶ merge(HF形式)──▶ vLLM OpenAI互換 serve
   (二層報酬)          │ 200step ごと HF へ push(最良eval3+直近3を保持)  │
   EarlyStopping(3)    │ 終了時 merged-final/ へ merge 版 + eval_record  │
                       └──────────────▶ HF ◀──────── DL ────────────────┘
                       (中断時は HF から restore して resume)  ▼
                          τ 評価:  phase0 ─▶ phase1 ─▶ phase2
```

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| `prepare_bfcl_data.py` | BFCL の question+possible_answer を verl parquet に変換 |
| `bfcl_dataset_and_interaction.py` | `build_verl_dataset` と user 役 `BFCLInteraction` |
| `bfcl_verl_tool.py` | verl `BaseTool`(`BFCLMultiTurnTool`) |
| `bfcl_backend.py` | BFCL バックエンドクラスの動的ロード/実行/状態比較 |
| `mt_reward.py` | 二層報酬(process × λ(step) + outcome × γ 後方伝播) |
| `bfcl_multiturn_grpo.yaml` / `bfcl_tool_config.yaml` / `bfcl_interaction_config.yaml` | verl 設定 |
| `run_bfcl_grpo_h100x8.sh` | 学習起動(env 駆動、LoRA r=32、FP8 rollout トグル、resume_mode=auto) |
| `train_monitor.py` | train.log 監視: val 指標抽出・EarlyStopping(patience=3)・λ 減衰用 step 供給 |
| `hf_checkpoint_uploader.py` | 200step ごと HF push(最良3+直近3保持)/ resume 復元 / merge 版最終 push |
| `Dockerfile` / `entrypoint.sh` | 学習イメージ(sglang) |
| `Dockerfile.eval` / `eval/entrypoint_eval.sh` | 評価イメージ(vLLM) |
| `eval/` | 評価ハーネス(下記) |

---

## 学習(training)

### データ準備
DOK 起動時は `entrypoint.sh` が **parquet が無ければ自動生成**する
(`BFCL_CATEGORIES` 環境変数で対象カテゴリを上書き可能)。手動で行う場合:
```bash
python prepare_bfcl_data.py \
  --bfcl-data-dir "$(python -c "import bfcl_eval,os;print(os.path.join(os.path.dirname(bfcl_eval.__file__),'data'))")" \
  --out-dir ./data \
  --categories multi_turn_base,multi_turn_miss_func,multi_turn_miss_param
```

### イメージビルド(CPU インスタンス、BuildKit 必須)
```bash
echo "hf_xxx" > hf_token.txt   # BuildKit secret で渡す。レイヤーに残さない。
DOCKER_BUILDKIT=1 docker build --secret id=hf_token,src=./hf_token.txt \
  -t <registry>/bfcl-mt-grpo:latest .
```

### DOK での起動(環境変数で制御)
`CODE_REPO`(このリポジトリの git URL)、`HF_REPO_ID`(checkpoint push 先)、
必要なハイパラを環境変数として設定して起動する。`entrypoint.sh` が
git clone → HF から最新 checkpoint を restore(resume)→ 背景アップローダ起動
→ 学習(出力は `${CKPT_DIR}/train.log` に記録、`train_monitor.py` が監視)
→ 終了時に最終同期 push + merge 版 push + バッファ待機。

- **checkpoint 保持**: HF の `checkpoints/global_step_N` に push し、
  「最良 eval 3 つ + 直近 3 つ」の最大 6 つだけ残す(それ以外は剪定)。
- **EarlyStopping**: val 指標(既定 reward 系)が patience=3 回連続で改善
  しなければ学習を停止(`early_stopped.json` を記録し正常終了扱い)。
- **merge 版**: 学習終了後、最良 checkpoint(eval 記録が無ければ最新)を
  HF 形式に merge して `merged-final/` へ push。どの CP か・eval の数字は
  `eval_record.json` に記録される。評価コンテナは
  `MERGED_MODEL_REPO=<HF_REPO_ID>` + `merged-final` をそのまま serve できる。

主な環境変数: `TRAIN_BATCH` `GROUP_SIZE` `LR` `TOTAL_EPOCHS` `SAVE_FREQ`(=200)
`UPLOAD_EVERY`(=200)`LORA_RANK`(=32、0 でフル FT)`ROLLOUT_FP8`(=0/1)
`EARLY_STOP`/`EARLY_STOP_PATIENCE`(=1/3)`KEEP_BEST`/`KEEP_LATEST`(=3/3)
`RESUME_FROM_HF`(=1)`MERGE_FINAL`(=1)。

---

## 評価(evaluation)— phase0 → phase1 → phase2

チェックポイントを HF 形式にマージ → vLLM を OpenAI 互換で起動 → τ を回す。
`eval/run_eval_pipeline.sh` が 1 チェックポイント分をまとめて実行する。

**技術的な要点**: 学習と同じく評価でも Qwen3 の thinking を発火させる。
τ はエージェント出力をパースするため、`<think>` が hermes の `tool_call` 抽出を
壊さないよう、vLLM を `--reasoning-parser qwen3`(`<think>` を分離)+
`--tool-call-parser hermes` で起動する。この整合を **phase0 で毎回検出**する。

### phase0 — mock + dummy_user(壊れ検出・API不要・CPごと毎回)
最も安価・高速。外部 API 不要。`<think>` と `tool_call` パースの整合を検査し、
破損(think が本文に漏れて `tool_calls` が空)なら **非ゼロ終了**して以降を止める。

```bash
CKPT_STEP_DIR=/workspace/code/checkpoints/.../global_step_600 \
MERGED_DIR=/workspace/merged/step600 \
PHASES="phase0" bash eval/run_eval_pipeline.sh
```
- 使用: `eval/phase0_smoke.py` + `eval/dummy_user.py`(公式ベンチ本体は使わない)

### phase1 — retail + ローカル user(傾向確認・API不要)
有望なチェックポイントで実施。user simulator も同じローカル vLLM を流用するため
**外部 API 不要**。tau-bench の `retail` を先頭数十タスクで回して傾向を見る。

```bash
PHASES="phase0 phase1" \
CKPT_STEP_DIR=... MERGED_DIR=... bash eval/run_eval_pipeline.sh
```
- 使用: `eval/run_tau_eval.py --phase phase1`(`--user-model-provider openai` = ローカル)

### phase2 — retail/airline + gemini-3.5-flash(公式準拠・最終CPのみ)
最終チェックポイントのみ。公式条件に準拠し、user simulator に
Google AI Studio の Gemini(litellm `google` provider)を使う。

```bash
# 実キーの渡し方は 2 通り(HF_TOKEN と同じくファイル管理を推奨)。
# (A) ファイル渡し【推奨】: キー値を環境変数に置かず、ファイル1枚に閉じ込める。
printf '%s' "$GEMINI_KEY" > /run/secrets/gemini_api_key   # 既定パス。または任意パス+ FILE 変数
# 任意パスにする場合: export GEMINI_API_KEY_FILE=/path/to/gemini_api_key
# (B) 従来どおり環境変数(後方互換):
# export GEMINI_API_KEY=...

PHASES="phase2" \
MERGE=0 MERGED_DIR=/workspace/merged/final bash eval/run_eval_pipeline.sh
```
> キー解決の優先順位は `GEMINI_API_KEY_FILE`(パス指定)→ 既定パス
> `/run/secrets/gemini_api_key` → 環境変数 `GEMINI_API_KEY`。ファイルが読めれば
> それを優先し、キー値は設定ファイル・コマンドライン・広い環境変数に残さない
> ([eval/eval_config.yaml](eval/eval_config.yaml) の phase2 `user` で切替可能)。
- 使用: `eval/run_tau_eval.py --phase phase2`(`--user-model-provider google`)
- telecom(τ2-bench)を測る場合は `eval/eval_config.yaml` の phase2 を
  `bench: tau2 / domain: telecom` に変更。

### 評価イメージのビルド
```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.eval -t <registry>/bfcl-eval:latest .
# コンテナ内でマージも行うなら: --build-arg WITH_VERL=1
```

### eval/ の中身
| ファイル | 役割 |
|---|---|
| `eval/eval_config.yaml` | phase0/1/2 の定義(domain・user・タスク数) |
| `eval/merge_checkpoint.sh` | verl FSDP checkpoint → HF 形式 |
| `eval/serve_agent_vllm.sh` | vLLM OpenAI 互換起動(reasoning/tool パーサ) |
| `eval/dummy_user.py` / `eval/phase0_smoke.py` | phase0 壊れ検出 |
| `eval/run_tau_eval.py` | phase1/2 の公式ランナー起動 + 集計 |
| `eval/run_eval_pipeline.sh` | 1CP分: merge→serve→phase0/1/2→停止 |

---

## 注意・未検証点

- **バージョン依存が強い**: verl / sglang / vLLM / bfcl_eval / tau-bench の API と
  CLI はバージョンで変わる。各所に env の逃げ道を用意している
  (`ROLLOUT_EXTRA_ARGS`、`TAU_RUN_CMD`、`REASONING_PARSER=""` など)。
- **LoRA(r=32)+ sglang rollout** の組合せは verl のバージョン依存が強い
  (`load_format=safetensors` 要件など)。動かない場合は `LORA_RANK=0` で
  フル FT に退避できる。merge は `verl.model_merger` →失敗時 peft
  `merge_and_unload` の 2 段構え(`LORA_MERGE=0` でスキップ)。
- **EarlyStopping / eval 抽出**は verl console logger の
  `step:N - key:value` 形式を前提に train.log を正規表現で読む
  (`train_monitor.py`)。形式が違う場合は `VAL_METRIC_KEY` /
  `VAL_LINE_REGEX` で調整する。
- **FP8 rollout** は既定 OFF。actor は bf16 のまま rollout エンジンのみ FP8。
  `ROLLOUT_FP8=1` で有効化し、phase0 で効果を実測してから採用する。
- **公式 tau-bench の結果フォーマット**に合わせて `run_tau_eval.py:parse_results`
  を微調整する必要がある(pass は `reward>=1` を成功とみなす慣例)。
- ローカル開発機に Python/Docker が無いため、**実機での起動確認が前提**。
