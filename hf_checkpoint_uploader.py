# -*- coding: utf-8 -*-
"""
hf_checkpoint_uploader.py

verl が書き出す checkpoint を HF Hub と同期する。設計書(entrypoint.sh の設計):
  - 学習途中の 200 steps ごとに checkpoint を HF へアップロード
  - 保持ポリシー: 「最良 eval 3 つ + 直近 3 つ」の合計最大 6 つを残す
    (eval_loss 最良 ≠ ベンチ最良の可能性があるため複数残す。重複時は 6 未満)
  - resume 採用: 途中中断時に最新 checkpoint から再開できるようにする
  - 学習終了後、merge 版 checkpoint を HF へアップロード
    (どの checkpoint か・eval の数字を eval_record.json に記録)
  - 非同期アップロードの取りこぼし防止 → final は同期 push + entrypoint がバッファ待機

repo 内レイアウト(HF_REPO_ID 直下):
  checkpoints/global_step_N/...   verl checkpoint(最大 6 つ保持)
  checkpoints/val_metrics.jsonl   train_monitor が抽出した eval 履歴
  checkpoints/train.log           学習ログ(resume 時に復元し末尾へ追記)
  checkpoints/entrypoint-*.log    起動時ログ(entrypoint.sh。起動ごとに 1 本)
  checkpoints/latest_checkpointed_iteration.txt   resume 用
  merged-final/...                学習終了後の merge 版(HF 形式)+ eval_record.json

モード:
  watch   : entrypoint が背景起動。CKPT_DIR を polling し、書き込み完了済みの
            step % UPLOAD_EVERY == 0 checkpoint を push → HF 側を保持ポリシーで剪定。
            併せて CKPT_DIR 直下の *.log を LOG_UPLOAD_SECONDS ごとに push。
  final   : 学習終了後に「今ある最新」を step 条件無視で同期 push + 剪定 + ログ push。
  restore : HF 上の最新 checkpoint を CKPT_DIR へ DL(resume 用。無ければ何もしない)。
            checkpoint を戻さない場合(HF_PUSH_CONTENTS=lora)でも train.log と
            val_metrics.jsonl だけは戻す(空ログで上書きして過去分を失わない)。
  merged  : 最良(無ければ最新)checkpoint を merge → merged-final/ へ同期 push。
  logs    : ログだけを同期 push(checkpoint の有無に関係なく)。

環境変数(引数でも指定可):
  CKPT_DIR        verl の default_local_dir(global_step_* が並ぶ場所)
  HF_REPO_ID      アップロード先 repo(例: user/qwen3-8b-bfcl-mt-grpo)
  HF_CKPT_PREFIX  repo 内 checkpoint 置き場(既定 checkpoints)
  UPLOAD_EVERY    アップロード間隔 step(既定 SAVE_FREQ、それも無ければ 10)
  KEEP_BEST / KEEP_LATEST  保持数(既定 3 / 3)
  POLL_SECONDS    watch の polling 間隔秒(既定 60)
  MERGED_DIR      merged モードの出力先(既定 /workspace/merged/final)
  HF_MERGED_PREFIX merged の repo 内パス(既定 merged-final)
  HF_PUSH_CONTENTS HF へ上げる中身(既定 lora)。lora=LoRA adapter と
                  tokenizer/config のみ / full=FSDP shard・optimizer も含む全部。
                  lora のときは optimizer 状態が無いため HF からの resume は不可。
  HF_TOKEN        書き込みトークン(未指定なら hf のログイン情報を使用)
  HF_LOG_FILES    HF へ上げるログの明示指定(":" or "," 区切り)。
                  既定は CKPT_DIR 直下の *.log すべて。
  LOG_UPLOAD_SECONDS  watch のログ push 間隔秒(既定 300)
  HF_LOG_MAX_MB   1 ファイルの上限 MB(既定 50)。超える分は「中央」を中略する
                  (先頭を捨てると DOK コンソールと同じ「先頭が切れたログ」になる)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

_STEP_RE = re.compile(r"global_step_(\d+)$")
_LATEST_FILE = "latest_checkpointed_iteration.txt"
_METRICS_FILE = "val_metrics.jsonl"
# HF へ上げる中身(HF_PUSH_CONTENTS)。
#   lora: LoRA adapter + tokenizer/config だけ(r=32 で ~200MB/個)。
#   full: FSDP shard と optimizer 状態も含む全部(8B フル FT なら ~80GB/個)。
# verl の checkpoint レイアウトは global_step_N/actor/{model,optim,extra}_world_size_*.pt
# + huggingface/(config・tokenizer)+ lora_adapter/(LoRA 学習時のみ)。
_LORA_ONLY_PATTERNS = ["*/lora_adapter/*", "*/huggingface/*"]
# resume を跨いで学習経過を1本のログに追記し続けるため、train.log も
# checkpoint と一緒に push/restore する(コンテナ再起動で workspace が
# 消えても過去分が末尾追記の起点として復元される)。
_TRAIN_LOG_FILE = "train.log"
# ログとして扱うファイル(CKPT_DIR 直下)。train.log と entrypoint-*.log。
_LOG_GLOB = "*.log"


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def log(msg: str) -> None:
    print(f"[uploader] {msg}", flush=True)


# --------------------------------------------------------------- ローカル側
def local_checkpoints(ckpt_dir: str) -> List[Tuple[int, str]]:
    """CKPT_DIR 直下の global_step_* を (step, path) 昇順で返す。"""
    out: List[Tuple[int, str]] = []
    if not os.path.isdir(ckpt_dir):
        return out
    for name in os.listdir(ckpt_dir):
        m = _STEP_RE.match(name)
        path = os.path.join(ckpt_dir, name)
        if m and os.path.isdir(path):
            out.append((int(m.group(1)), path))
    return sorted(out)


def completed_step(ckpt_dir: str) -> int:
    """verl が書き終えた step(latest_checkpointed_iteration.txt)。無ければ -1。

    これ以下の step の checkpoint だけを push 対象にする(書きかけ防止)。
    """
    p = os.path.join(ckpt_dir, _LATEST_FILE)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return -1


def load_val_metrics(ckpt_dir: str) -> List[Dict]:
    """train_monitor が書く val_metrics.jsonl を読む。無ければ空。"""
    p = os.path.join(ckpt_dir, _METRICS_FILE)
    out: List[Dict] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


def metric_for_step(metrics: List[Dict], ckpt_step: int) -> Optional[float]:
    """checkpoint step に対応する eval 値(その step 以前で最も近い記録)。"""
    best: Optional[Tuple[int, float]] = None
    for m in metrics:
        s, v = int(m.get("step", -1)), m.get("value")
        if v is None or s < 0 or s > ckpt_step:
            continue
        if best is None or s > best[0]:
            best = (s, float(v))
    return None if best is None else best[1]


def select_keep_steps(steps: List[int], metrics: List[Dict],
                      keep_best: int, keep_latest: int) -> Set[int]:
    """保持ポリシー: 最良 eval keep_best 個 + 直近 keep_latest 個(和集合)。"""
    keep: Set[int] = set(sorted(steps)[-keep_latest:]) if keep_latest > 0 else set()
    scored = [(metric_for_step(metrics, s), s) for s in steps]
    scored = [(v, s) for v, s in scored if v is not None]
    scored.sort(key=lambda t: (-t[0], -t[1]))  # 値が大きいほど良い前提(reward 系)
    keep.update(s for _, s in scored[:keep_best])
    return keep


# ------------------------------------------------------------------- ログ側
def log_files(ckpt_dir: str) -> List[str]:
    """HF へ上げるログファイルの実パス一覧。

    既定は CKPT_DIR 直下の *.log(train.log と entrypoint-*.log)。
    HF_LOG_FILES(":" または "," 区切り)を指定するとそのパスだけを対象にする。

    毎回 glob し直すのは、watch を起動した時点ではまだ train.log が
    存在しないため(entrypoint は uploader を学習より先に起動する)。
    """
    spec = _env("HF_LOG_FILES")
    if spec:
        paths = [p.strip() for p in re.split(r"[:,]", spec) if p.strip()]
    else:
        paths = sorted(glob.glob(os.path.join(ckpt_dir, _LOG_GLOB)))
    out: List[str] = []
    seen: Set[str] = set()
    for p in paths:
        base = os.path.basename(p)
        if base in seen or not os.path.isfile(p):
            continue
        seen.add(base)      # repo 側は basename で置くため同名は 1 つだけ
        out.append(p)
    return out


def capped_log_bytes(path: str, max_bytes: int) -> bytes:
    """ログを max_bytes 以内に収めて読む。超える場合は「中央」を中略する。

    末尾だけ残す(= 先頭を捨てる)実装にしないのは、それでは DOK コンソールと
    同じ「先頭が切れたログ」になり、起動時の設定・データ準備・最初の step という
    一番読みたい部分が消えるため。
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if max_bytes <= 0 or size <= max_bytes:
            return f.read()
        half = max_bytes // 2
        head = f.read(half)
        f.seek(size - half)
        tail = f.read(half)
    marker = (f"\n==== [uploader] 中略: {size - 2 * half} bytes 省略 "
              f"(元サイズ {size} bytes) ====\n").encode("utf-8")
    return head + marker + tail


class LogSync:
    """学習ログを checkpoint の push とは独立に HF へ送る。

    なぜ独立させるか: DOK のコンソールは先頭が切り落とされるため、ログは repo 側
    に残さないと読めない。checkpoint push に相乗りさせるだけでは
      - 最初の checkpoint(SAVE_FREQ step 目)より前に落ちた場合
      - HF_PUSH_CONTENTS=lora で adapter が無く push をスキップした場合
      - push が失敗した場合
    にログが 1 バイトも残らず、原因調査ができない。

    サイズ・mtime が変わっていないファイルは送らない(空コミットを積まない)。
    """

    def __init__(self, api, repo_id: str, prefix: str, ckpt_dir: str,
                 interval: int = 300, max_bytes: int = 50 * 1024 * 1024):
        self.api = api
        self.repo_id = repo_id
        self.prefix = prefix
        self.ckpt_dir = ckpt_dir
        self.interval = interval
        self.max_bytes = max_bytes
        self._sent: Dict[str, Tuple[int, int]] = {}   # path -> (size, mtime)
        self._next_at = 0.0
        self._repo_ready = False

    def sync(self, force: bool = False, reason: str = "periodic") -> int:
        """対象ログを push する。force=False なら interval を待つ。送った数を返す。"""
        now = time.time()
        if not force and now < self._next_at:
            return 0
        self._next_at = now + max(self.interval, 10)
        return sum(self._sync_one(p, reason) for p in log_files(self.ckpt_dir))

    def _sync_one(self, path: str, reason: str) -> int:
        try:
            st = os.stat(path)
        except OSError:
            return 0
        sig = (st.st_size, int(st.st_mtime))
        if self._sent.get(path) == sig:
            return 0                      # 変化なし
        name = os.path.basename(path)
        try:
            payload = capped_log_bytes(path, self.max_bytes)
            if not self._repo_ready:
                self.api.create_repo(self.repo_id, repo_type="model",
                                     exist_ok=True, private=True)
                self._repo_ready = True
            self.api.upload_file(
                path_or_fileobj=payload,
                repo_id=self.repo_id,
                path_in_repo=f"{self.prefix}/{name}",
                commit_message=f"log: {name} ({reason})",
            )
        except Exception as e:  # noqa: BLE001  ログ送信の失敗で学習を止めない
            print(f"[uploader] ログ push 失敗 {name}: {e}", file=sys.stderr,
                  flush=True)
            return 0
        self._sent[path] = sig
        log(f"ログ push: {self.prefix}/{name} ({st.st_size}B, {reason})")
        return 1


def _log_sync(api, args) -> LogSync:
    return LogSync(api, args.repo_id, args.prefix, args.ckpt_dir,
                   interval=args.log_every,
                   max_bytes=max(args.log_max_mb, 0) * 1024 * 1024)


def _move_meta(tmp: str, prefix: str, ckpt_dir: str,
               names: Tuple[str, ...]) -> List[str]:
    """DL 済み一時ディレクトリから CKPT_DIR へ meta を移す。移せた名前を返す。

    ローカルに既に同名がある場合は触らない(今回の起動で書き始めたものを
    HF 上の古い版で潰さない)。
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    moved: List[str] = []
    for name in names:
        src = os.path.join(tmp, prefix, name)
        dst = os.path.join(ckpt_dir, name)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.move(src, dst)
            moved.append(name)
    return moved


def restore_meta(ckpt_dir: str, repo_id: str, prefix: str,
                 token: Optional[str]) -> List[str]:
    """train.log と val_metrics.jsonl だけを HF から戻す(checkpoint は触らない)。

    なぜ checkpoint を戻さないケースでも必要か: train.log は repo 側で固定名
    (checkpoints/train.log)なので、復元せずに学習を始めると LogSync が
    「今回の起動分だけ」の短いログを同名で push し、HF 上の過去の学習経過を
    上書きして消してしまう。DOK コンソールは先頭が切れるため、そうなると
    起動〜最初の step の記録がどこにも残らない。

    latest_checkpointed_iteration.txt は戻さない(戻すと resume_mode=auto が
    存在しない checkpoint を掴んで壊れる)。
    """
    from huggingface_hub import snapshot_download

    want = tuple(n for n in (_TRAIN_LOG_FILE, _METRICS_FILE)
                 if not os.path.isfile(os.path.join(ckpt_dir, n)))
    if not want:
        return []
    tmp = os.path.join(ckpt_dir, "_restore_meta_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[f"{prefix}/{n}" for n in want],
            local_dir=tmp,
            token=token,
        )
        moved = _move_meta(tmp, prefix, ckpt_dir, want)
    except Exception as e:  # noqa: BLE001  repo 未作成・初回起動など
        log(f"ログ/val 履歴の restore をスキップ(初回なら正常): {e}")
        moved = []
    shutil.rmtree(tmp, ignore_errors=True)
    if moved:
        log(f"restore: {', '.join(moved)} を復元(以降は末尾へ追記)")
    else:
        log("HF 上に過去の train.log / val 履歴なし(新規として開始)")
    return moved


# ------------------------------------------------------------------- HF 側
def _api(token: Optional[str]):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def remote_steps(api, repo_id: str, prefix: str) -> Set[int]:
    """repo 内 {prefix}/global_step_N/ に存在する step 集合。"""
    steps: Set[int] = set()
    try:
        files = api.list_repo_files(repo_id, repo_type="model")
    except Exception as e:  # noqa: BLE001  repo 未作成など
        log(f"list_repo_files 失敗(未作成なら無視可): {e}")
        return steps
    pat = re.compile(re.escape(prefix) + r"/global_step_(\d+)/")
    for f in files:
        m = pat.match(f)
        if m:
            steps.add(int(m.group(1)))
    return steps


def has_lora_adapter(step_path: str) -> bool:
    """checkpoint 内に LoRA adapter があるか(actor/lora_adapter/)。"""
    for root, dirs, _ in os.walk(step_path):
        if "lora_adapter" in dirs:
            return True
    return False


def push_checkpoint(api, repo_id: str, prefix: str, step: int, path: str,
                    ckpt_dir: str, contents: str = "full",
                    log_max_bytes: int = 50 * 1024 * 1024) -> None:
    """global_step_N を repo へ push し、resume 用メタも同時に上げる。

    contents="lora" のときは LoRA adapter と tokenizer/config だけを上げる
    (FSDP shard と optimizer 状態は上げない = 途中再開はできなくなる)。
    """
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
    allow = _LORA_ONLY_PATTERNS if contents == "lora" else None
    api.upload_folder(
        folder_path=path,
        repo_id=repo_id,
        path_in_repo=f"{prefix}/global_step_{step}",
        allow_patterns=allow,
        commit_message=f"checkpoint step={step}"
                       + (" (lora only)" if contents == "lora" else ""),
    )
    for meta in (_LATEST_FILE, _METRICS_FILE, _TRAIN_LOG_FILE):
        mp = os.path.join(ckpt_dir, meta)
        if not os.path.isfile(mp):
            continue
        # ログは巨大化し得るので上限付きで読む(LogSync と同じ扱い)。
        body = capped_log_bytes(mp, log_max_bytes) if meta.endswith(".log") else mp
        api.upload_file(
            path_or_fileobj=body,
            repo_id=repo_id,
            path_in_repo=f"{prefix}/{meta}",
            commit_message=f"meta update (step={step})",
        )


def prune_remote(api, repo_id: str, prefix: str, metrics: List[Dict],
                 keep_best: int, keep_latest: int) -> None:
    """HF 側を保持ポリシー(最良 keep_best + 直近 keep_latest)へ剪定する。"""
    steps = sorted(remote_steps(api, repo_id, prefix))
    if not steps:
        return
    keep = select_keep_steps(steps, metrics, keep_best, keep_latest)
    for s in steps:
        if s in keep:
            continue
        log(f"剪定: {prefix}/global_step_{s} を削除")
        try:
            api.delete_folder(
                path_in_repo=f"{prefix}/global_step_{s}",
                repo_id=repo_id,
                commit_message=f"prune step={s} (keep best{keep_best}+latest{keep_latest})",
            )
        except Exception as e:  # noqa: BLE001  剪定失敗で学習を止めない
            log(f"剪定失敗 step={s}: {e}")
    log(f"保持中: {sorted(keep & set(steps))}")


# ----------------------------------------------------------------- 各モード
def _push_once(api, args, step: int, path: str) -> bool:
    contents = getattr(args, "contents", "full")
    if contents == "lora" and not has_lora_adapter(path):
        # LoRA 学習でなければ adapter は生まれない。ここで full に落とすと
        # 意図せず 80GB 級を上げてしまうので、上げずに知らせるだけにする。
        log(f"step={step}: lora_adapter が無いため push をスキップ"
            f"(HF_PUSH_CONTENTS=lora)。フル checkpoint を上げるなら "
            f"HF_PUSH_CONTENTS=full を指定してください。")
        return False
    log(f"push step={step} [{contents}] -> "
        f"{args.repo_id}/{args.prefix}/global_step_{step}")
    try:
        push_checkpoint(api, args.repo_id, args.prefix, step, path,
                        args.ckpt_dir, contents=contents,
                        log_max_bytes=max(getattr(args, "log_max_mb", 50), 0)
                        * 1024 * 1024)
        prune_remote(api, args.repo_id, args.prefix,
                     load_val_metrics(args.ckpt_dir), args.keep_best, args.keep_latest)
        log(f"done step={step}")
        return True
    except Exception as e:  # noqa: BLE001  push 失敗で学習を止めない
        print(f"[uploader] FAILED step={step}: {e}", file=sys.stderr, flush=True)
        return False


def mode_watch(args, token: Optional[str]) -> int:
    api = _api(token)
    uploaded: Set[int] = remote_steps(api, args.repo_id, args.prefix)
    logs = _log_sync(api, args)
    log(f"watch start dir={args.ckpt_dir} every={args.every} "
        f"keep=best{args.keep_best}+latest{args.keep_latest} "
        f"log_every={args.log_every}s 既存remote={sorted(uploaded)}")
    while True:
        done = completed_step(args.ckpt_dir)
        for step, path in local_checkpoints(args.ckpt_dir):
            if step in uploaded or step > done:
                continue
            if args.every > 0 and step % args.every != 0:
                continue
            if _push_once(api, args, step, path):
                uploaded.add(step)
        # checkpoint が 1 つも生まれていなくてもログは送る(落ちたときに
        # DOK コンソールしか残っていないと先頭が切れて原因が読めない)。
        logs.sync()
        time.sleep(max(args.poll, 5))


def mode_final(args, token: Optional[str]) -> int:
    """学習終了後: 最新 checkpoint を step 条件無視で同期 push(取りこぼし防止)。

    checkpoint の有無・push の成否に関わらず、先にログを送る(学習が
    checkpoint を作る前に落ちたケースが一番ログを読みたい)。
    """
    api = _api(token)
    logs = _log_sync(api, args)
    logs.sync(force=True, reason="final")

    ckpts = local_checkpoints(args.ckpt_dir)
    if not ckpts:
        log("ローカル checkpoint 無し。final push をスキップ。")
        return 0
    done = completed_step(args.ckpt_dir)
    valid = [c for c in ckpts if done < 0 or c[0] <= done]
    if not valid:
        log("書き込み完了済み checkpoint 無し。スキップ。")
        return 0
    step, path = valid[-1]
    if step in remote_steps(api, args.repo_id, args.prefix):
        log(f"最新 step={step} は push 済み。剪定のみ実施。")
        prune_remote(api, args.repo_id, args.prefix,
                     load_val_metrics(args.ckpt_dir), args.keep_best, args.keep_latest)
        return 0
    return 0 if _push_once(api, args, step, path) else 1


def mode_logs(args, token: Optional[str]) -> int:
    """ログだけを同期 push する。

    entrypoint の final_sync 末尾から呼ぶことで、最終 push / merge の結果や
    終了理由(EarlyStopping・OOM など)まで含めた最後の数十行を repo に残す。
    """
    logs = _log_sync(_api(token), args)
    n = logs.sync(force=True, reason=args.reason)
    if n == 0:
        log(f"push するログがありません(dir={args.ckpt_dir})")
    return 0


def mode_restore(args, token: Optional[str]) -> int:
    """resume 用: HF 上の最新 checkpoint を CKPT_DIR へ戻す(無ければ何もしない)。"""
    from huggingface_hub import snapshot_download

    if getattr(args, "contents", "full") == "lora":
        # lora push では FSDP shard / optimizer 状態を上げていないため、
        # HF 側の checkpoint から verl の学習を再開することはできない。
        # 中途半端に戻して resume_mode=auto に拾わせると壊れるので何もしない。
        log("HF_PUSH_CONTENTS=lora のため HF からの resume は行いません"
            "(adapter のみで optimizer 状態が無いため)。fresh start します。")
        # ただしログと val 履歴は戻す。戻さないと今回の短い train.log を
        # 同名で push して HF 上の前回起動分を上書きしてしまう。
        restore_meta(args.ckpt_dir, args.repo_id, args.prefix, token)
        return 0

    api = _api(token)
    steps = remote_steps(api, args.repo_id, args.prefix)
    if not steps:
        # checkpoint が無くても前回起動の train.log は repo にあり得る
        # (最初の save より前に落ちた場合)。上書きで消さないよう戻す。
        log("HF 上に checkpoint 無し。fresh start します。")
        restore_meta(args.ckpt_dir, args.repo_id, args.prefix, token)
        return 0
    step = max(steps)
    local = os.path.join(args.ckpt_dir, f"global_step_{step}")
    if os.path.isdir(local):
        log(f"global_step_{step} はローカルに存在。restore 不要。")
        restore_meta(args.ckpt_dir, args.repo_id, args.prefix, token)
        return 0
    log(f"restore: {args.repo_id}/{args.prefix}/global_step_{step} -> {local}")
    tmp = os.path.join(args.ckpt_dir, "_restore_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        allow_patterns=[f"{args.prefix}/global_step_{step}/**",
                        f"{args.prefix}/{_METRICS_FILE}",
                        f"{args.prefix}/{_TRAIN_LOG_FILE}"],
        local_dir=tmp,
        token=token,
    )
    os.makedirs(args.ckpt_dir, exist_ok=True)
    shutil.move(os.path.join(tmp, args.prefix, f"global_step_{step}"), local)
    # val 履歴と train.log も戻す(保持ポリシー・merge 選定・ログ追記の連続性)。
    _move_meta(tmp, args.prefix, args.ckpt_dir, (_METRICS_FILE, _TRAIN_LOG_FILE))
    shutil.rmtree(tmp, ignore_errors=True)
    # verl の resume_mode=auto はこのファイルを見る。
    with open(os.path.join(args.ckpt_dir, _LATEST_FILE), "w", encoding="utf-8") as f:
        f.write(str(step))
    log(f"restore 完了: step={step}(trainer.resume_mode=auto で再開)")
    return 0


def mode_merged(args, token: Optional[str]) -> int:
    """最良(無ければ最新)checkpoint を merge し merged-final/ へ同期 push。

    設計書: 学習終了後、merge 版 checkpoint を HF へアップロード
            (どの checkpoint か・eval の数字を記録)。
    """
    ckpts = local_checkpoints(args.ckpt_dir)
    if not ckpts:
        log("ローカル checkpoint 無し。merged push をスキップ。")
        return 0
    metrics = load_val_metrics(args.ckpt_dir)
    steps = [s for s, _ in ckpts]

    # eval 値のある step から最良を選ぶ。無ければ最新。
    scored = [(metric_for_step(metrics, s), s) for s in steps]
    scored = [(v, s) for v, s in scored if v is not None]
    if scored:
        best_value, best_step = max(scored)
        reason = "best_eval"
    else:
        best_value, best_step = None, steps[-1]
        reason = "latest(eval 記録なし)"
    step_dir = dict(ckpts)[best_step]
    log(f"merge 対象: step={best_step} ({reason}, eval={best_value})")

    merged_dir = args.merged_dir or "/workspace/merged/final"
    here = os.path.dirname(os.path.abspath(__file__))
    merge_sh = os.path.join(here, "eval", "merge_checkpoint.sh")
    env = dict(os.environ, CKPT_STEP_DIR=step_dir, MERGED_DIR=merged_dir)
    rc = subprocess.run(["bash", merge_sh], env=env, check=False).returncode
    if rc != 0:
        print(f"[uploader] merge 失敗 (rc={rc})", file=sys.stderr, flush=True)
        return 1

    # どの CP か・eval の数字を記録(設計書要件)。
    record = {
        "merged_step": best_step,
        "selection": reason,
        "eval_value": best_value,
        "metric_key": next((m.get("key") for m in metrics), None),
        "val_history": metrics,
    }
    es = os.path.join(args.ckpt_dir, "early_stopped.json")
    if os.path.isfile(es):
        try:
            with open(es, "r", encoding="utf-8") as f:
                record["early_stopped"] = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    with open(os.path.join(merged_dir, "eval_record.json"), "w",
              encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    api = _api(token)
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True, private=True)
    log(f"merged push -> {args.repo_id}/{args.merged_prefix}")
    api.upload_folder(
        folder_path=merged_dir,
        repo_id=args.repo_id,
        path_in_repo=args.merged_prefix,
        delete_patterns="*",  # merged-final は常に 1 世代のみ(上書き)
        commit_message=f"merged final (step={best_step}, eval={best_value})",
    )
    log("merged push 完了")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["watch", "final", "restore", "merged", "logs"],
                    default="watch")
    ap.add_argument("--ckpt-dir", default=_env("CKPT_DIR"))
    ap.add_argument("--repo-id", default=_env("HF_REPO_ID"))
    ap.add_argument("--prefix", default=_env("HF_CKPT_PREFIX", "checkpoints"))
    # SAVE_FREQ の倍数でしか checkpoint は生まれないため、既定は SAVE_FREQ に追従する
    # (両者がズレると step % every == 0 が一度も成立せず、何も upload されない)。
    ap.add_argument("--every", type=int,
                    default=int(_env("UPLOAD_EVERY", _env("SAVE_FREQ", "10"))))
    ap.add_argument("--keep-best", type=int, default=int(_env("KEEP_BEST", "3")))
    ap.add_argument("--keep-latest", type=int, default=int(_env("KEEP_LATEST", "3")))
    ap.add_argument("--poll", type=int, default=int(_env("POLL_SECONDS", "60")))
    ap.add_argument("--merged-dir", default=_env("MERGED_DIR", "/workspace/merged/final"))
    ap.add_argument("--merged-prefix", default=_env("HF_MERGED_PREFIX", "merged-final"))
    # ログ push(DOK コンソールは先頭が切れるため repo 側に必ず残す)。
    # checkpoint 間隔(--every, step 単位)とは別に、秒単位で送る。
    ap.add_argument("--log-every", type=int,
                    default=int(_env("LOG_UPLOAD_SECONDS", "300")),
                    help="watch でログを push する間隔秒(既定 300)")
    ap.add_argument("--log-max-mb", type=int,
                    default=int(_env("HF_LOG_MAX_MB", "50")),
                    help="ログ 1 ファイルの上限 MB。超過分は中央を中略する")
    ap.add_argument("--reason", default="manual",
                    help="mode=logs のコミットメッセージに載せる理由")
    ap.add_argument("--contents", choices=["lora", "full"],
                    default=_env("HF_PUSH_CONTENTS", "lora"),
                    help="HF へ上げる中身。lora=adapter と tokenizer/config のみ"
                         "(既定)、full=FSDP shard と optimizer も含む全部。")
    args = ap.parse_args()

    if not args.ckpt_dir or not args.repo_id:
        print("[uploader] CKPT_DIR / HF_REPO_ID 未設定のためスキップします。",
              file=sys.stderr, flush=True)
        return 0

    token = _env("HF_TOKEN")
    if args.mode == "watch":
        return mode_watch(args, token)
    if args.mode == "final":
        return mode_final(args, token)
    if args.mode == "restore":
        return mode_restore(args, token)
    if args.mode == "logs":
        return mode_logs(args, token)
    return mode_merged(args, token)


if __name__ == "__main__":
    raise SystemExit(main())
