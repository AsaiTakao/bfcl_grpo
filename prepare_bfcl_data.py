# -*- coding: utf-8 -*-
"""
prepare_bfcl_data.py

bfcl_dataset_and_interaction.build_verl_dataset で BFCL v3 の
question + possible_answer を結合し、verl が読む parquet を書き出す。

マルチターン(主目的)とシングルターン(既存能力の維持・退行検知)を混ぜる:
  - train: シングルターンを --st-train-ratio の割合だけ混ぜる。
           multi_turn 特化 RL による破滅的忘却をリプレイで抑える。
  - val  : 両方を必ず含める(層化)。train_monitor がこの 2 系統の val 指標を
           combined score にまとめ、シングルターンが退行したら EarlyStopping
           側で減点されるようにする。

--holdout-classes を指定すると multi_turn の val を「API クラス単位」で切り出す。
BFCL multi_turn の舞台は 8 クラスしかないため、ランダム分割の val は train と同じ
API を使う = 未知 API への汎化を測れない。最終評価(τ-bench retail/airline)は
完全に別ドメインなので、OOD 転移を見るならクラス単位の holdout が要る。

使い方:
  python prepare_bfcl_data.py \
      --bfcl-data-dir $(python -c "import bfcl_eval,os;print(os.path.join(os.path.dirname(bfcl_eval.__file__),'data'))") \
      --out-dir ./data \
      --categories multi_turn_base,multi_turn_miss_func,multi_turn_miss_param \
      --st-categories simple,multiple,parallel,live_simple \
      --holdout-classes TravelAPI,VehicleControlAPI \
      --val-ratio 0.1
"""
import argparse, os, random
import bfcl_dataset_and_interaction as di

# pandas は parquet 書き出しにしか使わない。分割ロジックを pandas 無しの環境から
# ユニットテストできるよう、import は main() 内に遅延させる。


# BFCL のバージョン間でのカテゴリ改称。v4 で simple は言語別に分割された。
# 旧名のまま指定されたとき黙って取り逃さないよう、代替名へ読み替える。
# java / javascript は AST の値表現が Python 系と異なり ast_match が誤判定する
# ため、simple の読み替え先は simple_python のみとする。
_CATEGORY_ALIASES = {
    "simple": ["simple_python"],
}


def _load(categories: str, bfcl_data_dir: str, prefix: str) -> list:
    """カンマ区切りのカテゴリ名から行を集める。読めないカテゴリは警告して飛ばす。"""
    rows = []
    missing = []
    available = set(di.available_categories(bfcl_data_dir, prefix))
    for cat in categories.split(","):
        cat = cat.strip()
        if not cat:
            continue

        # 旧名が使えない場合だけ、この BFCL バージョンでの相当カテゴリへ読み替える。
        if cat not in available:
            alias = next((a for a in _CATEGORY_ALIASES.get(cat, [])
                          if a in available), None)
            if alias:
                print(f"[{cat}] は {prefix} に存在しないため {alias} に読み替えます")
                cat = alias

        try:
            got = di.build_verl_dataset(cat, bfcl_data_dir, prefix)
        except (OSError, KeyError, ValueError) as e:
            # バージョンによって存在しないカテゴリがあるため、欠損は致命傷にしない。
            print(f"[{cat}] 読み込み失敗のためスキップ: {e}")
            missing.append(cat)
            continue
        print(f"[{cat}] {len(got)} エントリ")
        rows.extend(got)
    if missing:
        # カテゴリ名は BFCL のバージョンで変わる(v4 では simple が
        # simple_python / simple_java / simple_javascript に分割された)。
        avail = di.available_categories(bfcl_data_dir, prefix)
        print(f"[hint] {prefix} で実際に読めるカテゴリ: {avail}")
    return rows


def _split(rows: list, val_ratio: float, val_size: int) -> tuple:
    """(val, train) に分割する。val_size > 0 ならそちらを優先。"""
    n_val = val_size if val_size > 0 else int(len(rows) * val_ratio)
    n_val = max(0, min(n_val, len(rows)))
    return rows[:n_val], rows[n_val:]


def _row_classes(row: dict) -> set:
    """行が使う BFCL バックエンドクラス名の集合。シングルターンは空。"""
    ck = (row.get("extra_info", {})
             .get("tools_kwargs", {})
             .get(di.TOOL_NAME, {})
             .get("create_kwargs", {}))
    # involved_classes は parquet 対策で JSON 文字列として持っている。
    return set(di.decode_field(ck.get("involved_classes"), []))


def _split_by_holdout(rows: list, holdout: set) -> tuple:
    """API クラス単位で (val, train) に分割する。

    なぜランダム分割ではなくこれが要るか:
      BFCL multi_turn の舞台は 8 個のバックエンドクラスしかない。ランダム分割だと
      val は train と同じクラスを使うため「同じ API の未知シナリオ」しか測れず、
      未知 API への汎化(= 最終評価の τ-bench retail/airline は完全に別ドメイン)は
      検知できない。クラスごと丸ごと val へ寄せることで OOD 転移を測る。

    リークを避けるため、holdout クラスを 1 つでも含む行は val 側に送る
    (multi_turn は 1 エントリが複数クラスにまたがることがある)。
    """
    val = [r for r in rows if _row_classes(r) & holdout]
    train = [r for r in rows if not (_row_classes(r) & holdout)]
    return val, train


# プロンプト長の見積り係数。関数一覧(英語 schema + 日本語指示)は
# 1 トークン ≒ 3 文字程度。厳密さは要らず、桁が合っていればよい。
_CHARS_PER_TOKEN = 3.0
#: run_bfcl_grpo_h100x8.sh の MAX_PROMPT_LEN 既定値。
_MAX_PROMPT_TOKENS_HINT = 8192


def _report_prompt_size(rows: list, limit_tokens: int = _MAX_PROMPT_TOKENS_HINT) -> None:
    """プロンプト長を報告する。

    なぜ: system prompt に involved_classes の全関数シグネチャを載せるように
    したため、プロンプトが数千トークンになる。data.max_prompt_length を超えた行は
    filter_overlong_prompts=True によって学習時に静かに捨てられるので、
    データ準備の時点で気付けるようにしておく。
    """
    if not rows:
        return
    chars = [sum(len(m.get("content", "")) for m in r["prompt"]) for r in rows]
    est = [c / _CHARS_PER_TOKEN for c in chars]
    over = sum(1 for t in est if t > limit_tokens)
    print(f"prompt 長: 平均 {sum(chars)//len(chars)} 文字 / 最大 {max(chars)} 文字 "
          f"(≒ {max(est):.0f} tok)")
    if over:
        print(f"[warn] max_prompt_length={limit_tokens} を超えそうな行が {over} 件あります。"
              "filter_overlong_prompts=True だと学習時に静かに捨てられます。"
              "MAX_PROMPT_LEN を上げてください。")


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--bfcl-data-dir", required=True)
    ap.add_argument("--out-dir", default="./data")
    ap.add_argument("--categories",
                    default="multi_turn_base,multi_turn_miss_func,multi_turn_miss_param",
                    help="マルチターン(主目的)のカテゴリ")
    ap.add_argument("--st-categories",
                    default="simple_python,multiple,parallel,live_simple",
                    help="シングルターン(既存能力の維持)のカテゴリ。空文字で無効化。"
                         "simple_java / simple_javascript は AST の値表現が Python 系と"
                         "異なり ast_match が誤判定するため既定から外している")
    ap.add_argument("--version-prefix", default="",
                    help="データファイルの接頭辞(BFCL_v3 / BFCL_v4)。"
                         "既定は空 = データディレクトリから自動検出")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--holdout-classes", default="",
                    help="val 専用にする BFCL バックエンドクラス名(カンマ区切り)。"
                         "例: TravelAPI,VehicleControlAPI。指定すると multi_turn の "
                         "val はこのクラスを含む行のみになり、未知 API への汎化"
                         "(OOD 転移)を測れる。未指定ならランダム分割")
    ap.add_argument("--st-train-ratio", type=float, default=0.3,
                    help="train に占めるシングルターンの割合。0 で混ぜない")
    ap.add_argument("--val-st-size", type=int, default=0,
                    help="シングルターン val の件数を明示指定(0 = --val-ratio に従う)。"
                         "シングルターンは 1 ターンで終わるため eval が安価で、"
                         "退行検知のサンプル数を稼ぐならここを大きくする")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    prefix = args.version_prefix or di.detect_version_prefix(args.bfcl_data_dir)
    print(f"BFCL バージョン接頭辞: {prefix}"
          f"{'(自動検出)' if not args.version_prefix else ''}")

    mt_rows = _load(args.categories, args.bfcl_data_dir, prefix)
    st_rows = (_load(args.st_categories, args.bfcl_data_dir, prefix)
               if args.st_categories else [])
    if not mt_rows:
        raise SystemExit(
            "マルチターンの行が 0 件です。--bfcl-data-dir と --categories を確認してください。\n"
            f"  検出した接頭辞: {prefix}\n"
            f"  読めるカテゴリ: {di.available_categories(args.bfcl_data_dir, prefix)}"
        )

    rng.shuffle(mt_rows)
    rng.shuffle(st_rows)

    # val はマルチターン・シングルターンの両方を必ず含める(層化)。
    holdout = {c.strip() for c in args.holdout_classes.split(",") if c.strip()}
    if holdout:
        available = set().union(*(_row_classes(r) for r in mt_rows)) or set()
        unknown = holdout - available
        if unknown:
            # typo を黙って通すと val が空(または OOD でない)まま学習が走る。
            raise SystemExit(
                f"--holdout-classes に未知のクラス: {sorted(unknown)} / "
                f"データ中に存在するのは {sorted(available)}"
            )
        mt_val, mt_train = _split_by_holdout(mt_rows, holdout)
        print(f"holdout クラス {sorted(holdout)} を val 専用にしました "
              f"(val {len(mt_val)} / train {len(mt_train)})")
        if not mt_train:
            raise SystemExit("holdout の指定で train が空になりました。対象クラスを減らしてください。")
        if not mt_val:
            raise SystemExit("holdout の指定で val が空になりました。")
        frac = len(mt_val) / len(mt_rows)
        if frac > 0.5:
            print(f"[warn] multi_turn の {frac:.0%} が val 側に行きました。"
                  "学習データが薄くなりすぎていないか確認してください。")
    else:
        print("[warn] --holdout-classes 未指定: val は train と同じ API クラスを使うため、"
              "未知 API への汎化は測れません(同一 API 内の未知シナリオのみ)。")
        mt_val, mt_train = _split(mt_rows, args.val_ratio, 0)

    st_val, st_train_pool = _split(st_rows, args.val_ratio, args.val_st_size)

    # train のシングルターン量は「割合」で決める。混ぜすぎるとマルチターン
    # (主目的)の勾配が薄まるため、既定は 30% に抑える。
    if args.st_train_ratio > 0 and st_train_pool:
        # n_st / (n_mt + n_st) = ratio  ->  n_st = n_mt * ratio / (1 - ratio)
        ratio = min(args.st_train_ratio, 0.9)
        n_st = int(len(mt_train) * ratio / (1.0 - ratio))
        st_train = st_train_pool[:n_st]
    else:
        st_train = []

    train = mt_train + st_train
    val = mt_val + st_val
    rng.shuffle(train)
    rng.shuffle(val)

    _report_prompt_size(train + val)

    train_path = os.path.join(args.out_dir, "bfcl_mt_train.parquet")
    val_path = os.path.join(args.out_dir, "bfcl_mt_val.parquet")
    pd.DataFrame(train).to_parquet(train_path)
    pd.DataFrame(val).to_parquet(val_path)
    print(f"train {len(train)} (multi_turn {len(mt_train)} + single_turn {len(st_train)}) "
          f"-> {train_path}")
    print(f"val   {len(val)} (multi_turn {len(mt_val)} + single_turn {len(st_val)}) "
          f"-> {val_path}")
    if not st_val:
        print("[warn] シングルターン val が 0 件です。"
              "既存能力の退行は検知できません(--st-categories を確認)。")


if __name__ == "__main__":
    main()
