# -*- coding: utf-8 -*-
"""
prepare_bfcl_data.py

bfcl_dataset_and_interaction.build_verl_dataset で BFCL v3 の
question + possible_answer を結合し、verl が読む parquet を書き出す。

使い方:
  python prepare_bfcl_data.py \
      --bfcl-data-dir $(python -c "import bfcl_eval,os;print(os.path.join(os.path.dirname(bfcl_eval.__file__),'data'))") \
      --out-dir ./data \
      --categories multi_turn_base,multi_turn_miss_func,multi_turn_miss_param \
      --val-ratio 0.1
"""
import argparse, os, random
import pandas as pd
import bfcl_dataset_and_interaction as di


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfcl-data-dir", required=True)
    ap.add_argument("--out-dir", default="./data")
    ap.add_argument("--categories",
                    default="multi_turn_base,multi_turn_miss_func,multi_turn_miss_param")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    all_rows = []
    for cat in args.categories.split(","):
        cat = cat.strip()
        rows = di.build_verl_dataset(cat, args.bfcl_data_dir)
        print(f"[{cat}] {len(rows)} エントリ")
        all_rows.extend(rows)

    rng.shuffle(all_rows)
    n_val = int(len(all_rows) * args.val_ratio)
    val, train = all_rows[:n_val], all_rows[n_val:]

    # 主対象 miss_func/miss_param を train に厚く残すため、val は base 中心にしたい場合は
    # ここで層化するが、まずは単純 split。
    train_path = os.path.join(args.out_dir, "bfcl_mt_train.parquet")
    val_path = os.path.join(args.out_dir, "bfcl_mt_val.parquet")
    pd.DataFrame(train).to_parquet(train_path)
    pd.DataFrame(val).to_parquet(val_path)
    print(f"train {len(train)} -> {train_path}")
    print(f"val   {len(val)} -> {val_path}")


if __name__ == "__main__":
    main()
