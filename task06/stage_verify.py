"""
stage_verify.py — 最終資料驗證與 Token 統計
"""
import argparse
import json
import os
from glob import glob
import pyarrow.parquet as pq

import pyarrow.compute as pc

def count_tokens_dir(d):
    files = glob(f"{d}/*.parquet")
    total = 0
    for f in files:
        try:
            t = pq.read_table(f, columns=["input_ids"])
            total += pc.sum(pc.list_value_length(t["input_ids"])).as_py() or 0
        except Exception as e:
            print(f"  ⚠ 讀取 {f} 失敗：{e}")
    return total, len(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_a_dir", required=True)
    parser.add_argument("--type_b_dir", required=True)
    parser.add_argument("--cot_dir",    required=True)
    parser.add_argument("--logit_dir",  required=True)
    parser.add_argument("--stats_out",  required=True)
    args = parser.parse_args()

    print("=" * 62)
    print("  📊 Task06 數據集驗證報告")
    print("=" * 62)

    a_tok, a_files = count_tokens_dir(args.type_a_dir)
    b_tok, b_files = count_tokens_dir(args.type_b_dir)
    c_tok, c_files = count_tokens_dir(args.cot_dir)

    shard_files = glob(f"{args.logit_dir}/*.safetensors")
    shard_size  = sum(os.path.getsize(f) for f in shard_files)

    print(f"\n  Data-Type A（蒸餾型 ≤2048 tok）:")
    print(f"    檔案數：{a_files}　Token：{a_tok/1e9:.2f}B / 40.0B　完成率：{100*a_tok/40e9:.1f}%")

    print(f"\n  Data-Type B（長文 32K~128K tok）:")
    print(f"    檔案數：{b_files}　Token：{b_tok/1e9:.2f}B / 10.0B　完成率：{100*b_tok/10e9:.1f}%")

    print(f"\n  CoT 思考鏈子集（含於 Type A）:")
    print(f"    檔案數：{c_files}　Token：{c_tok/1e9:.2f}B / 6.0B　完成率：{100*c_tok/6e9:.1f}%")

    print(f"\n  Teacher Logit Shards（Top-128）:")
    print(f"    Shard：{len(shard_files)} / 100　大小：{shard_size/1e9:.1f} GB")

    total = a_tok + b_tok
    print(f"\n{'='*62}")
    print(f"  🎉 總計：{total/1e9:.2f}B / 50.0B　完成率：{100*total/50e9:.1f}%")

    # 警告
    if a_tok < 30e9:
        print(f"\n  ⚠  Type A 不足（{a_tok/1e9:.1f}B < 30B），建議補充 CoT 或代碼數據")
    if b_tok < 8e9:
        print(f"\n  ⚠  Type B 不足（{b_tok/1e9:.1f}B < 8B），建議補充長文數據")
    if len(shard_files) < 100:
        print(f"\n  ⚠  Logit Shards 未完整（{len(shard_files)}/100），請繼續 Stage 3")
    if total >= 45e9 and len(shard_files) >= 90:
        print(f"\n  ✅ 資料充足，可開始訓練：python trainDNA16B_distill.py --phase 1")

    stats = {
        "type_A_tokens": a_tok, "type_A_files": a_files,
        "type_B_tokens": b_tok, "type_B_files": b_files,
        "cot_tokens":    c_tok, "cot_files":    c_files,
        "logit_shards":  len(shard_files),
        "logit_size_gb": round(shard_size/1e9, 2),
        "total_tokens":  total,
        "completion_pct": round(100*total/50e9, 1),
    }
    with open(args.stats_out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  統計已存：{args.stats_out}")
    print("=" * 62)

if __name__ == "__main__":
    main()
