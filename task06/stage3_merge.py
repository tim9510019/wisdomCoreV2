"""
stage3_merge.py — 合併所有 Type A Parquet 成單一文件
供 Teacher 推論使用，完成後永久保留可換 Teacher 重用
"""
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from glob import glob
from pathlib import Path

SCHEMA = pa.schema([
    pa.field("input_ids",      pa.list_(pa.int32())),
    pa.field("labels",         pa.list_(pa.int32())),
    pa.field("n_split_index",  pa.int32()),
    pa.field("data_type",      pa.string()),
    pa.field("logit_shard_id", pa.int32()),
    pa.field("logit_offset",   pa.int64()),
    pa.field("think_mask",     pa.list_(pa.bool_())),
    pa.field("category",       pa.string()),
    pa.field("difficulty",     pa.int8()),
])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_a_dir", required=True)
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()

    files = sorted(glob(f"{args.type_a_dir}/*.parquet"))
    print(f"合併 {len(files)} 個 Type A Parquet 文件...")

    writer = pq.ParquetWriter(args.output, SCHEMA, compression="snappy")
    total  = 0

    for f in files:
        try:
            t = pq.read_table(f)
            # 補齊缺少欄位
            for field in SCHEMA:
                if field.name not in t.schema.names:
                    n = len(t)
                    if field.type == pa.list_(pa.bool_()):
                        arr = pa.array([[False] * len(r.as_py()) for r in t["input_ids"]], type=field.type)
                    elif field.type == pa.list_(pa.int32()):
                        arr = pa.array([[] for _ in range(n)], type=field.type)
                    elif field.type == pa.int32():
                        arr = pa.array([-1] * n, type=pa.int32())
                    elif field.type == pa.int64():
                        arr = pa.array([-1] * n, type=pa.int64())
                    elif field.type == pa.int8():
                        arr = pa.array([3] * n, type=pa.int8())
                    else:
                        arr = pa.array(["unknown"] * n, type=pa.string())
                    t = t.append_column(field.name, arr)
            t = t.select([f.name for f in SCHEMA]).cast(SCHEMA)
            writer.write_table(t)
            total += len(t)
            print(f"  [{total:>10,}] {Path(f).name}", end="\r")
        except Exception as e:
            print(f"\n  ⚠ {f} 跳過：{e}")

    writer.close()
    print(f"\n✅ 合併完成：{total:,} 樣本 → {args.output}")

if __name__ == "__main__":
    main()
