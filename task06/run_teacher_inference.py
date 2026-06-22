"""
run_teacher_inference.py — Qwen3-Coder-30B Teacher 推論引擎
================================================================
功能：
  1. 批量讀取 Data-Type A 訓練樣本
  2. 用 Qwen3-Coder-30B 前向傳播，提取每個 Token 位置的 Top-128 Logits
  3. 自動識別 <think>...</think> 區域，生成 think_mask
  4. 將所有 Logits 切割為 100 份 Shard 存入 safetensors

使用方式：
  python run_teacher_inference.py \\
      --input_parquet /data/train_type_A.parquet \\
      --output_dir    /data/logit_shards/ \\
      --shard_count   100 \\
      --top_k         128 \\
      --batch_size    4   \\
      --max_length    2048
"""

import os
import sys
import argparse
import math
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.torch import save_file, load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ============================================================
# 配置區
# ============================================================
TEACHER_MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_TOP_K    = 128
DEFAULT_BATCH    = 4
DEFAULT_SHARDS   = 100
DEFAULT_MAX_LEN  = 2048
DISTILL_TEMP     = 1.0    # Teacher logits 不做溫度縮放（原始 logits 輸出）

# Qwen3 特殊 Token IDs（用於識別 think 區域）
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN   = "</think>"

# ============================================================
# 工具函數
# ============================================================

def get_think_mask(token_ids: list[int], tokenizer) -> list[bool]:
    """
    對 token_ids 建立思考段落遮罩。
    think_mask[i] = True 表示該 Token 屬於 <think>...</think> 區域。
    """
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    mask = [False] * len(token_ids)

    # 用 offset mapping 定位 think 區域
    in_think = False
    running_ids = []
    for i, tid in enumerate(token_ids):
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        running_ids.append(tok_str)
        decoded_so_far = "".join(running_ids)

        if THINK_START_TOKEN in decoded_so_far and not in_think:
            in_think = True
        if THINK_END_TOKEN in decoded_so_far and in_think:
            in_think = False

        mask[i] = in_think

    return mask


@torch.no_grad()
def extract_top_k_logits(
    model,
    input_ids: torch.Tensor,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = DISTILL_TEMP,
):
    """
    前向傳播並提取 Top-K Logits。
    
    Args:
        model: Teacher 模型
        input_ids: (B, L)
        top_k: 保留的頂部詞彙數
        temperature: 溫度（不影響存儲，後續在 KL 中使用）
    
    Returns:
        values:  float16  (B, L, top_k)
        indices: int32    (B, L, top_k)
    """
    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits  # (B, L, vocab_size) float32

    # Top-K 選取
    values, indices = torch.topk(logits, k=top_k, dim=-1)
    
    return values.to(torch.float16).cpu(), indices.to(torch.int32).cpu()


def process_batch(
    batch_input_ids: list[list[int]],
    model,
    tokenizer,
    top_k: int,
    device: str,
    max_length: int,
):
    """
    處理一個 Batch，返回：
      - logit_values:  (B, L, top_k) float16
      - logit_indices: (B, L, top_k) int32
      - think_masks:   (B, L)        bool
    """
    B = len(batch_input_ids)
    
    # 截斷 + Padding（右側補 pad）
    max_len = min(max(len(ids) for ids in batch_input_ids), max_length)
    pad_id = tokenizer.pad_token_id or 0

    padded_ids = []
    for ids in batch_input_ids:
        ids = ids[:max_length]
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [pad_id] * pad_len)

    input_tensor = torch.tensor(padded_ids, dtype=torch.long, device=device)

    # 前向傳播
    values, indices = extract_top_k_logits(model, input_tensor, top_k=top_k)

    # 生成 think_masks
    think_masks = []
    for ids in batch_input_ids:
        ids_truncated = ids[:max_length]
        mask = get_think_mask(ids_truncated, tokenizer)
        # 補齊到 max_len
        mask += [False] * (max_len - len(mask))
        think_masks.append(mask)

    think_masks_tensor = torch.tensor(think_masks, dtype=torch.bool)

    return values, indices, think_masks_tensor


def save_shard(
    shard_id: int,
    output_dir: Path,
    all_values: torch.Tensor,
    all_indices: torch.Tensor,
    all_masks: torch.Tensor,
    metadata_list: list[dict],
):
    """
    將當前 Shard 的 Logits 存入 safetensors，Metadata 存入 JSON。
    """
    shard_name = f"logit_shard_{shard_id:03d}"
    
    # 存 safetensors
    shard_path = output_dir / f"{shard_name}.safetensors"
    save_file({
        "logit_values":  all_values,
        "logit_indices": all_indices,
        "think_mask":    all_masks,
    }, str(shard_path))

    # 存 Metadata JSON
    meta_path = output_dir / f"{shard_name}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "shard_id": shard_id,
            "n_samples": len(metadata_list),
            "max_seq_len": all_values.shape[1],
            "top_k": all_values.shape[2],
            "samples": metadata_list,
        }, f, ensure_ascii=False, indent=2)

    print(f"  ✅ Shard {shard_id:03d} 已存儲: {shard_path} ({all_values.nbytes / 1e6:.1f} MB)")
    return str(shard_path), str(meta_path)


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-Coder-30B Teacher 推論引擎")
    parser.add_argument("--input_parquet", type=str, required=True,
                        help="輸入的 Data-Type A Parquet 文件路徑")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Logit Shard 輸出目錄")
    parser.add_argument("--teacher_model", type=str, default=TEACHER_MODEL_ID,
                        help="Teacher 模型 ID 或本地路徑")
    parser.add_argument("--shard_count", type=int, default=DEFAULT_SHARDS,
                        help="切割份數（默認 100）")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                        help="Top-K Logits 數量（默認 128）")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH,
                        help="推論 Batch Size")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LEN,
                        help="最大序列長度（默認 2048）")
    parser.add_argument("--shard_start", type=int, default=0,
                        help="開始的 Shard ID")
    parser.add_argument("--shard_end", type=int, default=100,
                        help="結束的 Shard ID")
    parser.add_argument("--device", type=str, default="auto",
                        help="設備（auto/cuda/cpu）")
    parser.add_argument("--skip_update_parquet", action="store_true",
                        help="是否跳過更新 Parquet 中的 logit 索引資訊")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 載入數據集 ──
    print(f"📦 載入數據集：{args.input_parquet}")
    ds = load_dataset("parquet", data_files=args.input_parquet, split="train")
    total_samples = len(ds)
    samples_per_shard = math.ceil(total_samples / args.shard_count)
    print(f"   總樣本數：{total_samples:,}，每份 Shard ≈ {samples_per_shard:,} 樣本")

    # ── 載入 Teacher 模型 ──
    print(f"\n🧠 載入 Teacher 模型：{args.teacher_model}")
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print(f"   ✅ Teacher 模型就緒（設備：{device}，參數量：{sum(p.numel() for p in model.parameters())/1e9:.1f}B）")

    # ── 批量推論 ──
    print(f"\n🔬 開始批量 Teacher 推論（Top-{args.top_k} Logits）...")
    t_start = time.time()

    for shard_id in range(args.shard_start, args.shard_end):
        shard_start = shard_id * samples_per_shard
        shard_end   = min(shard_start + samples_per_shard, total_samples)

        if shard_start >= total_samples:
            break

        shard_ds = ds.select(range(shard_start, shard_end))
        print(f"\n  📁 Shard {shard_id:03d} ({shard_start}~{shard_end-1}, {len(shard_ds)} 樣本)")

        shard_len = len(shard_ds)
        # 預先獲取序列長度，計算該 Shard 中的最大 sequence length
        lengths = [min(len(x), args.max_length) for x in shard_ds["input_ids"]]
        max_seq = max(lengths)

        # 預先分配 CPU 記憶體張量，防止列表存儲和拼接時的記憶體高峰與記憶體洩漏
        all_values = torch.zeros((shard_len, max_seq, args.top_k), dtype=torch.float16)
        all_indices = torch.zeros((shard_len, max_seq, args.top_k), dtype=torch.int32)
        all_masks = torch.zeros((shard_len, max_seq), dtype=torch.bool)
        shard_meta = []

        for batch_start in range(0, len(shard_ds), args.batch_size):
            batch_end  = min(batch_start + args.batch_size, len(shard_ds))
            batch_rows = shard_ds.select(range(batch_start, batch_end))

            batch_input_ids = [row["input_ids"] for row in batch_rows]

            values, indices, think_masks = process_batch(
                batch_input_ids, model, tokenizer, args.top_k, device, args.max_length
            )

            # 逐樣本拷貝至預分配 Tensor 中
            for i in range(len(batch_input_ids)):
                L = min(len(batch_input_ids[i]), args.max_length)
                offset = batch_start + i
                all_values[offset, :L, :] = values[i, :L, :]
                all_indices[offset, :L, :] = indices[i, :L, :]
                all_masks[offset, :L] = think_masks[i, :L]
                shard_meta.append({
                    "global_idx":   shard_start + batch_start + i,
                    "shard_offset": len(shard_meta),
                    "seq_len":      L,
                    "n_split_index": batch_rows[i].get("n_split_index", 0) if hasattr(batch_rows[i], "get") else 0,
                })

            # 適當的休眠以防止 GPU 過熱
            time.sleep(1.0)

            # 進度顯示
            if (batch_start // args.batch_size) % 50 == 0:
                elapsed = time.time() - t_start
                done = shard_start + batch_end
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total_samples - done) / speed if speed > 0 else 0
                print(f"    進度: {done}/{total_samples} ({100*done/total_samples:.1f}%) | "
                      f"速度: {speed:.0f} 樣/s | ETA: {eta/3600:.1f}h")

        # 存 Shard
        save_shard(shard_id, output_dir, all_values, all_indices, all_masks, shard_meta)

    total_time = time.time() - t_start
    print(f"\n🎉 全部 {args.shard_count} 份 Logit Shard 生成完畢！")
    print(f"   總耗時：{total_time/3600:.2f} 小時")
    print(f"   輸出目錄：{output_dir}")

    # ── 更新 input parquet 中的 logit_shard_id 和 logit_offset ──
    if not args.skip_update_parquet:
        print(f"\n📝 正在更新 {args.input_parquet} 中的 logit_shard_id 與 logit_offset 欄位...")
        try:
            table = pq.read_table(args.input_parquet)
            logit_shard_ids = []
            logit_offsets = []
            for i in range(total_samples):
                shard_id = i // samples_per_shard
                offset = i % samples_per_shard
                logit_shard_ids.append(shard_id)
                logit_offsets.append(offset)

            shard_id_idx = table.schema.get_field_index("logit_shard_id")
            offset_idx = table.schema.get_field_index("logit_offset")

            if shard_id_idx != -1:
                table = table.set_column(shard_id_idx, "logit_shard_id", pa.array(logit_shard_ids, type=pa.int32()))
            else:
                table = table.append_column("logit_shard_id", pa.array(logit_shard_ids, type=pa.int32()))

            if offset_idx != -1:
                table = table.set_column(offset_idx, "logit_offset", pa.array(logit_offsets, type=pa.int64()))
            else:
                table = table.append_column("logit_offset", pa.array(logit_offsets, type=pa.int64()))

            pq.write_table(table, args.input_parquet, compression="snappy")
            print(f"   ✅ 更新成功！")
        except Exception as e:
            print(f"   ❌ 更新失敗：{e}")



if __name__ == "__main__":
    main()
