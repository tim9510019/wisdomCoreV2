"""
pack_distill_dataset.py — 蒸餾數據集打包工具
================================================================
功能：
  1. 收集所有原始文本/代碼數據，分類為 Data-Type A / B
  2. 用 Qwen3-Coder 的 Tokenizer 做 Token 化
  3. 切割 n_split_index（A/B 區塊分割點）
  4. 統計 Token 總量，確保 40B 規劃達標
  5. 輸出最終 Parquet 數據集（Type A 附帶 logit_shard_id/offset，Type B 不需要）

使用方式：
  python pack_distill_dataset.py \\
      --config   dataset_config.json \\
      --output   /data/train_packed/ \\
      --tokenizer Qwen/Qwen3-Coder-30B \\
      --max_len_A 2048 \\
      --max_len_B 32768
"""

import os
import sys
import json
import math
import argparse
import random
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

# ============================================================
# Schema 定義
# ============================================================

SCHEMA_A = pa.schema([
    pa.field("input_ids",      pa.list_(pa.int32())),
    pa.field("labels",         pa.list_(pa.int32())),
    pa.field("n_split_index",  pa.int32()),
    pa.field("data_type",      pa.string()),
    pa.field("logit_shard_id", pa.int32()),      # 待 Teacher 推論後填入
    pa.field("logit_offset",   pa.int64()),       # 待 Teacher 推論後填入
    pa.field("think_mask",     pa.list_(pa.bool_())),
    pa.field("category",       pa.string()),
    pa.field("difficulty",     pa.int8()),
])

SCHEMA_B = pa.schema([
    pa.field("input_ids",     pa.list_(pa.int32())),
    pa.field("labels",        pa.list_(pa.int32())),
    pa.field("n_split_index", pa.int32()),
    pa.field("data_type",     pa.string()),
    pa.field("source",        pa.string()),
    pa.field("token_length",  pa.int32()),
])

# ============================================================
# 數據配置 (對應規劃書各子類別)
# ============================================================

DEFAULT_CONFIG = {
    "data_type_A": [
        # ─── 第一類：推理與思考鏈 ───
        {
            "name": "cot_qwen3_self",
            "hf_dataset": None,        # 自生成，由 prepare_cot_distill.py 產出
            "local_path": None,
            "category": "cot",
            "target_tokens": 6_000_000_000,
            "difficulty": 4,
        },
        {
            "name": "cot_competitive",
            "hf_dataset": "open-r1/codeforces",
            "subset": None,
            "split": "train",
            "text_column": "solution",
            "category": "cot",
            "target_tokens": 3_000_000_000,
            "difficulty": 5,
        },
        {
            "name": "cot_math_algo",
            "hf_dataset": "AI-MO/NuminaMath-CoT",
            "subset": None,
            "split": "train",
            "text_column": "solution",
            "category": "cot",
            "target_tokens": 2_000_000_000,
            "difficulty": 4,
        },
        {
            "name": "cot_debug",
            "hf_dataset": None,
            "local_path": None,
            "category": "cot",
            "target_tokens": 2_000_000_000,
            "difficulty": 3,
        },
        {
            "name": "cot_architecture",
            "hf_dataset": None,
            "local_path": None,
            "category": "cot",
            "target_tokens": 2_000_000_000,
            "difficulty": 4,
        },
        {
            "name": "cot_selfplay",
            "hf_dataset": None,
            "local_path": None,
            "category": "cot",
            "target_tokens": 1_000_000_000,
            "difficulty": 5,
        },
        # ─── 第二類：代碼 A 型 ───
        {
            "name": "code_stack_v2_short",
            "hf_dataset": "bigcode/the-stack-v2-train-smol-ids",
            "subset": None,
            "split": "train",
            "text_column": "content",
            "category": "code",
            "target_tokens": 3_000_000_000,
            "max_token_filter": 2048,
            "difficulty": 2,
        },
        {
            "name": "code_qwen_synth",
            "hf_dataset": "Qwen/Qwen2.5-Coder-instruct-generated",
            "subset": None,
            "split": "train",
            "text_column": "output",
            "category": "code",
            "target_tokens": 2_000_000_000,
            "difficulty": 3,
        },
        {
            "name": "code_review",
            "hf_dataset": "codeparrot/github-code-clean",
            "subset": "Python",
            "split": "train",
            "text_column": "code",
            "category": "code",
            "target_tokens": 2_000_000_000,
            "difficulty": 3,
        },
        # ─── 第三類：互動式指令 ───
        {
            "name": "instruction_qwen3",
            "hf_dataset": "Qwen/Qwen2.5-72B-Instruct-generated-magpie-en",
            "subset": None,
            "split": "train",
            "text_column": "conversations",
            "is_chat": True,
            "category": "instruction",
            "target_tokens": 2_000_000_000,
            "difficulty": 3,
        },
        {
            "name": "instruction_multiturn",
            "hf_dataset": "HuggingFaceH4/ultrachat_200k",
            "subset": None,
            "split": "train_sft",
            "text_column": "messages",
            "is_chat": True,
            "category": "instruction",
            "target_tokens": 1_500_000_000,
            "difficulty": 3,
        },
        {
            "name": "instruction_agent",
            "hf_dataset": "Agent-TÜLU-3-DPO-IF",
            "subset": None,
            "split": "train",
            "text_column": "messages",
            "is_chat": True,
            "category": "instruction",
            "target_tokens": 1_000_000_000,
            "difficulty": 4,
        },
        {
            "name": "instruction_tdd",
            "hf_dataset": None,
            "local_path": None,
            "category": "instruction",
            "target_tokens": 500_000_000,
            "difficulty": 3,
        },
    ],
    "data_type_B": [
        # ─── 第二類：代碼 B 型 (長文) ───
        {
            "name": "code_github_repo",
            "hf_dataset": "bigcode/the-stack-v2-train-full",
            "subset": "Python",
            "split": "train",
            "text_column": "content",
            "category": "code_long",
            "target_tokens": 3_000_000_000,
            "min_token_filter": 8192,
            "max_token_filter": 131072,
            "source": "github_repo",
        },
        {
            "name": "code_stack_v2_long",
            "hf_dataset": "bigcode/the-stack-v2-train-smol-ids",
            "subset": None,
            "split": "train",
            "text_column": "content",
            "category": "code_long",
            "target_tokens": 2_000_000_000,
            "min_token_filter": 4096,
            "source": "github_code",
        },
        # ─── 第四類：理論長文 ───
        {
            "name": "theory_rfc",
            "hf_dataset": "bigbio/pubmed_abstractive_summarization",
            "local_path": None,
            "category": "theory",
            "target_tokens": 1_500_000_000,
            "source": "rfc_docs",
        },
        {
            "name": "theory_textbook",
            "hf_dataset": "open-phi/textbooks",
            "subset": None,
            "split": "train",
            "text_column": "textbook",
            "category": "theory",
            "target_tokens": 1_000_000_000,
            "source": "textbook",
        },
        {
            "name": "theory_sysdesign",
            "hf_dataset": None,
            "local_path": None,
            "category": "theory",
            "target_tokens": 800_000_000,
            "source": "sys_design",
        },
        {
            "name": "theory_papers",
            "hf_dataset": "allenai/peS2o",
            "subset": "s2orc",
            "split": "train",
            "text_column": "text",
            "category": "theory",
            "target_tokens": 700_000_000,
            "source": "cs_papers",
        },
    ],
}

# ============================================================
# 核心處理函數
# ============================================================

def tokenize_and_chunk(
    texts: list[str],
    tokenizer,
    chunk_size: int,
    overlap: int = 0,
    n_split_ratio: float = 0.75,
):
    """
    將文本 Tokenize 後切成固定長度 Chunk。
    n_split_index = chunk_size * n_split_ratio（A 區塊 / B 區塊分割點）
    """
    chunks = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        
        for start in range(0, len(ids) - chunk_size + 1, chunk_size - overlap):
            chunk = ids[start : start + chunk_size]
            if len(chunk) < chunk_size // 2:
                continue  # 丟棄過短的 Chunk
            
            n_split = int(len(chunk) * n_split_ratio)
            chunks.append({
                "input_ids":     chunk,
                "labels":        [-100] * n_split + chunk[n_split:],
                "n_split_index": n_split,
            })
    
    return chunks


def format_chat_as_text(messages: list[dict], tokenizer) -> str:
    """
    將 Chat 格式轉換為 tokenizer 的 chat_template 文本。
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            pass
    # Fallback：手動拼接
    result = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        result.append(f"<|{role}|>\n{content}")
    return "\n".join(result)


def count_tokens(parquet_path: str) -> int:
    """統計 Parquet 文件中的總 Token 數量。"""
    ds = load_dataset("parquet", data_files=parquet_path, split="train")
    total = sum(len(row["input_ids"]) for row in ds)
    return total


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="蒸餾數據集打包工具")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON 配置文件（不指定則使用默認配置）")
    parser.add_argument("--output", type=str, required=True,
                        help="輸出目錄")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                        help="Tokenizer 路徑")
    parser.add_argument("--max_len_A", type=int, default=2048,
                        help="Data-Type A 最大 Token 長度")
    parser.add_argument("--max_len_B", type=int, default=32768,
                        help="Data-Type B 最大 Token 長度")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry_run", action="store_true",
                        help="試運行：只統計 Token 數，不寫文件")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入配置
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG
        print("⚠️  使用默認配置，建議生產環境提供 --config")

    # 載入 Tokenizer
    print(f"🔤 載入 Tokenizer：{args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ─── 處理 Data-Type A ───
    print("\n📦 處理 Data-Type A（蒸餾型，≤ 2048 Token）...")
    total_A_tokens = 0
    type_A_writers = {}

    for src_cfg in config["data_type_A"]:
        name     = src_cfg["name"]
        category = src_cfg.get("category", "code")
        target   = src_cfg.get("target_tokens", 0)
        is_chat  = src_cfg.get("is_chat", False)
        
        print(f"\n  ▶ 處理 [{name}] 目標: {target/1e9:.1f}B tokens")

        # 嘗試從 HF 載入
        hf_id = src_cfg.get("hf_dataset")
        if hf_id is None:
            print(f"    ⚠️  {name} 需要本地生成腳本（跳過）")
            continue

        try:
            ds_kwargs = {"split": src_cfg.get("split", "train")}
            if src_cfg.get("subset"):
                ds_kwargs["name"] = src_cfg["subset"]
            
            ds_raw = load_dataset(hf_id, **ds_kwargs, streaming=True, trust_remote_code=True)
            text_col = src_cfg.get("text_column", "text")

            rows_A = []
            token_count = 0

            for sample in tqdm(ds_raw, desc=f"  {name}", unit=" samples"):
                if token_count >= target:
                    break

                if is_chat:
                    msgs = sample.get(text_col, [])
                    if isinstance(msgs, str):
                        text = msgs
                    else:
                        text = format_chat_as_text(msgs, tokenizer)
                else:
                    text = sample.get(text_col, "")

                if not text or len(text) < 50:
                    continue

                chunks = tokenize_and_chunk([text], tokenizer, chunk_size=args.max_len_A)
                for chunk in chunks:
                    chunk["data_type"]      = "A"
                    chunk["category"]       = category
                    chunk["difficulty"]     = src_cfg.get("difficulty", 3)
                    chunk["logit_shard_id"] = -1   # 待 Teacher 推論後填入
                    chunk["logit_offset"]   = -1
                    chunk["think_mask"]     = [False] * len(chunk["input_ids"])
                    rows_A.append(chunk)
                    token_count += len(chunk["input_ids"])

                    if token_count >= target:
                        break

                # 批量寫出（每 10K 樣本）
                if len(rows_A) >= 10000 and not args.dry_run:
                    out_path = output_dir / f"type_A_{name}_{token_count//1e6:.0f}M.parquet"
                    _write_type_A(rows_A, out_path)
                    rows_A = []

            # 寫出剩餘
            if rows_A and not args.dry_run:
                out_path = output_dir / f"type_A_{name}_final.parquet"
                _write_type_A(rows_A, out_path)

            total_A_tokens += token_count
            print(f"    ✅ {name}: {token_count/1e9:.2f}B tokens")

        except Exception as e:
            print(f"    ❌ {name} 載入失敗：{e}")
            continue

    # ─── 處理 Data-Type B ───
    print("\n📦 處理 Data-Type B（長文自回歸型，32K~128K Token）...")
    total_B_tokens = 0

    for src_cfg in config["data_type_B"]:
        name   = src_cfg["name"]
        target = src_cfg.get("target_tokens", 0)
        source = src_cfg.get("source", "unknown")

        print(f"\n  ▶ 處理 [{name}] 目標: {target/1e9:.1f}B tokens")

        hf_id = src_cfg.get("hf_dataset")
        if hf_id is None:
            print(f"    ⚠️  {name} 需要本地生成腳本（跳過）")
            continue

        try:
            ds_kwargs = {"split": src_cfg.get("split", "train")}
            if src_cfg.get("subset"):
                ds_kwargs["name"] = src_cfg["subset"]

            ds_raw = load_dataset(hf_id, **ds_kwargs, streaming=True, trust_remote_code=True)
            text_col = src_cfg.get("text_column", "text")
            min_tok  = src_cfg.get("min_token_filter", 4096)
            max_tok  = src_cfg.get("max_token_filter", args.max_len_B)

            rows_B = []
            token_count = 0

            for sample in tqdm(ds_raw, desc=f"  {name}", unit=" samples"):
                if token_count >= target:
                    break

                text = sample.get(text_col, "")
                if not text or len(text) < 1000:
                    continue

                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) < min_tok:
                    continue

                # 截斷到 max_tok
                ids = ids[:max_tok]
                L   = len(ids)

                row = {
                    "input_ids":     ids,
                    "labels":        ids.copy(),  # 全部自回歸
                    "n_split_index": 0,
                    "data_type":     "B",
                    "source":        source,
                    "token_length":  L,
                }
                rows_B.append(row)
                token_count += L

                # 批量寫出
                if len(rows_B) >= 1000 and not args.dry_run:
                    out_path = output_dir / f"type_B_{name}_{token_count//1e6:.0f}M.parquet"
                    _write_type_B(rows_B, out_path)
                    rows_B = []

            if rows_B and not args.dry_run:
                out_path = output_dir / f"type_B_{name}_final.parquet"
                _write_type_B(rows_B, out_path)

            total_B_tokens += token_count
            print(f"    ✅ {name}: {token_count/1e9:.2f}B tokens")

        except Exception as e:
            print(f"    ❌ {name} 載入失敗：{e}")
            continue

    # ─── 統計報告 ───
    total_tokens = total_A_tokens + total_B_tokens
    print(f"\n{'='*60}")
    print(f"📊 數據集打包完成！")
    print(f"   Data-Type A: {total_A_tokens/1e9:.2f}B tokens")
    print(f"   Data-Type B: {total_B_tokens/1e9:.2f}B tokens")
    print(f"   總計:        {total_tokens/1e9:.2f}B / 40.0B tokens")
    print(f"   完成率:      {100*total_tokens/40e9:.1f}%")
    print(f"   輸出目錄:    {output_dir}")

    # 寫統計報告
    stats = {
        "total_tokens": total_tokens,
        "type_A_tokens": total_A_tokens,
        "type_B_tokens": total_B_tokens,
        "target_tokens": 40_000_000_000,
        "completion_pct": 100 * total_tokens / 40e9,
    }
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def _write_type_A(rows: list[dict], path: Path):
    """寫出 Data-Type A Parquet。"""
    table = pa.table({
        "input_ids":      [row["input_ids"] for row in rows],
        "labels":         [row["labels"] for row in rows],
        "n_split_index":  [row["n_split_index"] for row in rows],
        "data_type":      [row["data_type"] for row in rows],
        "logit_shard_id": [row["logit_shard_id"] for row in rows],
        "logit_offset":   [row["logit_offset"] for row in rows],
        "think_mask":     [row["think_mask"] for row in rows],
        "category":       [row["category"] for row in rows],
        "difficulty":     [row["difficulty"] for row in rows],
    }, schema=SCHEMA_A)
    pq.write_table(table, str(path), compression="snappy")


def _write_type_B(rows: list[dict], path: Path):
    """寫出 Data-Type B Parquet。"""
    table = pa.table({
        "input_ids":     [row["input_ids"] for row in rows],
        "labels":        [row["labels"] for row in rows],
        "n_split_index": [row["n_split_index"] for row in rows],
        "data_type":     [row["data_type"] for row in rows],
        "source":        [row["source"] for row in rows],
        "token_length":  [row["token_length"] for row in rows],
    }, schema=SCHEMA_B)
    pq.write_table(table, str(path), compression="snappy")


if __name__ == "__main__":
    main()
