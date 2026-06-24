"""
prepare_bcot_distill.py — Type B-CoT 數據生成引擎
================================================================
功能：
  1. 從 HuggingFace 載入中等長度代碼/文本（512~4096 tokens）
  2. 用 Teacher（Qwen3-Coder-30B, enable_thinking=True）生成真實 <think> 摘要
  3. 組合：[<think>真實摘要</think>][原始代碼]
  4. labels：THINK 段 mask=-100，原始代碼段正常自回歸
  5. 輸出 SCHEMA_B_COT 格式 Parquet（含 think_mask、n_split_index）

與 Type A CoT 的差異：
  - Type A CoT：學「如何思考並解題」（think + answer 都學）
  - Type B-CoT：學「看過思考摘要後，如何生成對應的代碼」
                （think 段 mask=-100，只學 code 輸出部分）
  → 強化 THINK ↔ 代碼 的語義對齊，避免 THINK 內容與代碼脫節

使用方式：
  python prepare_bcot_distill.py \\
      --output_dir  ~/task06_data/type_B_cot \\
      --target_tokens 1750000000 \\
      --teacher_model Qwen/Qwen3-Coder-30B-A3B-Instruct \\
      --max_body_tokens 4096 \\
      --min_body_tokens 512 \\
      --max_think_tokens 768 \\
      --temperature 0.6
"""

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import os
import sys
import json
import random
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

TEACHER_MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
RANDOM_SEED = 2026
random.seed(RANDOM_SEED)

# ============================================================
# Schema
# ============================================================

SCHEMA_B_COT = pa.schema([
    pa.field("input_ids",      pa.list_(pa.int32())),
    pa.field("labels",         pa.list_(pa.int32())),
    pa.field("n_split_index",  pa.int32()),
    pa.field("data_type",      pa.string()),
    pa.field("logit_shard_id", pa.int32()),
    pa.field("logit_offset",   pa.int64()),
    pa.field("think_mask",     pa.list_(pa.bool_())),
    pa.field("source",         pa.string()),
    pa.field("token_length",   pa.int32()),
])

# ============================================================
# 分析 Prompt 模板庫（依來源類型選用）
# ============================================================

PROMPT_TEMPLATES = {
    "code": (
        "請閱讀以下程式碼，在 <think> 標籤中進行深度分析：\n"
        "1. 核心功能與用途\n"
        "2. 關鍵演算法與資料結構的選擇理由\n"
        "3. 潛在問題、邊界條件或可改進之處\n\n"
        "```\n{body}\n```\n\n"
        "請只輸出 <think> 標籤內的分析，不需要給出修改後的代碼。"
    ),
    "theory": (
        "請閱讀以下技術文本，在 <think> 標籤中進行深度摘要：\n"
        "1. 核心概念與主要論點\n"
        "2. 關鍵術語與定義\n"
        "3. 與其他概念的關聯或實際應用\n\n"
        "{body}\n\n"
        "請只輸出 <think> 標籤內的分析，不需要重複原文。"
    ),
    "math": (
        "請閱讀以下數學問題，在 <think> 標籤中分析解題思路：\n"
        "1. 問題的核心考點\n"
        "2. 可能的解法方向\n"
        "3. 關鍵步驟與注意事項\n\n"
        "問題：{body}\n\n"
        "請只輸出 <think> 標籤內的分析。"
    ),
}

# ============================================================
# 數據來源配置
# ============================================================

DATA_SOURCES = [
    {
        "id": "codeparrot/github-code-clean",
        "format": "parquet",
        "data_files": "hf://datasets/codeparrot/github-code-clean/data/train-*.parquet",
        "split": "train",
        "text_col": "code",
        "lang_col": "language",
        "langs": {"python", "javascript", "typescript", "go", "rust", "c++"},
        "prompt_type": "code",
        "source": "github_code",
        "weight": 0.60,
    },
    {
        "id": "open-phi/textbooks",
        "split": "train",
        "text_col": "textbook",
        "prompt_type": "theory",
        "source": "textbook",
        "weight": 0.25,
    },
    {
        "id": "AI-MO/NuminaMath-CoT",
        "split": "train",
        "text_col": "problem",
        "prompt_type": "math",
        "source": "math_problem",
        "weight": 0.15,
    },
]

# ============================================================
# 工具函數
# ============================================================

def extract_think_block(generated_text: str) -> str | None:
    """
    從 Teacher 生成文本中提取 <think>...</think> 區塊。
    若找不到完整標籤則回傳 None。
    """
    s = generated_text.find("<think>")
    e = generated_text.find("</think>")
    if s == -1 or e == -1 or e <= s:
        return None
    # 包含結束標籤與換行
    return generated_text[s : e + len("</think>")] + "\n"


def load_samples(src_cfg: dict, tokenizer, min_tok: int, max_tok: int,
                 needed: int) -> list[dict]:
    """
    從 HF 資料集載入文本樣本，過濾至 [min_tok, max_tok] token 範圍。
    回傳最多 needed 個樣本。
    """
    samples = []
    text_col = src_cfg["text_col"]

    try:
        if src_cfg.get("format") == "parquet":
            ds = load_dataset(
                "parquet",
                data_files=src_cfg["data_files"],
                split=src_cfg["split"],
                streaming=True,
            )
        else:
            ds = load_dataset(src_cfg["id"], split=src_cfg["split"], streaming=True)

        lang_filter = src_cfg.get("langs")
        lang_col    = src_cfg.get("lang_col")

        for sample in tqdm(ds, desc=f"  載入 {src_cfg['id'][:40]}", unit=" docs"):
            if len(samples) >= needed:
                break

            # 語言過濾（代碼類）
            if lang_filter and lang_col:
                if str(sample.get(lang_col, "")).lower() not in lang_filter:
                    continue

            text = str(sample.get(text_col, "") or "")
            if len(text) < 200:
                continue

            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < min_tok or len(ids) > max_tok:
                continue

            samples.append({
                "body": text,
                "body_ids": ids[:max_tok],
                "source": src_cfg["source"],
                "prompt_type": src_cfg["prompt_type"],
            })

    except Exception as ex:
        print(f"  ⚠️  {src_cfg['id']} 載入失敗：{ex}")

    return samples


def write_batch(rows: list[dict], path: Path):
    """寫出 B-CoT Parquet。"""
    if not rows:
        return
    table = pa.table({
        "input_ids":      [r["input_ids"]      for r in rows],
        "labels":         [r["labels"]          for r in rows],
        "n_split_index":  [r["n_split_index"]   for r in rows],
        "data_type":      [r["data_type"]        for r in rows],
        "logit_shard_id": [r["logit_shard_id"]  for r in rows],
        "logit_offset":   [r["logit_offset"]     for r in rows],
        "think_mask":     [r["think_mask"]       for r in rows],
        "source":         [r["source"]           for r in rows],
        "token_length":   [r["token_length"]     for r in rows],
    }, schema=SCHEMA_B_COT)
    pq.write_table(table, str(path), compression="snappy")
    print(f"  💾 寫出 {len(rows)} 筆 → {path.name}")


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Type B-CoT 數據生成引擎")
    parser.add_argument("--teacher_model",   type=str,   default=TEACHER_MODEL_ID)
    parser.add_argument("--output_dir",      type=str,   required=True)
    parser.add_argument("--target_tokens",   type=int,   default=1_750_000_000,
                        help="目標 body token 總量（不含 think 段）")
    parser.add_argument("--min_body_tokens", type=int,   default=512,
                        help="篩選代碼/文本的最小 token 長度")
    parser.add_argument("--max_body_tokens", type=int,   default=4096,
                        help="篩選代碼/文本的最大 token 長度")
    parser.add_argument("--max_think_tokens",type=int,   default=768,
                        help="Teacher 生成 THINK 摘要的最大 token 數")
    parser.add_argument("--temperature",     type=float, default=0.6)
    parser.add_argument("--vllm_batch_size", type=int,   default=256,
                        help="vLLM 每次並發推論批次大小")
    parser.add_argument("--seed",            type=int,   default=RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 載入 vLLM + Tokenizer ──
    print(f"\n🔤 載入 Teacher 模型 (vLLM)：{args.teacher_model}")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.teacher_model,
        tensor_parallel_size=2,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=args.max_body_tokens + args.max_think_tokens + 512,  # prompt + think + margin
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=args.max_think_tokens,
        temperature=args.temperature,
        top_p=0.95,
        repetition_penalty=1.05,
    )

    # ── 按 weight 計算各來源需要的樣本數（先多撈 2x，過濾後才能達標）──
    # 假設平均 body token = 2000，每個樣本
    est_samples_needed = int(args.target_tokens / 2000 * 2)  # 2x 緩衝
    print(f"\n📦 估計需要收集 {est_samples_needed:,} 個樣本...")

    all_samples = []
    for src in DATA_SOURCES:
        n = int(est_samples_needed * src["weight"])
        print(f"\n  ▶ 載入 [{src['id']}]（目標 {n:,} 個樣本）")
        samples = load_samples(
            src, tokenizer,
            min_tok=args.min_body_tokens,
            max_tok=args.max_body_tokens,
            needed=n,
        )
        all_samples.extend(samples)
        print(f"    ✅ 收集到 {len(samples):,} 個樣本")

    random.shuffle(all_samples)
    print(f"\n📊 總樣本池：{len(all_samples):,} 個，開始 Teacher 推論...")

    # ── 預先 tokenize THINK 起始標籤（用於組合 input_ids）──
    think_open_ids  = tokenizer.encode("<think>",  add_special_tokens=False)
    think_close_ids = tokenizer.encode("</think>\n", add_special_tokens=False)

    # ── 主推論迴圈 ──
    total_body_tokens = 0
    total_samples_ok  = 0
    batch_id          = 0
    rows_buf          = []
    sample_idx        = 0

    while total_body_tokens < args.target_tokens and sample_idx < len(all_samples):
        # 取一個 batch
        batch_items = all_samples[sample_idx : sample_idx + args.vllm_batch_size]
        sample_idx += len(batch_items)
        if not batch_items:
            break

        # 建構分析 Prompt
        formatted_prompts = []
        for item in batch_items:
            template = PROMPT_TEMPLATES.get(item["prompt_type"], PROMPT_TEMPLATES["code"])
            # 截斷 body 文本到合理長度（避免 prompt 過長）
            body_preview = item["body"][:3000]
            user_content = template.format(body=body_preview)
            messages = [{"role": "user", "content": user_content}]
            try:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except Exception:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            formatted_prompts.append(formatted)

        # Teacher 批次生成
        try:
            outputs = llm.generate(formatted_prompts, sampling_params)
        except Exception as ex:
            print(f"  ⚠️  vLLM 批次生成失敗：{ex}")
            continue

        # 處理每個輸出
        for item, out in zip(batch_items, outputs):
            generated = out.outputs[0].text

            # 提取 <think>...</think> 區塊
            think_text = extract_think_block(generated)
            if think_text is None:
                continue  # 沒有完整 think 標籤，跳過此樣本

            # 對 think 文本 tokenize
            think_ids = tokenizer.encode(think_text, add_special_tokens=False)
            if len(think_ids) < 20:
                continue  # think 太短，品質不佳

            body_ids  = item["body_ids"]
            full_ids  = think_ids + body_ids
            think_len = len(think_ids)
            body_len  = len(body_ids)
            L         = len(full_ids)

            # labels：THINK 段全部 mask=-100，body 段正常自回歸
            labels = [-100] * think_len + list(body_ids)

            # think_mask：THINK 段 True（KL 加權），body 段 False
            think_mask = [True] * think_len + [False] * body_len

            rows_buf.append({
                "input_ids":      list(full_ids),
                "labels":         labels,
                "n_split_index":  think_len,
                "data_type":      "B_COT",
                "logit_shard_id": -1,
                "logit_offset":   -1,
                "think_mask":     think_mask,
                "source":         item["source"],
                "token_length":   L,
            })
            total_body_tokens += body_len
            total_samples_ok  += 1

        # 批量寫出（每 2000 筆）
        if len(rows_buf) >= 2000:
            out_path = output_dir / f"bcot_batch_{batch_id:05d}.parquet"
            write_batch(rows_buf, out_path)
            rows_buf = []
            batch_id += 1
            print(
                f"  📈 進度：{total_body_tokens/1e6:.1f}M / "
                f"{args.target_tokens/1e6:.1f}M body tokens  "
                f"({total_samples_ok:,} 個有效樣本 | "
                f"成功率 {100*total_samples_ok/max(sample_idx,1):.1f}%)"
            )

    # 寫出剩餘
    if rows_buf:
        out_path = output_dir / f"bcot_batch_{batch_id:05d}_final.parquet"
        write_batch(rows_buf, out_path)

    # ── 統計 ──
    est_gb = (total_body_tokens * 4.5) / (1024 ** 3)
    print(f"\n{'='*60}")
    print(f"✅ Type B-CoT 數據生成完畢！")
    print(f"   有效樣本：  {total_samples_ok:,} 個")
    print(f"   Body tokens：{total_body_tokens/1e9:.3f}B")
    print(f"   估算儲存：  {est_gb:.1f} GB")
    print(f"   成功率：    {100*total_samples_ok/max(sample_idx,1):.1f}%")
    print(f"   輸出目錄：  {output_dir}")
    print(f"{'='*60}")

    # 寫統計 JSON
    stats = {
        "total_body_tokens":  total_body_tokens,
        "total_samples_ok":   total_samples_ok,
        "total_processed":    sample_idx,
        "success_rate_pct":   round(100 * total_samples_ok / max(sample_idx, 1), 2),
        "est_gb":             round(est_gb, 2),
    }
    with open(output_dir / "bcot_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
