"""
stage1_download.py — Stage 1 數據下載主程序
由 prepare_data_task06.slurm / prepare_data_task06_test.slurm 調用

支援 --max_gb 參數：
  指定硬碟容量上限後，自動等比縮放各 Step 的 target_tokens，
  確保全部 Parquet 輸出不超過指定的 GB 數。
  例如：--max_gb 100  (預設：不限制)
"""
import os
import sys
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# ============================================================
# 儲存成本估算（Snappy 壓縮後的有效 bytes/token）
# Type A：input_ids(4B)+labels(4B)+think_mask(1B)+metadata ≈ 9B → ~4.5B 壓縮
# Type B：input_ids(4B)+labels(4B)+minimal metadata         ≈ 8B → ~4.0B 壓縮
# ============================================================
BYTES_PER_TOKEN_A = 4.5
BYTES_PER_TOKEN_B = 4.0


def auto_scale_tokens(targets: dict, max_gb: float) -> float:
    """
    根據 --max_gb 等比縮放各 Step 的 target_tokens。

    Args:
        targets: dict，key 為 step 名稱（'1a'~'1e'），value 為 token 數（會原地修改）
        max_gb:  硬碟上限 (GB)

    Returns:
        scale_factor (float)：實際套用的縮放係數
    """
    max_bytes = max_gb * (1024 ** 3)

    # 計算自然總 bytes（1a/1b/1c 為 Type A，1d/1e 為 Type B）
    natural_bytes = 0.0
    for step, tok in targets.items():
        bpt = BYTES_PER_TOKEN_B if step in ('1d', '1e') else BYTES_PER_TOKEN_A
        natural_bytes += tok * bpt

    scale = min(1.0, max_bytes / natural_bytes) if natural_bytes > 0 else 1.0

    for step in targets:
        targets[step] = int(targets[step] * scale)

    natural_gb = natural_bytes / (1024 ** 3)
    scaled_gb  = natural_gb * scale
    return scale, natural_gb, scaled_gb


def print_budget_table(targets: dict, scale: float, natural_gb: float, scaled_gb: float, max_gb: float):
    """打印儲存預算調配表。"""
    w = 62
    print(f"\n{'='*w}")
    print(f"💾  儲存預算自動調配  (縮放係數 = {scale:.4f})")
    print(f"    原始規劃：{natural_gb:.1f} GB  →  調配後：{scaled_gb:.1f} GB  (上限 {max_gb} GB)")
    print(f"{'─'*w}")
    print(f"  {'Step':<6} {'類型':<8} {'調配後 Tokens':>16}  {'估算 GB':>8}")
    print(f"{'─'*w}")
    for step, tok in targets.items():
        bpt  = BYTES_PER_TOKEN_B if step in ('1d', '1e') else BYTES_PER_TOKEN_A
        typ  = 'Type-B' if step in ('1d', '1e') else 'Type-A'
        gb   = tok * bpt / (1024 ** 3)
        print(f"  {step:<6} {typ:<8} {tok/1e9:>14.3f}B  {gb:>8.2f}")
    print(f"{'─'*w}")
    print(f"  {'Total':<6} {'':<8} {sum(targets.values())/1e9:>14.3f}B  {scaled_gb:>8.2f}")
    print(f"  預算使用率：{100*scaled_gb/max_gb:.1f}%")
    print(f"{'='*w}\n")

def get_schema_A():
    return pa.schema([
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

def get_schema_B():
    return pa.schema([
        pa.field("input_ids",     pa.list_(pa.int32())),
        pa.field("labels",        pa.list_(pa.int32())),
        pa.field("n_split_index", pa.int32()),
        pa.field("data_type",     pa.string()),
        pa.field("source",        pa.string()),
        pa.field("token_length",  pa.int32()),
    ])

def flush_rows_A(rows, out_dir, tag, batch_id, schema):
    if not rows:
        return batch_id
    t = pa.table({k: [r[k] for r in rows] for k in rows[0].keys()}, schema=schema)
    path = os.path.join(out_dir, f"{tag}_{batch_id:05d}.parquet")
    pq.write_table(t, path, compression="snappy")
    print(f"  💾 寫出 {len(rows):,} 筆 → {os.path.basename(path)}")
    return batch_id + 1

def flush_rows_B(rows, out_dir, tag, batch_id, schema):
    if not rows:
        return batch_id
    t = pa.table({k: [r[k] for r in rows] for k in rows[0].keys()}, schema=schema)
    path = os.path.join(out_dir, f"{tag}_{batch_id:05d}.parquet")
    pq.write_table(t, path, compression="snappy")
    print(f"  💾 寫出 {len(rows):,} 筆 → {os.path.basename(path)}")
    return batch_id + 1

def chunk_ids(ids, chunk_size, category, difficulty):
    rows = []
    for start in range(0, len(ids) - chunk_size // 4, chunk_size):
        chunk = ids[start: start + chunk_size]
        if len(chunk) < chunk_size // 4:
            continue
            
        # 尋找 assistant 標籤或 <think> 標籤作為精確切分點
        n_split = -1
        # 1. 尋找 <|im_start|>assistant
        for i in range(len(chunk) - 2, -1, -1):
            if chunk[i] == 151644 and chunk[i+1] == 77091: # im_start assistant
                if i + 2 < len(chunk) and chunk[i+2] == 198: # newline
                    n_split = i + 3
                else:
                    n_split = i + 2
                break
                
        # 2. 如果沒找到，尋找 <think> 標籤 (151667)
        if n_split == -1:
            for i in range(len(chunk) - 1, -1, -1):
                if chunk[i] == 151667:
                    n_split = i
                    break
                    
        # 3. 最終備用：原先的 75% 切分
        if n_split == -1:
            n_split = int(len(chunk) * 0.75)
            
        rows.append({
            "input_ids":      list(chunk),
            "labels":         [-100] * n_split + list(chunk[n_split:]),
            "n_split_index":  n_split,
            "data_type":      "A",
            "logit_shard_id": -1,
            "logit_offset":   -1,
            "think_mask":     [False] * len(chunk),
            "category":       category,
            "difficulty":     difficulty,
        })
    return rows

def format_sample_to_text(sample, tokenizer, default_col=None):
    # 1. Chat structures
    for chat_col in ["messages", "conversations"]:
        if chat_col in sample and isinstance(sample[chat_col], list) and len(sample[chat_col]) > 0:
            try:
                return tokenizer.apply_chat_template(sample[chat_col], tokenize=False, add_generation_prompt=False)
            except Exception:
                # Fallback format
                parts = []
                for msg in sample[chat_col]:
                    role = msg.get("role", "user")
                    content = msg.get("content", msg.get("value", ""))
                    parts.append(f"<|{role}|>\n{content}")
                return "\n".join(parts)

    # 2. Pair structures (prompt + response)
    pairs = [
        ("problem", "solution"),
        ("instruction", "response"),
        ("query", "answer"),
        ("prompt", "generation")
    ]
    for prompt_key, response_key in pairs:
        if prompt_key in sample and response_key in sample:
            p_val = sample[prompt_key]
            r_val = sample[response_key]
            if p_val and r_val:
                messages = [
                    {"role": "user", "content": str(p_val)},
                    {"role": "assistant", "content": str(r_val)}
                ]
                try:
                    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                except Exception:
                    return f"<|user|>\n{p_val}\n<|assistant|>\n{r_val}"

    # 3. Default single text column
    if default_col and default_col in sample:
        val = sample[default_col]
        if isinstance(val, list):
            try:
                return tokenizer.apply_chat_template(val, tokenize=False, add_generation_prompt=False)
            except Exception:
                return " ".join(str(x) for x in val)
        return str(val)

    # Fallback to other common text columns
    for fallback in ["text", "content", "code"]:
        if fallback in sample and sample[fallback]:
            return str(sample[fallback])

    return ""

def step_1a_stack_short(tokenizer, type_a_dir, target_tokens=3_000_000_000):
    """1A: Codeparrot Github Code Clean 短文代碼 (≤ 4096 tok) → Type A"""
    print("\n[1A] Codeparrot Github Code Clean 短文代碼...")
    schema   = get_schema_A()
    CHUNK    = 2048
    LANGS    = {"python", "javascript", "typescript", "go", "rust", "c++"}
    # Mapping to standard tags
    LANG_MAP = {
        "python": "python", "javascript": "js",
        "typescript": "ts",  "go": "go",
        "rust": "rust",      "c++": "cpp",
    }
    total    = 0
    rows     = []
    batch_id = 0
    lang_counts = {}

    print("  載入 codeparrot/github-code-clean (parquet)...")
    try:
        ds = load_dataset(
            "parquet",
            data_files="hf://datasets/codeparrot/github-code-clean/data/train-*.parquet",
            split="train",
            streaming=True,
        )
        
        batch_size = 1000
        batch_samples = []
        
        for sample in tqdm(ds, desc="  Github Clean Short", unit=" files"):
            if total >= target_tokens:
                break

            lang = str(sample.get("language", "")).lower()
            if lang not in LANGS:
                continue

            code = sample.get("code", "")
            if not code or len(code) < 100:
                continue

            batch_samples.append((sample, lang))
            
            if len(batch_samples) >= batch_size:
                codes = [s[0].get("code", "") for s in batch_samples]
                encodings = tokenizer(codes, add_special_tokens=False, verbose=False)["input_ids"]
                
                for (s, l), ids in zip(batch_samples, encodings):
                    if len(ids) > CHUNK * 4:
                        continue
                    
                    tag = LANG_MAP.get(l, l[:6])
                    lang_counts[tag] = lang_counts.get(tag, 0) + 1
                    new_rows = chunk_ids(ids, CHUNK, "code", 2)
                    rows.extend(new_rows)
                    total += sum(len(r["input_ids"]) for r in new_rows)
                
                batch_samples = []

                if len(rows) >= 50000:
                    batch_id = flush_rows_A(rows, type_a_dir, f"stack_short_{tag}", batch_id, schema)
                    rows = []
                    print(f"  進度：{total/1e9:.2f}B tokens | {lang_counts}")
                    
        # Flush remaining batch
        if batch_samples:
            codes = [s[0].get("code", "") for s in batch_samples]
            encodings = tokenizer(codes, add_special_tokens=False, verbose=False)["input_ids"]
            for (s, l), ids in zip(batch_samples, encodings):
                if len(ids) > CHUNK * 4:
                    continue
                tag = LANG_MAP.get(l, l[:6])
                lang_counts[tag] = lang_counts.get(tag, 0) + 1
                new_rows = chunk_ids(ids, CHUNK, "code", 2)
                rows.extend(new_rows)
                total += sum(len(r["input_ids"]) for r in new_rows)

    except Exception as e:
        print(f"  ⚠ Github Clean 短文載入失敗：{e}")

    batch_id = flush_rows_A(rows, type_a_dir, "stack_short_final", batch_id, schema)
    print(f"  ✅ 1A 完成：{total/1e9:.2f}B tokens | 語言分布：{lang_counts}")
    return total

def step_1b_competitive(tokenizer, type_a_dir, target_tokens=3_000_000_000):
    """1B: 競程題 + 推理數據 → Type A"""
    print("\n[1B] 競程題 + 推理數據...")
    schema   = get_schema_A()
    CHUNK    = 2048
    total    = 0
    rows     = []
    batch_id = 0

    # Source config: (dataset_id, split, default_col, category, difficulty)
    SOURCES = [
        ("open-r1/codeforces-cots",           "train",     "messages",     "cot", 5),
        ("AI-MO/NuminaMath-CoT",              "train",     "solution",     "cot", 4),
        ("HuggingFaceH4/ultrachat_200k",      "train_sft", "messages",     "instruction", 3),
    ]

    for (ds_name, split, col, cat, diff) in SOURCES:
        if total >= target_tokens:
            break
        print(f"  ► {ds_name}")
        try:
            ds = load_dataset(ds_name, split=split, streaming=True)
            for sample in tqdm(ds, desc=f"  {ds_name[:35]}", unit=" samples"):
                if total >= target_tokens:
                    break
                text = format_sample_to_text(sample, tokenizer, default_col=col)
                if not text or len(text) < 100:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                new_rows = chunk_ids(ids, CHUNK, cat, diff)
                rows.extend(new_rows)
                total += sum(len(r["input_ids"]) for r in new_rows)
                if len(rows) >= 50000:
                    batch_id = flush_rows_A(rows, type_a_dir, "competitive", batch_id, schema)
                    rows = []
        except Exception as e:
            print(f"  ⚠ {ds_name} 失敗：{e}")

    batch_id = flush_rows_A(rows, type_a_dir, "competitive_final", batch_id, schema)
    print(f"  ✅ 1B 完成：{total/1e9:.2f}B tokens")
    return total

def step_1c_instruction(tokenizer, type_a_dir, target_tokens=4_000_000_000):
    """1C: 多輪指令對話 → Type A"""
    print("\n[1C] 多輪指令數據...")
    schema   = get_schema_A()
    CHUNK    = 2048
    total    = 0
    rows     = []
    batch_id = 0

    SOURCES = [
        ("Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1",          "train", "conversations"),
        ("ise-uiuc/Magicoder-OSS-Instruct-75K",              "train", "problem"),
        ("m-a-p/CodeFeedback-Filtered-Instruction",          "train", "query"),
    ]

    for (ds_name, split, col) in SOURCES:
        if total >= target_tokens:
            break
        print(f"  ► {ds_name}")
        try:
            ds = load_dataset(ds_name, split=split, streaming=True)
            for sample in tqdm(ds, desc=f"  {ds_name[:35]}", unit=" samples"):
                if total >= target_tokens:
                    break
                text = format_sample_to_text(sample, tokenizer, default_col=col)
                if not text or len(text) < 100:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                new_rows = chunk_ids(ids, CHUNK, "instruction", 3)
                rows.extend(new_rows)
                total += sum(len(r["input_ids"]) for r in new_rows)
                if len(rows) >= 50000:
                    batch_id = flush_rows_A(rows, type_a_dir, "instruction", batch_id, schema)
                    rows = []
        except Exception as e:
            print(f"  ⚠ {ds_name} 失敗：{e}")

    batch_id = flush_rows_A(rows, type_a_dir, "instruction_final", batch_id, schema)
    print(f"  ✅ 1C 完成：{total/1e9:.2f}B tokens")
    return total

def step_1d_code_long(tokenizer, type_b_dir, target_tokens=5_000_000_000):
    """1D: 長文代碼 (>4K tok) → Type B"""
    print("\n[1D] 長文代碼 (Type B)...")
    schema    = get_schema_B()
    MAX_CHUNK = 32768
    MIN_LEN   = 4096
    LANGS     = {"python", "javascript", "typescript", "go", "rust", "c++"}
    LANG_MAP  = {
        "python": "python", "javascript": "js",
        "typescript": "ts",  "go": "go",
        "rust": "rust",      "c++": "cpp",
    }
    total    = 0
    rows     = []
    batch_id = 0

    print("  載入 codeparrot/github-code-clean (parquet) 長文部分...")
    try:
        ds = load_dataset(
            "parquet",
            data_files="hf://datasets/codeparrot/github-code-clean/data/train-*.parquet",
            split="train",
            streaming=True,
        )
        
        batch_size = 500
        batch_samples = []
        
        for sample in tqdm(ds, desc="  Github Clean Long", unit=" files"):
            if total >= target_tokens:
                break

            lang = str(sample.get("language", "")).lower()
            if lang not in LANGS:
                continue

            code = sample.get("code", "")
            if not code or len(code) < 2000:
                continue

            batch_samples.append((sample, lang))
            
            if len(batch_samples) >= batch_size:
                codes = [s[0].get("code", "") for s in batch_samples]
                encodings = tokenizer(codes, add_special_tokens=False, verbose=False)["input_ids"]
                
                for (s, l), ids in zip(batch_samples, encodings):
                    if len(ids) < MIN_LEN:
                        continue

                    ids = ids[:MAX_CHUNK]
                    L   = len(ids)
                    tag = LANG_MAP.get(l, l[:6])
                    rows.append({
                        "input_ids":     list(ids),
                        "labels":        list(ids),
                        "n_split_index": 0,
                        "data_type":     "B",
                        "source":        f"github_{tag}",
                        "token_length":  L,
                    })
                    total += L
                
                batch_samples = []
                
                if len(rows) >= 1000:
                    batch_id = flush_rows_B(rows, type_b_dir, f"code_long_{tag}", batch_id, schema)
                    rows = []

        # Flush remaining batch
        if batch_samples:
            codes = [s[0].get("code", "") for s in batch_samples]
            encodings = tokenizer(codes, add_special_tokens=False, verbose=False)["input_ids"]
            for (s, l), ids in zip(batch_samples, encodings):
                if len(ids) < MIN_LEN:
                    continue
                ids = ids[:MAX_CHUNK]
                L   = len(ids)
                tag = LANG_MAP.get(l, l[:6])
                rows.append({
                    "input_ids":     list(ids),
                    "labels":        list(ids),
                    "n_split_index": 0,
                    "data_type":     "B",
                    "source":        f"github_{tag}",
                    "token_length":  L,
                })
                total += L

    except Exception as e:
        print(f"  ⚠ Github Clean 長文載入失敗：{e}")

    batch_id = flush_rows_B(rows, type_b_dir, "code_long_final", batch_id, schema)
    print(f"  ✅ 1D 完成：{total/1e9:.2f}B tokens")
    return total

def step_1e_theory(tokenizer, type_b_dir, target_tokens=4_000_000_000):
    """1E: CS 理論長文 → Type B"""
    print("\n[1E] CS 理論長文 (Type B)...")
    schema    = get_schema_B()
    MAX_CHUNK = 32768
    MIN_LEN   = 2048
    total     = 0
    rows      = []
    batch_id  = 0

    SOURCES = [
        ("open-phi/textbooks",                  "train", "textbook",  {}),
        ("vikp/textbook_quality_programming",   "train", "markdown",  {}),
        ("allenai/peS2o",                       "train", "text",      {"format": "json", "data_files": "hf://datasets/allenai/peS2o/data/v1/train-*.json.gz"}),
        ("izumi-lab/open-text-books",           "train", "text",      {}),
    ]

    for (ds_name, split, col, extra_kw) in SOURCES:
        if total >= target_tokens:
            break
        print(f"  ► {ds_name}")
        try:
            if extra_kw.get("format") == "json":
                ds = load_dataset("json", data_files=extra_kw["data_files"], split=split, streaming=True)
            else:
                ds = load_dataset(ds_name, split=split, streaming=True)
            
            src_tag = ds_name.split("/")[-1].replace("-", "_")[:20]
            for sample in tqdm(ds, desc=f"  {src_tag}", unit=" docs"):
                if total >= target_tokens:
                    break
                text = format_sample_to_text(sample, tokenizer, default_col=col)
                if not text or len(text) < 1000:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) < MIN_LEN:
                    continue
                ids = ids[:MAX_CHUNK]
                L   = len(ids)
                rows.append({
                    "input_ids":     list(ids),
                    "labels":        list(ids),
                    "n_split_index": 0,
                    "data_type":     "B",
                    "source":        src_tag,
                    "token_length":  L,
                })
                total += L
                if len(rows) >= 500:
                    batch_id = flush_rows_B(rows, type_b_dir, f"theory_{src_tag}", batch_id, schema)
                    rows = []
        except Exception as e:
            print(f"  ⚠ {ds_name} 失敗：{e}")

    batch_id = flush_rows_B(rows, type_b_dir, "theory_final", batch_id, schema)
    print(f"  ✅ 1E 完成：{total/1e9:.2f}B tokens")
    return total

def main():
    parser = argparse.ArgumentParser(description="Stage 1 數據下載主程序")
    parser.add_argument("--type_a_dir",  required=True)
    parser.add_argument("--type_b_dir",  required=True)
    parser.add_argument("--tokenizer",   default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--steps",       default="1a,1b,1c,1d,1e",
                        help="逗號分隔要執行的子步驟，例如 1a,1b 或 all")
    parser.add_argument("--target_tokens_1a", type=int, default=3_000_000_000)
    parser.add_argument("--target_tokens_1b", type=int, default=3_000_000_000)
    parser.add_argument("--target_tokens_1c", type=int, default=4_000_000_000)
    parser.add_argument("--target_tokens_1d", type=int, default=5_000_000_000)
    parser.add_argument("--target_tokens_1e", type=int, default=4_000_000_000)
    parser.add_argument("--max_gb",      type=float, default=None,
                        help=(
                            "硬碟容量上限（GB）。指定後自動等比縮放各 Step 的 target_tokens，"
                            "確保輸出 Parquet 總量不超過此值。"
                            "例如：--max_gb 100"
                        ))
    args = parser.parse_args()

    steps = set(args.steps.lower().split(",")) if args.steps != "all" else {"1a","1b","1c","1d","1e"}

    # ── 儲存預算自動調配 ──
    # 只對「本次要執行的 steps」計算縮放，跳過的 step 不納入預算
    targets = {}
    if "1a" in steps: targets["1a"] = args.target_tokens_1a
    if "1b" in steps: targets["1b"] = args.target_tokens_1b
    if "1c" in steps: targets["1c"] = args.target_tokens_1c
    if "1d" in steps: targets["1d"] = args.target_tokens_1d
    if "1e" in steps: targets["1e"] = args.target_tokens_1e

    if args.max_gb is not None:
        scale, natural_gb, scaled_gb = auto_scale_tokens(targets, args.max_gb)
        print_budget_table(targets, scale, natural_gb, scaled_gb, args.max_gb)
        if scale < 1.0:
            print(f"⚠️  原始規劃 {natural_gb:.1f} GB 超出限制 {args.max_gb} GB，"
                  f"已等比縮放至 {scaled_gb:.1f} GB（係數 {scale:.4f})")
        else:
            print(f"✅ 原始規劃 {natural_gb:.1f} GB 在預算 {args.max_gb} GB 以內，無需縮放")
    else:
        print("ℹ️  未指定 --max_gb，使用各 Step 的原始 target_tokens（無儲存上限）")

    print(f"\n🔤 載入 Tokenizer：{args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    print(f"  詞表大小：{tokenizer.vocab_size:,}")

    totals = {}
    if "1a" in steps: totals["1a"] = step_1a_stack_short(tokenizer, args.type_a_dir, target_tokens=targets.get("1a", args.target_tokens_1a))
    if "1b" in steps: totals["1b"] = step_1b_competitive(tokenizer, args.type_a_dir, target_tokens=targets.get("1b", args.target_tokens_1b))
    if "1c" in steps: totals["1c"] = step_1c_instruction(tokenizer, args.type_a_dir, target_tokens=targets.get("1c", args.target_tokens_1c))
    if "1d" in steps: totals["1d"] = step_1d_code_long(tokenizer, args.type_b_dir,  target_tokens=targets.get("1d", args.target_tokens_1d))
    if "1e" in steps: totals["1e"] = step_1e_theory(tokenizer, args.type_b_dir,      target_tokens=targets.get("1e", args.target_tokens_1e))

    # ── 完成統計 ──
    type_a_total = sum(v for k, v in totals.items() if k in ("1a", "1b", "1c"))
    type_b_total = sum(v for k, v in totals.items() if k in ("1d", "1e"))
    est_gb_a     = type_a_total * BYTES_PER_TOKEN_A / (1024 ** 3)
    est_gb_b     = type_b_total * BYTES_PER_TOKEN_B / (1024 ** 3)
    est_gb_total = est_gb_a + est_gb_b

    print("\n" + "="*60)
    print("  📊 Stage 1 完成統計")
    print("="*60)
    for k, v in totals.items():
        bpt = BYTES_PER_TOKEN_B if k in ('1d', '1e') else BYTES_PER_TOKEN_A
        gb  = v * bpt / (1024 ** 3)
        print(f"  Step {k}: {v/1e9:.3f}B tokens  ≈ {gb:.1f} GB")
    print(f"\n  Type A 合計：{type_a_total/1e9:.3f}B tokens  ≈ {est_gb_a:.1f} GB")
    print(f"  Type B 合計：{type_b_total/1e9:.3f}B tokens  ≈ {est_gb_b:.1f} GB")
    print(f"  總計：       {(type_a_total+type_b_total)/1e9:.3f}B tokens  ≈ {est_gb_total:.1f} GB")
    if args.max_gb:
        print(f"  預算上限：   {args.max_gb} GB  (使用率 {100*est_gb_total/args.max_gb:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    main()
