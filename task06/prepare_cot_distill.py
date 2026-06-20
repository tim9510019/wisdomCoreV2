"""
prepare_cot_distill.py — CoT 思考鏈數據生成引擎
================================================================
功能：
  1. 使用 Qwen3-Coder-30B 的 Enable Thinking 模式生成 CoT 推理鏈
  2. 輸入：各種 Coding 任務（競程題/Debug/系統設計等）
  3. 輸出：帶完整 <think>...</think><answer>...</answer> 結構的文本
  4. 存入 Data-Type A Parquet，供後續 Teacher 推論蒸餾

Sub-類別：
  - cot_qwen3_self:    Qwen3 自生成 CoT  (6B tokens)
  - cot_competitive:   競程題解題鏈      (3B tokens)
  - cot_debug:         Debug 推理鏈      (2B tokens)
  - cot_architecture:  架構設計推理      (2B tokens)
  - cot_selfplay:      Self-Play 反駁    (1B tokens)
"""

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

import torch
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

TEACHER_MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
RANDOM_SEED = 2026
random.seed(RANDOM_SEED)

# ============================================================
# 提示模板庫
# ============================================================

COT_PROMPT_TEMPLATES = {
    # 1. 自由 Coding CoT 生成
    "cot_coding": """\
請用 Python 解決以下問題，並展示完整的思考過程：

問題：{problem}

請先在 <think> 標籤中展示你的推理過程，然後給出最終代碼：
""",

    # 2. Debug 推理鏈
    "cot_debug": """\
以下代碼存在 Bug，請找出問題並修復：

```python
{buggy_code}
```

錯誤信息：{error_msg}

請先在 <think> 標籤中分析 Bug 原因，然後提供修復後的代碼：
""",

    # 3. 系統架構設計
    "cot_architecture": """\
設計一個系統：{system_description}

請在 <think> 標籤中進行以下分析：
1. 需求分解
2. 架構選型理由
3. 關鍵技術決策
4. 潛在風險與應對

最後給出完整的架構設計方案和核心代碼：
""",

    # 4. Self-Play 反駁改進
    "cot_selfplay": """\
以下是一個初始解決方案：

```python
{initial_solution}
```

請在 <think> 標籤中：
1. 批判性分析這個方案的缺陷
2. 提出改進思路
3. 實現改進版本

最終給出更優的實現：
""",

    # 5. 算法複雜度分析
    "cot_complexity": """\
分析以下算法並優化：

```python
{algorithm_code}
```

在 <think> 標籤中分析時間/空間複雜度，然後給出優化版本：
""",
}

# ============================================================
# Coding 問題種子庫（用於生成 CoT 數據）
# ============================================================

PROBLEM_SEEDS = [
    # 算法類
    "實現一個 LRU Cache，支持 O(1) 的 get 和 put 操作",
    "設計一個線程安全的生產者消費者隊列",
    "實現 Trie 字典樹，支持前綴匹配和通配符搜索",
    "設計一個分布式 ID 生成器（Snowflake 算法）",
    "實現一個高性能的事件循環（Event Loop）",
    "設計一個 ORM 框架的查詢構建器",
    "實現一個支持撤銷/重做的文本編輯器數據結構",
    "設計一個 SQL 解析器（AST 構建）",
    "實現一個簡單的 JIT 編譯器原型",
    "設計一個分布式鎖服務",
    # 系統設計類
    "設計 Twitter 的 Feed 推薦系統",
    "設計一個實時協作編輯器（Google Docs）",
    "設計一個高可用的消息隊列系統",
    "設計一個分布式文件系統",
    "設計一個 API 網關",
    # Debug 類
    "修復一個死鎖問題：兩個線程相互等待對方的鎖",
    "修復一個內存洩漏：循環引用導致 GC 無法回收",
    "修復一個競態條件：多線程下的計數器錯誤",
]

DEBUG_PROBLEMS = [
    {
        "buggy_code": """
def find_longest_palindrome(s):
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    max_len = 1
    start = 0
    
    for i in range(n):
        dp[i][i] = True
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                if length == 2:
                    dp[i][j] = True
                else:
                    dp[i][j] = dp[i+1][j-1]  # Bug: 索引越界
            if dp[i][j] and length > max_len:
                max_len = length
                start = i
    
    return s[start:start + max_len]
""",
        "error_msg": "IndexError: list index out of range"
    },
    {
        "buggy_code": """
import threading

class BankAccount:
    def __init__(self, balance):
        self.balance = balance
    
    def transfer(self, other, amount):
        if self.balance >= amount:
            time.sleep(0.001)  # 模擬網絡延遲
            self.balance -= amount
            other.balance += amount  # 競態條件 Bug
            return True
        return False
""",
        "error_msg": "多線程測試發現餘額不一致：預期 1000，實際 850"
    },
]

# ============================================================
# 主生成函數
# ============================================================

@torch.no_grad()
def generate_cot(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    device: str = "cuda",
) -> str:
    """使用 Qwen3 的 Enable Thinking 模式生成 CoT。"""
    # 構建帶 thinking 的消息格式
    messages = [
        {"role": "user", "content": prompt}
    ]

    # 使用 Qwen3 的 chat template（enable_thinking=True）
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Qwen3 特有參數
        )
    except Exception:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    # 只取新生成的部分
    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def get_think_mask_from_text(text: str, tokenizer) -> tuple[list[int], list[bool]]:
    """
    對生成文本進行 Token 化並標記 think_mask。
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    mask = [False] * len(ids)

    # 找到 <think> 和 </think> 的 token 位置
    decoded_so_far = ""
    in_think = False

    for i, tid in enumerate(ids):
        tok = tokenizer.decode([tid], skip_special_tokens=False)
        decoded_so_far += tok

        if "<think>" in decoded_so_far and not in_think:
            in_think = True
        if "</think>" in decoded_so_far and in_think:
            in_think = False

        mask[i] = in_think

    return ids, mask


def write_batch_to_parquet(
    batch_data: list[dict],
    output_path: str,
    tokenizer,
    chunk_size: int = 1024,
):
    """
    將生成的 CoT 文本分塊並存入 Parquet。
    """
    schema = pa.schema([
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

    rows = []
    for item in batch_data:
        text     = item["text"]
        category = item.get("category", "cot")
        difficulty = item.get("difficulty", 4)

        ids, mask = get_think_mask_from_text(text, tokenizer)

        # 切割成 chunk_size 大小
        n_split_ratio = 0.75
        for start in range(0, len(ids) - chunk_size // 2, chunk_size):
            chunk_ids  = ids[start:start + chunk_size]
            chunk_mask = mask[start:start + chunk_size]

            if len(chunk_ids) < chunk_size // 4:
                continue

            n_split = int(len(chunk_ids) * n_split_ratio)
            labels  = [-100] * n_split + list(chunk_ids[n_split:])

            rows.append({
                "input_ids":      list(chunk_ids),
                "labels":         labels,
                "n_split_index":  n_split,
                "data_type":      "A",
                "logit_shard_id": -1,
                "logit_offset":   -1,
                "think_mask":     chunk_mask,
                "category":       category,
                "difficulty":       difficulty,
            })

    if not rows:
        return 0

    table = pa.table({
        "input_ids":      [r["input_ids"] for r in rows],
        "labels":         [r["labels"] for r in rows],
        "n_split_index":  [r["n_split_index"] for r in rows],
        "data_type":      [r["data_type"] for r in rows],
        "logit_shard_id": [r["logit_shard_id"] for r in rows],
        "logit_offset":   [r["logit_offset"] for r in rows],
        "think_mask":     [r["think_mask"] for r in rows],
        "category":       [r["category"] for r in rows],
        "difficulty":     [r["difficulty"] for r in rows],
    }, schema=schema)

    pq.write_table(table, output_path, compression="snappy")
    return len(rows)


def get_diverse_prompts(tokenizer, limit=40000):
    """
    從多個高品質數據集中流式提取輸入 Prompt，組合成多元的程式設計與邏輯推理問題庫。
    """
    print("\n📦 正在從多個高品質數據集中收集多元 Prompt...")
    prompts = []
    
    # 數據源配置：(dataset_name, split, prompt_columns, template_format)
    # prompt_columns 為可能存有 Prompt 的欄位順序或特殊處理
    SOURCES = [
        {
            "id": "ise-uiuc/Magicoder-OSS-Instruct-75K",
            "split": "train",
            "keys": ["problem"],
            "weight": 15000,
            "category": "cot_coding",
            "difficulty": 3
        },
        {
            "id": "open-r1/codeforces-cots",
            "split": "train",
            "keys": ["messages", "problem"],
            "weight": 10000,
            "category": "cot_coding",
            "difficulty": 5
        },
        {
            "id": "AI-MO/NuminaMath-CoT",
            "split": "train",
            "keys": ["problem"],
            "weight": 8000,
            "category": "cot_math",
            "difficulty": 4
        },
        {
            "id": "m-a-p/CodeFeedback-Filtered-Instruction",
            "split": "train",
            "keys": ["query"],
            "weight": 7000,
            "category": "cot_coding",
            "difficulty": 3
        }
    ]

    for src in SOURCES:
        ds_id = src["id"]
        needed = src["weight"]
        print(f"  📥 讀取 {ds_id} (目標 {needed} 題)...")
        count = 0
        try:
            ds = load_dataset(ds_id, split=src["split"], streaming=True)
            for sample in ds:
                if count >= needed or len(prompts) >= limit:
                    break
                
                # 尋找與提取 Prompt 文本
                raw_prompt = ""
                for k in src["keys"]:
                    val = sample.get(k)
                    if not val:
                        continue
                    if isinstance(val, list):
                        # 如果是 chat 格式，只拿第一輪 User 說的話
                        if len(val) > 0 and isinstance(val[0], dict):
                            raw_prompt = val[0].get("content", val[0].get("value", ""))
                        else:
                            raw_prompt = str(val[0])
                    else:
                        raw_prompt = str(val)
                    
                    if raw_prompt:
                        break
                
                if not raw_prompt or len(raw_prompt) < 15:
                    continue
                
                # 構建 Prompt
                if src["category"] == "cot_math":
                    prompt_text = f"請解決以下數學與邏輯推理問題，並展示完整的思考過程：\n\n問題：{raw_prompt}\n\n請先在 <think> 標籤中展示你的推理與推導過程，然後給出最終答案："
                else:
                    prompt_text = f"請解決以下程式設計或演算法問題，並展示完整的思考過程：\n\n問題：{raw_prompt}\n\n請先在 <think> 標籤中展示你的推理與代碼分析過程，然後給出最終的代碼實現："

                prompts.append({
                    "prompt": prompt_text,
                    "category": src["category"],
                    "difficulty": src["difficulty"]
                })
                count += 1
        except Exception as e:
            print(f"  ⚠️ 讀取 {ds_id} 失敗：{e}")
            continue
            
    # 隨機打散 Prompt
    random.shuffle(prompts)
    print(f"✅ 收集完成！總共獲得 {len(prompts):,} 題多元 Prompts。")
    return prompts


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CoT 思考鏈數據生成引擎")
    parser.add_argument("--teacher_model", type=str, default=TEACHER_MODEL_ID)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_tokens", type=int, default=60_000_000) # 預設改為 60M tokens
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    # 1. 獲取多元 Prompt 池
    prompts_pool = get_diverse_prompts(None, limit=40000)
    if not prompts_pool:
        print("❌ 無法取得任何多元 Prompt，請檢查網路與數據集連線！")
        sys.exit(1)

    print(f"\n🔤 載入 Teacher 模型 (vLLM)：{args.teacher_model}")
    from vllm import LLM, SamplingParams
    
    # 建立 vLLM LLM 實例
    llm = LLM(
        model=args.teacher_model,
        tensor_parallel_size=2,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=0.95,
        repetition_penalty=1.1,
    )

    total_tokens = 0
    batch_id = 0
    batch_data = []

    print(f"\n🧠 開始生成 CoT 數據（目標：{args.target_tokens/1e6:.1f}M tokens）...")

    VLLM_BATCH_SIZE = 512
    prompt_idx = 0
    
    while total_tokens < args.target_tokens and prompt_idx < len(prompts_pool):
        current_batch_items = prompts_pool[prompt_idx : prompt_idx + VLLM_BATCH_SIZE]
        prompt_idx += len(current_batch_items)
        
        if not current_batch_items:
            break
            
        formatted_prompts = []
        for item in current_batch_items:
            messages = [{"role": "user", "content": item["prompt"]}]
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except Exception:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            formatted_prompts.append(text)
            
        try:
            outputs = llm.generate(formatted_prompts, sampling_params)
        except Exception as e:
            print(f"  ⚠️  vLLM 批次生成失敗：{e}")
            continue
            
        for item, out in zip(current_batch_items, outputs):
            raw_prompt = item["prompt"]
            generated = out.outputs[0].text
            category = item["category"]
            difficulty = item["difficulty"]
            
            combined_text = raw_prompt + generated
            batch_data.append({
                "text": combined_text,
                "category": category,
                "difficulty": difficulty,
            })
            
            token_estimate = len(tokenizer.encode(combined_text, add_special_tokens=False))
            total_tokens += token_estimate
            
        while len(batch_data) >= 1000 or (total_tokens >= args.target_tokens and len(batch_data) > 0):
            chunk = batch_data[:1000]
            batch_data = batch_data[1000:]
            
            out_path = output_dir / f"cot_batch_{batch_id:05d}.parquet"
            n_rows = write_batch_to_parquet(chunk, str(out_path), tokenizer)
            print(f"  💾 Batch {batch_id}: {n_rows} 行 | 累計: {total_tokens/1e6:.2f}M tokens | 已使用 Prompt 數: {prompt_idx}/{len(prompts_pool)}")
            batch_id += 1

    print(f"\n✅ CoT 數據生成完畢！總計：{total_tokens/1e6:.2f}M tokens")


if __name__ == "__main__":
    main()
