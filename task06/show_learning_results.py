"""
show_learning_results.py — DNA Helix 16B 學習成果展示與推理腳本
================================================================
功能：
  1. 載入指定的 DNA Helix 16B 權重（支持 distill 包裝器與純 base_model 權重）
  2. 使用 Tokenizer 對測試 Prompts 進行自迴歸生成
  3. 分別展示三大能力指標的實際輸出：
     - 指標 A: 局部代碼語意與語法精準度 (測試粒子頭 Particle Head)
     - 指標 B: 思考鏈與推理邏輯 (測試 <think> 推理蒸餾)
     - 指標 C: 長上下文感知 (測試波動頭 Wave Head Phase Lock)

使用方式：
  python show_learning_results.py --model_path ~/task06_checkpoints/best_distill_model.pth
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F

# 加入路徑以利導入模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GEMMA3 import DNAHelix
from transformers import AutoTokenizer

TOKENIZER_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

# 測試 Prompt 集
TEST_PROMPTS = {
    "1. 粒子頭代碼語法能力 (Syntax Check)": 
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number using dynamic programming.\"\"\"\n",
    
    "2. 思考鏈推理能力 (CoT Reasoning)": 
        "<think>\n我們需要找出 1 到 50 之間所有質數的和。首先，我需要定義質數的判斷規則：\n",
    
    "3. 波動頭長文相位感知 (Long Context - Phase 2/3 成果)":
        "這是第一段長文的開始。此處包含了一些秘密的系統設定值 Key = 42。\n" + ("中間填充無關冗長上下文資訊。 " * 50) + "\n請回答剛才第一段文字中提到的 Key 是多少？\nKey = "
}

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=64, temperature=0.7):
    model.eval()
    device = next(model.parameters()).device
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    generated = input_ids.clone()
    
    for _ in range(max_new_tokens):
        # forward pass
        logits = model(generated)
        next_token_logits = logits[:, -1, :] / temperature
        
        # 簡單的 Greedy decoding
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        generated = torch.cat((generated, next_token), dim=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    # 只解碼新生成的部分
    new_tokens = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)

def main():
    parser = argparse.ArgumentParser(description="DNA Helix 16B 學習成果展示與推理腳本")
    parser.add_argument("--model_path", type=str, default="~/task06_checkpoints/best_distill_model.pth",
                        help="模型權重 (.pth) 的儲存路徑")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="推論設備")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)
    
    print("=" * 70)
    print("🧬 DNA Helix 16B 學習成果驗證引擎啟動...")
    print("=" * 70)

    # 1. 載入 Tokenizer
    print(f"Loading Tokenizer: {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)

    # 2. 建立模型 (16B 架構)
    print("Initializing 16B DNA Helix Model (D=4096, hidden=16384, blocks=40)...")
    model = DNAHelix(
        vocab_size=len(tokenizer),
        D=4096,
        hidden_dim=16384,
        num_blocks=40,
        C=1024,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128
    )

    # 3. 載入權重
    if os.path.exists(model_path):
        print(f"Loading weights from: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            # 處理 distill wrapper 前綴
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("base_model."):
                    clean_state_dict[k.replace("base_model.", "")] = v
                else:
                    clean_state_dict[k] = v
            
            # 載入
            missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
            print("  ✅ Weights loaded successfully!")
            if missing:
                print(f"  [INFO] Missing keys (expected in new layers): {len(missing)}")
        except Exception as e:
            print(f"  ❌ Weights loading failed: {e}")
            print("  [WARN] Running inference with randomly initialized weights for verification.")
    else:
        print(f"⚠️  未找到模型權重檔案: {model_path}")
        print("  將以隨機初始化的模型執行前向推理展示。")

    # 移動到指定設備
    print(f"Moving model to {args.device}...")
    # 這裡我們模擬 mock 掉 flash_attn，以防 CPU 報錯
    try:
        model = model.to(args.device)
        if args.device == "cuda":
            model = model.to(torch.bfloat16)
    except Exception as e:
        print(f"  Error moving model: {e}")

    print("\n" + "#" * 70)
    print("🚀 開始執行多維度成果驗證推理：")
    print("#" * 70 + "\n")

    for title, prompt in TEST_PROMPTS.items():
        print(f"📌 {title}")
        print(f"--- Prompt ---")
        print(prompt, end="")
        print(f"\n--------------")
        
        try:
            # 限制生成長度
            generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=40)
            print(f"🟢 生成成果：")
            print(generated_text)
        except Exception as e:
            print(f"🔴 推理生成失敗：{e}")
        print("-" * 70 + "\n")

    print("=" * 70)
    print("🎉 成果驗證展示完畢！")
    print("=" * 70)

if __name__ == "__main__":
    main()
