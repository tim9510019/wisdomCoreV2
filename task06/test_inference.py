import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_header(title):
    print("\n" + "="*80)
    print(f"🌟 {title}")
    print("="*80 + "\n")

@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_new_tokens=2048, temperature=0.7, top_p=0.9):
    # 使用 Qwen 標準 Chat Template 格式化輸入
    messages = [
        {"role": "user", "content": prompt}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
    
    # 追蹤思維鏈狀態，用於在終端機渲染灰色字體
    in_think_block = False
    generated_text = ""
    
    print("🤖 模型回應：")
    
    for _ in range(max_new_tokens):
        # 進行模型前向傳播
        logits = model(input_ids)
        next_token_logits = logits[0, -1, :]
        
        # 採樣與解碼
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float("inf")
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
        token_id = next_token.item()
        
        # 檢查是否為結束符號或 Padding 符號
        if token_id == tokenizer.eos_token_id or (tokenizer.pad_token_id is not None and token_id == tokenizer.pad_token_id):
            break
            
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        if "<|endoftext|>" in token_str or "endoftext" in token_str:
            break
            
        generated_text += token_str
        
        # 實時文字渲染控制：偵測思考區段並以灰色標示
        if "<think>" in token_str or "think" in token_str.lower() and "<" in generated_text[-10:]:
            in_think_block = True
            sys.stdout.write("\033[90m" + token_str) # 終端機灰色
        elif "</think>" in token_str or "think" in token_str.lower() and "</" in generated_text[-10:]:
            in_think_block = False
            sys.stdout.write(token_str + "\033[0m")  # 恢復正常顏色
        else:
            if in_think_block:
                sys.stdout.write(token_str)
            else:
                sys.stdout.write(token_str)
                
        sys.stdout.flush()
        
        # 將生成的 token 串接到輸入序列中
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        
        # 安全防護：如果模型解碼出結束標籤，主動退出
        if "<|im_end|>" in generated_text[-15:] or "<|endoftext|>" in generated_text[-15:]:
            break
            
    print("\n" + "-"*80)
    # 確保終端機顏色回復正常
    sys.stdout.write("\033[0m")
    sys.stdout.flush()

def main():
    print_header("DNA Helix Ultimate 16B 蒸餾模型推理測試終端")
    
    # 1. 載入模型權重路徑
    default_ckpt_path = os.path.expanduser("~/task06_checkpoints/best_distill_model.pth")
    ckpt_path = input(f"請輸入模型權重檔案路徑 [預設: {default_ckpt_path}]: ").strip()
    if not ckpt_path:
        ckpt_path = default_ckpt_path
        
    if not os.path.exists(ckpt_path):
        print(f"❌ 找不到權重檔案：{ckpt_path}，請檢查路徑。")
        sys.exit(1)
        
    # 2. 初始化 Tokenizer
    tokenizer_id = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    print(f"🔄 正在載入 Tokenizer: {tokenizer_id}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    
    # 3. 初始化 16B 骨幹模型
    print("🧬 正在初始化 DNA Helix Ultimate 16B 骨幹結構...")
    from GEMMA3 import DNAHelixUltimate
    model = DNAHelixUltimate(
        vocab_size=151936,
        D=4096,
        C=2048,
        hidden_dim=16384,
        num_blocks=40,
    )
    
    # 4. 載入訓練好的權重
    print(f"💾 正在載入權重檔案: {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    
    # 自動適應並剝除可能存在的 wrapper 前綴 (如 base_model.)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("base_model."):
            cleaned_state_dict[k[len("base_model."):]] = v
        else:
            cleaned_state_dict[k] = v
            
    try:
        model.load_state_dict(cleaned_state_dict)
        print("✅ 權重成功載入！")
    except Exception as e:
        print(f"❌ 載入權重失敗，錯誤訊息：\n{e}")
        sys.exit(1)
        
    model = model.cuda().eval().to(torch.bfloat16)
    
    # 預設測試案例
    test_cases = [
        "請用 Python 寫一個以 Manacher 演算法尋找最長迴文子字串（Longest Palindromic Substring）的函數，時間複雜度需為 O(n)。",
        "一個袋子有 3 個紅球與 7 個藍球。若不放回地連續抽取兩球，兩球皆為紅球的機率是多少？請逐步推導。",
        "請分析以下 Python 程式碼的錯誤並給出正確寫法：\ndef append_to(element, target=[]):\n    target.append(element)\n    return target",
        "自訂測試（手動輸入您的 Prompt）"
    ]
    
    while True:
        print("\n📢 請選擇測試 Case:")
        for idx, case in enumerate(test_cases):
            print(f"  [{idx + 1}] {case[:60]}...")
            
        try:
            choice_str = input("請輸入編號 (或輸入 q 退出): ").strip()
            if choice_str.lower() == 'q':
                break
            choice = int(choice_str) - 1
            if choice < 0 or choice >= len(test_cases):
                print("❌ 無效選擇")
                continue
        except ValueError:
            print("❌ 請輸入數字")
            continue
            
        if choice == len(test_cases) - 1:
            prompt = input("\n請輸入您的測試 Prompt: ").strip()
        else:
            prompt = test_cases[choice]
            
        if not prompt:
            print("❌ Prompt 不能為空")
            continue
            
        print(f"\n🚀 輸入的 Prompt: {prompt}\n")
        generate_response(model, tokenizer, prompt)

if __name__ == "__main__":
    main()
