import os
import torch
from transformers import AutoTokenizer

# 關閉 bitsandbytes 煩人的歡迎訊息
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# ⚠️ 注意：請確保 train.py 裡面的 transplant_and_freeze 函數：
# (1) 迴圈已改為 for i in range(26):
# (2) 載入精度 dtype 已改為 torch.float16
from train import AGIV2GForCausalLM, transplant_and_freeze
from AGIV2G import AGIV2G

def test_generation():
    # 🚀 動態設備偵測：自動判斷硬體環境並進行指派
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("=" * 50)
    print(f"🖥️ 當前掛載運算設備: {device}")
    print("=" * 50)

    model_id = "google/gemma-3-1b-it"
    
    print("⏳ [1/3] 載入 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    print(f"⏳ [2/3] 初始化 AGIV2G 模型並掛載 {model_id} 權重...")
    # 鎖死物理詞表大小為 262144
    base = AGIV2G(vocab_size=262144, D=1152, hidden_dim=6912, num_blocks=26)
    
    # 進行權重移植與凍結
    base = transplant_and_freeze(model_id, base)
    
    # 🚀 將模型精準推送至目標設備 (GPU 或 CPU)
    model = AGIV2GForCausalLM(base).to(torch.float16).to(device)
    model.eval() # 關閉 Dropout，並確保 Zero-Gating 處於純粹的 0 狀態

    # 測試提示詞 (Prompt)
    prompt = "The capital of France is Paris. The capital of Japan is"
    
    # 🚀 將輸入張量推送至目標設備
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    print(f"\n⏳ [3/3] 開始生成 (Prompt: '{prompt}')...")
    print("-" * 50)
    print(prompt, end="") # 先印出 prompt

    max_new_tokens = 10
    
    # 最純粹的 Greedy Decoding (貪婪解碼) 迴圈
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # 取得目前的 logits (只拿最後一個 token 的預測結果)
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs["logits"][:, -1, :] 
            
            # 找出機率最大的一個字
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # 將新預測的字接回 input_ids，準備預測下一個字
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            # 印出這個新預測的字
            next_word = tokenizer.decode(next_token_id[0])
            print(next_word, end="", flush=True)
            
            # 如果預測出 EOS token，代表模型認為句子結束了
            if next_token_id.item() == tokenizer.eos_token_id:
                break
                
    print("\n" + "-" * 50)
    print("🎯 生成測試完畢。")

if __name__ == "__main__":
    test_generation()