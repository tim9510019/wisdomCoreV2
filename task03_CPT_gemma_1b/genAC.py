"""
genAC.py — AGIV2 嚴格自迴歸生成腳本 (終極防禦版)
===========================================================
核心機制修正：
1. 物理因果律：採用嚴格 Token-by-Token 生成。
2. 空間對齊裝甲：動態 Padding 確保序列長度永遠是 Chunk Size 的倍數，防止頻率碎裂。
3. 破冰採樣：引入溫度與 Top-K 採樣，並加上重複懲罰，粉碎數字迴圈。
4. 乾草堆模擬：在 Main 區塊模擬長文本環境，觸發 M_global 路由機制。
"""
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# 確保路徑正確
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from AGIV2GAC import AGIV2G

# --- 配置參數 ---
MODEL_ID = "google/gemma-3-1b-it"
CHECKPOINT_PATH = "./agiv2_zerogate_ac_checkpoints_1K/best_model.pth"
MAX_TOTAL_LEN = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 256  # 必須與你訓練時的 Chunk Size 設定對齊

def load_ac_model():
    print(f"🧬 [載入] 正在初始化 AGIV2GAC 物理隔離架構...")
    # 初始化與訓練時完全一致的架構
    model = AGIV2G(
        vocab_size=262144, 
        D=1152, 
        hidden_dim=6912, 
        num_blocks=26,
        head_dim=256
    )
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"📬 [權重] 正在從 {CHECKPOINT_PATH} 載入最佳模型...")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        # 自動處理 Trainer 儲存時可能產生的 prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("base_model.model.", "").replace("base_model.", "").replace("model.", "")
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ 權重對齊完成。")
    else:
        print(f"⚠️ 找不到權重檔：{CHECKPOINT_PATH}，將使用隨機初始化狀態。")
        
    return model.to(DEVICE).to(torch.bfloat16).eval()

@torch.no_grad()
def generate_ac_robust(prompt_text, max_gen=64, chunk_size=CHUNK_SIZE):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load_ac_model()
    
    # 確保有可用的 pad_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)
    print(f"\n🚀 [開始生成] 初始長度 N={input_ids.shape[1]}")
    print("-" * 50)
    
    # 為了版面整潔，不印出幾千字的雜訊，只印出最後的問題
    display_prompt = prompt_text.split("結合主體")[-1]
    print(f"結合主體{display_prompt}", end="", flush=True)

    generated_ids = input_ids
    
    for step in range(max_gen):
        current_len = generated_ids.shape[1]
        
        # [防護 1：動態空間填充] 確保長度能被 chunk_size 整除，維持物理相位完美
        pad_len = (chunk_size - (current_len % chunk_size)) % chunk_size
        if pad_len > 0:
            pad_tensor = torch.full((1, pad_len), pad_id, device=DEVICE)
            model_input = torch.cat([generated_ids, pad_tensor], dim=1)
        else:
            model_input = generated_ids
            
        # 前向傳播
        logits = model(model_input)
        
        # 精準提取 "真實序列的最後一個 Token" 的 Logit (扣除 padding 的影響)
        next_token_logits = logits[0, current_len - 1, :]
        
        # [防護 2：重複懲罰 (Repetition Penalty)]
        # 強制壓制最近 20 個輸出過的 Token 機率，阻斷復讀機
        for prev_id in generated_ids[0, -20:].tolist():
            # 針對 logits 大於 0 的部分進行懲罰
            if next_token_logits[prev_id] > 0:
                next_token_logits[prev_id] /= 1.5 
            else:
                next_token_logits[prev_id] *= 1.5
            
        # [防護 3：Top-K 採樣] 放棄貪婪搜索，引入微小熱力學隨機性
        temperature = 0.7
        top_k = 10
        
        next_token_logits = next_token_logits / temperature
        values, indices = torch.topk(next_token_logits, top_k)
        probs = F.softmax(values, dim=-1)
        
        next_token_index = torch.multinomial(probs, num_samples=1)
        next_token_id = indices[next_token_index].item()
        
        # 輸出
        output_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
        print(output_text, end="", flush=True)
        
        # 遇到 EOS 停止
        if next_token_id == tokenizer.eos_token_id:
            print("\n\n[INFO] 檢測到 EOS，生成停止。")
            break
            
        # 拼接回原序列
        next_token_tensor = torch.tensor([[next_token_id]], device=DEVICE)
        generated_ids = torch.cat([generated_ids, next_token_tensor], dim=1)

    print("\n" + "-" * 50)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # 1. 模擬訓練時的背景雜訊 (把長度墊高，模擬 32K 乾草堆，避免模型無特徵可抓)
    background_noise = "System log initialization. Process running normally. No errors detected. " * 300
    
    # 2. 注入目標實體與解碼值 (這就是那根 "針")
    needle = " \n[SECURE DATA] 實體標識為「Quantum Router」。該實體的解碼核心值為：TOKEN_9921。\n "
    
    # 3. 繼續加入干擾雜訊
    more_noise = "Memory dump complete. Buffer allocated. Awaiting next instruction sequence. " * 300
    
    # 4. 提出最終問題
    question = "\n結合主體 'Quantum Router' 與其對應的解碼值，請給出最終執行碼。若邏輯門顯示無效，回應 'REDACTED'。結果："
    
    # 組合完整的 Prompt
    full_test_prompt = background_noise + needle + more_noise + question
    
    # 執行推論
    final_output = generate_ac_robust(full_test_prompt, max_gen=32)