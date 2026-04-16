"""
genAC.py — AC 範式區塊並行生成腳本 (N -> N+B Block Generation)
===========================================================
核心機制：
1. 物理隔離推理：利用提示詞 N 生成一次 M_global，直接產出未來 B 個 Token。
2. 靜態全息快照：在生成 Chunk 期間，M_global 保持不變，實現 O(1) 推理。
3. 權重自動對齊：自動修正與載入訓練時的 best_model 權重。
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

# --- 配置參數 (對齊 Roadmap 階段一) ---
MODEL_ID = "google/gemma-3-1b-it"
CHECKPOINT_PATH = "./agiv2_zerogate_ac_checkpoints/best_model.pth"
BLOCK_SIZE = 512  # 對齊 trainRouterAC.py 的 b_size
MAX_TOTAL_LEN = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        
        # 自動處理 Trainer 儲存時可能產生的 prefix (base_model. 或 model.)
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
def generate_ac(prompt_text, max_gen=1024):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load_ac_model()
    
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)
    cur_len = input_ids.shape[1]
    
    print(f"\n🚀 [開始生成] 初始長度 N={cur_len}，採用 N -> N+B 區塊並行模式...")
    print("-" * 50)
    print(f"PROMPT: {prompt_text}", end="", flush=True)

    generated_ids = input_ids
    
    while generated_ids.shape[1] < MAX_TOTAL_LEN and (generated_ids.shape[1] - cur_len) < max_gen:
        # [Step 1] 物理隔離：將當前 N 作為 X_past，擴增 B 個 Slot 作為 X_future
        N_len = generated_ids.shape[1]
        # 填充 PAD 作為未來 Slot，防止因果洩漏 (雖然模型已無遮罩，但 PAD 能保持頻譜穩定)
        pad_ids = torch.full((1, BLOCK_SIZE), tokenizer.pad_token_id, device=DEVICE)
        model_input = torch.cat([generated_ids, pad_ids], dim=1)
        
        # [Step 2] 全局共振：模型在內部生成一次 M_global，並一次性預測整個區塊
        logits = model(model_input)
        
        # [Step 3] 區塊解碼：取最後 B 個位置的 Logits
        # 在 AC 訓練中，標籤是對齊未來 B 個位置的
        next_block_logits = logits[0, N_len : N_len + BLOCK_SIZE, :]
        
        # 進行取樣 (可以換成 greedy: torch.argmax)
        next_block_ids = []
        for i in range(BLOCK_SIZE):
            # 簡單 Greedy 解碼提取區塊內容
            token_id = torch.argmax(next_block_logits[i], dim=-1).item()
            next_block_ids.append(token_id)
            
            # 檢查是否提前遇到 EOS
            if token_id == tokenizer.eos_token_id:
                break
        
        # [Step 4] 刷新序列：將產生的整個區塊併入，準備下一次 M_global 更新
        new_tokens_tensor = torch.tensor([next_block_ids], device=DEVICE)
        generated_ids = torch.cat([generated_ids, new_tokens_tensor], dim=1)
        
        # 即時顯示
        output_text = tokenizer.decode(next_block_ids, skip_special_tokens=True)
        print(output_text, end="", flush=True)
        
        if tokenizer.eos_token_id in next_block_ids:
            print("\n\n[INFO] 檢測到 EOS，生成停止。")
            break

    print("\n" + "-" * 50)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    test_prompt = "分析主體 'Quantum Router' 的數據流。警告：若撤銷門開啟，請立即停止。"
    final_output = generate_ac(test_prompt)