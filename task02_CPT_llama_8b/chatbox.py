import os
import sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AGIV2L import AGIV2L
from train import transplant_and_freeze

# GPU 1 對齊 train.py
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ==========================================
# 1. 補齊與 Train 完全相同的模型外殼 (Llama 版)
# ==========================================
class AGIV2LForCausalLM(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids):
        hidden_states = self.base_model.embedding(input_ids)
        # 注意：Llama 架構不用像 Gemma 那樣乘以 math.sqrt(D)
        
        for i, block in enumerate(self.base_model.blocks):
            shift_size = (block.C // 2) if (i % 2 != 0) else 0
            hidden_states = block(hidden_states, shift_size=shift_size)
                
        hidden_states = self.base_model.final_norm(hidden_states)
        logits = self.base_model.fc_out(hidden_states)
            
        return {"logits": logits}

# ==========================================
# 2. 流水式文字產生 (Token-by-Token)
# ==========================================
@torch.no_grad()
def stream_generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_k=50, device="cuda"):
    model.eval()
    
    # Text Generation Template (針對 Llama 3 基礎版)
    # 若是 Instruct 版可以用 apply_chat_template，基礎版用傳統對話格式引導
    formatted_prompt = f"User: {prompt}\nAssistant: "
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
    
    print("🤖 模型: ", end="", flush=True)
    generated_text = ""
    
    for _ in range(max_new_tokens):
        outputs = model(input_ids)
        logits = outputs["logits"][:, -1, :]  # 取最後一個 Token 的 Logits
        
        if temperature > 0.0:
            logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # 轉換回文字並印出
        token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)
        print(token_str, end="", flush=True)
        generated_text += token_str
        
        # 如果模型吐出結束符號，則停止
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    print("\n")
    return generated_text

# ==========================================
# 3. 主程式 Chatbox
# ==========================================
def main():
    model_id = "NousResearch/Meta-Llama-3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔄 正在初始化 Tokenizer ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("🔄 正在建立模型架構...")
    base = AGIV2L(vocab_size=len(tokenizer), D=4096, hidden_dim=14336, num_blocks=32)
    model = AGIV2LForCausalLM(base).to(torch.bfloat16).to(device)
    
    # 嘗試尋找訓練過的安全存檔
    save_dir = "./agiv2-cpt"
    final_path = os.path.join(save_dir, "final.pth")
    best_path = os.path.join(save_dir, "best_model.pth")
    
    loaded = False
    for path in [best_path, final_path]:
        if os.path.exists(path):
            print(f"📦 偵測到訓練權重！正在載入 {path} ...")
            model.load_state_dict(torch.load(path, map_location=device), strict=False)
            loaded = True
            break
            
    if not loaded:
        print("⚠️ 未偵測到訓練好的權重 (final.pth / best_model.pth)")
        print("🔄 改為載入 Llama 3 初始預訓練權重做示範...")
        base = transplant_and_freeze(model_id, base)
        model = AGIV2LForCausalLM(base).to(torch.bfloat16).to(device)

    print("=========================================================")
    print("🎉 Llama-8B Chatbox 就緒！輸入你的對話 (輸入 'quit' 或 'exit' 離開)")
    print("=========================================================")
    
    while True:
        try:
            user_input = input("\n👤 提問: ")
            if user_input.strip().lower() in ['quit', 'exit']:
                print("👋 掰掰！")
                break
            if not user_input.strip():
                continue
                
            stream_generate(
                model=model, 
                tokenizer=tokenizer, 
                prompt=user_input, 
                max_new_tokens=300, 
                temperature=0.7, 
                top_k=50,
                device=device
            )
            
        except KeyboardInterrupt:
            print("\n👋 掰掰！")
            break

if __name__ == "__main__":
    main()
