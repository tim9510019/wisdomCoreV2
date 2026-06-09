import os
import sys
import torch
import torch.nn.functional as F

# 載入 GEMMA4
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from GEMMA4 import GEMMA4
from inference import get_safetensors_path, transplant_gemma4
from transformers import AutoTokenizer

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safetensors_path = get_safetensors_path("google/gemma-4-E2B-it")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")

    # 初始化模型
    model = GEMMA4(
        vocab_size=262144,
        D=1536,
        hidden_dim=12288,
        num_blocks=35,
        C=512,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        image_token_id=258880,
        audio_token_id=258881,
        patch_size=16,
        in_channels=3
    ).to(device)

    model = transplant_gemma4(safetensors_path, model)
    model = model.to(dtype=torch.bfloat16)
    model.eval()

    # 構造 Prompt
    messages = [
        {"role": "user", "content": "What is the capital city of France? Answer in one word."}
    ]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)
    input_ids = torch.tensor([encoded["input_ids"]]).to(device)

    print("\n" + "="*50)
    print(f"Prompt IDs: {input_ids[0].tolist()}")
    print(f"Decoded Prompt: {tokenizer.decode(input_ids[0].tolist())}")
    print("="*50)

    # 測試多種 Scaling 策略
    scaling_strategies = {
        "1. Standard Gemma scale (math.sqrt(D))": lambda d: math.sqrt(d),
        "2. No scale (1.0)": lambda d: 1.0,
        "3. Inverse scale (1.0 / math.sqrt(D))": lambda d: 1.0 / math.sqrt(d),
        "4. D scale (D)": lambda d: d
    }

    import math
    for name, scale_fn in scaling_strategies.items():
        scale = scale_fn(model.D)
        print(f"\n🧪 Testing Strategy: {name} (scale factor = {scale:.6f})")
        
        # 覆寫 forward 過程中的 embed_multimodal 進行測試
        with torch.no_grad():
            # 我們手動跑一下前向傳播的一部分
            # 1. 取得 PLE
            pl_embeds = model.embed_multimodal_per_layer(input_ids)
            
            # 2. 取得文字 Embedding 並乘上測試的 scale
            x_text = model.embedding(input_ids) * scale
            
            # 3. 逐層前向傳播
            out = x_text
            for i, block in enumerate(model.blocks):
                layer_pl = pl_embeds[..., i * 256 : (i + 1) * 256]
                out = block(out, per_layer_embed=layer_pl)
                
            out = model.final_norm(out)
            logits = model.fc_out(out)
            
            # 應用 capping
            logits = logits / 30.0
            logits = torch.tanh(logits) * 30.0
            
            # 獲取最後一個預測位置
            next_token_logits = logits[0, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            
            # 取得 Top-5 tokens
            top_probs, top_indices = torch.topk(probs, 5)
            for rank in range(5):
                token_id = top_indices[rank].item()
                prob = top_probs[rank].item()
                token_text = tokenizer.decode([token_id])
                print(f"   Rank {rank+1}: Token ID {token_id:6d} | Prob: {prob*100:6.2f}% | Decoded: '{token_text}'")

if __name__ == "__main__":
    test()
