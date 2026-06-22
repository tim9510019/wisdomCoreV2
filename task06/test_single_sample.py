import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_section(title):
    print("\n" + "="*80)
    print(f"🔬 {title}")
    print("="*80 + "\n")

def main():
    print_section("DNA Helix Ultimate 16B 單樣本診斷與過擬合測試器")

    # 1. 載入 Tokenizer
    tokenizer_id = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    print(f"🔄 正在載入 Tokenizer: {tokenizer_id}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    # 2. 讀取數據集中的第一筆樣本
    parquet_path = os.path.expanduser("~/task06_data/train_type_A_combined.parquet")
    if not os.path.exists(parquet_path):
        print(f"❌ 找不到訓練 Parquet 文件：{parquet_path}")
        sys.exit(1)

    print(f"📦 正在從 {parquet_path} 讀取第一筆數據...")
    table = pq.read_table(parquet_path)
    
    # 取得第一筆數據
    row = table.to_batches()[0].to_pydict()
    input_ids_list = row["input_ids"][0]
    n_split = row["n_split_index"][0]
    
    prompt_ids = input_ids_list[:n_split]
    target_ids = input_ids_list[n_split:]
    
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    target_text = tokenizer.decode(target_ids, skip_special_tokens=False)
    
    print("\n--- 📖 [訓練集樣本分析] ---")
    print(f"總 Token 長度: {len(input_ids_list)}")
    print(f"Prompt 切分點 (n_split_index): {n_split}")
    print(f"Prompt (前 {n_split} tokens):")
    print(repr(prompt_text))
    print(f"\nTarget Answer (剩餘 {len(target_ids)} tokens):")
    print(repr(target_text))
    print("-------------------------\n")

    # 3. 初始化模型
    print("🧬 正在初始化 DNA Helix Ultimate 16B 模型結構...")
    from GEMMA3 import DNAHelixUltimate
    model = DNAHelixUltimate(
        vocab_size=151936,
        D=4096,
        C=2048,
        hidden_dim=16384,
        num_blocks=40,
    )

    # 4. 載入現有權重進行測試
    default_ckpt_path = os.path.expanduser("~/task06_checkpoints/best_distill_model.pth")
    ckpt_path = input(f"請輸入模型權重檔案路徑（測試推理，或按 Enter 跳過載入）: ").strip()
    
    if ckpt_path:
        if os.path.exists(ckpt_path):
            print(f"💾 正在載入權重檔案: {ckpt_path}...")
            state_dict = torch.load(ckpt_path, map_location="cpu")
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
        else:
            print(f"⚠️ 找不到權重檔案 {ckpt_path}，將使用隨機權重進行測試。")
            
    model = model.cuda()
    
    # 5. 測試現有模型在該 Prompt 下的輸出
    print("\n🚀 [測試 1: 現有模型推理生成]")
    model.eval()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long).cuda()
    generated_ids = []
    
    with torch.no_grad():
        for i in range(len(target_ids) + 10): # 多生成 10 個 token 觀察
            logits = model(input_ids)
            next_token = torch.argmax(logits[0, -1, :]).item()
            generated_ids.append(next_token)
            
            token_str = tokenizer.decode([next_token], skip_special_tokens=False)
            print(token_str, end="", flush=True)
            
            if next_token == tokenizer.eos_token_id or next_token == 151643: # eos or pad
                break
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long).cuda()], dim=-1)
            
    print("\n\n" + "-"*40)

    # 6. 單樣本過擬合測試 (Diagnostic Overfitting Loop)
    print("\n🔥 [測試 2: 單樣本過擬合（Overfitting）診斷訓練] ...")
    print("本測試將使用梯度下降法對該樣本進行 100 步的極速訓練。")
    print("若 Loss 能順利下降至趨近於 0，證明模型的前向傳播與反向傳播結構無 Bug；若 Loss 停滯或 NaN，代表架構代碼有 Bug。")
    
    # 開啟訓練模式與梯度
    for p in model.parameters():
        p.requires_grad = True
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # 準備訓練 Batch (Batch size = 1)
    train_ids = torch.tensor([input_ids_list], dtype=torch.long).cuda()
    labels = torch.tensor([[-100] * n_split + target_ids], dtype=torch.long).cuda()
    
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    
    for step in range(1, 101):
        optimizer.zero_grad()
        
        # 前向傳播
        logits = model(train_ids)
        
        # 計算偏移
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        if torch.isnan(loss):
            print(f"❌ Step {step}: Loss 變為 NaN！架構中存在數值溢出或 Bug。")
            break
            
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0 or step == 1:
            print(f"   Step {step:3d} / 100 | Loss: {loss.item():.6f}")

    print("\n🎉 過擬合訓練結束！")
    
    # 7. 再次測試推理輸出
    print("\n🚀 [測試 3: 過擬合後模型推理生成]")
    model.eval()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long).cuda()
    
    with torch.no_grad():
        for i in range(len(target_ids) + 10):
            logits = model(input_ids)
            next_token = torch.argmax(logits[0, -1, :]).item()
            token_str = tokenizer.decode([next_token], skip_special_tokens=False)
            print(token_str, end="", flush=True)
            if next_token == tokenizer.eos_token_id or next_token == 151643:
                break
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long).cuda()], dim=-1)
            
    print("\n\n" + "="*80)

if __name__ == "__main__":
    main()
