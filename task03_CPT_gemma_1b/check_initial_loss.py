import os
import sys

# 🚀 修正：對齊 1080 Ti 的硬體環境與索引
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer

# 確保從正確的模組匯入我們升級後的包裝器與拓撲
from train import AGIV2GForCausalLM, transplant_and_freeze
from AGIV2G import AGIV2G

def verify_initial_state_with_qa():
    # 動態設備偵測
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("=" * 50)
    print(f"🖥️ 當前掛載運算設備: {device}")
    print("=" * 50)

    # 指定為 Gemma 3 1B 官方模型
    model_id = "google/gemma-3-1b-it"
    
    print("⏳ [1/4] 載入 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print(f"⏳ [2/4] 初始化 AGIV2G 模型並掛載 {model_id} 權重...")
    # 🚀 修正：鎖死物理詞表大小 262144 與 26 層結構
    base = AGIV2G(vocab_size=262144, D=1152, hidden_dim=6912, num_blocks=26)
    base = transplant_and_freeze(model_id, base)
    
    # 🚀 修正：強制降維至 fp16 (float16) 以完美匹配 Pascal 架構
    model = AGIV2GForCausalLM(base).to(torch.float16).to(device)
    model.eval() # 測試初始 Loss 時，關閉 Dropout 與 GC 干擾，確保絕對零狀態

    print("⏳ [3/4] 載入簡單邏輯與數學 QA 測試語料...")
    # 使用極度規律的內容，這對擁有強大常識的 Gemma 來說應該要有極低的 Loss
    simple_qa_text = (
        "Question: What is 1 + 1?\nAnswer: 2.\n"
        "Question: What is 5 + 5?\nAnswer: 10.\n"
        "Question: What is 10 + 10?\nAnswer: 20.\n"
        "Question: What is the capital of Japan?\nAnswer: Tokyo.\n"
        "Question: What is the capital of France?\nAnswer: Paris.\n"
    )
    
    tokens = tokenizer(simple_qa_text, return_tensors="pt")["input_ids"].to(device)
    labels = tokens.clone()

    print(f"⏳ [4/4] 執行單步前向傳播 (序列長度: {tokens.size(1)})...")
    with torch.no_grad():
        outputs = model(input_ids=tokens, labels=labels)
        
    loss = outputs["loss"].item()
    
    # 觀測閘控初始值 (Zero-Gating 機制驗證)
    gate_fft_vals = [b.gate_fft.item() for b in base.blocks if hasattr(b, 'gate_fft')]
    gate_mem_vals = [b.gate_mem.item() for b in base.blocks if hasattr(b, 'gate_mem')]
    
    avg_gate_fft = sum(gate_fft_vals) / len(gate_fft_vals) if gate_fft_vals else 0.0
    avg_gate_mem = sum(gate_mem_vals) / len(gate_mem_vals) if gate_mem_vals else 0.0

    print("\n" + "="*50)
    print("🎯 物理狀態與初始條件驗證報告 (QA 測試)")
    print("="*50)
    print(f"🔹 測試文本: \n{simple_qa_text.strip()}")
    print("-" * 50)
    # 若拓撲完美，Loss 應該低於 2.0
    print(f"🔹 初始 Loss 值: {loss:.4f}")
    # 驗證新架構特徵空間尚未干擾原始邏輯
    print(f"🔹 閘控 FFT 平均: {avg_gate_fft:.6f} (預期: 0.000000)")
    print(f"🔹 閘控 Mem 平均: {avg_gate_mem:.6f} (預期: 0.000000)")
    print("="*50)

if __name__ == "__main__":
    verify_initial_state_with_qa()